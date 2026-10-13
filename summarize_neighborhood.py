"""Common-sampling validation summary for the frozen neighborhood ablation."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import ROOT
from completion import load_checkpoint
from training_data import read_manifest, dataset_fingerprint, evaluate_entries
from organize_figures import organize


def main():
    torch.set_num_threads(2)
    dataset=ROOT/'outputs/datasets/bronze_20261013_seed42'
    output=ROOT/'outputs/repeats/bronze_neighborhood_confirmation'
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    manifest=read_manifest(dataset); fingerprint=dataset_fingerprint(dataset,manifest)
    entries=[e for e in manifest['samples'] if e['split']=='val']
    variants={'wide':'bronze_wide1140','pointwise_xyz':'bronze_pointwise_xyz','local':'bronze_local'}
    rows=[]
    for seed in protocol['training_seeds']:
        for label,prefix in variants.items():
            run=ROOT/'outputs/runs'/f'{prefix}_seed{seed}'
            last=torch.load(run/'last.pt',weights_only=True,map_location='cpu')
            assert last['epoch']==20 and last['dataset_fingerprint']==fingerprint
            for field,value in dict(seed=seed,input_points=512,output_points=128,target_points=128,batch_size=8,lr=.001,amp=False,output_mode='tanh',boundary_weight=0.,decoder_width=1140).items():
                assert last['config'][field]==value
            assert last['config'].get('architecture','mlp')==('mlp' if label=='wide' else label)
            assert not last['config'].get('repulsion_weight',0)
            if label!='wide':
                assert last['config']['encoder_anchors']==32 and last['config']['encoder_neighbors']==16
            model,checkpoint=load_checkpoint(run/'best.pt')
            assert checkpoint['selection_split']=='val'
            metrics=evaluate_entries(model,dataset,entries,512,43,torch.device('cpu'))
            if not (run/'evaluation_val_common43.json').exists():
                (run/'evaluation_val_common43.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
            rows.append(dict(seed=seed,label=label,best_epoch=checkpoint['epoch'],parameters=sum(p.numel() for p in model.parameters()),
                             cd=metrics['mean_local_cd_squared'],fscore=metrics['mean_local_fscore'],model_metrics=metrics['models']))
    assert {r['parameters'] for r in rows if r['label']!='wide'}=={615236}
    means={label:{field:dict(mean=float(np.mean([r[field] for r in rows if r['label']==label])),
                           sample_std=float(np.std([r[field] for r in rows if r['label']==label],ddof=1))) for field in ('cd','fscore')} for label in variants}
    pairs=[]
    for seed in protocol['training_seeds']:
        selected={r['label']:r for r in rows if r['seed']==seed}
        pairs.append(dict(seed=seed,cd_reduction_vs_control=1-selected['local']['cd']/selected['pointwise_xyz']['cd'],
                          cd_reduction_vs_wide=1-selected['local']['cd']/selected['wide']['cd']))
    qualities={}
    for label,prefix in variants.items():
        path=ROOT/'outputs/runs'/f'{prefix}_seed42'/'quality_val.json'
        qualities[label]=json.loads(path.read_text(encoding='utf-8'))['summary']
    reduction=1-means['local']['cd']['mean']/means['pointwise_xyz']['cd']['mean']
    report=dict(protocol=protocol,rows=rows,means=means,paired=pairs,seed42_quality=qualities,
                reduction_vs_matched_control=reduction,
                limitation='Two fixed validation objects; no independent object replication or blind test. Different best epochs are selected per native validation sampling, final comparison uses seed43.')
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    figures=ROOT/'outputs/experiment_figures'
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,label in enumerate(variants):
        selected=[r for r in rows if r['label']==label]
        for ax,field in zip(axes,('cd','fscore')):
            ax.bar(np.arange(3)+(i-1)*.25,[r[field] for r in selected],width=.25,label=label)
            ax.set(xticks=np.arange(3),xticklabels=['42','43','44'],xlabel='Training seed',ylabel=field)
            ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Neighborhood encoding / 20 epochs / same decoder / evaluation seed43')
    fig.tight_layout();fig.savefig(figures/'bronze_neighborhood_three_seeds.png',dpi=180);plt.close(fig)
    lines=['# 邻域几何编码：参数相同的消融','',
           '## 文献与改造范围','',
           '[PointNet++论文第3.2节](https://proceedings.neurips.cc/paper_files/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)与[作者sample_and_group / pointnet_sa_module](https://github.com/charlesq34/pointnet2/blob/master/utils/pointnet_util.py)是邻域分组与局部池化的依据。原任务是分类/分割；本实验是局部补全改造，不是完整PointNet++复现。','',
           'local：可见输入512点→FPS32中心→每中心16个kNN（包含中心）→相对中心坐标3→64→128 MLP→邻域max→拼接中心在缺口坐标中的XYZ→131→128 MLP→全局max；解码器沿用132→1140→384。','',
           'pointwise_xyz：输入每点绝对缺口坐标3→64→128，拼接该点XYZ→131→128→全局max；与local完全同参数615236、同解码器及同种子初始权重。它用于容量对照。既有wide为614852参数，差384。','',
           '工程选择：单尺度32/16、kNN、无BN/MSG/特征传播、最近球心点启动的确定性FPS、半径坐标、球内tanh与CD/R²。这些固定预算选择不是原论文最佳超参数。kNN局部物理尺度会随密度变化，不能称为固定半径邻域。','',
           '三种子42/43/44，各20轮，Adam lr0.001，batch8；数据与输入采样和损失固定，无排斥/边缘项。每运行以验证CD²选择best，最终统一evaluation seed43。协议在训练前保存，未根据结果搜索邻域数。','',
           '## 全验证集结果','',
           '|种子|方案|最佳轮|CD²↓|F-score↑|','|---|---|---:|---:|---:|']
    lines += [f'|{r["seed"]}|{r["label"]}|{r["best_epoch"]}|{r["cd"]:.8f}|{r["fscore"]:.6f}|' for r in rows]
    lines += ['','|方案|CD²均值±标准差|F-score均值±标准差|','|---|---:|---:|']
    for label,m in means.items():
        lines.append(f'|{label}|{m["cd"]["mean"]:.8f} ± {m["cd"]["sample_std"]:.8f}|{m["fscore"]["mean"]:.6f} ± {m["fscore"]["sample_std"]:.6f}|')
    lines += ['',f'local相对同参数pointwise_xyz的平均CD²降低：{reduction:.2%}（负值表示变差）。每个种子的配对结果见summary.json，标准差不表示器物级置信区间。',
              f'结论：本轮local平均CD²比同参数逐点对照高{-reduction:.2%}，三个种子均变差，F-score也下降。seed42聚集比例与离面距离均更差，因此当前轻量邻域改造未获得支持，不替换原型。这个负结果不否定完整PointNet++，也不能证明局部几何没有价值。',
              '当前local训练误差也高于逐点对照，可能涉及信息汇总方式或训练预算；尚未分离这些原因。下一步如做收敛对照，应同步延长两个方案预算，不针对单个方案选择性续训来宣称结构收益。','',
              '## seed42全40缺口的补充诊断','',
              '|方案|DCD↓|聚集比例↓|网格距离/R↓|','|---|---:|---:|---:|']
    for label,q in qualities.items():
        lines.append(f'|{label}|{q["dcd_equal_points_mean"]:.6f}|{q["prediction_close_fraction_0_02_radius"]:.2%}|{q["prediction_mesh_mean_distance_over_radius"]:.6f}|')
    lines += ['','网格仅用于评价，未用于训练、选取邻域或预测投影。聚集比例阈值0.02R是工程诊断。补充诊断只覆盖seed42，不能推广为三个种子的密度/离面结论。','',
              '![三种子对照](../../experiment_figures/bronze_neighborhood_three_seeds.png)','',
              '[同缺口三方案图与逐案例指标](../../diagnostics/neighborhood_cases_seed42/README.md)','',
              '目前只有两个验证器物，而且验证集已参与多轮方法探索；三训练种子不能补足器物数量，已见过的测试集也不能再作为全新盲测证据。已交付原型保持冻结。']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (figures/'bronze_neighborhood_notes.md').write_text('# 邻域编码图片说明\n\nbronze_neighborhood_three_seeds.png：相同解码器的邻域编码、逐点容量对照与宽MLP三种子验证结果，左CD²右F-score。用于方法改进消融段。\n\n[完整结果](../repeats/bronze_neighborhood_confirmation/README.md)',encoding='utf-8')
    organize(); print(json.dumps(dict(means=means,paired=pairs,seed42_quality=qualities),indent=2))


if __name__=='__main__':
    main()
