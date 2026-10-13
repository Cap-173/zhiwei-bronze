"""固定验证采样汇总分层解码的容量对照与重复种子。"""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from completion import load_checkpoint
from main import ROOT
from organize_figures import organize
from training_data import dataset_fingerprint, evaluate_entries, read_manifest

DATASET = ROOT/'outputs/datasets/bronze_20261013_seed42'
OUTPUT = ROOT/'outputs/repeats/bronze_hierarchy_confirmation'
VARIANTS = {'mlp': 'bronze_cd128', 'wide': 'bronze_wide1140', 'hier_aux': 'bronze_hier32_aux01'}


def summarize():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    manifest=read_manifest(DATASET)
    fingerprint=dataset_fingerprint(DATASET,manifest)
    entries=[e for e in manifest['samples'] if e['split']=='val']
    torch.set_num_threads(2)
    rows=[]
    for seed in [protocol['exploratory_seed']]+protocol['confirmation_seeds']:
        for label,prefix in VARIANTS.items():
            run=ROOT/'outputs/runs'/f'{prefix}_seed{seed}'
            last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
            assert last['epoch']==20 and last['dataset_fingerprint']==fingerprint
            assert last['config']['seed']==seed
            for field,value in dict(input_points=512,output_points=128,target_points=128,batch_size=8,lr=.001,amp=False,output_mode='tanh',boundary_weight=0.).items():
                assert last['config'][field]==value,(run,field)
            if label=='hier_aux':
                assert last['config']['architecture']=='hierarchical' and last['config']['coarse_weight']==.01
            else:
                assert last['config'].get('architecture','mlp')=='mlp'
                assert last['config'].get('decoder_width',256)==(1140 if label=='wide' else 256)
            model,checkpoint=load_checkpoint(run/'best.pt')
            assert checkpoint['selection_split']=='val'
            metrics=evaluate_entries(model,DATASET,entries,512,protocol['evaluation_seed'],torch.device('cpu'))
            evaluation=run/'evaluation_val_common43.json'
            if not evaluation.exists():
                evaluation.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
            rows.append(dict(seed=seed,label=label,run=str(run),best_epoch=checkpoint['epoch'],
                             parameters=sum(p.numel() for p in model.parameters()),
                             cd_squared=metrics['mean_local_cd_squared'],fscore=metrics['mean_local_fscore']))
    aggregates={}
    for label in VARIANTS:
        selected=[r for r in rows if r['label']==label]
        aggregates[label]={field:dict(mean=float(np.mean([r[field] for r in selected])),
                                     sample_std=float(np.std([r[field] for r in selected],ddof=1)))
                           for field in ('cd_squared','fscore')}
    paired=[]
    for seed in (42,43,44):
        selected={r['label']:r for r in rows if r['seed']==seed}
        paired.append(dict(seed=seed,
                           reduction_vs_small=1-selected['hier_aux']['cd_squared']/selected['mlp']['cd_squared'],
                           reduction_vs_wide=1-selected['hier_aux']['cd_squared']/selected['wide']['cd_squared']))
    confirmatory={label:{field:float(np.mean([r[field] for r in rows if r['label']==label and r['seed'] in protocol['confirmation_seeds']]))
                        for field in ('cd_squared','fscore')} for label in VARIANTS}
    report=dict(protocol=protocol,rows=rows,aggregates=aggregates,paired=paired,
                confirmatory=confirmatory,
                note='seed42 exploratory; seeds43/44 confirmatory. Fixed 2 validation objects, not independent object replication or statistical significance.')
    with (OUTPUT/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    figures=ROOT/'outputs/experiment_figures'
    filename='bronze_hierarchy_three_seeds.png'
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for index,(label,color) in enumerate(zip(VARIANTS,('#4b9cd3','#9b7bb5','#78b86a'))):
        selected=[r for r in rows if r['label']==label]
        axes[0].bar(np.arange(3)+(index-1)*.25,[r['cd_squared'] for r in selected],width=.25,label=label,color=color)
        axes[1].bar(np.arange(3)+(index-1)*.25,[r['fscore'] for r in selected],width=.25,label=label,color=color)
    for ax,title in zip(axes,('Validation CD squared (lower is better)','Validation F-score (higher is better)')):
        ax.set(title=title,xticks=np.arange(3),xticklabels=['42 (explore)','43','44'],xlabel='Training seed')
        field='cd_squared' if ax is axes[0] else 'fscore'
        ax.set_ylim(0,max(r[field] for r in rows)*1.4)
        ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed split / 20 epochs / common evaluation seed43')
    fig.tight_layout();fig.savefig(figures/filename,dpi=180);plt.close(fig)
    lines=['# 分层解码改进：重复种子与容量对照','',
           '数据划分固定，输入512点、输出/训练目标128点、CD损失、tanh、20轮、CPU、batch8、学习率0.001。训练种子42用于探索，43与44用于固定方案的复验。最终评价输入采样统一evaluation-seed43；最佳轮次仍依据各训练过程的验证CD²选取。','',
           '| 训练种子 | 小MLP CD² | 宽MLP CD² | 分层+粗监督 CD² | 相对宽MLP降低 |','|---|---:|---:|---:|---:|']
    for pair in paired:
        selected={r['label']:r for r in rows if r['seed']==pair['seed']}
        lines.append(f'| {pair["seed"]} | {selected["mlp"]["cd_squared"]:.8f} | {selected["wide"]["cd_squared"]:.8f} | {selected["hier_aux"]["cd_squared"]:.8f} | {pair["reduction_vs_wide"]:.2%} |')
    lines+=['','| 方案 | CD²均值 ± 样本标准差 | F-score均值 ± 样本标准差 | 参数量 |','|---|---:|---:|---:|']
    for label in VARIANTS:
        a=aggregates[label];parameter=next(r['parameters'] for r in rows if r['label']==label)
        lines.append(f'| {label} | {a["cd_squared"]["mean"]:.8f} ± {a["cd_squared"]["sample_std"]:.8f} | {a["fscore"]["mean"]:.6f} ± {a["fscore"]["sample_std"]:.6f} | {parameter:,} |')
    lines+=['','## 固定方案复验结论','',
            f'只看新增种子43/44：宽MLP平均CD²={confirmatory["wide"]["cd_squared"]:.8f}，分层平均CD²={confirmatory["hier_aux"]["cd_squared"]:.8f}。分层相对宽MLP的收益未在这两个种子中重复出来，不能将seed42的4.02%提升作为稳定结构收益。',
            f'复验F-score：宽MLP={confirmatory["wide"]["fscore"]:.6f}，分层={confirmatory["hier_aux"]["fscore"]:.6f}。保留宽MLP作为容量基线，分层模型仍是研究候选。','',
            '宽MLP与分层模型参数量相差约0.038%，用于检查增大容量的影响。同等轮数不等于同等计算量；分层模型有FPS粗层监督开销。三个种子的标准差不是置信区间。','',
            '## 文献与工程改动','',
            '参考PF-Net（CVPR2020）父点+残差子点及多阶段监督。原方法包含多尺度编码器、三级解码和可选对抗模块；本实验只改解码器，保留原编码器，采用32父点→128子点的两级简化，粗层目标从训练缺失目标做FPS。',
            '父点特征64维，子点偏移以0.25R×tanh限制，子点球外做径向裁剪。这些是本项目的工程改动；0.01粗层监督参考作者前30轮alpha1设置，但层数/分辨率不同，不能声称完全复现PF-Net。',
            '[作者生成器](https://github.com/zztianzz/PF-Net-Point-Fractal-Network/blob/master/model_PFNet.py) · [作者训练代码](https://github.com/zztianzz/PF-Net-Point-Fractal-Network/blob/master/Train_PFNet.py)','',
            '## 图和进一步诊断','',f'![三种子对照](../../experiment_figures/{filename})','',
            '[seed42结构与粗监督消融](../../comparisons/bronze_hierarchy_seed42/README.md)','',
            '验证仅2个器物，复验增加训练随机性证据，没有增加形状多样性。此前原型测试集已经使用，本轮不再用该测试集选型；下一阶段需要保留新的独立器物作最终确认。','']
    (OUTPUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (figures/'bronze_hierarchy_图片说明.md').write_text('# 分层解码实验图片说明\n\n- bronze_hierarchy_three_seeds.png：三个训练种子的验证CD²与F-score，统一评价采样种子43，用于可靠性分析章节。seed42是探索种子，43/44是复验。\n- bronze_hierarchy_seed42_validation_comparison.png：结构与粗层监督消融，包含小MLP、容量相近宽MLP、分层、分层+粗监督。用于方法消融章节。\n',encoding='utf-8')
    organize()
    print(json.dumps(dict(aggregates=aggregates,paired=paired),indent=2))


if __name__=='__main__':
    summarize()
