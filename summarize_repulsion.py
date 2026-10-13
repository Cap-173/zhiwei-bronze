"""Summarize fixed-budget PU-Net-inspired repulsion versus capacity-matched CD."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from completion import load_checkpoint
from training_data import read_manifest, evaluate_entries, dataset_fingerprint
from main import ROOT
from organize_figures import organize


def main():
    torch.set_num_threads(2)
    dataset=ROOT/'outputs/datasets/bronze_20261013_seed42'
    manifest=read_manifest(dataset)
    entries=[e for e in manifest['samples'] if e['split']=='val']
    output=ROOT/'outputs/repeats/bronze_repulsion_confirmation'
    output.mkdir(parents=True, exist_ok=True)
    rows=[]
    for seed in (42,43,44):
        for label,prefix in [('CD','bronze_wide1140'),('CD + repulsion','bronze_repulsion01')]:
            run=ROOT/'outputs/runs'/f'{prefix}_seed{seed}'
            last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
            assert last['epoch']==20 and last['dataset_fingerprint']==dataset_fingerprint(dataset,manifest)
            for key,value in dict(seed=seed,input_points=512,output_points=128,target_points=128,batch_size=8,lr=.001,output_mode='tanh',decoder_width=1140,amp=False,boundary_weight=0.).items():
                assert last['config'][key]==value
            assert last['config'].get('repulsion_weight',0.)==(.01 if prefix=='bronze_repulsion01' else 0.)
            model,checkpoint=load_checkpoint(run/'best.pt')
            metrics=evaluate_entries(model,dataset,entries,512,43,torch.device('cpu'))
            (run/'evaluation_val_common43.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
            rows.append(dict(seed=seed,label=label,best_epoch=checkpoint['epoch'],cd=metrics['mean_local_cd_squared'],fscore=metrics['mean_local_fscore']))
    aggregates={label:{field:float(np.mean([r[field] for r in rows if r['label']==label])) for field in ('cd','fscore')} for label in ('CD','CD + repulsion')}
    report=dict(rows=rows,means=aggregates,evaluation_seed=43,split='val',note='2 validation objects; repeated training seeds, not independent object replication')
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    figures=ROOT/'outputs/experiment_figures'
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,label in enumerate(aggregates):
        selected=[r for r in rows if r['label']==label]
        for ax,field in zip(axes,('cd','fscore')):
            ax.bar(np.arange(3)+(i-.5)*.35,[r[field] for r in selected],width=.35,label=label)
            ax.set(xticks=np.arange(3),xticklabels=['42','43','44'],xlabel='Training seed',ylabel=field)
            ax.legend()
    fig.suptitle('20 epochs / matched wide MLP / validation sampling seed 43')
    fig.tight_layout(); fig.savefig(figures/'bronze_repulsion_three_seeds.png',dpi=180); plt.close(fig)
    lines=['# PU-Net借鉴排斥项：固定预算对照','',
           '参考[PU-Net CVPR2018论文](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.pdf)与[作者代码](https://github.com/yulequan/PU-Net/blob/master/code/model_utils.py)。','',
           '采用0.07-r×exp(-r²/0.03²)的邻域平均能量，权重0.01。工程改动：距离以缺口半径归一化；以精确4个非自身近邻替代作者球查询20候选后top5去首点及补齐；重建项使用本项目双向平方CD/R²。属于借鉴消融，并非PU-Net完整复现。','',
           '复用宽MLP CD基线，参数614852，输入512、输出/目标128，batch8，Adam lr0.001，tanh，20轮，固定数据划分。最佳轮次按各运行验证CD²选取，汇总统一采样seed43；测试集未参与本轮。','',
           '|种子|方案|最佳轮|CD²↓|F-score↑|','|---|---|---:|---:|---:|']
    lines += [f'|{r["seed"]}|{r["label"]}|{r["best_epoch"]}|{r["cd"]:.8f}|{r["fscore"]:.6f}|' for r in rows]
    lines += ['',f'三种子平均：CD² {aggregates["CD"]["cd"]:.8f} → {aggregates["CD + repulsion"]["cd"]:.8f}；F-score {aggregates["CD"]["fscore"]:.6f} → {aggregates["CD + repulsion"]["fscore"]:.6f}。',
              '结论：三种子平均CD²降低约2.92%，各种子同方向改善，但F-score均值略降。seed42的近邻聚集比例从28.398%升至29.004%，DCD从0.917684升至0.918043，到网格距离/R从0.071411升至0.071659，均未改善。本轮不能宣称排斥项解决聚集，也不足以替换原型。没有搜索额外权重或带宽。',
              '只有两个验证器物，种子重复不能证明跨器物泛化或统计显著性；聚集与网格距离另见seed42的quality_val.json。已交付原型保持冻结。','',
              '![三种子对照](../../experiment_figures/bronze_repulsion_three_seeds.png)']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (figures/'bronze_repulsion_notes.md').write_text('图 bronze_repulsion_three_seeds.png：宽MLP的CD与CD加排斥三种子验证对照，左CD²右F-score。用于方法改进消融结果段，数据见 outputs/repeats/bronze_repulsion_confirmation/summary.json。',encoding='utf-8')
    organize()
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
