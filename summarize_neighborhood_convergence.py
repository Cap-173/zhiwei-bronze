"""Compare fixed 20/40/60 epoch prefixes and preserve convergence evidence."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from completion import load_checkpoint
from training_data import read_manifest, dataset_fingerprint, evaluate_entries
from run_neighborhood_convergence import sha256
from organize_figures import organize
from main import ROOT


def main():
    torch.set_num_threads(2)
    output=ROOT/'outputs/repeats/bronze_neighborhood_convergence'
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    dataset=ROOT/'outputs/datasets/bronze_20261013_seed42'
    manifest=read_manifest(dataset); fingerprint=dataset_fingerprint(dataset,manifest)
    entries=[e for e in manifest['samples'] if e['split']=='val']
    rows=[]; histories={}
    for seed in protocol['training_seeds']:
        for label in protocol['architectures']:
            original=ROOT/'outputs/runs'/f'bronze_{label}_seed{seed}'
            run=ROOT/'outputs/runs'/f'bronze_{label}_e60_seed{seed}'
            provenance=json.loads((run/'resume_source.json').read_text(encoding='utf-8'))
            assert sha256(original/'last.pt')==provenance['last_sha256']
            assert sha256(original/'best.pt')==provenance['best_sha256']
            original_last=torch.load(original/'last.pt',map_location='cpu',weights_only=True)
            final=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
            assert final['epoch']==60 and final['history'][:20]==original_last['history']
            assert final['config']==original_last['config']
            assert final['dataset_fingerprint']==fingerprint
            histories[(seed,label)]=final['history']
            for budget in protocol['budgets']:
                directory=original if budget==20 else run/'milestones/epoch40' if budget==40 else run
                last=torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
                assert last['epoch']==budget and last['history']==final['history'][:budget]
                assert last['config']==final['config']
                for kind in ('best','last'):
                    model,checkpoint=load_checkpoint(directory/f'{kind}.pt')
                    assert checkpoint['dataset_fingerprint']==fingerprint
                    assert sum(p.numel() for p in model.parameters())==615236
                    metrics=evaluate_entries(model,dataset,entries,512,43,torch.device('cpu'))
                    rows.append(dict(seed=seed,label=label,budget=budget,checkpoint=kind,epoch=checkpoint['epoch'],
                                     cd=metrics['mean_local_cd_squared'],fscore=metrics['mean_local_fscore'],
                                     train_cd_native=checkpoint['history'][-1]['train_cd_squared'],
                                     model_metrics=metrics['models']))
    aggregates=[]
    for budget in protocol['budgets']:
        for label in protocol['architectures']:
            for kind in ('best','last'):
                selected=[r for r in rows if r['budget']==budget and r['label']==label and r['checkpoint']==kind]
                aggregates.append(dict(budget=budget,label=label,checkpoint=kind,
                    **{field:dict(mean=float(np.mean([r[field] for r in selected])),sample_std=float(np.std([r[field] for r in selected],ddof=1))) for field in ('cd','fscore')}))
    means={(a['budget'],a['label'],a['checkpoint']):a for a in aggregates}
    quality=[]
    for budget in (20,60):
        for label in protocol['architectures']:
            directory=ROOT/'outputs/runs'/f'bronze_{label}{"_e60" if budget==60 else ""}_seed42'
            q=json.loads((directory/'quality_val.json').read_text(encoding='utf-8'))
            assert q['sample_count']==40 and q['model_count']==2 and q['split']=='val'
            quality.append(dict(budget=budget,label=label,**q['summary']))
    final_ratio=means[(60,'local','best')]['cd']['mean']/means[(60,'pointwise_xyz','best')]['cd']['mean']-1
    report=dict(protocol=protocol,rows=rows,aggregates=aggregates,quality_seed42=quality,
                relative_cd_excess_local_vs_control_60=final_ratio,
                warning='Validation revisited in previous exploratory rounds. Three training seeds are not independent validation objects or confidence intervals. No test-set access.')
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    figures=ROOT/'outputs/experiment_figures'
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for label,color in zip(protocol['architectures'],('#2ca02c','#ff7f0e')):
        for ax,field in zip(axes[:2],('train_cd_squared','val_cd_squared')):
            data=np.array([[h[field] for h in histories[(seed,label)]] for seed in protocol['training_seeds']])
            ax.plot(np.arange(1,61),data.mean(0),label=label,color=color)
            ax.fill_between(np.arange(1,61),data.min(0),data.max(0),alpha=.12,color=color)
            ax.set(xlabel='Epoch',ylabel=field);ax.legend();ax.grid(alpha=.2)
        selected=[means[(budget,label,'best')] for budget in protocol['budgets']]
        axes[2].errorbar(protocol['budgets'],[a['cd']['mean'] for a in selected],
                        yerr=[a['cd']['sample_std'] for a in selected],marker='o',label=label,color=color)
    axes[0].set_title('Full train CD2 (native sampling per seed)')
    axes[1].set_title('Validation CD2 (native sampling per seed)')
    axes[2].set(title='Best validation CD2 (common sampling seed43)',xlabel='Total epoch budget',ylabel='CD squared')
    axes[2].legend();axes[2].grid(alpha=.2)
    for ax in axes[:2]:
        ax.axvline(20,ls='--',color='#777',alpha=.6);ax.axvline(40,ls='--',color='#777',alpha=.6)
    fig.suptitle('Matched 615236 parameters / constant Adam lr0.001 / three training seeds')
    fig.tight_layout();filename='bronze_neighborhood_convergence_20_40_60.png'
    fig.savefig(figures/filename,dpi=180);plt.close(fig)
    lines=['# 邻域编码收敛对照：20→40→60轮','',
           '两个方案、三个种子42/43/44同步增加预算，数据、学习率、损失、解码器、点数和参数量均保持。以20轮last.pt恢复优化器和随机状态，新运行目录保留40轮快照；原20轮检查点哈希和历史前缀检查通过。不是新文献方法，而是检查训练预算影响的工程对照。','',
           '60轮及中间40轮为开始前固定预算；没有按表现选择性延长某一方案。每个预算都报告此前验证CD²最优best，以及恰好结束轮last。最终评价采样统一seed43；曲线沿用各种子原训练过程的seed+1采样，图中阴影为三种子最小到最大，不是置信区间。','',
           '结构依据和改造范围见[上一轮记录](../bronze_neighborhood_confirmation/README.md)，包括PointNet++作者代码、单尺度FPS32+kNN16、相同参数逐点编码控制。','',
           '## 各预算的验证均值','',
           '|总轮数|方案|检查点|CD²均值↓|F-score均值↑|','|---|---|---|---:|---:|']
    lines += [f'|{a["budget"]}|{a["label"]}|{a["checkpoint"]}|{a["cd"]["mean"]:.8f}|{a["fscore"]["mean"]:.6f}|' for a in aggregates]
    lines += ['',f'60轮best：邻域方案CD²相对同参数逐点对照变化为{final_ratio:+.2%}（正值为误差增加）。','',
              '## 结果解释','',
              '逐点对照20→60轮的平均best CD²降低约9.60%，邻域方案仅降低约0.48%。邻域三个种子的原验证最优轮次均在40轮以内，40→60轮没有更新best。训练CD²持续下降而验证曲线停滞，提示当前数据和编码改造的泛化问题；不能将此前差距简单归因为只训练20轮。这个观察不证明无限预算下的收敛，也不否定完整PointNet++。',
              '暂缓该邻域改造，保留逐点60轮方案为研究候选。下一步优先做器物级划分复验并补充独立器物，而不是继续在同两个验证器物上搜索邻域参数；当前原型保持冻结。','',
              '## 每个种子的60轮结果','',
              '|种子|方案|best轮次|best验证CD²|last验证CD²|末轮训练CD²（原采样）|','|---|---|---:|---:|---:|---:|']
    for seed in protocol['training_seeds']:
        for label in protocol['architectures']:
            best=next(r for r in rows if r['seed']==seed and r['label']==label and r['budget']==60 and r['checkpoint']=='best')
            last=next(r for r in rows if r['seed']==seed and r['label']==label and r['budget']==60 and r['checkpoint']=='last')
            lines.append(f'|{seed}|{label}|{best["epoch"]}|{best["cd"]:.8f}|{last["cd"]:.8f}|{last["train_cd_native"]:.8f}|')
    lines += ['','## seed42补充诊断（全40验证缺口）','',
              '|预算|方案|DCD↓|聚集比例↓|到网格距离/R↓|','|---|---|---:|---:|---:|']
    lines += [f'|{q["budget"]}|{q["label"]}|{q["dcd_equal_points_mean"]:.6f}|{q["prediction_close_fraction_0_02_radius"]:.2%}|{q["prediction_mesh_mean_distance_over_radius"]:.6f}|' for q in quality]
    lines += ['','网格仅用于评价。聚集阈值为0.02R，离面与聚集仅评价seed42；不能代替三个种子的几何结论。','',
              f'![训练预算对照](../../experiment_figures/{filename})','',
              '[60轮同缺口图](../../diagnostics/neighborhood60_cases_seed42/README.md)','',
              '不能由60轮结果宣称理论或无限预算下收敛；验证只有两个器物，且已多轮探索，结果需要更多独立器物确认。原型仍冻结。完整配置、逐种子/逐器物指标与标准差见summary.json。']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (figures/'bronze_neighborhood_convergence_notes.md').write_text('# 训练预算对照图\n\n'+filename+'：左为训练CD²，中为验证CD²（原采样），右为20/40/60轮best的统一采样验证CD²。曲线阴影为三种子极值，右误差棒为样本标准差。用于解释训练预算影响。\n\n[完整结果](../repeats/bronze_neighborhood_convergence/README.md)',encoding='utf-8')
    organize();print(json.dumps(dict(aggregates=aggregates,relative_excess=final_ratio,quality_seed42=quality),indent=2))


if __name__=='__main__':
    main()
