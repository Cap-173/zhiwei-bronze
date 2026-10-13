"""Separate data integrity, fixed-hole optimization and one-object generalization."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from completion import load_checkpoint, context_points, farthest_subset, sample_points
from training_data import checked_sample, read_manifest, evaluate_entries, nearest_squared
from organize_figures import organize
from main import ROOT


def point_metrics(prediction,target,radius):
    p,t=(torch.tensor(x,dtype=torch.float32) for x in (prediction,target))
    a,b=nearest_squared(p,t),nearest_squared(t,p)
    precision=float((a<=.02**2).float().mean());recall=float((b<=.02**2).float().mean())
    cd=float(a.mean()+b.mean())
    return dict(cd=cd,cd_over_R2=cd/radius**2,fscore=2*precision*recall/(precision+recall) if precision+recall else 0.)


def sampling_reference(source,entries):
    rows=[]
    for index,entry in enumerate(entries):
        sample=checked_sample(source/entry['path']/'training.npz')
        target=sample['local_target'];count=min(128,len(target))
        pred=farthest_subset(torch.tensor(target,dtype=torch.float32)[None],count)[0].numpy()
        if count<128:pred=np.concatenate((pred,sample_points(target,128-count,np.random.default_rng(index+42))))
        metrics=point_metrics(pred,target,float(sample['radius']))
        random_metrics=[point_metrics(sample_points(target,128,np.random.default_rng(np.random.SeedSequence([31415,index,repeat]))),target,float(sample['radius'])) for repeat in range(5)]
        rows.append(dict(path=entry['path'],fps_reference=metrics,random_reference={f:float(np.mean([r[f] for r in random_metrics])) for f in metrics}))
    return dict(samples=rows,fps_mean={field:float(np.mean([r['fps_reference'][field] for r in rows])) for field in ('cd','cd_over_R2','fscore')},
                random_mean={field:float(np.mean([r['random_reference'][field] for r in rows])) for field in ('cd','cd_over_R2','fscore')},
                note='Truth-only sampling reference, not a model or strict theoretical optimum. FPS may not minimize CD.')


def render_case(sample,predictions,name,figures):
    center,radius=sample['center'],float(sample['radius'])
    visible=(sample['input']-center)/radius;visible=visible[np.linalg.norm(visible,axis=1)<=1.5]
    target=(sample['local_target']-center)/radius
    _,_,basis=np.linalg.svd(np.concatenate((visible,target)),full_matrices=False)
    visible,target=visible@basis.T,target@basis.T
    panels=[('Reference',target),('Visible input',None)]+[(label,((pred-center)/radius)@basis.T) for label,pred in predictions.items()]
    fig,axes=plt.subplots(3,len(panels),figsize=(3.25*len(panels),10))
    for column,(label,points) in enumerate(panels):
        for row,(a,b) in enumerate(((0,1),(0,2),(1,2))):
            ax=axes[row,column];ax.scatter(visible[:,a],visible[:,b],s=3,c='#6aa7bb',alpha=.35)
            if points is not None:
                if column==0:ax.scatter(points[:,a],points[:,b],s=8,c='#209b62',alpha=.65)
                else:
                    ax.scatter(target[:,a],target[:,b],s=7,facecolors='none',edgecolors='#209b62',alpha=.3,linewidths=.5)
                    ax.scatter(points[:,a],points[:,b],s=12,c='#e77d20')
            ax.add_patch(plt.Circle((0,0),1,fill=False,ls='--',color='#999'))
            ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),aspect='equal',xlabel=f'PC{a+1}/R',ylabel=f'PC{b+1}/R')
            ax.grid(alpha=.15)
            if row==0:ax.set_title(label,fontsize=10)
    fig.suptitle(name+' | blue: visible; green: missing target; orange: prediction',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));filename=f'bronze_basic_fit_{name}.png'
    fig.savefig(figures/filename,dpi=180);plt.close(fig)
    return filename


def main():
    torch.set_num_threads(2)
    output=ROOT/'outputs/diagnostics/basic_fit';figures=ROOT/'outputs/experiment_figures'
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    audit=json.loads((output/'data_audit.json').read_text(encoding='utf-8'))
    source=Path(protocol['source_dataset']);dataset=Path(protocol['single_object_dataset'])
    roles={'seen_train16':protocol['training_holes'],'unseen_same_object4':protocol['unseen_same_object_holes'],'cross_object20':protocol['cross_object_evaluation']}
    references={role:sampling_reference(source,entries) for role,entries in roles.items()}
    evaluations={};models={};histories={};fixed={};fixed_predictions={}
    sample=checked_sample(source/protocol['single_hole']['path']/'training.npz')
    for label in protocol['architectures']:
        run=ROOT/'outputs/runs'/f'bronze_single_object_{label}_seed42'
        model,checkpoint=load_checkpoint(run/'last.pt')
        assert checkpoint['epoch']==200 and checkpoint['selection_split']=='train'
        assert len(read_manifest(dataset)['samples'])==16
        models[label]=model;histories[label]=checkpoint['history']
        evaluations[label]={role:evaluate_entries(model,source,entries,512,43,torch.device('cpu')) for role,entries in roles.items()}
        fixed[label]={}
        for objective,prefix in [('CD','single_hole'),('ordered MSE','single_hole_mse')]:
            with np.load(output/f'{prefix}_{label}.npz') as data:
                pred=data['prediction'];target=data['target']
            metrics=point_metrics(pred,target,float(sample['radius']))
            fixed[label][objective]=dict(fixed_target=metrics,full_target=point_metrics(pred,sample['local_target'],float(sample['radius'])),
                history=json.loads((output/f'{prefix}_{label}.json').read_text(encoding='utf-8'))['history'])
            fixed_predictions[f'{label} {objective}']=pred
    examples=[]
    for role,entries in roles.items():
        entry=entries[0];s=checked_sample(source/entry['path']/'training.npz')
        context=context_points(s['input'],s['center'],512,np.random.default_rng(np.random.SeedSequence([43,0,0])))
        region=np.r_[s['center'],s['radius']]
        with torch.no_grad():
            preds={label:model(torch.tensor(context,dtype=torch.float32)[None],torch.tensor(region,dtype=torch.float32)[None])[0].numpy() for label,model in models.items()}
        filename=render_case(s,preds,role,figures)
        np.savez_compressed(output/f'{role}_case.npz',input=s['input'],target=s['local_target'],center=s['center'],radius=s['radius'],context=context,**preds)
        examples.append(dict(role=role,path=entry['path'],figure=filename))
    fixed_figure=render_case(sample,fixed_predictions,'fixed_hole_objective_diagnostic',figures)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for label in protocol['architectures']:
        for objective in ('CD','ordered MSE'):
            h=fixed[label][objective]['history']
            metric='loss' if objective=='CD' else 'cd_over_R2'
            axes[0].plot([r['step'] for r in h],[max(r[metric],1e-12) for r in h],label=label+' '+objective)
        axes[1].plot([r['epoch'] for r in histories[label]],[r['train_cd_squared'] for r in histories[label]],label=label)
    axes[0].set(yscale='log',xlabel='Optimizer steps',ylabel='CD squared / R squared',title='Fixed context and128 target points')
    axes[1].set(xlabel='Epoch',ylabel='Full-target training CD squared',title='Single-object16-hole CD training')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();curve='bronze_basic_fit_learning_curves.png';fig.savefig(figures/curve,dpi=180);plt.close(fig)
    report=dict(protocol=protocol,data_audit=audit['summary'],fixed_hole=fixed,evaluations=evaluations,
        sampling_references=references,examples=examples,
        ordered_diagnostic='Added after fixed CD plateau. Same initialization, context, fixed target order, Adam lr and steps. Checks representational/optimization sanity only, not a legitimate unordered completion objective.',
        limitation='One object and training seed; same-object heldout holes are correlated with training object. No new independent object-generalization evidence.')
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 基础排查与单器物拟合','',
        '依据[Stanford CS231n的tiny-subset sanity check](https://cs231n.github.io/neural-networks-3/#sanity-checks)先检查小样本能否拟合。本轮属于工程诊断，不是新补全方法；小样本能拟合也不能证明整条实现和泛化都正确。','',
        '## 数据检查','',
        f'原训练/验证12器物的240个样本：{audit["summary"]["passed"]}/240通过有限数、完整/可见/缺失分区、球形掩码、可见输入归一化和同器物坐标恢复检查。原两个测试器物未读取。',
        f'目标点数{audit["summary"]["min_target_points"]}–{audit["summary"]["max_target_points"]}，其中19个缺口少于128点，训练会有放回采样；输入抽样重复比例均值{audit["summary"]["mean_context_duplicate_fraction"]:.2%}。',
        f'FPS32中心位于缺口周围2R内的比例均值{audit["summary"]["mean_fps_anchors_within_2R"]:.2%}，每组16近邻的平均不同坐标数{audit["summary"]["mean_knn_unique_points"]:.2f}。这些是采样诊断，不证明失败原因。',
        '不同OBJ没有统一语义朝向；同器物各缺口没有坐标翻转。PCA轴不等同于器物朝向，因此本检查不能认定跨器物朝向已经合适，仍需单独旋转/对齐对照。','',
        '## 固定缺口检查','',
        '司母戊鼎第一个缺口：固定512输入和128目标点、seed42、Adam lr0.001、1000次更新，两个模型参数均615236。CD出现平台后追加同配置有序坐标MSE诊断。目标顺序固定，MSE结果不作为真实无序补全方案，也不参与16缺口训练。','',
        '|模型|诊断损失|最终固定目标CD²/R²↓|完整缺失目标CD²↓|','|---|---|---:|---:|']
    for label,results in fixed.items():
        for objective,result in results.items():
            lines.append(f'|{label}|{objective}|{result["fixed_target"]["cd_over_R2"]:.8f}|{result["full_target"]["cd"]:.8f}|')
    lines += ['',f'![拟合曲线](../../experiment_figures/{curve})','',f'![固定缺口](../../experiment_figures/{fixed_figure})','',
        '## 单器物多个缺口','',
        '仅用司母戊鼎前16个缺口训练200轮、batch4、CD/R²，后4个新缺口与商代青铜觥20缺口只用于末轮评价。训练清单只有一个真实器物ID且全部标train，未伪造器物ID或放宽原器物划分防护。模型从头训练，未用固定缺口诊断权重暖启动。','',
        '|评价范围|模型|样本数|CD²↓|F-score↑|真实目标FPS128采样参考CD²|','|---|---|---:|---:|---:|---:|']
    for role in roles:
        for label in protocol['architectures']:
            metric=evaluations[label][role]
            lines.append(f'|{role}|{label}|{metric["sample_count"]}|{metric["mean_local_cd_squared"]:.8f}|{metric["mean_local_fscore"]:.6f}|{references[role]["fps_mean"]["cd"]:.8f}|')
    lines += ['','FPS128与随机128采样参考直接使用真值，仅帮助解释输出点数和采样误差，不能称为理论下界或实际模型。目标不足128时覆盖全部真实点后补重复点。跨器物只评价一个器物，结果不可推广；后4个新缺口也不是独立新器物。','']
    lines += ['## 本轮判断','',
        '数据基本不变量没有发现错误，但不证明全部实现都正确。两个网络都能在固定顺序诊断中把同一缺口的CD²/R²降至约3×10⁻⁷，说明至少有记住该缺口的表达和优化能力；固定CD训练却停在0.01084/0.01772。仅一个缺口的诊断不能证明全部困难都来自CD。',
        '单器物训练16缺口F-score约0.662/0.581，同器物4个新缺口约0.140/0.139，存在明显记忆与迁移差距。缺口位置和大小也可能影响该差距。下一步优先核对一对一点匹配损失的文献及实现并做等预算对照；有序MSE只作诊断，不能直接用于无序补全目标。','']
    for example in examples:
        lines += [f'### {example["role"]}',f'样本：`{example["path"]}`。用途：区分训练记忆、同器物新缺口与换器物的形状恢复。',f'![{example["role"]}](../../experiment_figures/{example["figure"]})','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (figures/'bronze_basic_fit_notes.md').write_text('\n'.join(['# 基础拟合诊断图片','',
        '[完整指标与说明](../diagnostics/basic_fit/README.md)',curve+'：固定缺口两损失的CD曲线与单器物16缺口训练曲线。',
        fixed_figure+'：固定缺口两模型、两诊断损失的同视角结果，仅用于训练链路排查。']+
        [e['figure']+'：'+e['role']+'，用于训练记忆和泛化差距分析。' for e in examples]),encoding='utf-8')
    organize();print(json.dumps(dict(fixed_target={label:{objective:r['fixed_target'] for objective,r in results.items()} for label,results in fixed.items()},
        evaluations={label:{role:dict(cd=m['mean_local_cd_squared'],fscore=m['mean_local_fscore']) for role,m in results.items()} for label,results in evaluations.items()}),indent=2))


if __name__=='__main__':main()
