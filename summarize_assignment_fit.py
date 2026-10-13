"""Same-input CD versus exact-matching diagnostics and single-object evaluation."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assignment_loss import assignment_per_sample
from completion import load_checkpoint, context_points
from training_data import checked_sample, evaluate_entries, read_manifest, dataset_fingerprint
from summarize_basic_fit import point_metrics, render_case
from organize_figures import organize
from main import ROOT


def main():
    torch.set_num_threads(2)
    output=ROOT/'outputs/diagnostics/assignment_fit';basic=ROOT/'outputs/diagnostics/basic_fit'
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    source=Path(protocol['source_dataset']);dataset=Path(protocol['single_object_dataset'])
    assert dataset_fingerprint(dataset,read_manifest(dataset))==protocol['dataset_fingerprint']
    runs={'CD':ROOT/'outputs/runs/bronze_single_object_pointwise_xyz_seed42',
          'Exact matching':ROOT/'outputs/runs/bronze_single_object_emd_exact_seed42'}
    assert hashlib.sha256((runs['CD']/'last.pt').read_bytes()).hexdigest()==protocol['baseline_checkpoint_sha256']
    roles={'seen_train16':protocol['training_holes'],'unseen_same_object4':protocol['unseen_same_object_holes'],
           'cross_object20':protocol['cross_object_evaluation']}
    models={};histories={};evaluations={};fixed={};examples=[]
    fixed_sample=checked_sample(source/protocol['single_hole']['path']/'training.npz')
    fixed_predictions={}
    for label,run in runs.items():
        model,checkpoint=load_checkpoint(run/'last.pt')
        assert checkpoint['epoch']==200 and checkpoint['selection_split']=='train'
        assert checkpoint['dataset_fingerprint']==protocol['dataset_fingerprint']
        assert sum(p.numel() for p in model.parameters())==615236
        assert checkpoint['config']['loss_mode']=='emd_exact' if label=='Exact matching' else 'loss_mode' not in checkpoint['config']
        models[label]=model;histories[label]=checkpoint['history']
        evaluations[label]={role:evaluate_entries(model,source,entries,512,43,torch.device('cpu')) for role,entries in roles.items()}
        path=basic/'single_hole_pointwise_xyz.npz' if label=='CD' else output/'single_hole_emd_exact.npz'
        with np.load(path) as data:pred,target=data['prediction'],data['target']
        radius=float(fixed_sample['radius'])
        region=torch.tensor(np.r_[fixed_sample['center'],radius],dtype=torch.float32)[None]
        emd=float(assignment_per_sample(torch.tensor(pred,dtype=torch.float32)[None],torch.tensor(target,dtype=torch.float32)[None],region)[0])
        distances=((pred[:,None]-target[None,:])**2).sum(-1)
        fixed[label]=dict(**point_metrics(pred,target,radius),exact_distance_over_R=emd,
                         nearest_distinct_target_sites=int(len(np.unique(distances.argmin(1)))),
                         full_target=point_metrics(pred,fixed_sample['local_target'],radius))
        fixed_predictions[label]=pred
    sources={r['id']:r['source'] for r in json.loads((ROOT/'outputs/local_models/import_20261013_125555_685154/manifest.json').read_text(encoding='utf-8'))['models']}
    meshes={};quality={label:{} for label in runs}
    figures=ROOT/'outputs/experiment_figures'
    for role,entries in roles.items():
        rows={label:[] for label in runs}
        for index,entry in enumerate(entries):
            sample=checked_sample(source/entry['path']/'training.npz')
            context=context_points(sample['input'],sample['center'],512,np.random.default_rng(np.random.SeedSequence([43,0,index])))
            region=np.r_[sample['center'],sample['radius']]
            mid=entry['model_id']
            if mid not in meshes:
                print(f'Mesh reference {mid}',flush=True)
                meshes[mid]=trimesh.load(sources[mid],force='mesh',process=False,skip_materials=True)
            predictions={}
            for label,model in models.items():
                with torch.no_grad():
                    pred=model(torch.tensor(context,dtype=torch.float32)[None],torch.tensor(region,dtype=torch.float32)[None])[0]
                    distances=torch.cdist(pred[None],pred[None],compute_mode='donot_use_mm_for_euclid_dist')[0]
                    distances.fill_diagonal_(float('inf'))
                    close=float((distances.min(1).values<float(sample['radius'])*.02).float().mean())
                predictions[label]=pred.numpy()
                original=pred.numpy()*float(sample['scale'])+sample['offset']
                _,mesh_dist,_=trimesh.proximity.closest_point(meshes[mid],original)
                if not np.isfinite(mesh_dist).all():raise ValueError('非有限网格距离。')
                rows[label].append(dict(path=entry['path'],close_fraction=close,
                    mesh_distance_over_R=float(mesh_dist.mean()/(float(sample['radius'])*float(sample['scale'])))))
            if index==0:
                name='assignment_'+role
                filename=render_case(sample,predictions,name,figures)
                np.savez_compressed(output/f'{role}_case.npz',input=sample['input'],target=sample['local_target'],
                    center=sample['center'],radius=sample['radius'],context=context,**predictions)
                examples.append(dict(role=role,path=entry['path'],figure=filename))
        for label in runs:
            quality[label][role]=dict(samples=rows[label],means={field:float(np.mean([r[field] for r in rows[label]])) for field in ('close_fraction','mesh_distance_over_R')})
    fixed_figure=render_case(fixed_sample,fixed_predictions,'assignment_fixed_hole',figures)
    report=dict(protocol=protocol,fixed_hole=fixed,evaluations=evaluations,geometry=quality,examples=examples,
        interpretation='Single object/seed, fixed final epoch; no model selected using unseen holes. Exact Euclidean bijection in hole units, not a full PCN reproduction. Target replacement can repeat geometric coordinates despite unique indices.')
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for label in runs:
        history=json.loads((basic/'single_hole_pointwise_xyz.json' if label=='CD' else output/'single_hole_emd_exact.json').read_text(encoding='utf-8'))['history']
        field='loss' if label=='CD' else 'cd_over_R2'
        axes[0].plot([r['step'] for r in history],[max(r[field],1e-12) for r in history],label=label)
        axes[1].plot([r['epoch'] for r in histories[label]],[r['train_cd_squared'] for r in histories[label]],label=label)
    axes[0].set(yscale='log',xlabel='Optimizer steps',ylabel='CD squared / R squared',title='Same fixed128 target points')
    axes[1].set(xlabel='Epoch',ylabel='Full-target CD squared',title='16-hole training / same model and sampled data')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();curve='bronze_assignment_fit_curves.png';fig.savefig(figures/curve,dpi=180);plt.close(fig)
    lines=['# 一对一点匹配损失对照','',
        '## 文献与实现区别','',
        '参考[PCN（3DV2018）第4.3节](https://arxiv.org/pdf/1808.00671)与[作者tf_util.py](https://github.com/wentaoyuan/pcn/blob/master/tf_util.py)。原文粗层使用等点数的平均欧氏EMD，作者采用CUDA近似匹配。本实验用[SciPy精确方阵最小代价匹配](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)，算法为修改的Jonker-Volgenant，不是作者近似求解器。','',
        '成本为欧氏距离，匹配后求平均并除以缺口半径R，不是平方距离/R²。离散对应不求导，匹配距离对坐标求导。只有128缺失点、原逐点编码/tanh/解码1140，没有PCN多阶段全物体结构和细层CD。本实验是借鉴对照。','',
        'CD仍使用本项目双向平方CD/R²。两种损失的原始数值不能直接比较；所有结果用相同完整缺失目标CD²、F-score及几何诊断评价。等轮数不代表等计算成本。','',
        '## 固定缺口1000步','',
        '复用相同初始化seed42、512输入、固定128目标、Adam lr0.001及1000步；目标顺序不作为匹配约束。','',
        '|损失|CD²/R²↓|F-score↑|精确平均匹配距离/R↓|最近对应不同目标位置|','|---|---:|---:|---:|---:|']
    lines += [f'|{label}|{m["cd_over_R2"]:.8f}|{m["fscore"]:.6f}|{m["exact_distance_over_R"]:.6f}|{m["nearest_distinct_target_sites"]}/128|' for label,m in fixed.items()]
    lines += ['',f'![固定缺口](../../experiment_figures/{fixed_figure})','',
        '## 同器物16缺口200轮','',
        '同一司母戊鼎前16缺口训练、后4缺口评价，另一个青铜觥20缺口评价。只改变训练损失：seed42、batch4、lr0.001、输入512/输出与目标128、615236参数均固定，从头训练200轮，取last.pt；不使用新缺口挑选轮数或权重。复用CD基线并校验模型哈希与数据指纹。','',
        '|范围|损失|CD²↓|F-score↑|聚集比例↓|网格距离/R↓|','|---|---|---:|---:|---:|---:|']
    for role in roles:
        for label in runs:
            m=evaluations[label][role];q=quality[label][role]['means']
            lines.append(f'|{role}|{label}|{m["mean_local_cd_squared"]:.8f}|{m["mean_local_fscore"]:.6f}|{q["close_fraction"]:.2%}|{q["mesh_distance_over_R"]:.6f}|')
    lines += ['',f'![学习曲线](../../experiment_figures/{curve})','',
        '聚集为最近预测点距离<0.02R的比例，是工程诊断。网格仅用于评价，不用于预测投影或训练。目标点少于128时原采样会重复，因此一对一点索引不保证目标坐标都不同。','']
    for example in examples:
        lines += [f'### {example["role"]}',f'样本：`{example["path"]}`；两方法相同采样与PCA视角，真值PCA仅用于展示。',f'![{example["role"]}](../../experiment_figures/{example["figure"]})','']
    lines += ['## 限制','',
        '只用一个器物与一个训练种子；同器物新缺口只有4个，换器物只评价一个器物。本轮不能声称稳定青铜器泛化收益，原型保持冻结。固定缺口拟合的改善也不能直接转述成新缺口补全改善。']
    lines += ['','## 本轮判断','',
        '固定目标覆盖与训练样本拟合改善，但同器物新缺口的CD²、F-score及离面评价未改善，换器物评价也变差，因此当前配置不采用。对照同时改变对应方式、距离指数和半径缩放，不能把收益或退化全部归因于单独的一对一对应。下一步优先做固定优化更新数的训练缺口数量对照；数据覆盖不足只是待验证假设。']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (figures/'bronze_assignment_fit_notes.md').write_text('\n'.join(['# 一对一匹配对照图片','',
        '[完整报告](../diagnostics/assignment_fit/README.md)',curve+'：同固定目标的CD评价曲线和单器物训练CD曲线。',
        fixed_figure+'：固定缺口CD与精确匹配的三视角对照，仅用于拟合诊断。']+
        [e['figure']+'：'+e['role']+'，用于检查训练覆盖与新缺口迁移。' for e in examples]),encoding='utf-8')
    organize();print(json.dumps(dict(fixed=fixed,evaluations={label:{role:dict(cd=m['mean_local_cd_squared'],fscore=m['mean_local_fscore']) for role,m in results.items()} for label,results in evaluations.items()},geometry={label:{role:q['means'] for role,q in results.items()} for label,results in quality.items()}),indent=2))


if __name__=='__main__':main()
