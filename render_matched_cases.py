"""Same validation holes, context sampling, scales and views across five methods."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from completion import context_points, load_checkpoint
from training_data import checked_sample, read_manifest, dataset_fingerprint
from analyze_validation import diagnose
from prepare_dataset import restore
from organize_figures import organize
from main import ROOT

RUNS = {'CD small':'bronze_cd128_seed42', 'CD wide':'bronze_wide1140_seed42',
        'DCD':'bronze_dcd128_seed42', 'Hierarchical':'bronze_hier32_aux01_seed42',
        'Repulsion':'bronze_repulsion01_seed42'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['five_methods','neighborhood','neighborhood60'],default='five_methods')
    args=parser.parse_args()
    convergence=args.experiment=='neighborhood60'
    neighborhood=args.experiment in ('neighborhood','neighborhood60')
    active_runs=({'CD wide':'bronze_wide1140_seed42','Pointwise control':'bronze_pointwise_xyz_seed42',
                  'Local neighborhoods':'bronze_local_seed42'} if neighborhood else RUNS)
    if convergence:
        active_runs={'Pointwise control':'bronze_pointwise_xyz_e60_seed42','Local neighborhoods':'bronze_local_e60_seed42'}
    namespace='neighborhood60' if convergence else 'neighborhood' if neighborhood else 'matched'
    torch.set_num_threads(2)
    dataset=ROOT/'outputs/datasets/bronze_20261013_seed42'
    output=ROOT/'outputs/diagnostics'/f'{namespace}_cases_seed42'
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录已有结果，请保留后选择新目录。')
    output.mkdir(parents=True,exist_ok=True)
    figures=ROOT/'outputs/experiment_figures'
    manifest=read_manifest(dataset)
    fingerprint=dataset_fingerprint(dataset,manifest)
    entries=[e for e in manifest['samples'] if e['split']=='val']
    models={}
    provenance={}
    for label,run in active_runs.items():
        path=ROOT/'outputs/runs'/run/'best.pt'
        model,checkpoint=load_checkpoint(path)
        assert checkpoint['dataset_fingerprint']==fingerprint
        assert checkpoint['input_points']==512 and checkpoint['output_points']==128 and checkpoint['seed']==42
        models[label]=model
        provenance[label]=dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               epoch=checkpoint['epoch'],parameters=sum(p.numel() for p in model.parameters()),config=checkpoint['config'])
    cases=[]
    for index,entry in enumerate(entries):
        sample=checked_sample(dataset/entry['path']/'training.npz')
        context=context_points(sample['input'],sample['center'],512,np.random.default_rng(np.random.SeedSequence([43,0,index])))
        region=np.r_[sample['center'],sample['radius']]
        with torch.no_grad():
            predictions={label:model(torch.tensor(context,dtype=torch.float32)[None],torch.tensor(region,dtype=torch.float32)[None])[0].numpy() for label,model in models.items()}
        metrics={label:diagnose(pred,sample['local_target'].astype(np.float32),sample['center'],float(sample['radius'])) for label,pred in predictions.items()}
        cases.append(dict(index=index,entry=entry,sample=sample,context=context,predictions=predictions,metrics=metrics))
    chosen=[]
    if convergence:
        previous=json.loads((ROOT/'outputs/diagnostics/matched_cases_seed42/comparison.json').read_text(encoding='utf-8'))
        for record in previous['cases']:
            chosen.append((record['name'],next(c for c in cases if c['entry']['path']==record['entry']['path'])))
    else:
        for mid in sorted({e['model_id'] for e in entries}):
            ordered=sorted([c for c in cases if c['entry']['model_id']==mid],key=lambda c:c['metrics']['CD wide']['local_cd_squared'])
            chosen += [(f'{mid}_median',ordered[len(ordered)//2]),(f'{mid}_highest',ordered[-1])]
    sources={r['id']:r['source'] for r in json.loads((ROOT/'outputs/local_models/import_20261013_125555_685154/manifest.json').read_text(encoding='utf-8'))['models']}
    meshes={mid:trimesh.load(sources[mid],force='mesh',process=False,skip_materials=True) for mid in {c['entry']['model_id'] for _,c in chosen}}
    records=[]
    overview,overview_axes=plt.subplots(2,2,figsize=(12,10),subplot_kw={'projection':'3d'})
    for number,(name,case) in enumerate(chosen,1):
        print(f'Rendering {name}: {case["entry"]["path"]}',flush=True)
        sample=case['sample']; center=sample['center']; radius=float(sample['radius'])
        visible=(sample['input']-center)/radius
        visible=visible[np.linalg.norm(visible,axis=1)<=1.5]
        target=(sample['local_target']-center)/radius
        # Ground truth defines display axes only; predictions have already been computed.
        _,_,basis=np.linalg.svd(np.concatenate((visible,target)),full_matrices=False)
        context_view=visible@basis.T; target_view=target@basis.T
        panels=[('Complete reference',target_view,None),('Damaged input',None,None)]
        row_metrics={}
        arrays=dict(input=sample['input'],target=sample['target'],local_target=sample['local_target'],center=center,radius=radius,context=case['context'],display_basis=basis)
        for label,pred in case['predictions'].items():
            local=(pred-center)/radius
            distance=np.linalg.norm(local[:,None]-local[None,:],axis=-1)
            np.fill_diagonal(distance,np.inf)
            close=distance.min(axis=1)<.02
            original=restore(pred,sample['offset'],sample['scale'])
            _,mesh_distance,_=trimesh.proximity.closest_point(meshes[case['entry']['model_id']],original)
            metrics={**case['metrics'][label], 'close_fraction_0_02R':float(close.mean()),
                     'mesh_distance_over_radius':float(mesh_distance.mean()/(radius*float(sample['scale'])))}
            row_metrics[label]=metrics
            panels.append((label,local@basis.T,close))
            arrays[label.replace(' ','_')]=pred
        np.savez_compressed(output/f'{name}.npz',**arrays)
        fig,axes=plt.subplots(3,len(panels),figsize=(3.15*len(panels),10))
        for column,(label,points,close) in enumerate(panels):
            for row,(a,b) in enumerate(((0,1),(0,2),(1,2))):
                ax=axes[row,column]
                ax.scatter(context_view[:,a],context_view[:,b],s=3,c='#6aa7bb',alpha=.35)
                if points is not None:
                    if column==0:
                        ax.scatter(points[:,a],points[:,b],s=8,c='#209b62',alpha=.65)
                    else:
                        ax.scatter(target_view[:,a],target_view[:,b],s=7,facecolors='none',edgecolors='#209b62',alpha=.3,linewidths=.5)
                        ax.scatter(points[:,a],points[:,b],s=12,c=np.where(close,'#d62828','#e77d20'),alpha=.9)
                ax.add_patch(plt.Circle((0,0),1,fill=False,ls='--',color='#999999'))
                ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),aspect='equal',xlabel=f'PC{a+1}/R',ylabel=f'PC{b+1}/R')
                ax.grid(alpha=.15)
                if row==0:
                    title=label
                    if label in row_metrics:
                        m=row_metrics[label]
                        title+=f'\nCD2={m["local_cd_squared"]:.5f}; close={m["close_fraction_0_02R"]:.1%}\nmesh/R={m["mesh_distance_over_radius"]:.3f}'
                    ax.set_title(title,fontsize=10)
        fig.suptitle(f'{name} | same hole, context seed43 and view | blue: visible; green: target; orange: generated; red: NN < 0.02R',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.95))
        filename=f'bronze_{namespace}_{name}.png'
        fig.savefig(figures/filename,dpi=180); plt.close(fig)
        ax=overview_axes.flat[number-1]
        full=sample['target']; inp=sample['input']
        ax.scatter(inp[:,0],inp[:,1],inp[:,2],s=1,c='#6aa7bb',alpha=.3)
        missing=sample['local_target']
        ax.scatter(missing[:,0],missing[:,1],missing[:,2],s=7,c='#209b62')
        low=full.min(axis=0); high=full.max(axis=0); mid=(low+high)/2; span=(high-low).max()*.55
        ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[1]-span,mid[1]+span),zlim=(mid[2]-span,mid[2]+span),title=name)
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=20,azim=35)
        records.append(dict(name=name,entry=case['entry'],validation_index=case['index'],figure=filename,metrics=row_metrics))
    overview.suptitle('Selected validation holes: visible input (blue), missing reference (green)')
    overview.tight_layout();overview.savefig(figures/f'bronze_{namespace}_hole_locations.png',dpi=180);plt.close(overview)
    report=dict(selection='per validation object median (rank11/20) and highest wide-CD error; all methods use identical cases',
                dataset_fingerprint=fingerprint,evaluation_seed=43,provenance=provenance,cases=records,
                all_validation_metrics=[dict(entry=c['entry'],metrics=c['metrics']) for c in cases],
                limitations='Selected examples, not random population samples. PCA uses target for display only. Ball boundary coverage is not a physical seam metric. Exact duplicate points may overlap in images. Models differ in parameter count; use wide CD for capacity comparison.')
    if convergence:
        report['selection']='Reuse the four archived 20-epoch wide-CD cases; no reselection using 60-epoch predictions'
        report['budget']=60
        report['limitations']='Two matched parameter-count methods at the same 60-epoch budget; displayed checkpoints are per-run validation best within budget, not necessarily epoch60. Selected validation cases, not random population samples; PCA target is display only, not inference.'
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 同一缺口方案对照','',
           '方案：'+', '.join(active_runs)+'。另附完整局部参考和残缺输入，三个投影视角。所有模型seed42，输入抽样统一seed43、每样本索引；同一案例使用完全相同的512点模型输入。可见展示为完整局部上下文，不等于仅512点模型输入。','',
           '## 案例选择与图例','',
           '每个验证器物按宽MLP CD²排序，选择第11/20及最高误差，共四例；不能把这些图当作随机代表案例。仅使用验证集。每图统一PCA视角、范围±1.5R和点大小。真实目标仅用于评价和展示，未用于预测。','',
           '蓝色为可见点，绿色为缺失目标，橙色为预测；红色为具有距离小于0.02R近邻的预测点，并不直接表示离面。虚线圆为球形缺口半径。三视角用于减少单一投影遮挡。','',
           '图注建议：同一验证缺口不同补全方案的三视角局部对照。所有方法使用相同输入采样和显示坐标；绿色参考只用于评价。','',
           f'![缺口位置](../../experiment_figures/bronze_{namespace}_hole_locations.png)','',
           '## 案例指标','',
           '|案例|方案|CD²↓|聚集比例↓|网格距离/R↓|球外缘覆盖↑|','|---|---|---:|---:|---:|---:|']
    if convergence:
        lines.insert(2,'本组为60轮预算内验证best检查点；沿用原20轮宽CD选定的四例，没有依据60轮结果重新挑图，best实际轮次见comparison.json。')
    for record in records:
        for label,m in record['metrics'].items():
            coverage=m['boundary_coverage_at_0_1_radius']
            edge='无外缘目标' if coverage is None else f'{coverage:.1%}'
            lines.append(f'|{record["name"]}|{label}|{m["local_cd_squared"]:.6f}|{m["close_fraction_0_02R"]:.1%}|{m["mesh_distance_over_radius"]:.4f}|{edge}|')
    for record in records:
        lines += ['',f'### {record["name"]}',f'样本：`{record["entry"]["path"]}`。报告位置：结果分析—同缺口定性对照。',
                  f'![{record["name"]}](../../experiment_figures/{record["figure"]})']
    lines += ['','## 如何解读','',
              '先观察三投影中预测是否覆盖绿色目标，再用网格距离检查离面。红点用于定位局部聚集。球外缘覆盖只是人工球形挖洞的覆盖诊断，不能当作真实破损接缝连续性。不能从二维图或到完整网格距离单独断言接缝已修复。','',
              ('主要容量对照是Pointwise control与Local neighborhoods，二者参数完全相同。' if neighborhood else '容量公平的主要对照是宽MLP、分层和排斥；DCD和小CD参数更少。')+'各图并非独立器物重复，也不能取代全40缺口及三训练种子的汇总。原始预测、完整参考、PCA矩阵和实际模型输入保存于同目录NPZ，指标和模型哈希见comparison.json。']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (figures/f'bronze_{namespace}_cases_notes.md').write_text('\n'.join(['# 同缺口方案图：用途与来源','',
        f'[完整说明及指标](../diagnostics/{output.name}/README.md)','',
        f'bronze_{namespace}_hole_locations.png：四个案例在器物中的位置，用于结果分析开头。']+
        [f'{r["figure"]}：{r["entry"]["path"]}，三视角局部对照，用于补全覆盖、聚集和离面问题分析。' for r in records]),encoding='utf-8')
    organize()
    print(output/'README.md')


if __name__=='__main__':
    main()
