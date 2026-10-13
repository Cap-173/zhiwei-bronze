"""Fixed-epoch out-of-fold metrics, mesh diagnostics and matched report figures."""
import json
import gc
from pathlib import Path
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from completion import load_checkpoint, context_points, sample_points
from density_loss import density_chamfer
from prepare_dataset import restore
from training_data import read_manifest, checked_sample, dataset_fingerprint, evaluate_entries
from organize_figures import organize
from main import ROOT


def display_name(source):
    path=Path(source)
    parts=path.parts
    if '青铜器模型_obj' in parts:
        return parts[parts.index('青铜器模型_obj')+1]
    return path.parent.name


def render_case(sample, predictions, name, figures):
    center,radius=sample['center'],float(sample['radius'])
    visible=(sample['input']-center)/radius
    visible=visible[np.linalg.norm(visible,axis=1)<=1.5]
    target=(sample['local_target']-center)/radius
    _,_,basis=np.linalg.svd(np.concatenate((visible,target)),full_matrices=False)
    visible,target=visible@basis.T,target@basis.T
    panels=[('Reference',target),('Damaged',None)]+[(label,((points-center)/radius)@basis.T) for label,points in predictions.items()]
    fig,axes=plt.subplots(3,4,figsize=(13,10))
    for column,(label,points) in enumerate(panels):
        for row,(a,b) in enumerate(((0,1),(0,2),(1,2))):
            ax=axes[row,column]
            ax.scatter(visible[:,a],visible[:,b],s=3,c='#6aa7bb',alpha=.35)
            if points is not None:
                if column==0:
                    ax.scatter(points[:,a],points[:,b],s=8,c='#209b62',alpha=.65)
                else:
                    ax.scatter(target[:,a],target[:,b],s=7,facecolors='none',edgecolors='#209b62',linewidths=.5,alpha=.3)
                    distances=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
                    np.fill_diagonal(distances,np.inf)
                    ax.scatter(points[:,a],points[:,b],s=12,c=np.where(distances.min(1)<.02,'#d62828','#e77d20'))
            ax.add_patch(plt.Circle((0,0),1,fill=False,ls='--',color='#999'))
            ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),aspect='equal',xlabel=f'PC{a+1}/R',ylabel=f'PC{b+1}/R')
            ax.grid(alpha=.15)
            if row==0: ax.set_title(label)
    fig.suptitle(f'{name} | out-of-fold epoch60 | blue: visible; green: target; orange: prediction; red: NN < 0.02R')
    fig.tight_layout(rect=(0,0,1,.95))
    filename=f'bronze_cv4_{name}.png'
    fig.savefig(figures/filename,dpi=180);plt.close(fig)
    return filename,basis


def main():
    torch.set_num_threads(2)
    base=ROOT/'outputs/datasets/bronze_object_cv4'
    protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    output=ROOT/'outputs/repeats/bronze_object_cv4'
    if (output/'summary.json').exists():
        raise FileExistsError('已有完整CV汇总，不覆盖。')
    output.mkdir(parents=True,exist_ok=True)
    figures=ROOT/'outputs/experiment_figures'
    sources={r['id']:r['source'] for r in json.loads((ROOT/'outputs/local_models/import_20261013_125555_685154/manifest.json').read_text(encoding='utf-8'))['models']}
    rows=[];fold_rows=[];examples=[]
    for fold in protocol['folds']:
        dataset=Path(fold['dataset']);manifest=read_manifest(dataset)
        entries=[e for e in manifest['samples'] if e['split']=='val']
        assert len(entries)==60 and len({e['model_id'] for e in entries})==3
        assert not set(manifest['splits'])&set(protocol['excluded_original_test_models'])
        assert not set(fold['train_models'])&set(fold['heldout_models'])
        models={};evaluations={}
        for label in protocol['architectures']:
            run=ROOT/'outputs/runs'/f'bronze_cv4_fold{fold["fold"]}_{label}_seed42'
            model,checkpoint=load_checkpoint(run/'last.pt')
            assert checkpoint['epoch']==60 and checkpoint['seed']==42
            assert checkpoint['dataset_fingerprint']==dataset_fingerprint(dataset,manifest)==fold['dataset_fingerprint']
            assert sum(p.numel() for p in model.parameters())==615236
            assert checkpoint['config']['architecture']==label
            for field,value in protocol['fixed_config'].items():
                assert checkpoint['config'][field]==value
            metric=evaluate_entries(model,dataset,entries,512,43,torch.device('cpu'))
            (run/'oof_epoch60.json').write_text(json.dumps(metric,indent=2),encoding='utf-8')
            models[label]=model;evaluations[label]={r['path']:r for r in metric['samples']}
            fold_rows.append(dict(fold=fold['fold'],label=label,heldout_models=fold['heldout_models'],
                cd=metric['model_macro_cd_squared'],fscore=metric['model_macro_fscore'],epoch=60))
        # Meshes are loaded one at a time; only this fold's heldout objects.
        for mid in fold['heldout_models']:
            print(f'OOF mesh evaluation fold{fold["fold"]} {mid}',flush=True)
            mesh=trimesh.load(sources[mid],force='mesh',process=False,skip_materials=True)
            for index,entry in enumerate(entries):
                if entry['model_id']!=mid: continue
                sample=checked_sample(dataset/entry['path']/'training.npz')
                context=context_points(sample['input'],sample['center'],512,np.random.default_rng(np.random.SeedSequence([43,0,index])))
                region=np.r_[sample['center'],sample['radius']]
                predicted={}
                for label,model in models.items():
                    with torch.no_grad():
                        pred=model(torch.tensor(context,dtype=torch.float32)[None],torch.tensor(region,dtype=torch.float32)[None])[0]
                        dcd=[]
                        for repeat in range(5):
                            target=sample_points(sample['local_target'],128,np.random.default_rng(np.random.SeedSequence([31415,index,repeat])))
                            dcd.append(float(density_chamfer(pred[None],torch.tensor(target,dtype=torch.float32)[None],alpha=1000,exponent=1)))
                        distances=torch.cdist(pred[None],pred[None],compute_mode='donot_use_mm_for_euclid_dist')[0]
                        distances.fill_diagonal_(float('inf'))
                        nearest=distances.min(1).values/float(sample['radius'])
                    prediction=pred.numpy();predicted[label]=prediction
                    _,mesh_distance,_=trimesh.proximity.closest_point(mesh,restore(prediction,sample['offset'],sample['scale']))
                    if not np.isfinite(mesh_distance).all() or not np.isfinite(dcd).all():
                        raise ValueError(f'非有限几何诊断: {mid} {entry["path"]} {label}')
                    metric=evaluations[label][entry['path']]
                    rows.append(dict(fold=fold['fold'],label=label,model_id=mid,
                        original_path=entry['original_sample_path'],fold_path=entry['path'],
                        cd=metric['local_cd_squared'],fscore=metric['local_fscore'],
                        dcd=float(np.mean(dcd)),close_fraction=float((nearest<.02).float().mean()),
                        mesh_distance_over_radius=float(mesh_distance.mean()/(float(sample['radius'])*float(sample['scale']))),
                        dcd_target_sampling_with_replacement=len(sample['local_target'])<128))
                # Deterministic example independent of prediction errors.
                if mid==fold['heldout_models'][0] and entry==next(e for e in entries if e['model_id']==mid):
                    name=f'fold{fold["fold"]}_{mid}_sample0'
                    filename,basis=render_case(sample,predicted,name,figures)
                    np.savez_compressed(output/f'{name}.npz',input=sample['input'],target=sample['local_target'],
                        center=sample['center'],radius=sample['radius'],context=context,display_basis=basis,**predicted)
                    examples.append(dict(name=name,figure=filename,model_id=mid,original_path=entry['original_sample_path']))
            del mesh;gc.collect()
        del models;gc.collect()
    fields=('cd','fscore','dcd','close_fraction','mesh_distance_over_radius')
    objects=[]
    for label in protocol['architectures']:
        for mid in protocol['development_models']:
            selected=[r for r in rows if r['label']==label and r['model_id']==mid]
            assert len(selected)==20 and len({r['fold'] for r in selected})==1
            objects.append(dict(label=label,model_id=mid,name=display_name(sources[mid]),
                fold=selected[0]['fold'],samples=20,**{field:float(np.mean([r[field] for r in selected])) for field in fields}))
    means={label:{field:float(np.mean([r[field] for r in objects if r['label']==label])) for field in fields} for label in protocol['architectures']}
    pairs=[]
    for mid in protocol['development_models']:
        selected={r['label']:r for r in objects if r['model_id']==mid}
        pairs.append(dict(model_id=mid,name=selected['local']['name'],fold=selected['local']['fold'],
            cd_relative_excess=selected['local']['cd']/selected['pointwise_xyz']['cd']-1,
            cd_difference=selected['local']['cd']-selected['pointwise_xyz']['cd']))
    assert len(rows)==480 and len(objects)==24
    report=dict(protocol=protocol,checkpoint='last.pt at fixed epoch60',sample_count_per_method=240,
        object_count=12,fold_metrics=fold_rows,object_metrics=objects,object_macro_means=means,
        paired_objects=pairs,samples=rows,examples=examples,
        interpretation='Exploratory CV with a single training seed on previously studied development objects; not new independent blind data. Holes correlated within each object; training sets overlap across folds. No independence/significance claim.')
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for i,label in enumerate(protocol['architectures']):
        selected=[next(r for r in objects if r['label']==label and r['model_id']==mid) for mid in protocol['development_models']]
        for ax,field in zip(axes,('cd','fscore')):
            ax.bar(np.arange(12)+(i-.5)*.36,[r[field] for r in selected],width=.36,label=label)
            ax.set(xticks=np.arange(12),xticklabels=[mid.replace('bronze_','') for mid in protocol['development_models']],xlabel='Heldout object ID',ylabel=field)
            ax.legend();ax.grid(axis='y',alpha=.2)
    axes[0].set_yscale('log');axes[0].set_title('Out-of-fold CD squared (log scale)')
    axes[1].set_title('Out-of-fold F-score')
    fig.suptitle('4 folds / train9 heldout3 / fixed epoch60 / training seed42 / matched615236 parameters')
    fig.tight_layout();filename='bronze_object_cv4_per_object.png';fig.savefig(figures/filename,dpi=180);plt.close(fig)
    excess=means['local']['cd']/means['pointwise_xyz']['cd']-1
    wins=sum(p['cd_difference']<0 for p in pairs)
    lines=['# 器物级四折交叉验证','',
        '用原训练/验证12个器物，排除原测试bronze_06与bronze_07。每折9器物180缺口训练、3器物60缺口留出；每个器物恰好一次留出，总计240个折外缺口/方案。原NPZ逐字节复制、哈希校验，不重新挖洞或归一化。','',
        '[分组交叉验证原则](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)：将器物ID作为组，避免同器物的相关缺口跨训练与留出。这里是固定均衡四折的自定义实现，非调用sklearn。','',
        '训练种子固定42、划分种子20261013，两个方法每折均从头训练60轮，无旧模型暖启动；参数均615236，其余配置保持。主结果固定第60轮last.pt，不用留出器物选择best或调整预算。训练代码会保存best.pt，但它不进入主结果。曲线监测不改变学习率或训练规则。','',
        '此前三种子检查训练随机性；本轮固定一个种子隔离划分变化，没有进行四折×三种子复验。所有12器物已用于此前开发，故属于探索性复验，不能称为新独立盲测。','',
        '## 每折名单','', '|折|训练器物|留出器物|','|---|---|---|']
    lines += [f'|{f["fold"]}|{", ".join(f["train_models"])}|{", ".join(f["heldout_models"])}|' for f in protocol['folds']]
    lines += ['','## 器物宏平均','', '|方案|CD²↓|F-score↑|DCD↓|聚集比例↓|网格距离/R↓|','|---|---:|---:|---:|---:|---:|']
    lines += [f'|{label}|{m["cd"]:.8f}|{m["fscore"]:.6f}|{m["dcd"]:.6f}|{m["close_fraction"]:.2%}|{m["mesh_distance_over_radius"]:.6f}|' for label,m in means.items()]
    lines += ['',f'邻域local相对逐点对照的宏平均CD²变化：{excess:+.2%}（正值为误差更大）。local在{wins}/12个器物上CD²更低，逐器物结果如下。','',
        '|器物|名称|折|逐点CD²|邻域CD²|邻域误差变化|','|---|---|---:|---:|---:|---:|']
    for pair in pairs:
        selected={r['label']:r for r in objects if r['model_id']==pair['model_id']}
        lines.append(f'|{pair["model_id"]}|{pair["name"]}|{pair["fold"]}|{selected["pointwise_xyz"]["cd"]:.8f}|{selected["local"]["cd"]:.8f}|{pair["cd_relative_excess"]:+.2%}|')
    lines += ['',f'![逐器物对照](../../experiment_figures/{filename})','',
        '每器物先平均20个缺口，再对12器物等权平均。DCD固定128目标点抽样5次，alpha1000、指数1；抽样不足时允许重复并记录。聚集阈值0.02R是工程诊断；源OBJ仅用于评价，未用于训练、推理投影或参数选择。','',
        '## 固定案例图','', '每折展示按ID排序的第一个留出器物的第一个缺口，不按误差挑图。蓝为可见、绿为真值、橙为预测、红为近邻小于0.02R的聚集点。真值PCA仅用于显示。']
    for example in examples:
        lines += ['',f'### {example["name"]}',f'样本：`{example["original_path"]}`。报告位置：器物级泛化复验。',f'![{example["name"]}](../../experiment_figures/{example["figure"]})']
    lines += ['','## 解释限制','',
        '240个缺口不是240个独立器物；四折训练集有重叠，不能把四折当作完全独立实验来计算显著性。结果只支持已知小规模器物池上的划分敏感性分析。器型差异、分组种子及训练随机性仍需更多独立器物与重复划分核对。原交付模型保持冻结。']
    lines += ['', '本轮判断：当前邻域改造在这组开发器物与固定训练预算下未得到支持，保留逐点对照作为研究基线。这个结论不等于否定完整PointNet++。分组边界、原始样本复制与最终数值有限性应一并审查。']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (figures/'bronze_object_cv4_notes.md').write_text('\n'.join(['# 器物级交叉验证图','',
        filename+'：12个器物各自折外预测的CD²与F-score，左纵轴对数尺度。用于器物级划分复验。',
        '[完整报告、名单和指标](../repeats/bronze_object_cv4/README.md)']+
        [f'{e["figure"]}：{e["original_path"]}，固定案例三投影对照。' for e in examples]),encoding='utf-8')
    organize();print(json.dumps(dict(means=means,local_cd_relative_excess=excess,local_cd_wins=wins),indent=2))


if __name__=='__main__':
    main()
