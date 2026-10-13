"""Data invariants and frozen single-object fitting protocol; no original test access."""
import json
import hashlib
from pathlib import Path
import shutil
import numpy as np
import torch
from completion import context_points
from neighborhood_encoder import local_groups
from prepare_dataset import fit_transform, restore
from training_data import read_manifest, checked_sample
from main import ROOT


def check_invariants(sample):
    full=sample['target'];mask=sample['mask']
    radius=float(sample['radius']);center=sample['center']
    checks=dict(finite=all(np.isfinite(sample[k]).all() for k in ('input','target','local_target','center','offset','scale','radius')),
        partition=(mask.dtype==bool and len(mask)==len(full) and np.array_equal(full[~mask],sample['input']) and np.array_equal(full[mask],sample['local_target'])),
        target_in_sphere=bool((np.linalg.norm(sample['local_target']-center,axis=1)<=radius+1e-9).all()),
        visible_outside_sphere=bool((np.linalg.norm(sample['input']-center,axis=1)>radius-1e-9).all()))
    offset,scale=fit_transform(restore(sample['input'],sample['offset'],sample['scale']))
    checks['visible_only_normalization']=bool(np.allclose(offset,sample['offset'],rtol=1e-9,atol=1e-10) and np.isclose(scale,float(sample['scale']),rtol=1e-9))
    return checks


def main():
    torch.set_num_threads(2)
    source=ROOT/'outputs/datasets/bronze_20261013_seed42'
    output=ROOT/'outputs/diagnostics/basic_fit'
    output.mkdir(parents=True,exist_ok=False)
    manifest=read_manifest(source)
    entries=[e for e in manifest['samples'] if e['split']!='test']
    rows=[];references={};orientation={}
    for index,entry in enumerate(entries):
        sample=checked_sample(source/entry['path']/'training.npz')
        checks=check_invariants(sample)
        raw=restore(sample['target'],sample['offset'],sample['scale'])
        mid=entry['model_id']
        if mid not in references:
            references[mid]=raw
            _,singular,basis=np.linalg.svd(raw-raw.mean(0),full_matrices=False)
            orientation[mid]=dict(raw_bbox_extent=np.ptp(raw,axis=0).tolist(),pca_basis=basis.tolist(),
                singular_values=singular.tolist(),note='PCA sign and axes are not semantic orientation labels; no cross-object pose alignment was applied.')
        checks['same_object_coordinates_unchanged']=bool(np.allclose(raw,references[mid],rtol=1e-9,atol=1e-10))
        context=context_points(sample['input'],sample['center'],512,np.random.default_rng(np.random.SeedSequence([43,0,index]))).astype(np.float32)
        relative=(context-sample['center'])/float(sample['radius'])
        centers,offsets=local_groups(torch.tensor(relative,dtype=torch.float32)[None],32,16)
        unique_neighbors=[len(torch.unique(group,dim=0)) for group in offsets[0]]
        rows.append(dict(model_id=mid,path=entry['path'],checks=checks,all_passed=all(checks.values()),
            full_points=len(raw),target_points=len(sample['local_target']),missing_fraction=len(sample['local_target'])/len(raw),
            normalized_radius=float(sample['radius']),target_radius_max_over_R=float(np.linalg.norm(sample['local_target']-sample['center'],axis=1).max()/float(sample['radius'])),
            context_duplicate_fraction=1-len(np.unique(context,axis=0))/512,
            context_within_2R=float((np.linalg.norm(relative,axis=1)<=2).mean()),
            fps_anchors_within_2R=float((centers[0].norm(dim=1)<=2).float().mean()),
            knn_unique_points_mean=float(np.mean(unique_neighbors)),target_sampling_replacement=len(sample['local_target'])<128))
    summary=dict(sample_count=len(rows),model_count=len(references),passed=sum(r['all_passed'] for r in rows),
        min_target_points=min(r['target_points'] for r in rows),max_target_points=max(r['target_points'] for r in rows),
        target_sampling_replacement_cases=sum(r['target_sampling_replacement'] for r in rows),
        missing_fraction_range=[min(r['missing_fraction'] for r in rows),max(r['missing_fraction'] for r in rows)],
        normalized_radius_range=[min(r['normalized_radius'] for r in rows),max(r['normalized_radius'] for r in rows)],
        mean_context_duplicate_fraction=float(np.mean([r['context_duplicate_fraction'] for r in rows])),
        mean_fps_anchors_within_2R=float(np.mean([r['fps_anchors_within_2R'] for r in rows])),
        mean_knn_unique_points=float(np.mean([r['knn_unique_points_mean'] for r in rows])))
    (output/'data_audit.json').write_text(json.dumps(dict(summary=summary,orientation=orientation,samples=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    if summary['passed']!=len(rows):
        raise ValueError('基础不变量失败，停止拟合并先检查数据。')
    selected=[e for e in entries if e['model_id']=='bronze_02']
    assert len(selected)==20
    train_entries,heldout=selected[:16],selected[16:]
    dataset=ROOT/'outputs/datasets/bronze_single_object_fit'
    dataset.mkdir(parents=True,exist_ok=False)
    copied=[]
    for index,entry in enumerate(train_entries):
        relative=f'train/sample_{index:04d}'
        destination=dataset/relative;destination.mkdir(parents=True)
        original=source/entry['path']/'training.npz'
        shutil.copy2(original,destination/'training.npz')
        assert original.read_bytes()==(destination/'training.npz').read_bytes()
        copied.append({**entry,'path':relative,'split':'train','original_sample_path':entry['path']})
    (dataset/'manifest.json').write_text(json.dumps(dict(schema_version=1,status='complete',seed=42,
        samples=copied,splits={'bronze_02':'train'},evaluation_ready=False,
        note='Training-only16 holes of one physical object. Four unseen holes kept outside this manifest; no fake IDs or weakened object-split guard.'),indent=2),encoding='utf-8')
    protocol=dict(object_id='bronze_02',object_name='司母戊鼎',source_dataset=str(source),single_object_dataset=str(dataset),
        single_hole=selected[0],training_holes=train_entries,unseen_same_object_holes=heldout,
        cross_object_evaluation=[e for e in entries if e['model_id']=='bronze_03'],
        architectures=['pointwise_xyz','local'],training_seed=42,evaluation_seed=43,
        single_hole_steps=1000,single_object_epochs=200,batch_size=4,learning_rate=.001,
        input_points=512,output_points=128,target_points=128,decoder_width=1140,encoder_anchors=32,encoder_neighbors=16,
        output_mode='tanh',loss='CD squared divided by R squared; no auxiliary loss',
        primary_checkpoint='final fixed step/epoch, not training-loss best or unseen-hole selected',
        target_floor='FPS128 and fixed5 random128 draws from full missing target; reference only, not a learned prediction',
        purpose='Engineering fit sanity check, not a new method or generalized bronze completion score',
        excluded_original_test_models=sorted({e['model_id'] for e in manifest['samples'] if e['split']=='test'}))
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
