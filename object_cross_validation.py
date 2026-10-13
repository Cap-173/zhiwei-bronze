"""Prepare object-disjoint development folds; reuse unchanged simulated holes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from main import ROOT
from training_data import read_manifest, dataset_fingerprint


def grouped_folds(model_ids, folds=4, seed=20261013):
    ids=sorted(model_ids)
    if len(ids)!=len(set(ids)) or not 2<=folds<=len(ids):
        raise ValueError('器物ID需唯一，折数至少2且不超过器物数。')
    order=np.random.default_rng(seed).permutation(ids)
    return [sorted(part.tolist()) for part in np.array_split(order,folds)]


def prepare(source, output, folds=4, split_seed=20261013):
    source,output=Path(source),Path(output)
    manifest=read_manifest(source)
    eligible=[e for e in manifest['samples'] if e['split'] in ('train','val')]
    ids=sorted({e['model_id'] for e in eligible})
    excluded=sorted({e['model_id'] for e in manifest['samples'] if e['split']=='test'})
    assignments=grouped_folds(ids,folds,split_seed)
    if output.exists():
        raise FileExistsError('交叉验证数据目录已存在，不覆盖。')
    output.mkdir(parents=True)
    source_hashes={e['path']:hashlib.sha256((source/e['path']/'training.npz').read_bytes()).hexdigest() for e in eligible}
    records=[]
    for number,heldout in enumerate(assignments):
        directory=output/f'fold_{number}'
        directory.mkdir()
        splits={mid:'val' if mid in heldout else 'train' for mid in ids}
        entries=[]
        for index,entry in enumerate(eligible):
            split=splits[entry['model_id']]
            relative=f'{split}/sample_{index:04d}'
            target=directory/relative
            target.mkdir(parents=True)
            shutil.copy2(source/entry['path']/'training.npz',target/'training.npz')
            assert hashlib.sha256((target/'training.npz').read_bytes()).hexdigest()==source_hashes[entry['path']]
            entries.append({**entry,'split':split,'path':relative,'original_sample_path':entry['path'],
                            'source_npz_sha256':source_hashes[entry['path']]})
        fold_manifest=dict(schema_version=1,status='complete',seed=manifest['seed'],
            point_limit=manifest.get('point_limit'),radius_range=manifest.get('radius_range'),
            fraction_range=manifest.get('fraction_range'),splits=splits,samples=entries,
            evaluation_ready=False,fold_number=number,split_seed=split_seed,
            source_dataset=str(source.resolve()),excluded_original_test_models=excluded,
            note='Development object CV. Only unchanged training.npz copied; no new damage or normalization. Primary evaluation uses fixed final epoch, not heldout-selected best.')
        (directory/'manifest.json').write_text(json.dumps(fold_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        read_manifest(directory)
        records.append(dict(fold=number,train_models=sorted(set(ids)-set(heldout)),heldout_models=heldout,
            train_samples=sum(e['split']=='train' for e in entries),heldout_samples=sum(e['split']=='val' for e in entries),
            dataset=str(directory.resolve()),dataset_fingerprint=dataset_fingerprint(directory,fold_manifest)))
    protocol=dict(status='prepared',fold_count=folds,split_seed=split_seed,development_models=ids,
        excluded_original_test_models=excluded,folds=records,source_dataset=str(source.resolve()),
        source_hashes=source_hashes,training_seed=42,epochs=60,evaluation_seed=43,
        architectures=['pointwise_xyz','local'],parameters=615236,
        primary_checkpoint='last.pt at fixed epoch60; heldout not used for stopping, lr or selection',
        secondary_checkpoint='best.pt is stored by trainer but excluded from primary OOF comparison',
        grouping_reference='https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data',
        fixed_config=dict(input_points=512,output_points=128,target_points=128,batch_size=8,lr=.001,
            amp=False,output_mode='tanh',decoder_width=1140,encoder_anchors=32,encoder_neighbors=16),
        scope='Exploratory development CV; same twelve known objects and fixed holes; one training seed; no new blind test or independent data collection')
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/datasets/bronze_20261013_seed42')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/datasets/bronze_object_cv4')
    args=parser.parse_args()
    protocol=prepare(args.source,args.output)
    print(json.dumps(dict(folds=protocol['folds'],excluded=protocol['excluded_original_test_models']),indent=2))


if __name__=='__main__':
    main()
