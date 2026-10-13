"""Train both matched encoders from scratch on one frozen development fold."""
import argparse
import json
from pathlib import Path
import time
from main import ROOT
from train import train
from training_data import read_manifest, dataset_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=int,required=True,choices=range(4))
    args=parser.parse_args()
    base=ROOT/'outputs/datasets/bronze_object_cv4'
    protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    fold=protocol['folds'][args.fold]
    dataset=Path(fold['dataset'])
    manifest=read_manifest(dataset)
    assert dataset_fingerprint(dataset,manifest)==fold['dataset_fingerprint']
    assert len(fold['train_models'])==9 and len(fold['heldout_models'])==3
    assert not set(manifest['splits'])&set(protocol['excluded_original_test_models'])
    for architecture in protocol['architectures']:
        output=ROOT/'outputs/runs'/f'bronze_cv4_fold{args.fold}_{architecture}_seed42'
        print(f'Fold {args.fold} | {architecture} | train9 / heldout3 | fixed60 epochs',flush=True)
        started=time.perf_counter()
        train(dataset,output,epochs=protocol['epochs'],seed=protocol['training_seed'],device='cpu',
              architecture=architecture,**protocol['fixed_config'])
        (output/'cv_provenance.json').write_text(json.dumps(dict(fold=args.fold,
            protocol=str(base/'protocol.json'),primary_checkpoint='last.pt',
            elapsed_seconds=time.perf_counter()-started),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
