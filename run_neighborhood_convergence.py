"""Resume both encoder variants with frozen 40/60-epoch milestones.

Original 20-epoch runs stay intact. Optimizer and RNG resume from copied last.pt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from main import ROOT
from train import train


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(run, destination):
    destination.mkdir(parents=True,exist_ok=False)
    for filename in ('best.pt','last.pt','history.json'):
        shutil.copy2(run/filename,destination/filename)
        assert sha256(run/filename)==sha256(destination/filename)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture',required=True,choices=['local','pointwise_xyz'])
    args=parser.parse_args()
    for seed in (42,43,44):
        source=ROOT/'outputs/runs'/f'bronze_{args.architecture}_seed{seed}'
        run=ROOT/'outputs/runs'/f'bronze_{args.architecture}_e60_seed{seed}'
        checkpoint=torch.load(source/'last.pt',weights_only=True,map_location='cpu')
        assert checkpoint['epoch']==20
        assert checkpoint['config']['architecture']==args.architecture and checkpoint['config']['seed']==seed
        snapshot(source,run)
        (run/'resume_source.json').write_text(json.dumps(dict(source=str(source),epoch=20,
            last_sha256=sha256(source/'last.pt'),best_sha256=sha256(source/'best.pt'),
            config=checkpoint['config']),indent=2),encoding='utf-8')
        for epochs in (40,60):
            train(ROOT/'outputs/datasets/bronze_20261013_seed42',run,
                  epochs=epochs,input_points=512,output_points=128,target_points=128,
                  batch_size=8,lr=.001,device='cpu',output_mode='tanh',decoder_width=1140,
                  architecture=args.architecture,encoder_anchors=32,encoder_neighbors=16,
                  seed=seed,resume=run/'last.pt')
            if epochs==40:
                snapshot(run,run/'milestones/epoch40')


if __name__=='__main__':
    main()
