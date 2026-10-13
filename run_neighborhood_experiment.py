"""Frozen 20-epoch, three-seed neighborhood encoder ablation."""
import argparse
import json
from main import ROOT
from train import train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture',required=True,choices=['local','pointwise_xyz'])
    args=parser.parse_args()
    for seed in (42,43,44):
        train(ROOT/'outputs/datasets/bronze_20261013_seed42',
              ROOT/'outputs/runs'/f'bronze_{args.architecture}_seed{seed}',
              epochs=20,input_points=512,output_points=128,target_points=128,
              batch_size=8,lr=.001,device='cpu',output_mode='tanh',decoder_width=1140,
              architecture=args.architecture,encoder_anchors=32,encoder_neighbors=16,seed=seed)


if __name__=='__main__':
    main()
