"""独立评价整个数据划分，不参与模型选择。"""
import argparse
import json
from pathlib import Path
import sys

from completion import load_checkpoint
from train import resolve_device
from training_data import dataset_fingerprint, evaluate_entries, read_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threshold', type=float, default=0.02)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if not 0 < args.threshold < float('inf'):
            raise ValueError('F-score 阈值必须为有限正数。')
        manifest = read_manifest(args.dataset)
        entries = [e for e in manifest['samples'] if e['split'] == args.split]
        if not entries:
            raise ValueError(f'没有 {args.split} 样本；不能用训练样本替代测试。')
        model, checkpoint = load_checkpoint(args.checkpoint)
        device = resolve_device(args.device)
        model.to(device)
        fingerprint = dataset_fingerprint(args.dataset, manifest)
        report = evaluate_entries(model, args.dataset, entries, checkpoint['input_points'],
                                  checkpoint['seed']+1, device, args.threshold)
        report.update(split=args.split, checkpoint=str(args.checkpoint.resolve()),
                      checkpoint_selection_split=checkpoint.get('selection_split', 'unknown'),
                      matches_training_dataset=fingerprint == checkpoint.get('dataset_fingerprint'),
                      dataset_fingerprint=fingerprint)
        output = args.output or args.checkpoint.parent / f'evaluation_{args.split}.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        # 保留每次评价报告，防止不同参数的结果被无意覆盖。
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(f'{args.split}: {report["sample_count"]} samples, {report["model_count"]} models')
        print(f'Local CD2={report["mean_local_cd_squared"]:.6f} | F-score={report["mean_local_fscore"]:.6f}')
        print(f'报告: {output.resolve()}')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'评价失败: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
