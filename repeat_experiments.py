"""配对多种子重复实验，只使用验证集，不重新划分数据。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from completion import load_checkpoint
from main import ROOT
from train import train
from training_data import dataset_fingerprint, evaluate_entries, read_manifest


def summarize(rows):
    seeds = sorted({row['seed'] for row in rows})
    if len(seeds) < 2:
        raise ValueError('重复实验至少需要两个种子。')
    paired, aggregate = [], {}
    for seed in seeds:
        pair = [row for row in rows if row['seed'] == seed]
        if len(pair) != 2 or {r['output_mode'] for r in pair} != {'soft', 'tanh'}:
            raise ValueError('每个种子必须有且只有一对 soft/tanh 结果。')
        baseline = next(row for row in pair if row['output_mode'] == 'soft')
        changed = next(row for row in pair if row['output_mode'] == 'tanh')
        a, b = baseline['val_cd_squared'], changed['val_cd_squared']
        paired.append(dict(seed=seed, baseline_cd_squared=a, tanh_cd_squared=b,
                           relative_reduction_percent=100*(a-b)/a if a else None,
                           tanh_better=b < a))
    for mode in ('soft', 'tanh'):
        selected = [r for r in rows if r['output_mode'] == mode]
        cd = [r['val_cd_squared'] for r in selected]
        fscore = [r['val_fscore'] for r in selected]
        aggregate[mode] = dict(mean_cd_squared=float(np.mean(cd)), sample_std_cd_squared=float(np.std(cd, ddof=1)),
                               mean_fscore=float(np.mean(fscore)), sample_std_fscore=float(np.std(fscore, ddof=1)))
    return dict(seeds=seeds, pairs=paired, aggregate=aggregate,
                improved_seeds=sum(p['tanh_better'] for p in paired), total_seeds=len(seeds),
                note='Sample standard deviation across training seeds, not confidence interval or significance test.')


def run_repeats(dataset, output, runs_root, seeds, epochs=20, evaluation_seed=43,
                reuse=None, device='cpu'):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError('至少两个不同的非负种子。')
    manifest = read_manifest(dataset)
    entries = [e for e in manifest['samples'] if e['split'] == 'val']
    if not entries:
        raise ValueError('需要独立验证集。')
    fingerprint = dataset_fingerprint(dataset, manifest)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    report = dict(status='running', split='val', seeds=seeds, epochs=epochs, evaluation_seed=evaluation_seed,
                  dataset=str(Path(dataset).resolve()), dataset_fingerprint=fingerprint, runs=rows,
                  note='Dataset split fixed; evaluation context sampling fixed across seeds. Test split not evaluated.')
    def save():
        temporary = output/'progress.tmp'
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(output/'results.json')
    save()
    try:
        for seed in seeds:
            for mode in ('soft', 'tanh'):
                directory = Path((reuse or {}).get((seed, mode), runs_root/f'repeat_{mode}_seed{seed}'))
                config = dict(input_points=512, output_points=128, seed=seed, lr=0.001,
                              batch_size=8, target_points=256, amp=False, output_mode=mode, boundary_weight=0.)
                if directory.exists():
                    saved = torch.load(directory/'last.pt', map_location='cpu', weights_only=True)
                    prior = {'output_mode': 'soft', 'boundary_weight': 0., **saved['config']}
                    if prior != config or saved['dataset_fingerprint'] != fingerprint or saved['epoch'] > epochs:
                        raise ValueError(f'已有运行预算或数据不匹配: {directory}')
                    if saved['epoch'] < epochs:
                        print(f'Resume seed={seed}, mode={mode}', flush=True)
                        train(dataset, directory, epochs=epochs, device=device, resume=directory/'last.pt', **config)
                else:
                    print(f'Start seed={seed}, mode={mode}', flush=True)
                    train(dataset, directory, epochs=epochs, device=device, **config)
                model, best = load_checkpoint(directory/'best.pt')
                if best.get('selection_split') != 'val':
                    raise ValueError('最佳模型必须依据验证集选择。')
                # 最终验证评价统一使用同一个采样种子，不混入评价采样波动。
                metrics = evaluate_entries(model, dataset, entries, 512, evaluation_seed, torch.device('cpu'))
                rows.append(dict(seed=seed, output_mode=mode, best_epoch=best['epoch'],
                                 run=str(directory.resolve()), val_cd_squared=metrics['mean_local_cd_squared'],
                                 val_fscore=metrics['mean_local_fscore']))
                save()
                print(f'Finished seed={seed}, mode={mode}: val CD2={metrics["mean_local_cd_squared"]:.6f}', flush=True)
        report['summary'] = summarize(rows)
        report['status'] = 'complete'
        save()
    except Exception:
        report['status'] = 'failed'
        save()
        raise
    figure = make_subplots(rows=1, cols=2, subplot_titles=('各训练种子的验证 CD²', '配对 CD² 降幅'))
    for mode in ('soft', 'tanh'):
        selected = [r for r in rows if r['output_mode'] == mode]
        figure.add_trace(go.Scatter(x=[r['seed'] for r in selected], y=[r['val_cd_squared'] for r in selected],
                                   mode='lines+markers', name=mode), row=1, col=1)
    figure.add_trace(go.Bar(x=[p['seed'] for p in report['summary']['pairs']],
                           y=[p['relative_reduction_percent'] for p in report['summary']['pairs']],
                           name='降幅 %'), row=1, col=2)
    figure.update_layout(template='plotly_dark', height=650, title='固定数据划分的配对多种子验证实验')
    figure.write_html(str(output/'comparison.html'), include_plotlyjs=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--evaluation-seed', type=int, default=43)
    parser.add_argument('--runs-root', type=Path, default=ROOT/'outputs'/'runs')
    parser.add_argument('--baseline42', type=Path)
    parser.add_argument('--tanh42', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        reuse = {}
        if args.baseline42:
            reuse[(42, 'soft')] = args.baseline42
        if args.tanh42:
            reuse[(42, 'tanh')] = args.tanh42
        output = args.output_dir or ROOT/'outputs'/'repeats'/datetime.now().strftime('repeat_%Y%m%d_%H%M%S_%f')
        report = run_repeats(args.dataset, output, args.runs_root, args.seeds, args.epochs, args.evaluation_seed, reuse)
        print(json.dumps(report['summary'], indent=2))
        print(f'重复实验报告: {output.resolve()}')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'重复实验失败: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
