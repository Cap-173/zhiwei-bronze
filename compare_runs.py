"""对照运行的验证集诊断，只比较相同数据与训练预算。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from analyze_validation import analyze
from main import ROOT


def compare(dataset, runs, output):
    if len(runs) < 2:
        raise ValueError('至少需要两个运行。')
    for label in runs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
            raise ValueError('实验标签仅允许英文字母、数字、下划线和连字符。')
    budget = None
    for path in runs.values():
        checkpoint = torch.load(Path(path)/'last.pt', weights_only=True, map_location='cpu')
        keys = ('seed', 'input_points', 'output_points', 'target_points', 'batch_size', 'lr', 'amp')
        current = dict(epoch=checkpoint['epoch'], dataset_fingerprint=checkpoint['dataset_fingerprint'],
                       **{key: checkpoint['config'][key] for key in keys})
        history_path = Path(path)/'history.json'
        current['device'] = json.loads(history_path.read_text(encoding='utf-8')).get('device', 'unknown') if history_path.exists() else 'unknown'
        if budget is not None and current != budget:
            raise ValueError('不同运行的数据、点数、种子、轮数或训练预算不同，不能作直接消融比较。')
        budget = current
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, directory in runs.items():
        best = torch.load(Path(directory)/'best.pt', weights_only=True, map_location='cpu')
        if best.get('selection_split') != 'val':
            raise ValueError('对照实验需要根据验证集选择 best.pt。')
        report = analyze(dataset, Path(directory)/'best.pt', output/label)
        summary = report['summary']
        rows.append(dict(label=label, run=str(Path(directory).resolve()), best_epoch=best['epoch'],
                         output_mode=best.get('output_mode', 'soft'), boundary_weight=best['config'].get('boundary_weight', 0.),
                         val_cd_squared=summary['local_cd_squared'],
                         val_boundary_coverage=summary['boundary_coverage_at_0_1_radius'],
                         val_boundary_error_over_radius=summary['boundary_mean_distance_over_radius'],
                         predicted_outer_fraction=summary['predicted_outer_fraction']))
    report = dict(split='val', budget=budget, runs=rows, note='One seed, short training; not proof of general improvement or surface continuity.')
    (output/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    figure = make_subplots(rows=1, cols=3, subplot_titles=('验证 CD²（越小越好）', '外缘覆盖率（越大越好）', '外缘距离 / R（越小越好）'))
    for column, field in enumerate(('val_cd_squared', 'val_boundary_coverage', 'val_boundary_error_over_radius'), 1):
        figure.add_trace(go.Bar(x=[row['label'] for row in rows], y=[row[field] for row in rows], showlegend=False), row=1, col=column)
    figure.update_layout(template='plotly_dark', height=650, title='相同数据与训练预算的验证集对照')
    figure.write_html(str(output/'comparison.html'), include_plotlyjs=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--runs', required=True, nargs='+', help='标签=运行目录')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        runs = {}
        for specification in args.runs:
            label, directory = specification.split('=', 1)
            if label in runs:
                raise ValueError('实验标签重复。')
            runs[label] = Path(directory)
        output = args.output_dir or ROOT/'outputs'/'comparisons'/datetime.now().strftime('compare_%Y%m%d_%H%M%S_%f')
        report = compare(args.dataset, runs, output)
        print(json.dumps(report['runs'], indent=2))
        print(f'对照报告: {output.resolve()}')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'比较失败: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
