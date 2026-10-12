"""加载补全模型并生成残缺/预测/参考三维预览。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from completion import chamfer_squared, context_points, load_checkpoint
from damage import write_cloud
from main import ROOT, read_cloud
from prepare_dataset import fit_transform, normalize, restore


def predict_cloud(model, checkpoint, points, center, radius):
    offset, scale = fit_transform(points)
    normalized = normalize(points, offset, scale)
    center_normalized = normalize(np.asarray(center), offset, scale)
    radius_normalized = radius / scale
    context = context_points(normalized, center_normalized, checkpoint['input_points'],
                             np.random.default_rng(checkpoint['seed'] + 1))
    with torch.no_grad():
        predicted = model(torch.tensor(context, dtype=torch.float32)[None],
                          torch.tensor(np.r_[center_normalized, radius_normalized], dtype=torch.float32)[None])[0].numpy()
    return restore(predicted, offset, scale), offset, scale


def preview(points, prediction, target, output):
    panels = [('残缺输入', points), ('补全结果：橙色为预测', points)]
    if target is not None:
        panels.append(('完整参考（仅用于评价）', target))
    figure = make_subplots(rows=1, cols=len(panels), specs=[[{'type': 'scene'}] * len(panels)],
                          subplot_titles=[title for title, _ in panels])
    all_points = np.concatenate((points, prediction, target if target is not None else points))
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    pad = max(float(np.max(high-low)) * 0.05, 1e-6)
    scene = dict(aspectmode='data', **{f'{axis}axis': dict(range=[low[i]-pad, high[i]+pad], title=axis.upper())
                                     for i, axis in enumerate(('x', 'y', 'z'))})
    layouts = {}
    for column, (_, cloud) in enumerate(panels, 1):
        shown = cloud
        if len(cloud) > 50000:
            shown = cloud[np.random.default_rng(42).choice(len(cloud), 50000, replace=False)]
        figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2], mode='markers',
                                     marker=dict(size=2, color='#67c5ce'), showlegend=False), row=1, col=column)
        layouts['scene' if column == 1 else f'scene{column}'] = scene
    figure.add_trace(go.Scatter3d(x=prediction[:, 0], y=prediction[:, 1], z=prediction[:, 2], mode='markers',
                                 marker=dict(size=3, color='#ff9c42'), name='预测补全点'), row=1, col=2)
    figure.update_layout(template='plotly_dark', title='基础局部补全结果', height=750, **layouts)
    figure.write_html(str(output), include_plotlyjs=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--sample', type=Path, help='包含 input.ply 和 region.json 的样本目录')
    parser.add_argument('--input', type=Path, help='真实残缺 PLY，必须同时指定球心半径')
    parser.add_argument('--center', type=float, nargs=3)
    parser.add_argument('--radius', type=float)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.input and (args.sample or args.center is None or args.radius is None):
        parser.error('--input 必须配合 --center 和 --radius，不能同时使用 --sample')
    try:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            runs = sorted((ROOT / 'outputs' / 'runs').glob('run_*/best.pt'))
            if not runs:
                raise ValueError('请先运行 train.py。')
            checkpoint_path = runs[-1]
        model, checkpoint = load_checkpoint(checkpoint_path)
        target = local_target = None
        if args.input:
            input_path, center, radius = args.input, args.center, args.radius
            scope = 'real input; no ground truth'
        else:
            sample = args.sample
            if sample is None:
                dataset = Path(checkpoint['dataset'])
                manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
                # 仅预览一个样本，不将其当作整个测试集成绩。
                entries = sorted(manifest['samples'], key=lambda e: {'test': 0, 'val': 1, 'train': 2}[e['split']])
                sample = dataset / entries[0]['path']
                scope = entries[0]['split'] + ' sample preview only'
            else:
                scope = 'user-selected sample; split not verified'
            metadata = json.loads((sample / 'region.json').read_text(encoding='utf-8'))
            input_path, center, radius = sample / 'input.ply', metadata['center'], metadata['radius']
            if (sample / 'target.ply').exists():
                target = read_cloud(sample / 'target.ply')[0]
            if (sample / 'removed.ply').exists():
                local_target = read_cloud(sample / 'removed.ply')[0]
        if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
            raise ValueError('球心必须有限，半径必须为有限正数。')
        points, _, _, _ = read_cloud(input_path)
        prediction, offset, scale = predict_cloud(model, checkpoint, points, center, radius)
        output = args.output_dir or ROOT / 'outputs' / 'predictions' / datetime.now().strftime('prediction_%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        write_cloud(output / 'generated.ply', prediction)
        write_cloud(output / 'completed.ply', np.concatenate((points, prediction)))
        metrics = dict(scope=scope, input_points=len(points), generated_points=len(prediction),
                       unedited_coordinates_preserved=True, checkpoint=str(checkpoint_path.resolve()),
                       input=str(input_path.resolve()))
        if local_target is not None and len(local_target):
            with torch.no_grad():
                score = float(chamfer_squared(torch.tensor(normalize(prediction, offset, scale), dtype=torch.float32)[None],
                                              torch.tensor(normalize(local_target, offset, scale), dtype=torch.float32)[None]))
            metrics['local_cd_squared_normalized'] = score
            metrics['local_cd_squared_original_units'] = score * scale ** 2
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        preview(points, prediction, target, output / 'comparison.html')
        print(f'输入点: {len(points)} | 生成点: {len(prediction)} | 评价范围: {scope}')
        print(f'结果: {output.resolve()}')
        print(f'预览: {(output / "comparison.html").resolve()}')
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'预测失败: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
