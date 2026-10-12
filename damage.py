"""球形选区、模拟局部缺失与训练样本保存。直接运行即可查看示例。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plyfile import PlyData, PlyElement

from main import ROOT, make_demo, read_cloud


def sphere_mask(points, center, radius):
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('球心必须是三个有限坐标。')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('半径必须为有限正数。')
    return np.linalg.norm(points - center, axis=1) <= radius


def write_cloud(path, points, colors=None):
    fields = [(axis, 'f8') for axis in ('x', 'y', 'z')]
    if colors is not None:
        fields += [(channel, 'u1') for channel in ('red', 'green', 'blue')]
    vertices = np.empty(len(points), dtype=fields)
    for i, axis in enumerate(('x', 'y', 'z')):
        vertices[axis] = points[:, i]
    if colors is not None:
        for i, channel in enumerate(('red', 'green', 'blue')):
            vertices[channel] = colors[:, i]
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(path))


def save_sample(directory, points, colors, mask, center, radius, source, invalid=0):
    # 所有验证先于文件写入，避免留下无效训练样本。
    if not mask.any():
        raise ValueError('选区没有包含任何点，请调整球心或增大半径。')
    if mask.all():
        raise ValueError('选区包含了全部点，请缩小半径，保留修复上下文。')
    directory.mkdir(parents=True, exist_ok=False)
    for name, selected in [('target', np.ones(len(points), dtype=bool)),
                           ('input', ~mask), ('removed', mask)]:
        write_cloud(directory / f'{name}.ply', points[selected],
                    colors[selected] if colors is not None else None)
    np.save(directory / 'mask.npy', mask)
    metadata = dict(schema_version=1, source=str(Path(source).resolve()),
                    center=[float(v) for v in center], radius=float(radius),
                    total_points=len(points), remaining_points=int((~mask).sum()),
                    removed_points=int(mask.sum()), invalid_points_excluded=invalid,
                    mask_reference='target.ply vertex order; true means removed',
                    selection_rule='Euclidean distance <= radius',
                    coordinate_units='same as source', task='local_completion')
    (directory / 'region.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (directory / 'instruction.txt').write_text('补全这个区域。\n', encoding='utf-8')
    return metadata


def comparison(points, colors, mask, center, radius, output, max_points=50000):
    figure = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                          subplot_titles=('完整点云：橙色为选区', '残缺点云：选区内点已删除'))
    rng = np.random.default_rng(42)
    # 使用同一组显示索引，确保选区外在两侧完全对应。
    selected = np.flatnonzero(mask)
    retained = np.flatnonzero(~mask)
    selected_budget = min(len(selected), max(1, int(max_points * len(selected) / len(points))))
    retained_budget = min(len(retained), max_points - selected_budget)
    selected = rng.choice(selected, selected_budget, replace=False)
    retained = rng.choice(retained, retained_budget, replace=False)
    if colors is None:
        color = '#67c5ce'
    else:
        color = [f'rgb({r},{g},{b})' for r, g, b in colors[retained]]
    for col in (1, 2):
        p = points[retained]
        figure.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
                                     marker=dict(size=2, color=color), name='选区外',
                                     showlegend=col == 1), row=1, col=col)
    p = points[selected]
    figure.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
                                 marker=dict(size=3, color='#ff9c42'), name='将删除的点'), row=1, col=1)
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 24))
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    for col in (1, 2):
        figure.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.12, showscale=False,
                                   colorscale=[[0, '#ff9c42'], [1, '#ff9c42']],
                                   hoverinfo='skip', name='球形选区'), row=1, col=col)
    low, high = points.min(axis=0), points.max(axis=0)
    span = high - low
    padding = np.maximum(span * 0.05, max(float(span.max()), 1e-6) * 0.01)
    scene = dict(aspectmode='data', camera=dict(eye=dict(x=2.2, y=0.6, z=1.0)),
                 **{f'{axis}axis': dict(title=axis.upper(), range=[low[i]-padding[i], high[i]+padding[i]])
                    for i, axis in enumerate(('x', 'y', 'z'))})
    figure.update_layout(template='plotly_dark', scene=scene, scene2=scene,
                         title=f'球形局部缺失 · 删除 {mask.sum():,} / {len(points):,} 点 · 半径 {radius:g}',
                         margin=dict(l=0, r=0, b=0, t=90), height=750)
    figure.write_html(str(output), include_plotlyjs=True, config=dict(scrollZoom=True, responsive=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path)
    parser.add_argument('--center', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--radius', type=float)
    parser.add_argument('--output-dir', type=Path, help='新的样本目录，已有目录不会覆盖')
    parser.add_argument('--max-points', type=int, default=50000)
    args = parser.parse_args(argv)
    if args.max_points < 2:
        parser.error('--max-points 至少为 2')
    if args.input is not None and (args.center is None or args.radius is None):
        parser.error('真实模型请同时指定 --center X Y Z 和 --radius R，单位与模型一致')
    try:
        path = args.input
        if path is None:
            path = ROOT / 'outputs' / 'demo_vessel.ply'
            if not path.exists():
                make_demo(path)
        points, colors, _, invalid = read_cloud(path)
        if not len(points):
            raise ValueError('文件没有有效点。')
        center = args.center if args.center is not None else [0.9, 0, 0.25]
        radius = args.radius if args.radius is not None else 0.4
        mask = sphere_mask(points, center, radius)
        directory = args.output_dir or ROOT / 'outputs' / 'samples' / datetime.now().strftime('sample_%Y%m%d_%H%M%S_%f')
        metadata = save_sample(directory, points, colors, mask, center, radius, path, invalid)
        comparison(points, colors, mask, center, radius, directory / 'comparison.html', args.max_points)
        print(f'球心: {center} | 半径: {radius}')
        print(f'完整点: {len(points):,} | 删除点: {metadata["removed_points"]:,} | 剩余点: {metadata["remaining_points"]:,}')
        print(f'样本已保存: {directory.resolve()}')
        print(f'用浏览器打开: {(directory / "comparison.html").resolve()}')
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'无法生成缺失样本: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
