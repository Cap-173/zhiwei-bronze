"""读取 PLY 点云，生成可离线旋转、缩放的 HTML 预览。"""
import argparse
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parent


def read_cloud(path):
    ply = PlyData.read(str(path))
    if 'vertex' not in ply:
        raise ValueError('PLY 没有 vertex 元素。')
    vertices = ply['vertex'].data
    names = vertices.dtype.names or ()
    if not all(axis in names for axis in ('x', 'y', 'z')):
        raise ValueError('顶点必须包含 x、y、z 坐标。')
    points = np.column_stack([vertices[a] for a in ('x', 'y', 'z')]).astype(float)
    finite = np.isfinite(points).all(axis=1)
    colors = None
    if all(c in names for c in ('red', 'green', 'blue')):
        colors = np.column_stack([vertices[c] for c in ('red', 'green', 'blue')])[finite]
        colors = np.clip(np.nan_to_num(colors.astype(float)), 0, 255).astype(np.uint8)
    return points[finite], colors, len(points), int((~finite).sum())


def make_demo(path):
    """生成器皿示例，不是真实扫描数据。"""
    rng = np.random.default_rng(42)
    z = rng.uniform(-1, 1, 12000)
    angle = rng.uniform(0, 2 * np.pi, len(z))
    radius = 0.65 + 0.30 * np.cos(z * np.pi / 2) + 0.12 * np.exp(-((z - 0.9) / 0.12) ** 2)
    points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    vertices = np.empty(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i, axis in enumerate(('x', 'y', 'z')):
        vertices[axis] = points[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, 'vertex')], text=True).write(str(path))


def render_cloud(points, colors, output, title, max_points):
    indices = np.arange(len(points))
    if len(points) > max_points:
        indices = np.random.default_rng(42).choice(len(points), max_points, replace=False)
    shown = points[indices]
    marker = dict(size=2, opacity=0.9)
    if colors is None:
        marker.update(color=shown[:, 2], colorscale='Viridis', showscale=True,
                      colorbar=dict(title='Z 坐标'))
    else:
        marker['color'] = [f'rgb({r},{g},{b})' for r, g, b in colors[indices]]
    figure = go.Figure(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2],
                                  mode='markers', marker=marker))
    figure.update_layout(title=f'{title} · 有效点 {len(points):,} · 显示 {len(shown):,}',
                         template='plotly_dark', margin=dict(l=0, r=0, b=0, t=60),
                         scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs=True,
                      config=dict(scrollZoom=True, responsive=True))
    return len(shown)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path, help='PLY 文件，省略则生成示例')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'point_cloud.html')
    parser.add_argument('--max-points', type=int, default=50000, help='预览点数上限')
    args = parser.parse_args(argv)
    if args.max_points <= 0:
        parser.error('--max-points 必须大于 0')
    try:
        path = args.input
        if path is None:
            path = ROOT / 'outputs' / 'demo_vessel.ply'
            make_demo(path)
        points, colors, total, invalid = read_cloud(path)
        if not len(points):
            raise ValueError('文件没有有效点，无法显示。')
        print(f'输入: {path.resolve()}')
        print(f'总点数: {total:,} | 有效点: {len(points):,} | 无效点: {invalid:,}')
        for i, axis in enumerate(('X', 'Y', 'Z')):
            print(f'{axis} 范围: {points[:, i].min():.6g} ~ {points[:, i].max():.6g}')
        displayed = render_cloud(points, colors, args.output, path.name, args.max_points)
        print(f'预览点数: {displayed:,}（仅预览采样，原始文件不变）')
        print(f'请用浏览器打开: {args.output.resolve()}')
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'无法生成预览: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
