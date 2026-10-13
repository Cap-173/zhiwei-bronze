"""在指定球形区域补全PLY点云；支持真实残缺输入及模拟删除。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np

from completion import load_checkpoint
from damage import sphere_mask, write_cloud
from main import ROOT, read_cloud
from predict import predict_cloud, preview


def complete(input_path, checkpoint_path, center, radius, output, remove_region=False):
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError('球心必须为三个有限坐标，半径必须为有限正数。')
    output = Path(output)
    if output.exists():
        raise FileExistsError('输出目录已存在，不覆盖。')
    points, colors, total, invalid = read_cloud(input_path)
    if remove_region:
        keep = ~sphere_mask(points, center, radius)
        if keep.all() or not keep.any():
            raise ValueError('模拟删除必须同时保留球内和球外点。')
        visible = points[keep]
        visible_colors = colors[keep] if colors is not None else None
    else:
        visible, visible_colors = points, colors
    if len(visible) < 2:
        raise ValueError('需要至少两个有效可见点。')
    model, checkpoint = load_checkpoint(checkpoint_path)
    prediction, offset, scale = predict_cloud(model, checkpoint, visible, center, radius)
    if not np.isfinite(prediction).all() or not (np.linalg.norm(prediction-center, axis=1) <= radius*(1+1e-5)).all():
        raise ValueError('预测点坐标或球形范围无效。')
    completed = np.concatenate((visible, prediction))
    completed_colors = None if visible_colors is None else np.concatenate((visible_colors, np.full((len(prediction), 3), 180, dtype=np.uint8)))
    output.mkdir(parents=True)
    write_cloud(output / 'input.ply', visible, visible_colors)
    write_cloud(output / 'generated.ply', prediction)
    write_cloud(output / 'completed.ply', completed, completed_colors)
    preview(visible, prediction, None, output / 'comparison.html')
    report = dict(input=str(Path(input_path).resolve()), checkpoint=str(Path(checkpoint_path).resolve()),
                  center=center.tolist(), radius=float(radius), remove_region=remove_region,
                  source_total_points=total, invalid_points_excluded=invalid,
                  input_points=len(visible), generated_points=len(prediction),
                  completed_points=len(completed), unedited_coordinates_preserved=True,
                  known_colors_preserved=visible_colors is not None,
                  generated_color='neutral gray placeholder; not learned' if visible_colors is not None else None,
                  offset=offset.tolist(), scale=scale, scope='inference only; no ground truth used',
                  output_type='point cloud; not reconstructed mesh')
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--center', required=True, type=float, nargs=3)
    p.add_argument('--radius', required=True, type=float)
    p.add_argument('--remove-region', action='store_true', help='完整输入上先删除球内点；残缺输入不要使用')
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    output = a.output_dir or ROOT / 'outputs' / 'completions' / datetime.now().strftime('completion_%Y%m%d_%H%M%S_%f')
    result = complete(a.input, a.checkpoint, a.center, a.radius, output, a.remove_region)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    print(output.resolve())
