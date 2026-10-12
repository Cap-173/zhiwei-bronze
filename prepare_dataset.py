"""批量生成局部补全数据。默认示例仅用于调试，不作泛化评估。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from damage import comparison, save_sample, sphere_mask
from main import ROOT, make_demo, read_cloud


def fit_transform(points):
    """只由可见输入确定变换，避免使用缺失区域的真实坐标。"""
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError('归一化需要非空、有限坐标。')
    offset = (points.min(axis=0) + points.max(axis=0)) / 2
    scale = float(np.linalg.norm(points - offset, axis=1).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('点云空间尺度为零或无效。')
    return offset, scale


def normalize(points, offset, scale):
    return (points - offset) / scale


def restore(points, offset, scale):
    return points * scale + offset


def choose_region(points, rng, radius_range, fraction_range):
    # 半径相对于完整样本包围盒对角线，用于构造模拟损坏，不作为模型输入。
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    for _ in range(200):
        center = points[rng.integers(len(points))].copy()
        radius = float(rng.uniform(*radius_range) * diagonal)
        mask = sphere_mask(points, center, radius)
        fraction = float(mask.mean())
        if fraction_range[0] <= fraction <= fraction_range[1]:
            return center, radius, mask
    raise ValueError('200 次尝试未找到符合缺失比例的球形选区，请调整半径或缺失比例。')


def split_models(model_ids, seed):
    if len(model_ids) < 3:
        return {model_id: 'train' for model_id in model_ids}
    order = np.random.default_rng(seed).permutation(len(model_ids))
    n_test = max(1, int(len(order) * 0.1))
    n_val = max(1, int(len(order) * 0.1))
    assignments = {}
    for i, index in enumerate(order):
        assignments[model_ids[index]] = 'test' if i < n_test else 'val' if i < n_test + n_val else 'train'
    return assignments


def load_training_sample(path):
    """无需 PyTorch 即可读取训练输入、局部目标、选区及逆变换。"""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def build_dataset(models, output, samples=10, point_limit=4096, seed=42,
                  radius_range=(0.08, 0.18), fraction_range=(0.02, 0.20), split_assignments=None):
    ids = [model_id for model_id, _ in models]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('需要模型，且每个模型 ID 必须唯一。')
    if samples < 1 or point_limit < 16:
        raise ValueError('每模型样本数至少为 1，点数上限至少为 16。')
    if not (0 < radius_range[0] <= radius_range[1] < 1):
        raise ValueError('半径比例需要满足 0 < 最小值 <= 最大值 < 1。')
    if not (0 < fraction_range[0] <= fraction_range[1] < 1):
        raise ValueError('缺失比例需要满足 0 < 最小值 <= 最大值 < 1。')
    if output.exists():
        raise FileExistsError(f'输出目录已存在，不会覆盖: {output}')
    if split_assignments is not None:
        if set(split_assignments) != set(ids) or any(s not in ('train', 'val', 'test') for s in split_assignments.values()):
            raise ValueError('指定划分必须覆盖全部模型且只包含 train/val/test。')
        if 'train' not in split_assignments.values():
            raise ValueError('指定划分必须包含训练模型。')
        splits = dict(split_assignments)
    else:
        splits = split_models(ids, seed)
    rng = np.random.default_rng(seed)
    prepared = []
    fingerprints = set()
    # 先检查全部源文件，精确重复的点云不能作为不同器物划分。
    for model_id, path in models:
        points, colors, total, invalid = read_cloud(path)
        if len(points) < 16:
            raise ValueError(f'{model_id}: 至少需要 16 个有效点。')
        ordered = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
        fingerprint = hashlib.sha256(ordered.tobytes()).hexdigest()
        if fingerprint in fingerprints:
            raise ValueError('存在相同坐标的模型，请勿将同一器物当作不同实例。')
        fingerprints.add(fingerprint)
        if len(points) > point_limit:
            indices = np.sort(rng.choice(len(points), point_limit, replace=False))
            points = points[indices]
            colors = colors[indices] if colors is not None else None
        fit_transform(points)
        prepared.append((model_id, path, points, colors, total, invalid))
    output.mkdir(parents=True)
    entries = []
    manifest = dict(schema_version=1, seed=seed, point_limit=point_limit,
                    radius_range=list(radius_range), fraction_range=list(fraction_range),
                    splits=splits, status='building', samples=entries,
                    evaluation_ready=all(s in splits.values() for s in ('train', 'val', 'test')),
                    note='Split by supplied physical model ID; explicit splits preserved when provided.')
    manifest_path = output / 'manifest.json'
    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    write_manifest()
    try:
        for model_index, (model_id, path, points, colors, total, invalid) in enumerate(prepared):
            for index in range(samples):
                center, radius, mask = choose_region(points, rng, radius_range, fraction_range)
                offset, scale = fit_transform(points[~mask])
                normalized = normalize(points, offset, scale)
                local_center = normalize(center, offset, scale)
                directory = output / splits[model_id] / f'model_{model_index:04d}_sample_{index:04d}'
                metadata = save_sample(directory, points, colors, mask, center, radius, path, invalid)
                metadata.update(model_id=model_id, split=splits[model_id], source_total_points=total,
                                normalization=dict(offset=offset.tolist(), scale=scale,
                                                   fitted_on='input.ply; visible points only'))
                (directory / 'region.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
                np.savez_compressed(directory / 'training.npz', input=normalized[~mask],
                                    local_target=normalized[mask], target=normalized, mask=mask,
                                    center=local_center, radius=radius / scale, offset=offset, scale=scale)
                if model_index == 0 and index == 0:
                    comparison(points, colors, mask, center, radius, directory / 'comparison.html')
                entries.append(dict(model_id=model_id, split=splits[model_id],
                                    path=directory.relative_to(output).as_posix(),
                                    input_points=int((~mask).sum()), target_points=int(mask.sum())))
                write_manifest()
        manifest['status'] = 'complete'
        write_manifest()
    except Exception:
        manifest['status'] = 'failed'
        write_manifest()
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, help='JSON 对象：器物实例 ID 到 PLY 路径；相对路径基于 JSON 目录')
    parser.add_argument('--splits', type=Path, help='可选 JSON 对象：模型 ID 到 train/val/test，用于保留公开数据官方划分')
    parser.add_argument('--samples-per-model', type=int, default=10)
    parser.add_argument('--points', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--radius-range', type=float, nargs=2, default=(0.08, 0.18))
    parser.add_argument('--missing-range', type=float, nargs=2, default=(0.02, 0.20))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.models:
            mapping = json.loads(args.models.read_text(encoding='utf-8'))
            if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
                raise ValueError('模型清单必须是 ID 到路径的 JSON 对象。')
            models = [(key, args.models.resolve().parent / value) for key, value in mapping.items()]
        else:
            path = ROOT / 'outputs' / 'demo_vessel.ply'
            if not path.exists():
                make_demo(path)
            models = [('demo_vessel', path)]
        output = args.output_dir or ROOT / 'outputs' / 'datasets' / datetime.now().strftime('dataset_%Y%m%d_%H%M%S_%f')
        manifest = build_dataset(models, output, args.samples_per_model, args.points, args.seed,
                                 args.radius_range, args.missing_range,
                                 json.loads(args.splits.read_text(encoding='utf-8')) if args.splits else None)
        print(f'生成完成: {len(manifest["samples"])} 个样本 | {len(models)} 个模型')
        if not manifest['evaluation_ready']:
            print('缺少验证或测试划分；不能报告完整的泛化评价。')
        print(f'数据目录: {output.resolve()}')
        print(f'首个预览: {output.resolve() / manifest["samples"][0]["path"] / "comparison.html"}')
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'无法准备数据: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
