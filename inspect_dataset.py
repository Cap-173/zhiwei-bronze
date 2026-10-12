"""检查训练样本、选区和坐标还原；不修改数据集。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from damage import sphere_mask
from main import ROOT, read_cloud
from prepare_dataset import restore
from training_data import checked_sample, read_manifest


def inspect_sample(directory, entry):
    sample = checked_sample(directory / 'training.npz')
    metadata = json.loads((directory / 'region.json').read_text(encoding='utf-8'))
    target, colors, total, invalid = read_cloud(directory / 'target.ply')
    if invalid:
        raise ValueError('完整目标 PLY 含无效坐标。')
    mask = np.load(directory / 'mask.npy', allow_pickle=False)
    if mask.dtype != np.bool_ or mask.shape != (len(target),) or not mask.any() or mask.all():
        raise ValueError('Mask 必须与完整点云对应且保留选区内外点。')
    if not np.array_equal(mask, sample['mask']):
        raise ValueError('NPZ 和 mask.npy 的选区不一致。')
    if not np.array_equal(mask, sphere_mask(target, metadata['center'], metadata['radius'])):
        raise ValueError('Mask 与球心/半径定义不一致。')
    for key, selected, filename in [('input', ~mask, 'input.ply'),
                                    ('local_target', mask, 'removed.ply'),
                                    ('target', np.ones(len(target), dtype=bool), 'target.ply')]:
        cloud, rgb, _, bad = read_cloud(directory / filename)
        if bad or not np.array_equal(cloud, target[selected]):
            raise ValueError(f'{filename} 未保持目标的坐标和顺序。')
        if colors is not None and (rgb is None or not np.array_equal(rgb, colors[selected])):
            raise ValueError(f'{filename} 颜色与目标不一致。')
        if sample[key].shape != cloud.shape or not np.allclose(
                restore(sample[key], sample['offset'], sample['scale']), cloud, rtol=1e-10, atol=1e-9):
            raise ValueError(f'{key} 无法按记录的变换还原。')
    if sample['offset'].shape != (3,) or not np.isfinite(sample['offset']).all() or sample['scale'].shape != () or not np.isfinite(sample['scale']) or sample['scale'] <= 0:
        raise ValueError('归一化参数无效。')
    if not np.allclose(restore(sample['center'], sample['offset'], sample['scale']), metadata['center']):
        raise ValueError('归一化球心无法还原。')
    if not np.isclose(sample['radius'] * sample['scale'], metadata['radius']):
        raise ValueError('归一化半径无法还原。')
    expected = dict(total_points=len(target), remaining_points=int((~mask).sum()), removed_points=int(mask.sum()))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('区域元数据点数不一致。')
    if entry['input_points'] != expected['remaining_points'] or entry['target_points'] != expected['removed_points']:
        raise ValueError('清单点数与样本不一致。')
    if metadata.get('model_id') != entry['model_id'] or metadata.get('split') != entry['split']:
        raise ValueError('区域元数据的模型 ID 或划分不一致。')
    return dict(path=entry['path'], model_id=entry['model_id'], split=entry['split'],
                total_points=total, removed_points=int(mask.sum()), missing_fraction=float(mask.mean()),
                radius=float(metadata['radius']), scale=float(sample['scale']), status='passed')


def inspect_dataset(dataset):
    manifest = read_manifest(dataset)
    rows, errors, warnings = [], [], []
    if not manifest['samples']:
        errors.append('清单没有样本。')
    for entry in manifest['samples']:
        try:
            if manifest['splits'].get(entry['model_id']) != entry['split']:
                raise ValueError('模型划分表与样本划分不一致。')
            row = inspect_sample(Path(dataset) / entry['path'], entry)
            limits = manifest.get('fraction_range', [0, 1])
            if not limits[0] <= row['missing_fraction'] <= limits[1]:
                warnings.append(f'{entry["path"]}: 缺失比例不在生成参数范围内。')
            rows.append(row)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'{entry["path"]}: {error}')
    splits = {}
    for split in ('train', 'val', 'test'):
        entries = [e for e in manifest['samples'] if e['split'] == split]
        splits[split] = dict(samples=len(entries), models=len({e['model_id'] for e in entries}))
    if not splits['train']['samples']:
        errors.append('没有训练样本。')
    if not splits['val']['samples'] or not splits['test']['samples']:
        warnings.append('缺少验证或测试集，当前数据仅可用于训练流程调试。')
    warnings.append('自动检查不判断模型原本是否完整、真实器物身份或数据授权；需人工核对。')
    return dict(dataset=str(Path(dataset).resolve()), passed=not errors, checked_samples=len(rows),
                expected_samples=len(manifest['samples']), splits=splits, errors=errors, warnings=warnings, samples=rows)


def render_report(report, output):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('各划分样本数', '局部缺失比例分布'))
    fig.add_trace(go.Bar(x=list(report['splits']), y=[s['samples'] for s in report['splits'].values()],
                         name='样本数'), row=1, col=1)
    fig.add_trace(go.Histogram(x=[r['missing_fraction'] for r in report['samples']], name='缺失比例'), row=1, col=2)
    fig.update_layout(template='plotly_dark', title=f'数据检查：{"通过" if report["passed"] else "失败"} · '
                      f'{report["checked_samples"]}/{report["expected_samples"]} 个样本通过', height=650)
    fig.update_xaxes(tickformat='.0%', row=1, col=2)
    fig.write_html(str(output), include_plotlyjs=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        dataset = args.dataset
        if dataset is None:
            paths = sorted((ROOT / 'outputs' / 'datasets').glob('dataset_*'))
            if not paths:
                raise ValueError('没有数据集，请先运行 prepare_dataset.py。')
            dataset = paths[-1]
        report = inspect_dataset(dataset)
        output = args.output_dir or ROOT / 'outputs' / 'reports' / datetime.now().strftime('data_%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        render_report(report, output / 'report.html')
        print(f'检查通过: {report["passed"]} | 样本: {report["checked_samples"]}/{report["expected_samples"]}')
        for issue in report['errors'] + report['warnings']:
            print(issue)
        print(f'报告目录: {output.resolve()}')
        return 0 if report['passed'] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'检查失败: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
