"""验证集局部补全诊断：边缘覆盖与径向收缩，不用于测试集调参。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

from completion import context_points, load_checkpoint, sample_points
from main import ROOT
from predict import preview
from prepare_dataset import restore
from training_data import checked_sample, dataset_fingerprint, nearest_squared, read_manifest


def diagnose(prediction, target, center, radius):
    """所有距离均在已归一化坐标内；边缘带为距球心 >= 0.8R。"""
    prediction, target = torch.as_tensor(prediction), torch.as_tensor(target)
    center = torch.as_tensor(center, dtype=prediction.dtype)
    forward, backward = nearest_squared(prediction, target), nearest_squared(target, prediction)
    target_radial = torch.linalg.vector_norm(target-center, dim=1)/radius
    predicted_radial = torch.linalg.vector_norm(prediction-center, dim=1)/radius
    boundary = target_radial >= 0.8
    # 单向覆盖诊断：真实边缘到预测点的最近距离，不宣称为标准边界损失。
    edge = backward[boundary]
    return dict(local_cd_squared=float(forward.mean()+backward.mean()),
                predicted_outer_fraction=float((predicted_radial >= 0.8).float().mean()),
                target_outer_fraction=float(boundary.float().mean()), target_boundary_points=int(boundary.sum()),
                predicted_radius_p95=float(torch.quantile(predicted_radial, 0.95)),
                target_radius_p95=float(torch.quantile(target_radial, 0.95)),
                boundary_mean_distance_over_radius=float(edge.sqrt().mean()/radius) if len(edge) else None,
                boundary_coverage_at_0_1_radius=float((edge <= (0.1*radius)**2).float().mean()) if len(edge) else None)


def analyze(dataset, checkpoint_path, output):
    manifest = read_manifest(dataset)
    entries = [e for e in manifest['samples'] if e['split'] == 'val']
    if not entries:
        raise ValueError('没有验证样本；不使用测试集代替验证集。')
    model, checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get('dataset_fingerprint') != dataset_fingerprint(dataset, manifest):
        raise ValueError('诊断数据与训练数据不一致，请使用该模型对应的验证集。')
    rows, cases = [], []
    with torch.no_grad():
        for index, entry in enumerate(entries):
            sample = checked_sample(Path(dataset)/entry['path']/'training.npz')
            rng = np.random.default_rng(np.random.SeedSequence([checkpoint['seed']+1, 0, index]))
            context = context_points(sample['input'], sample['center'], checkpoint['input_points'], rng)
            region = np.r_[sample['center'], sample['radius']]
            prediction = model(torch.tensor(context, dtype=torch.float32)[None],
                               torch.tensor(region, dtype=torch.float32)[None])[0]
            target = torch.tensor(sample['local_target'], dtype=torch.float32)
            metrics = diagnose(prediction, target, sample['center'], float(sample['radius']))
            # 仅作密度参考：直接从真实目标抽相同数量点，不能用于实际预测。
            oracle = sample_points(sample['local_target'], len(prediction), np.random.default_rng(index+42))
            reference = diagnose(torch.tensor(oracle, dtype=torch.float32), target, sample['center'], float(sample['radius']))
            metrics['oracle_sampled_boundary_coverage'] = reference['boundary_coverage_at_0_1_radius']
            metrics['oracle_sampled_outer_fraction'] = reference['predicted_outer_fraction']
            rows.append(dict(model_id=entry['model_id'], path=entry['path'], **metrics))
            cases.append((sample, prediction.numpy()))
    summary = {}
    for key in rows[0]:
        if key not in ('model_id', 'path'):
            values = [row[key] for row in rows if row[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
    summary['no_boundary_samples'] = sum(row['target_boundary_points'] == 0 for row in rows)
    order = sorted(range(len(rows)), key=lambda i: rows[i]['local_cd_squared'])
    chosen = [('lowest_cd', order[0]), ('median_cd', order[len(order)//2]), ('highest_cd', order[-1])]
    output.mkdir(parents=True, exist_ok=False)
    previews = {}
    for label, index in chosen:
        sample, predicted = cases[index]
        offset, scale = sample['offset'], sample['scale']
        cloud, generated, complete = (restore(p, offset, scale) for p in (sample['input'], predicted, sample['target']))
        preview(cloud, generated, complete, output/f'{label}.html')
        np.savez_compressed(output/f'{label}.npz', input=cloud, predicted=generated, target=complete,
                            local_target=restore(sample['local_target'], offset, scale),
                            center=restore(sample['center'], offset, scale), radius=sample['radius']*scale)
        previews[label] = dict(html=f'{label}.html', sample=rows[index])
    report = dict(split='val', sample_count=len(rows), model_count=len({r['model_id'] for r in rows}),
                  summary=summary, samples=rows, previews=previews, checkpoint=str(Path(checkpoint_path).resolve()),
                  definitions=dict(boundary='local target points with distance to sphere center >= 0.8R',
                                   coverage='fraction of boundary targets with a predicted neighbor within 0.1R',
                                   oracle='same-size random sample from true missing points; density reference only, not a predictive method',
                                   note='Diagnostic measures, not surface continuity proof or official benchmark metrics.'))
    (output/'diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.output_dir or ROOT/'outputs'/'diagnostics'/datetime.now().strftime('val_%Y%m%d_%H%M%S_%f')
        report = analyze(args.dataset, args.checkpoint, output)
        print(f'Validation: {report["model_count"]} models / {report["sample_count"]} samples')
        print(json.dumps(report['summary'], indent=2))
        print(f'报告及案例: {output.resolve()}')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'诊断失败: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
