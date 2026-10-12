"""按需读取样本，确定性采样和全量局部评价。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from completion import context_points, sample_points
from prepare_dataset import load_training_sample


def read_manifest(dataset):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('数据集未完成，不能训练或评价。')
    model_splits, paths = {}, set()
    for entry in manifest['samples']:
        if entry['split'] not in ('train', 'val', 'test'):
            raise ValueError('数据集划分名称无效。')
        model_id = entry['model_id']
        if model_id in model_splits and model_splits[model_id] != entry['split']:
            raise ValueError('同一器物跨越不同数据集划分。')
        model_splits[model_id] = entry['split']
        path = (dataset / entry['path']).resolve()
        if not path.is_relative_to(dataset) or path in paths:
            raise ValueError('样本路径越界或重复。')
        paths.add(path)
    return manifest


def dataset_fingerprint(dataset, manifest):
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode())
    for entry in manifest['samples']:
        with (Path(dataset) / entry['path'] / 'training.npz').open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def checked_sample(path):
    sample = load_training_sample(path)
    for key in ('input', 'local_target'):
        points = sample[key]
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f'{path}: {key} 必须是非空有限的 N×3 坐标。')
    if sample['center'].shape != (3,) or not np.isfinite(sample['center']).all():
        raise ValueError('无效的选区球心。')
    if sample['radius'].shape != () or not np.isfinite(sample['radius']) or sample['radius'] <= 0:
        raise ValueError('无效的选区半径。')
    return sample


class CompletionDataset(Dataset):
    def __init__(self, root, entries, input_points, target_points, seed):
        self.root, self.entries = Path(root), entries
        self.input_points, self.target_points, self.seed = input_points, target_points, seed
        self.epoch = 0

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        sample = checked_sample(self.root / self.entries[index]['path'] / 'training.npz')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        points = context_points(sample['input'], sample['center'], self.input_points, rng)
        target = sample_points(sample['local_target'], self.target_points, rng)
        region = np.r_[sample['center'], sample['radius']]
        return tuple(torch.tensor(value, dtype=torch.float32) for value in (points, region, target))


def nearest_squared(a, b, chunk=512):
    # 双方向分块，避免评价大目标时构造整个距离矩阵。
    result = []
    for start in range(0, len(a), chunk):
        query = a[start:start+chunk]
        closest = torch.full((len(query),), float('inf'), device=a.device)
        for end in range(0, len(b), chunk):
            distance = torch.cdist(query[None], b[end:end+chunk][None]).square()[0]
            closest = torch.minimum(closest, distance.amin(dim=1))
        result.append(closest)
    return torch.cat(result)


def evaluate_entries(model, root, entries, input_points, seed, device, threshold=0.02):
    if not entries:
        return None
    model.eval()
    rows = []
    with torch.no_grad():
        for index, entry in enumerate(entries):
            sample = checked_sample(Path(root) / entry['path'] / 'training.npz')
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0, index]))
            context = context_points(sample['input'], sample['center'], input_points, rng)
            region = np.r_[sample['center'], sample['radius']]
            pred = model(torch.tensor(context, dtype=torch.float32, device=device)[None],
                         torch.tensor(region, dtype=torch.float32, device=device)[None])[0]
            target = torch.tensor(sample['local_target'], dtype=torch.float32, device=device)
            forward, backward = nearest_squared(pred, target), nearest_squared(target, pred)
            cd = float(forward.mean() + backward.mean())
            precision = float((forward <= threshold**2).float().mean())
            recall = float((backward <= threshold**2).float().mean())
            fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.
            rows.append(dict(model_id=entry['model_id'], path=entry['path'], local_cd_squared=cd,
                             local_fscore=fscore, precision=precision, recall=recall))
    models = {}
    for model_id in sorted({row['model_id'] for row in rows}):
        selected = [row for row in rows if row['model_id'] == model_id]
        models[model_id] = dict(samples=len(selected),
                               local_cd_squared=float(np.mean([r['local_cd_squared'] for r in selected])),
                               local_fscore=float(np.mean([r['local_fscore'] for r in selected])))
    return dict(sample_count=len(rows), model_count=len(models),
                mean_local_cd_squared=float(np.mean([r['local_cd_squared'] for r in rows])),
                mean_local_fscore=float(np.mean([r['local_fscore'] for r in rows])),
                model_macro_cd_squared=float(np.mean([r['local_cd_squared'] for r in models.values()])),
                model_macro_fscore=float(np.mean([r['local_fscore'] for r in models.values()])),
                fscore_threshold=threshold, coordinate_space='normalized input coordinates',
                models=models, samples=rows)
