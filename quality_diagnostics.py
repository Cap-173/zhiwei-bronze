"""补充等点数DCD、聚集诊断及到源网格的距离，仅用于评价。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import trimesh

from completion import context_points, load_checkpoint, sample_points
from density_loss import density_chamfer
from prepare_dataset import restore
from training_data import checked_sample, read_manifest


def diagnose(dataset, checkpoint_path, source_manifest, split, output):
    dataset = Path(dataset)
    model, checkpoint = load_checkpoint(checkpoint_path)
    torch.set_num_threads(2)
    sources = {r['id']: r['source'] for r in json.loads(Path(source_manifest).read_text(encoding='utf-8'))['models']}
    entries = [e for e in read_manifest(dataset)['samples'] if e['split'] == split]
    if not entries:
        raise ValueError('评价划分为空。')
    meshes, rows = {}, []
    count = checkpoint['output_points']
    for index, entry in enumerate(entries):
        sample = checked_sample(dataset / entry['path'] / 'training.npz')
        rng = np.random.default_rng(np.random.SeedSequence([43, 0, index]))
        context = context_points(sample['input'], sample['center'], checkpoint['input_points'], rng)
        region = np.r_[sample['center'], sample['radius']]
        with torch.no_grad():
            pred = model(torch.tensor(context, dtype=torch.float32)[None], torch.tensor(region, dtype=torch.float32)[None])[0]
            dcd = []
            for repeat in range(5):
                target = sample_points(sample['local_target'], count, np.random.default_rng(np.random.SeedSequence([31415, index, repeat])))
                dcd.append(float(density_chamfer(pred[None], torch.tensor(target, dtype=torch.float32)[None], alpha=1000, exponent=1)))
            distances = torch.cdist(pred[None], pred[None], compute_mode='donot_use_mm_for_euclid_dist')[0]
            distances.fill_diagonal_(float('inf'))
            nearest = distances.min(dim=1).values / float(sample['radius'])
        mid = entry['model_id']
        if mid not in meshes:
            print(f'Loading reference mesh: {mid}', flush=True)
            meshes[mid] = trimesh.load(sources[mid], force='mesh', process=False, skip_materials=True)
        original = restore(pred.numpy(), sample['offset'], sample['scale'])
        _, mesh_distance, _ = trimesh.proximity.closest_point(meshes[mid], original)
        radius_original = float(sample['radius'] * sample['scale'])
        rows.append(dict(model_id=mid, path=entry['path'], dcd_equal_points_mean=float(np.mean(dcd)),
                         dcd_target_draw_std=float(np.std(dcd)),
                         target_sampling_with_replacement=len(sample['local_target']) < count,
                         prediction_nn_median_over_radius=float(nearest.median()),
                         prediction_close_fraction_0_02_radius=float((nearest < .02).float().mean()),
                         prediction_mesh_mean_distance_over_radius=float(mesh_distance.mean() / radius_original)))
    fields = ['dcd_equal_points_mean', 'prediction_nn_median_over_radius',
              'prediction_close_fraction_0_02_radius', 'prediction_mesh_mean_distance_over_radius']
    report = dict(split=split, checkpoint=str(Path(checkpoint_path).resolve()),
                  samples=rows, sample_count=len(rows), model_count=len(meshes),
                  summary={f: float(np.mean([r[f] for r in rows])) for f in fields},
                  protocol=dict(evaluation_seed=43, target_draw_seed=31415, target_draws=5,
                                dcd_alpha=1000, dcd_exponent=1, points=count,
                                coordinate_space='visible-input normalized coordinates',
                                mesh_reference='full original OBJ; evaluation only, no snapping',
                                note='Close-point fraction is an engineering diagnostic, not a paper metric; low mesh distance alone does not prove correct patch or continuity.'))
    with Path(output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report['summary'], indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--sources', required=True, type=Path)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    diagnose(a.dataset, a.checkpoint, a.sources, a.split, a.output)
