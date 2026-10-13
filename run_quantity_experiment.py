"""Nested damage coverage experiment with an equal optimizer-update budget."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from main import ROOT
from prepare_dataset import choose_region, fit_transform, normalize, restore
from training_data import checked_sample, read_manifest, dataset_fingerprint, evaluate_entries
from completion import load_checkpoint, context_points
from train import train
from summarize_basic_fit import render_case
from organize_figures import organize


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(protocol, output):
    source = Path(protocol['source_dataset'])
    original = Path(protocol['single_object_dataset'])
    base = read_manifest(original)['samples']
    sample = checked_sample(source / protocol['training_holes'][0]['path'] / 'training.npz')
    raw = restore(sample['target'], sample['offset'], sample['scale'])
    rng = np.random.default_rng(20261013)
    masks = {checked_sample(source / e['path'] / 'training.npz')['mask'].tobytes()
             for e in protocol['training_holes'] + protocol['unseen_same_object_holes']}
    additions = []
    while len(additions) < 112:
        center, radius, mask = choose_region(raw, rng, (.08, .18), (.02, .20))
        if mask.tobytes() in masks:
            continue
        masks.add(mask.tobytes())
        offset, scale = fit_transform(raw[~mask])
        normalized = normalize(raw, offset, scale)
        additions.append(dict(input=normalized[~mask], local_target=normalized[mask],
                              target=normalized, mask=mask, center=normalize(center, offset, scale),
                              radius=radius / scale, offset=offset, scale=scale))
    datasets = {}
    for count in (64, 128):
        root = ROOT / f'outputs/datasets/bronze_quantity_{count}'
        root.mkdir(parents=True, exist_ok=False)
        entries = []
        for i in range(count):
            path = f'train/sample_{i:04d}'
            directory = root / path
            directory.mkdir(parents=True)
            if i < 16:
                src = original / base[i]['path'] / 'training.npz'
                shutil.copyfile(src, directory / 'training.npz')
                assert src.read_bytes() == (directory / 'training.npz').read_bytes()
                data = checked_sample(src)
            else:
                data = additions[i - 16]
                np.savez_compressed(directory / 'training.npz', **data)
            restored = restore(data['input'], data['offset'], data['scale'])
            off, scale = fit_transform(restored)
            assert np.allclose(off, data['offset']) and np.isclose(scale, data['scale'])
            entries.append(dict(model_id='bronze_02', split='train', path=path,
                                input_points=len(data['input']), target_points=len(data['local_target'])))
        manifest = dict(schema_version=1, status='complete', seed=20261013, samples=entries)
        dump(root / 'manifest.json', manifest)
        datasets[str(count)] = dict(path=str(root), fingerprint=dataset_fingerprint(root, read_manifest(root)),
                                   epochs=800 // (count // 4))
    for i in range(64):
        a = Path(datasets['64']['path']) / f'train/sample_{i:04d}/training.npz'
        b = Path(datasets['128']['path']) / f'train/sample_{i:04d}/training.npz'
        assert a.read_bytes() == b.read_bytes()
    audit = {}
    for count in (16, 64, 128):
        train_masks = [checked_sample(original / e['path'] / 'training.npz')['mask'] for e in base]
        train_masks += [s['mask'] for s in additions[:max(0, count - 16)]]
        covered = np.logical_or.reduce(train_masks)
        audit[str(count)] = dict(surface_point_coverage=float(covered.mean()),
            heldout_target_overlap=[float(covered[checked_sample(source / e['path'] / 'training.npz')['mask']].mean())
                                    for e in protocol['unseen_same_object_holes']])
    result = dict(base_protocol=protocol, datasets=datasets, updates=800, batch_size=4,
                  generation_seed=20261013, audit=audit,
                  baseline_sha256=hashlib.sha256((ROOT / 'outputs/runs/bronze_single_object_pointwise_xyz_seed42/last.pt').read_bytes()).hexdigest(),
                  limitation='Nested same-object holes; different missing regions can overlap. No exact heldout mask copied. Not new objects or disjoint surface regions.')
    dump(output / 'protocol.json', result)
    return result


def main():
    torch.set_num_threads(2)
    output = ROOT / 'outputs/diagnostics/quantity_fit'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        frozen = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    else:
        frozen = prepare(json.loads((ROOT / 'outputs/diagnostics/basic_fit/protocol.json').read_text(encoding='utf-8')), output)
    p = frozen['base_protocol']
    runs = {'16': ROOT / 'outputs/runs/bronze_single_object_pointwise_xyz_seed42'}
    assert hashlib.sha256((runs['16'] / 'last.pt').read_bytes()).hexdigest() == frozen['baseline_sha256']
    for label, dataset in frozen['datasets'].items():
        root = Path(dataset['path'])
        assert dataset_fingerprint(root, read_manifest(root)) == dataset['fingerprint']
        run = ROOT / f'outputs/runs/bronze_quantity_{label}_seed42'
        runs[label] = run
        if not (run / 'last.pt').exists():
            train(root, run, epochs=dataset['epochs'], input_points=512, output_points=128,
                  target_points=128, seed=42, batch_size=4, lr=.001, device='cpu',
                  output_mode='tanh', architecture='pointwise_xyz', decoder_width=1140)
    roles = dict(seen_train16=p['training_holes'], unseen_same_object4=p['unseen_same_object_holes'],
                 cross_object20=p['cross_object_evaluation'])
    models, results, histories = {}, {}, {}
    for label, run in runs.items():
        model, checkpoint = load_checkpoint(run / 'last.pt')
        assert checkpoint['epoch'] * (int(label) // 4) == 800
        assert all(int(state['step']) == 800 for state in checkpoint['optimizer']['state'].values())
        assert sum(x.numel() for x in model.parameters()) == 615236
        if label != '16':
            assert checkpoint['dataset_fingerprint'] == frozen['datasets'][label]['fingerprint']
        models[label] = model
        histories[label] = checkpoint['history']
        results[label] = {role: evaluate_entries(model, Path(p['source_dataset']), entries, 512, 43, torch.device('cpu'))
                          for role, entries in roles.items()}
    figures = ROOT / 'outputs/experiment_figures'
    examples = []
    import trimesh
    sources = {r['id']: r['source'] for r in json.loads((ROOT / 'outputs/local_models/import_20261013_125555_685154/manifest.json').read_text(encoding='utf-8'))['models']}
    meshes, geometry = {}, {label: {} for label in runs}
    for role, entries in roles.items():
        rows = {label: [] for label in runs}
        for index, entry in enumerate(entries):
            sample = checked_sample(Path(p['source_dataset']) / entry['path'] / 'training.npz')
            context = context_points(sample['input'], sample['center'], 512, np.random.default_rng(np.random.SeedSequence([43, 0, index])))
            region = np.r_[sample['center'], sample['radius']]
            mid = entry['model_id']
            if mid not in meshes:
                meshes[mid] = trimesh.load(sources[mid], force='mesh', process=False, skip_materials=True)
            for label, model in models.items():
                with torch.no_grad():
                    pred = model(torch.tensor(context, dtype=torch.float32)[None], torch.tensor(region, dtype=torch.float32)[None])[0]
                    distances = torch.cdist(pred[None], pred[None], compute_mode='donot_use_mm_for_euclid_dist')[0]
                    distances.fill_diagonal_(float('inf'))
                    close = float((distances.min(1).values < float(sample['radius'])*.02).float().mean())
                _, mesh_dist, _ = trimesh.proximity.closest_point(meshes[mid], restore(pred.numpy(), sample['offset'], sample['scale']))
                assert np.isfinite(mesh_dist).all()
                rows[label].append(dict(path=entry['path'], close_fraction=close,
                    mesh_distance_over_R=float(mesh_dist.mean()/(float(sample['radius'])*float(sample['scale'])))))
        for label in runs:
            geometry[label][role] = dict(samples=rows[label], means={field: float(np.mean([r[field] for r in rows[label]])) for field in ('close_fraction', 'mesh_distance_over_R')})
    for role, entries in roles.items():
        sample = checked_sample(Path(p['source_dataset']) / entries[0]['path'] / 'training.npz')
        context = context_points(sample['input'], sample['center'], 512, np.random.default_rng(np.random.SeedSequence([43, 0, 0])))
        region = np.r_[sample['center'], sample['radius']]
        with torch.no_grad():
            predictions = {label+' holes': model(torch.tensor(context, dtype=torch.float32)[None],
                           torch.tensor(region, dtype=torch.float32)[None])[0].numpy() for label, model in models.items()}
        filename = render_case(sample, predictions, 'quantity_'+role, figures)
        np.savez_compressed(output / (role+'_case.npz'), context=context, target=sample['local_target'], **predictions)
        examples.append(dict(role=role, path=entries[0]['path'], figure=filename))
    dump(output / 'summary.json', dict(protocol=frozen, evaluations=results, geometry=geometry, examples=examples))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, history in histories.items():
        ax.plot([r['epoch'] * (int(label)//4) for r in history], [r['train_cd_squared'] for r in history], label=label+' holes')
    ax.set(xlabel='Optimizer updates', ylabel='Training full-target CD squared', title='Different training sets / equal 800 updates')
    ax.legend(); fig.tight_layout()
    curve = 'bronze_quantity_training_curves.png'
    fig.savefig(figures / curve, dpi=180); plt.close(fig)
    lines = ['# 固定更新次数的训练缺口数量对照', '',
        '司母戊鼎16、64、128个训练缺口，batch4，各800次更新，分别200、50、25轮；seed42、Adam lr0.001、逐点XYZ编码、tanh、解码1140、615236参数、512输入/128输出与目标、CD²/R²均相同。取最后权重，未按评价结果挑选轮数。复用16缺口基线并校验哈希。', '',
        '新增缺口基于相同4096完整点，半径为包围盒对角线0.08–0.18、缺失比例0.02–0.20，生成seed20261013。保留原16份NPZ字节不变，64是128的严格前缀。归一化只用可见点；三个方案均处理3200个训练样本，计算耗时不保证相等。', '',
        '这是同一器物的损坏覆盖实验，缺失区域允许重叠；4个原评价缺口未直接加入训练，但其表面点可能出现在其他训练目标中。不能视为不重叠区域验证或新增器物实验。评价输入seed43固定，原测试器物06/07未使用。', '',
        '|评价范围|训练缺口|CD²↓|F-score↑|聚集比例↓|网格距离/R↓|', '|---|---:|---:|---:|---:|---:|']
    for role in roles:
        for label in runs:
            m = results[label][role]
            q = geometry[label][role]['means']
            lines.append(f'|{role}|{label}|{m["mean_local_cd_squared"]:.8f}|{m["mean_local_fscore"]:.6f}|{q["close_fraction"]:.2%}|{q["mesh_distance_over_R"]:.6f}|')
    lines += ['', '聚集比例为预测点最近邻距离小于0.02R的比例；网格距离只用于评价，未用于训练或预测投影。', '', '|训练缺口|全表面点被训练目标覆盖|4个评价目标的覆盖比例|', '|---:|---:|---|']
    for label, a in frozen['audit'].items():
        lines.append(f'|{label}|{a["surface_point_coverage"]:.2%}|'+', '.join(f'{x:.2%}' for x in a['heldout_target_overlap'])+'|')
    base = results['16']['unseen_same_object4']['mean_local_cd_squared']
    lines += ['', '## 结果解释', '', '相同器物4个新缺口的CD²相对16缺口基线：'+ '；'.join(f'{label}个训练缺口 {(results[label]["unseen_same_object4"]["mean_local_cd_squared"]/base-1):+.2%}' for label in ('64', '128'))+'。负数代表下降。',
        '64缺口的同器物新位置CD、F-score和表面距离改善，但聚集比例由5.66%升至19.34%；128缺口F-score反而下降。数据覆盖带来部分收益，点聚集与薄壁形状仍未解决，当前不替换冻结原型。下一步将64缺口作为候选做配对种子复验，再决定是否扩展到多器物实验。该候选由本轮探索结果提出，复验仍不能称为全新盲测。',
        '单器物、单种子和仅4个新缺口，结果用于定位数据覆盖问题，不证明跨器物泛化。增加缺口同时减少每个缺口的重复训练，不能将变化归因于数量单一因素。', '', f'![训练曲线](../../experiment_figures/{curve})', '曲线分别评价各自训练集，不能当作相同训练样本的直接拟合比较。', '']
    for e in examples:
        lines += [f'### {e["role"]}', f'样本：`{e["path"]}`。相同上下文和展示视角；PCA仅用于绘图。', f'![对照](../../experiment_figures/{e["figure"]})', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    (figures / 'bronze_quantity_notes.md').write_text('\n'.join(['# 缺口数量实验图片', '[报告](../diagnostics/quantity_fit/README.md)', curve+'：固定更新数下各自训练集CD曲线。'] + [e['figure']+'：'+e['role']+'三种训练数量的同样本三视角对照。' for e in examples]), encoding='utf-8')
    organize()
    print(json.dumps({label:{role:dict(cd=m['mean_local_cd_squared'], fscore=m['mean_local_fscore']) for role,m in r.items()} for label,r in results.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
