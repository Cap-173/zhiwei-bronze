"""Paired training-seed confirmation of the 16/64-hole coverage diagnostic."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from completion import context_points, load_checkpoint
from main import ROOT
from organize_figures import organize
from prepare_dataset import restore
from run_quantity_experiment import dump
from summarize_basic_fit import render_case
from train import train
from training_data import checked_sample, dataset_fingerprint, evaluate_entries, read_manifest


def checkpoint_path(count, seed):
    if (count, seed) == (16, 42):
        return ROOT / 'outputs/runs/bronze_single_object_pointwise_xyz_seed42/last.pt'
    return ROOT / f'outputs/runs/bronze_quantity_{count}_seed{seed}/last.pt'


def freeze(output):
    prior_path = ROOT / 'outputs/diagnostics/quantity_fit/summary.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    p = prior['protocol']['base_protocol']
    datasets = {16: Path(p['single_object_dataset']),
                64: Path(prior['protocol']['datasets']['64']['path'])}
    frozen = dict(counts=[16, 64], exploratory_seed=42, confirmation_seeds=[43, 44],
                  updates=800, evaluation_seed=43,
                  prior_summary=str(prior_path), prior_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                  source_dataset=p['source_dataset'],
                  roles=dict(seen_train16=p['training_holes'],
                             unseen_same_object4=p['unseen_same_object_holes'],
                             cross_object20=p['cross_object_evaluation']),
                  datasets={str(count):dict(path=str(path), fingerprint=dataset_fingerprint(path, read_manifest(path)),
                            epochs=800//(count//4)) for count, path in datasets.items()},
                  config=dict(input_points=512, output_points=128, target_points=128,
                              batch_size=4, lr=.001, device='cpu', output_mode='tanh',
                              architecture='pointwise_xyz', decoder_width=1140),
                  checkpoint_policy='last at 800 updates; no evaluation-based model selection',
                  scope='Candidate64 selected after seed42 exploration; only training seeds vary, holes and evaluation examples fixed. Same-object targets overlap; one cross-object reference, no original test06/07.')
    dump(output / 'protocol.json', frozen)
    return frozen


def validate_checkpoint(path, count, seed, protocol):
    model, checkpoint = load_checkpoint(path)
    dataset = protocol['datasets'][str(count)]
    assert checkpoint['epoch'] == dataset['epochs']
    assert checkpoint['dataset_fingerprint'] == dataset['fingerprint']
    assert checkpoint['selection_split'] == 'train'
    assert checkpoint['config']['seed'] == seed
    assert sum(x.numel() for x in model.parameters()) == 615236
    assert checkpoint['optimizer']['state']
    assert all(int(s['step']) == protocol['updates'] for s in checkpoint['optimizer']['state'].values())
    for key, expected in protocol['config'].items():
        if key != 'device':
            assert checkpoint['config'][key] == expected, (key, checkpoint['config'][key], expected)
    assert 'loss_mode' not in checkpoint['config']
    assert not checkpoint['config']['boundary_weight']
    return model, checkpoint


def geometry(models, protocol):
    sources = {r['id']:r['source'] for r in json.loads((ROOT / 'outputs/local_models/import_20261013_125555_685154/manifest.json').read_text(encoding='utf-8'))['models']}
    meshes, result = {}, {label:{} for label in models}
    source = Path(protocol['source_dataset'])
    for role, entries in protocol['roles'].items():
        print(f'Geometry audit: {role}', flush=True)
        rows = {label:[] for label in models}
        for index, entry in enumerate(entries):
            sample = checked_sample(source / entry['path'] / 'training.npz')
            context = context_points(sample['input'], sample['center'], 512,
                np.random.default_rng(np.random.SeedSequence([protocol['evaluation_seed'], 0, index])))
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
                _, mesh_distance, _ = trimesh.proximity.closest_point(meshes[mid], restore(pred.numpy(), sample['offset'], sample['scale']))
                if not np.isfinite(mesh_distance).all():
                    raise ValueError('Non-finite mesh distance')
                rows[label].append(dict(path=entry['path'], close_fraction=close,
                    mesh_distance_over_R=float(mesh_distance.mean()/(float(sample['radius'])*float(sample['scale'])))))
        for label in models:
            result[label][role] = dict(samples=rows[label], means={field:float(np.mean([r[field] for r in rows[label]]))
                for field in ('close_fraction', 'mesh_distance_over_R')})
    return result


def main():
    torch.set_num_threads(2)
    output = ROOT / 'outputs/repeats/bronze_quantity_confirmation'
    output.mkdir(parents=True, exist_ok=True)
    p = json.loads((output / 'protocol.json').read_text(encoding='utf-8')) if (output / 'protocol.json').exists() else freeze(output)
    prior_path = Path(p['prior_summary'])
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == p['prior_sha256']
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    for dataset in p['datasets'].values():
        assert dataset_fingerprint(Path(dataset['path']), read_manifest(Path(dataset['path']))) == dataset['fingerprint']
    for seed in p['confirmation_seeds']:
        for count in p['counts']:
            path = checkpoint_path(count, seed)
            if path.exists():
                _, checkpoint = load_checkpoint(path)
                if checkpoint['epoch'] == p['datasets'][str(count)]['epochs']:
                    validate_checkpoint(path, count, seed, p)
                    continue
            print(f'Train paired candidate: seed={seed}, holes={count}', flush=True)
            train(Path(p['datasets'][str(count)]['path']), path.parent,
                  epochs=p['datasets'][str(count)]['epochs'], seed=seed,
                  resume=path if path.exists() else None, **p['config'])
    evaluations, quality, models, histories, hashes = {}, {}, {}, {}, {}
    for seed in [42]+p['confirmation_seeds']:
        for count in p['counts']:
            label = f'{count}_seed{seed}'
            path = checkpoint_path(count, seed)
            model, checkpoint = validate_checkpoint(path, count, seed, p)
            histories[label] = checkpoint['history']
            hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
            if seed == 42:
                evaluations[label] = prior['evaluations'][str(count)]
                quality[label] = prior['geometry'][str(count)]
            else:
                models[label] = model
                evaluations[label] = {role:evaluate_entries(model, Path(p['source_dataset']), entries,
                    512, p['evaluation_seed'], torch.device('cpu')) for role, entries in p['roles'].items()}
    geometry_path = output / 'geometry.json'
    if geometry_path.exists():
        cached = json.loads(geometry_path.read_text(encoding='utf-8'))
        assert cached['checkpoint_sha256'] == {k:hashes[k] for k in models}
        quality.update(cached['geometry'])
    else:
        fresh = geometry(models, p)
        quality.update(fresh)
        dump(geometry_path, dict(checkpoint_sha256={k:hashes[k] for k in models}, geometry=fresh))
    aggregated = {}
    for scope, seeds in [('confirmation_only', p['confirmation_seeds']), ('all_three', [42]+p['confirmation_seeds'])]:
        aggregated[scope] = {}
        for role in p['roles']:
            aggregated[scope][role] = {}
            for count in p['counts']:
                values = {field:[evaluations[f'{count}_seed{seed}'][role][field] for seed in seeds]
                          for field in ('mean_local_cd_squared', 'mean_local_fscore')}
                values.update({field:[quality[f'{count}_seed{seed}'][role]['means'][field] for seed in seeds]
                               for field in ('close_fraction', 'mesh_distance_over_R')})
                aggregated[scope][role][str(count)] = {field:dict(mean=float(np.mean(v)), std_population=float(np.std(v)), values=v)
                    for field,v in values.items()}
    figures = ROOT / 'outputs/experiment_figures'
    examples = []
    for seed in p['confirmation_seeds']:
        for role in ('unseen_same_object4', 'cross_object20'):
            entry = p['roles'][role][0]
            sample = checked_sample(Path(p['source_dataset']) / entry['path'] / 'training.npz')
            context = context_points(sample['input'], sample['center'], 512, np.random.default_rng(np.random.SeedSequence([43,0,0])))
            region = np.r_[sample['center'], sample['radius']]
            with torch.no_grad():
                predictions = {f'{count} holes / seed{seed}':models[f'{count}_seed{seed}'](torch.tensor(context, dtype=torch.float32)[None],
                    torch.tensor(region, dtype=torch.float32)[None])[0].numpy() for count in p['counts']}
            filename = render_case(sample, predictions, f'quantity_confirmation_seed{seed}_{role}', figures)
            np.savez_compressed(output / f'seed{seed}_{role}_case.npz', context=context, target=sample['local_target'],
                center=sample['center'], radius=sample['radius'], **predictions)
            examples.append(dict(seed=seed, role=role, path=entry['path'], figure=filename))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, role in enumerate(('unseen_same_object4', 'cross_object20')):
        for seed in [42]+p['confirmation_seeds']:
            axes[index].plot([16,64], [evaluations[f'{count}_seed{seed}'][role]['mean_local_cd_squared'] for count in p['counts']],
                marker='o', label=f'seed{seed}'+(' exploration' if seed==42 else ' confirmation'))
        axes[index].set(title=role, xlabel='Training holes / 800 updates', ylabel='Full-target CD squared', xticks=[16,64])
        axes[index].legend(); axes[index].grid(alpha=.2)
    fig.tight_layout(); curve = 'bronze_quantity_confirmation_paired_cd.png'
    fig.savefig(figures / curve, dpi=180); plt.close(fig)
    dump(output / 'summary.json', dict(protocol=p, checkpoint_sha256=hashes, evaluations=evaluations,
        geometry=quality, aggregates=aggregated, examples=examples))
    lines = ['# 64缺口方案的训练种子配对复验', '',
        'seed42探索后固定64缺口候选，新增训练seed43、44；每种子比较16/64个训练缺口，从头训练，batch4、800次更新，分别200/50轮，取last.pt。网络、学习率、CD²/R²、原数据和生成缺口均固定，评价采样seed43统一。校验数据指纹、模型参数615236、配置及所有Adam参数更新计数。原seed42权重和结果复用。', '',
        '只改变训练随机种子，未新增器物、缺口生成种子或评价样本。原4个新缺口目标与其他训练目标的表面重叠，换器物仅一个青铜觥。候选来源于已有评价结果，因此这是固定开发样本上的复验，不能称为全新盲测；原测试06/07未使用。', '',
        '|范围|训练seed|训练缺口|CD²↓|F-score↑|聚集比例↓|网格距离/R↓|', '|---|---:|---:|---:|---:|---:|---:|']
    for role in p['roles']:
        for seed in [42]+p['confirmation_seeds']:
            for count in p['counts']:
                label = f'{count}_seed{seed}'; m = evaluations[label][role]; q = quality[label][role]['means']
                lines.append(f'|{role}|{seed}|{count}|{m["mean_local_cd_squared"]:.8f}|{m["mean_local_fscore"]:.6f}|{q["close_fraction"]:.2%}|{q["mesh_distance_over_R"]:.6f}|')
    lines += ['', '## 汇总', '', '|统计范围|评价范围|训练缺口|CD²均值 ± 种子总体标准差|F-score均值|聚集均值|网格距离/R均值|', '|---|---|---:|---:|---:|---:|---:|']
    for scope, groups in aggregated.items():
        for role, counts in groups.items():
            for count, m in counts.items():
                lines.append(f'|{scope}|{role}|{count}|{m["mean_local_cd_squared"]["mean"]:.8f} ± {m["mean_local_cd_squared"]["std_population"]:.8f}|{m["mean_local_fscore"]["mean"]:.6f}|{m["close_fraction"]["mean"]:.2%}|{m["mesh_distance_over_R"]["mean"]:.6f}|')
    lines += ['', '总体标准差仅描述固定评价样本下的训练种子波动；两枚新增种子或三枚总种子不足以证明统计显著性。聚集定义为预测点最近邻距离<0.02R，网格只用于评价。', '', '## 配对变化', '']
    for seed in [42]+p['confirmation_seeds']:
        for role in ('unseen_same_object4', 'cross_object20'):
            old = evaluations[f'16_seed{seed}'][role]; new = evaluations[f'64_seed{seed}'][role]
            lines.append(f'- seed{seed} / {role}：CD² {(new["mean_local_cd_squared"]/old["mean_local_cd_squared"]-1):+.2%}；F-score差值 {new["mean_local_fscore"]-old["mean_local_fscore"]:+.6f}。')
    same = aggregated['confirmation_only']['unseen_same_object4']
    cross = aggregated['confirmation_only']['cross_object20']
    conclusion = (f'新增seed43/44均改善同器物新缺口与青铜觥CD²；仅新增种子的同器物CD²均值下降 '
        f'{(1-same["64"]["mean_local_cd_squared"]["mean"]/same["16"]["mean_local_cd_squared"]["mean"]):.2%}，'
        f'青铜觥下降 {(1-cross["64"]["mean_local_cd_squared"]["mean"]/cross["16"]["mean_local_cd_squared"]["mean"]):.2%}。'
        '同器物seed44的F-score下降，因此F-score收益未逐种子成立。'
        f'同器物聚集均值从 {same["16"]["close_fraction"]["mean"]:.2%} 增至 {same["64"]["close_fraction"]["mean"]:.2%}，'
        '表面距离改善但薄壁双层和点分布仍不正确。数据覆盖对距离指标的收益可重复，不能声称全面解决补全。')
    lines += ['', '## 判断', '', conclusion, '',
        '逐种子、指标和几何质量同时判断；CD改善不能替代薄壁结构正确性或点分布质量。增加缺口同时减少每个缺口重复训练，固定更新次数也不等于相同耗时。保留当前冻结原型，复验结果只作为下一轮多器物覆盖实验的依据。', '',
        f'![配对CD](../../experiment_figures/{curve})', '']
    for e in examples:
        lines += [f'### seed{e["seed"]} / {e["role"]}', f'样本：`{e["path"]}`；相同上下文与展示视角，PCA只用于绘图。', f'![对照](../../experiment_figures/{e["figure"]})', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    (figures / 'bronze_quantity_confirmation_notes.md').write_text('\n'.join(['# 缺口覆盖种子复验图片',
        '[报告](../repeats/bronze_quantity_confirmation/README.md)', curve+'：三个训练种子的16/64缺口配对CD，seed42为探索结果。']+
        [e['figure']+f'：seed{e["seed"]}，{e["role"]}，同样本三视角对照。' for e in examples]), encoding='utf-8')
    organize()
    print(json.dumps(aggregated, indent=2), flush=True)


if __name__ == '__main__':
    main()
