"""从官方 ModelNet40 ZIP 导入器皿子集，表面采样并保留官方测试集。"""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import trimesh

from damage import write_cloud
from main import ROOT

CATEGORIES = ['bottle', 'bowl', 'cup', 'flower_pot', 'vase']


def sample_off(raw, count, seed):
    # 少量 ModelNet 文件将 OFF 和顶点数写在同一行，先统一文件头。
    text = raw.decode('utf-8-sig')
    stripped = text.lstrip()
    if stripped.startswith('OFF') and not stripped[3:].startswith(('\n', '\r')):
        text = 'OFF\n' + stripped[3:].lstrip()
    mesh = trimesh.load(io.BytesIO(text.encode()), file_type='off', force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('网格为空或含无效坐标。')
    if not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError('网格表面积无效。')
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    if not np.isfinite(points).all():
        raise ValueError('表面采样产生无效点。')
    return points, dict(vertices=len(mesh.vertices), faces=len(mesh.faces),
                        watertight=bool(mesh.is_watertight), area=float(mesh.area))


def import_archive(archive, output, train_per_category=40, test_per_category=10, points=8192, seed=42, categories=None):
    categories = categories or CATEGORIES
    if min(train_per_category, test_per_category) < 1 or points < 16 or seed < 0:
        raise ValueError('每类模型数需为正，点数至少 16，种子非负。')
    if not categories or not set(categories) <= set(CATEGORIES):
        raise ValueError('请使用支持的器皿类别。')
    if output.exists():
        raise FileExistsError('导入目录已存在，不会覆盖。')
    rng = np.random.default_rng(seed)
    models, splits, rows, errors = {}, {}, [], []
    seen = set()
    output.mkdir(parents=True)
    with zipfile.ZipFile(archive) as package:
        files = package.namelist()
        for category in categories:
            for source_split, limit in [('train', train_per_category), ('test', test_per_category)]:
                candidates = sorted(name for name in files if name.startswith(f'ModelNet40/{category}/{source_split}/') and name.endswith('.off'))
                if not candidates:
                    raise ValueError(f'压缩包缺少 {category}/{source_split}。')
                order = rng.permutation(len(candidates))
                imported = []
                for index in order:
                    name = candidates[index]
                    model_id = f'modelnet40_{category}_{Path(name).stem}'
                    try:
                        raw = package.read(name)
                        checksum = hashlib.sha256(raw).hexdigest()
                        if checksum in seen:
                            errors.append(dict(source=name, reason='duplicate source bytes; skipped'))
                            continue
                        model_seed = int.from_bytes(hashlib.sha256(f'{seed}:{model_id}'.encode()).digest()[:4], 'little')
                        cloud, info = sample_off(raw, points, model_seed)
                        destination = output / 'ply' / f'{model_id}.ply'
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        write_cloud(destination, cloud)
                        models[model_id] = destination.relative_to(output).as_posix()
                        splits[model_id] = source_split
                        rows.append(dict(model_id=model_id, category=category, official_split=source_split,
                                         archive_path=name, source_sha256=checksum, sampling_seed=model_seed,
                                         points=len(cloud), **info))
                        seen.add(checksum)
                        imported.append(model_id)
                        if len(imported) >= limit:
                            break
                    except (ValueError, OSError, IndexError, TypeError) as error:
                        errors.append(dict(source=name, reason=str(error)))
                # 从官方训练模型中留出验证模型，不动官方测试模型。
                if source_split == 'train' and len(imported) >= 2:
                    for model_id in imported[:max(1, round(len(imported)*0.1))]:
                        splits[model_id] = 'val'
                print(f'{category}/{source_split}: {len(imported)} models', flush=True)
    if not models:
        raise ValueError('没有成功导入任何模型。')
    for filename, data in [('models.json', models), ('splits.json', splits)]:
        (output / filename).write_text(json.dumps(data, indent=2), encoding='utf-8')
    provenance = dict(dataset='ModelNet40 vessel subset', download_url='https://modelnet.cs.princeton.edu/ModelNet40.zip',
                      source_page='https://modelnet.cs.princeton.edu/download.html', seed=seed, surface_points=points,
                      official_test_preserved=True, selection='seeded random subset per category; validation from official train',
                      license_note='Original CAD authors retain copyright; ModelNet is provided for academic research.',
                      note='Not bronze scans, not an official completion benchmark; manually inspect mesh completeness.',
                      models=rows, skipped=errors)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'outputs' / 'downloads' / 'ModelNet40.zip')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--train-per-category', type=int, default=40)
    parser.add_argument('--test-per-category', type=int, default=10)
    parser.add_argument('--points', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    try:
        output = args.output_dir or ROOT / 'outputs' / 'public_data' / datetime.now().strftime('modelnet_%Y%m%d_%H%M%S_%f')
        report = import_archive(args.archive, output, args.train_per_category, args.test_per_category, args.points, args.seed)
        print(f'导入完成: {len(report["models"])} 个模型 | 跳过 {len(report["skipped"])} 个文件')
        print(f'清单目录: {output.resolve()}')
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f'导入失败: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
