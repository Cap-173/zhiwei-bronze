"""解压本地网格并按表面积采样为现有流程可读取的 PLY。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import trimesh

from damage import write_cloud
from main import ROOT, render_cloud


def extract_models(root):
    for archive_path in sorted(root.rglob('*.zip')):
        destination = archive_path.parent / (archive_path.stem + '_extracted')
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                name = member.filename
                if member.is_dir() or name.startswith('__MACOSX/'):
                    continue
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError(f'不安全的压缩包路径：{name}')
                raw = archive.read(member)
                if target.exists():
                    if target.read_bytes() != raw:
                        raise FileExistsError(f'解压目标已有不同内容：{target}')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)


def convert(root, output, count, seed):
    if count < 16 or seed < 0:
        raise ValueError('点数至少16，随机种子非负。')
    if output.exists():
        raise FileExistsError('输出目录已存在，不覆盖。')
    extract_models(root)
    sources = sorted(root.rglob('*.obj'))
    if not sources:
        raise ValueError('没有找到OBJ模型。')
    output.mkdir(parents=True)
    mapping, records, failures = {}, [], []
    for index, source in enumerate(sources):
        model_id = f'bronze_{index:02d}'
        print(f'[{index+1}/{len(sources)}] {source.relative_to(root)}', flush=True)
        try:
            mesh = trimesh.load(source, force='mesh', process=False, skip_materials=True)
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                raise ValueError('没有有效网格面。')
            if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area) or mesh.area <= 0:
                raise ValueError('网格坐标或表面积无效。')
            points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed + index)
            if not np.isfinite(points).all():
                raise ValueError('采样坐标无效。')
            cloud = output / 'ply' / f'{model_id}.ply'
            cloud.parent.mkdir(parents=True, exist_ok=True)
            write_cloud(cloud, points)
            preview = output / 'previews' / f'{model_id}.html'
            render_cloud(points, None, preview, source.parent.name + ' · 网格表面采样', 20000)
            mapping[model_id] = cloud.relative_to(output).as_posix()
            records.append(dict(id=model_id, source=str(source.resolve()),
                                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                vertices=len(mesh.vertices), faces=len(mesh.faces),
                                points=len(points), seed=seed+index,
                                bounds=mesh.bounds.tolist(), area=float(mesh.area),
                                ply=mapping[model_id], preview=preview.relative_to(output).as_posix()))
        except Exception as error:
            failures.append(dict(source=str(source), error=str(error)))
        (output / 'models.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
        (output / 'manifest.json').write_text(json.dumps(dict(models=records, failures=failures,
            method='area-weighted surface sampling', original_coordinates=True,
            texture_colors=False), ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 青铜器模型点云清单', '',
             '从原始OBJ按表面积采样，每个模型保留原始坐标；未采样纹理颜色。原始模型保留不改动。',
             '模型来源和物理尺寸尚未核实；网格采样不等于真实扫描。', '',
             '| ID | 原始模型 | 采样点数 | 预览 | PLY |', '|---|---|---:|---|---|']
    for row in records:
        name = Path(row['source']).relative_to(root.resolve()).as_posix()
        lines.append(f'| {row["id"]} | {name} | {row["points"]:,} | [打开]({row["preview"]}) | [点云]({row["ply"]}) |')
    lines += ['', f'成功 {len(records)} 个，失败 {len(failures)} 个。详情见 manifest.json。', '',
              'models.json 可传入 prepare_dataset.py 的 --models 参数；数据集构建时再进行归一化和模型级划分。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} 个模型转换失败，查看 manifest.json。')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'models' / '青铜器模型_obj')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--points', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = args.output or ROOT / 'outputs' / 'local_models' / datetime.now().strftime('import_%Y%m%d_%H%M%S_%f')
    print(convert(args.source, output, args.points, args.seed))
