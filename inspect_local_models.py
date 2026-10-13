"""检查本地 OBJ 坐标、面索引、材质引用及未解压模型，生成清单。"""
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def inspect(path):
    vertices = faces = invalid = 0
    largest_index = 0
    materials = []
    with path.open(encoding='utf-8-sig', errors='replace') as stream:
        for line in stream:
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] == 'v':
                vertices += 1
                try:
                    if len(fields) < 4 or not all(math.isfinite(float(v)) for v in fields[1:4]):
                        invalid += 1
                except ValueError:
                    invalid += 1
            elif fields[0] == 'f':
                faces += 1
                try:
                    indices = [int(f.split('/')[0]) for f in fields[1:]]
                    if len(indices) < 3 or any(i == 0 or (i < 0 and -i > vertices) for i in indices):
                        invalid += 1
                    largest_index = max(largest_index, max(indices, default=0))
                except ValueError:
                    invalid += 1
            elif fields[0] == 'mtllib':
                materials.append(line.strip()[7:].strip())
    missing = [name for name in materials if not (path.parent / name).is_file()]
    if largest_index > vertices:
        invalid += 1
    return dict(path=path.relative_to(ROOT).as_posix(), vertices=vertices, faces=faces,
                invalid_records=invalid, missing_materials=missing)


def main():
    records = [inspect(p) for p in sorted((ROOT / 'models').rglob('*.obj'))]
    archives = []
    for path in sorted((ROOT / 'models').rglob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            archives.append(dict(path=path.relative_to(ROOT).as_posix(),
                                 models=[n for n in archive.namelist() if n.endswith('.obj') and not n.startswith('__MACOSX/')]))
    output = ROOT / 'outputs' / 'model_inspection'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'inventory.json').write_text(json.dumps(dict(models=records, archives=archives), ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 本地三维模型检查', '', '检查范围：OBJ顶点坐标、面顶点索引、MTL文件引用及ZIP内模型清单；未验证网格封闭性或纹理渲染效果。', '',
             '| 模型 | 顶点 | 面 | 异常记录 | 缺失材质文件 |', '|---|---:|---:|---:|---|']
    for r in records:
        lines.append(f'| {Path(r["path"]).parent.name} | {r["vertices"]:,} | {r["faces"]:,} | {r["invalid_records"]} | {", ".join(r["missing_materials"]) or "无"} |')
    lines += ['', '## 待解压模型', '']
    for a in archives:
        lines += [f'- {a["path"]}：{", ".join(a["models"])}']
    lines += ['', '当前 main.py / prepare_dataset.py 读取 PLY 点云。OBJ网格需先按表面采样为PLY，不能直接作为现有训练输入。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(obj_count=len(records), archive_count=len(archives), problems=[r for r in records if r['invalid_records'] or r['missing_materials']]), ensure_ascii=True))


if __name__ == '__main__':
    main()
