"""汇总实验图和待截图页面，保留原始输出。"""
import ast
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def organize():
    folder = ROOT / 'outputs' / 'experiment_figures'
    folder.mkdir(parents=True, exist_ok=True)
    lines = ['# 实验图片索引', '', '报告用图统一放在本目录，原始输出保留用于追溯。', '',
             '## 已有图片', '',
             '蓝色：可见上下文；绿色：真实缺失目标；橙色：预测点；虚线圆：缺口半径。',
             '三列为验证CD²低、中、高误差案例，三行为不同局部PCA投影。案例按误差选择，并非随机代表案例。', '']
    for source in sorted((ROOT / 'outputs' / 'diagnostics').glob('val_*/local_views.png')):
        name = f'验证局部补全诊断_{source.parent.name}.png'
        if source.parent.name == 'val_20261012_132022_354683':
            name = '图F_原基线验证局部补全诊断.png'
        shutil.copy2(source, folder / name)
        relative = source.relative_to(ROOT).as_posix()
        lines += [f'### {name}', '', f'![{name}]({name})', '',
                  '用途：检查表面偏移、预测点聚集与缺口外围覆盖，用于报告结果分析。',
                  f'来源：[{relative}](../../{relative})', '']
    plan = ROOT / 'outputs/documents/20261012/add_figure_positions.py'
    if plan.exists():
        tree = ast.parse(plan.read_text(encoding='utf-8'))
        items = next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'items' for t in n.targets))
        lines += ['## 待截图页面（尚未导出图片）', '', '按下列文件名保存；保留图例，图E与图I保持相同视角和缩放。', '']
        for letter, title, _, location, source, capture, caption in items:
            lines += [f'### 图{letter}：{title}', '', f'文件名：`图{letter}_{title}.png`',
                      f'来源：[{title}](../../{source})', f'截取内容：{capture}。',
                      f'报告位置：{location}。', f'建议图注：{caption}', '']
    lines += ['## 后续整理', '', '命名使用“图号_实验或物体名_用途.png”，不同版本不要覆盖同名图片。',
              '运行 `python organize_figures.py` 更新索引；render_diagnostics.py 导出后也会自动汇总。',
              '文档分页检查图和工具图标不属于实验图，未收录。', '']
    extra_notes = sorted(p for p in folder.glob('*.md') if p.name != 'README.md')
    if extra_notes:
        lines += ['## 其他实验图片说明', '']
        lines += [f'- [{p.stem}]({p.name})' for p in extra_notes]
        lines += ['']
    index = folder / 'README.md'
    index.write_text('\n'.join(lines), encoding='utf-8')
    return index


if __name__ == '__main__':
    print(organize())
