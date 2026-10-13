"""将验证消融对照导出为集中归档的静态图。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main import ROOT
from organize_figures import organize


def render(directory):
    directory = Path(directory).resolve()
    report = json.loads((directory / 'comparison.json').read_text(encoding='utf-8'))
    rows = report['runs']
    fields = [('val_cd_squared', 'Validation CD squared (lower is better)'),
              ('val_boundary_coverage', 'Boundary coverage (higher is better)'),
              ('val_boundary_error_over_radius', 'Boundary distance / R (lower is better)')]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (field, title) in zip(axes, fields):
        values = [row[field] for row in rows]
        palette = ['#4b9cd3', '#78b86a', '#eea236', '#9b7bb5']
        axis.bar([row['label'] for row in rows], values,
                 color=[palette[i % len(palette)] for i in range(len(rows))])
        axis.set(title=title, ylim=(0, max(values) * 1.25))
        for i, value in enumerate(values):
            axis.text(i, value + max(values)*0.02, f'{value:.5f}', ha='center')
        axis.grid(axis='y', alpha=0.2)
    figure.suptitle(f'{directory.name}: fixed budget / seed {report["budget"]["seed"]}')
    figure.tight_layout()
    folder = ROOT / 'outputs' / 'experiment_figures'
    folder.mkdir(parents=True, exist_ok=True)
    filename = f'{directory.name}_validation_comparison.png'
    figure.savefig(folder / filename, dpi=180)
    plt.close(figure)
    lines = [f'# {directory.name} 图片说明', '', f'![验证对照]({filename})', '',
             '用途：报告消融实验章节。依次展示验证CD²、缺口外缘覆盖率和外缘距离/R。',
             '外缘指真实缺失目标中距离球心至少0.8R的点；覆盖指其在0.1R距离内有预测邻居的比例。',
             f'来源：{directory / "comparison.json"}',
             '相同数据划分、训练预算和随机种子，按各自验证CD²选择最佳检查点。单种子、2个验证器物，不能据此证明普遍改善。', '']
    (folder / f'{directory.name}_图片说明.md').write_text('\n'.join(lines), encoding='utf-8')
    organize()
    return folder / filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    print(render(parser.parse_args().directory))
