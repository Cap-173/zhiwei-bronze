"""导出训练曲线静态图，集中归档并写明用途及来源。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main import ROOT
from organize_figures import organize


def render(run):
    run = Path(run).resolve()
    history = json.loads((run / 'history.json').read_text(encoding='utf-8'))
    rows = history['history']
    if not rows:
        raise ValueError('训练历史为空。')
    folder = ROOT / 'outputs' / 'experiment_figures'
    folder.mkdir(parents=True, exist_ok=True)
    filename = f'{run.name}_training_curves.png'
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    epochs = [r['epoch'] for r in rows]
    axes[0].plot(epochs, [r['train_loss'] for r in rows], label='Train loss')
    for split in ('train', 'val'):
        axes[1].plot(epochs, [r[f'{split}_cd_squared'] for r in rows], label=split)
        axes[2].plot(epochs, [r[f'{split}_fscore'] for r in rows], label=split)
    dcd = history.get('config', {}).get('loss_mode') == 'dcd'
    loss_title = 'Density-aware Chamfer loss' if dcd else 'Loss / region radius squared'
    for ax, title in zip(axes, (loss_title, 'Local CD squared (lower is better)', 'F-score at 0.02 (higher is better)')):
        ax.set(title=title, xlabel='Epoch')
        ax.grid(alpha=0.2)
        ax.legend()
    axes[2].set_ylim(0, 1)
    figure.suptitle(run.name)
    figure.tight_layout()
    figure.savefig(folder / filename, dpi=180)
    plt.close(figure)
    best = min((r for r in rows if r['val_cd_squared'] is not None), key=lambda r: r['val_cd_squared'])
    notes = [f'# {run.name} 图片说明', '', f'![训练曲线]({filename})', '',
             '用途：报告训练过程与验证表现。左为DCD损失（DCD运行）或按区域半径平方缩放的CD损失（CD运行），中为局部CD²，右为F-score。不同损失的数值不直接比较。',
             f'数据来源：{run / "history.json"}',
             f'最佳验证CD²轮次：{best["epoch"]}；CD²={best["val_cd_squared"]:.8f}，F-score={best["val_fscore"]:.4f}。',
             '最佳检查点按验证CD²选取。曲线不包含测试集；该数据集验证部分只有2个器物。', '']
    (folder / f'{run.name}_图片说明.md').write_text('\n'.join(notes), encoding='utf-8')
    organize()
    return folder / filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    print(render(parser.parse_args().run))
