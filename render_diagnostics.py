"""生成验证案例的局部多视角静态图，用于人工检查。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from main import ROOT


def render(directory):
    figure, axes = plt.subplots(3, 3, figsize=(13, 12))
    for column, label in enumerate(('lowest_cd', 'median_cd', 'highest_cd')):
        with np.load(directory / f'{label}.npz') as data:
            center, radius = data['center'], float(data['radius'])
            context = (data['input']-center)/radius
            context = context[np.linalg.norm(context, axis=1) <= 1.5]
            predicted = (data['predicted']-center)/radius
            target = (data['local_target']-center)/radius
        # PCA 只用于统一展示角度，绝不回传到预测或训练。
        _, _, basis = np.linalg.svd(np.concatenate((context, target)), full_matrices=False)
        context, predicted, target = (points @ basis.T for points in (context, predicted, target))
        for row, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            axis = axes[row, column]
            axis.scatter(context[:, a], context[:, b], s=3, c='#6aa7bb', alpha=0.35, label='Visible context')
            axis.scatter(target[:, a], target[:, b], s=7, c='#209b62', alpha=0.45, label='Missing target')
            axis.scatter(predicted[:, a], predicted[:, b], s=12, c='#e77d20', alpha=0.85, label='Prediction')
            axis.add_patch(plt.Circle((0, 0), 1, fill=False, linestyle='--', color='#999999'))
            axis.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect='equal',
                     xlabel=f'Local PCA axis {a+1} / R', ylabel=f'Local PCA axis {b+1} / R')
            axis.grid(alpha=0.15)
            if row == 0:
                axis.set_title(label.replace('_', ' ').title())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.97))
    figure.suptitle('Validation diagnostics: local target vs predicted patch', y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = directory / 'local_views.png'
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    directory = args.directory or sorted((ROOT/'outputs'/'diagnostics').glob('val_*'))[-1]
    print(render(directory).resolve())
