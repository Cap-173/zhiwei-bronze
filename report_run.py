"""生成训练损失、局部 CD 和 F-score 的离线交互曲线。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from main import ROOT


def render_history(history_path, output):
    report = json.loads(history_path.read_text(encoding='utf-8'))
    rows = report['history']
    if not rows:
        raise ValueError('训练历史为空。')
    epochs = [row['epoch'] for row in rows]
    figure = make_subplots(rows=1, cols=3, subplot_titles=('训练损失（区域尺度）', '局部 CD²（越小越好）', '局部 F-score（越大越好）'))
    fields = [('train_loss', '训练损失', 1), ('train_cd_squared', '训练 CD²', 2),
              ('val_cd_squared', '验证 CD²', 2), ('train_fscore', '训练 F-score', 3), ('val_fscore', '验证 F-score', 3)]
    for key, name, column in fields:
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            figure.add_trace(go.Scatter(x=epochs, y=values, name=name, mode='lines+markers', connectgaps=False), row=1, col=column)
    figure.update_layout(template='plotly_dark', height=650,
                         title=f'训练报告 · {report.get("evaluation_scope", "评价范围未记录")} · {len(rows)} 轮')
    figure.update_xaxes(title_text='Epoch')
    figure.update_yaxes(range=[0, 1], row=1, col=3)
    figure.write_html(str(output), include_plotlyjs=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        run = args.run
        if run is None:
            histories = sorted((ROOT / 'outputs' / 'runs').glob('run_*/history.json'))
            if not histories:
                raise ValueError('没有训练记录，请先运行 train.py。')
            run = histories[-1].parent
        output = args.output_dir or ROOT / 'outputs' / 'reports' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        report = render_history(run / 'history.json', output / 'curves.html')
        (output / 'history.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'训练轮数: {len(report["history"])} | 评价范围: {report.get("evaluation_scope", "unknown")}')
        print(f'曲线预览: {(output / "curves.html").resolve()}')
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'生成报告失败: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
