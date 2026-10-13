"""组装补全原型交付包、冻结模型及实验报告。"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import torch

from main import ROOT

PACKAGE = ROOT / 'outputs' / 'prototype' / 'bronze_completion'
DATASET = ROOT / 'outputs' / 'datasets' / 'bronze_20261013_seed42'


def assemble():
    selection = json.loads((PACKAGE / 'selection.json').read_text(encoding='utf-8'))
    source = torch.load(selection['checkpoint'], map_location='cpu', weights_only=True)
    deployed = {k: source[k] for k in ('state_dict', 'output_points', 'input_points', 'seed', 'output_mode', 'model_version')}
    deployed['description'] = 'Local point cloud completion prototype; fixed validation-selected model'
    torch.save(deployed, PACKAGE / 'model.pt')
    code = PACKAGE / 'src'
    code.mkdir(exist_ok=True)
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('requirements*.txt')) + [ROOT/'LITERATURE_PLAN.md']
    provenance = {}
    for path in paths:
        shutil.copy2(path, code / path.name)
        provenance[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    tests = code/'tests'
    tests.mkdir(exist_ok=True)
    for path in (ROOT/'tests').glob('test_*.py'):
        shutil.copy2(path, tests/path.name)
        provenance[f'tests/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    entries = [e for e in manifest['samples'] if e['split']=='test']
    demos = []
    for mid in sorted({e['model_id'] for e in entries}):
        entry = next(e for e in entries if e['model_id']==mid)
        sample = DATASET / entry['path']
        directory = PACKAGE/'demo_inputs'/mid
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample/'input.ply', directory/'input.ply')
        meta = json.loads((sample/'region.json').read_text(encoding='utf-8'))
        config = dict(model_id=mid, input=f'demo_inputs/{mid}/input.ply', center=meta['center'], radius=meta['radius'],
                      source_sample=entry['path'], split='test')
        (directory/'region.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        demos.append(config)
    (PACKAGE/'demo_config.json').write_text(json.dumps(demos,indent=2),encoding='utf-8')
    (PACKAGE/'run_demo.py').write_text('''from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from complete_region import complete
for item in json.loads((ROOT / 'demo_config.json').read_text(encoding='utf-8')):
    directory = ROOT / 'demo_results' / item['model_id']
    if directory.exists():
        print(f'Existing result preserved: {directory}')
        continue
    complete(ROOT / item['input'], ROOT / 'model.pt', item['center'], item['radius'], directory)
    print(directory / 'comparison.html')
''',encoding='utf-8')
    provenance['model.pt'] = hashlib.sha256((PACKAGE/'model.pt').read_bytes()).hexdigest()
    (PACKAGE/'source_hashes.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    report_dir = PACKAGE/'evidence'
    report_dir.mkdir(exist_ok=True)
    run = Path(selection['checkpoint']).parent
    for name in ('history.json', 'evaluation_val.json', 'evaluation_test.json', 'quality_val.json', 'quality_test.json'):
        if (run/name).exists():
            shutil.copy2(run/name, report_dir/name)
    shutil.copy2(DATASET/'manifest.json', report_dir/'dataset_manifest.json')
    figures = PACKAGE/'figures'
    figures.mkdir(exist_ok=True)
    for name in ('bronze_cd128_seed42_training_curves.png', 'bronze_dcd128_seed42_training_curves.png',
                 'bronze_dcd_seed42_validation_comparison.png', '青铜器数据集_划分与缺失比例_seed42.png'):
        shutil.copy2(ROOT/'outputs/experiment_figures'/name, figures/name)
    for label in ('cd128', 'dcd128'):
        shutil.copy2(ROOT/f'outputs/comparisons/bronze_dcd_seed42/{label}/local_views.png', figures/f'{label}_local_views.png')
    print(PACKAGE)


def report():
    selection=json.loads((PACKAGE/'selection.json').read_text(encoding='utf-8'))
    selected=selection['selected']['run']
    run=ROOT/'outputs/runs'/selected
    test=json.loads((run/'evaluation_test.json').read_text(encoding='utf-8'))
    quality={name: json.loads((ROOT/'outputs/runs'/name/'quality_val.json').read_text(encoding='utf-8'))['summary']
             for name in ('bronze_cd128_seed42','bronze_dcd128_seed42')}
    qtest=json.loads((run/'quality_test.json').read_text(encoding='utf-8'))['summary']
    lines=['# 青铜器局部补全原型：交付与实验报告','',
           '交付范围：模型导入、模拟缺口、训练、验证选择、独立测试、真实残缺PLY推理、结果点云与预览。单种子小规模原型，不包含文本条件模型、交互选区界面或网格重建。','',
           '## 快速运行','',
           '在项目根目录执行：','', '```powershell',
           r'.\.venv\Scripts\python.exe outputs\prototype\bronze_completion\run_demo.py','```','',
           '该命令使用交付包内两个测试器物的残缺点云和选区参数，不读取完整目标。已有演示结果保留，避免覆盖。','',
           '使用自己的残缺PLY：','', '```powershell',
           r'.\.venv\Scripts\python.exe complete_region.py --input "你的残缺点云.ply" --checkpoint outputs/prototype/bronze_completion/model.pt --center 0 0 0 --radius 0.2',
           '```','',
           '球心与半径是示例，需替换为输入坐标单位下的实际选区。完整点云上模拟删除可加--remove-region。输出包含input.ply、generated.ply、completed.ply、result.json与comparison.html。可见点坐标保留；原输入含RGB时保留已知颜色，新点颜色是灰色占位，不是预测纹理。','',
           '## 数据与模型选择','',
           '14个器物，每器物20个球形缺口；train/val/test按器物划分10/2/2，样本200/40/40。完整样本4096点，模型输入512点，输出128点。自动一致性检查280/280通过。','',
           '所有候选均训练20轮、CPU、seed42、batch8、学习率0.001。模型和方案仅依据验证CD²选取，选择记录selection.json先于测试结果保存。较早边缘约束实验训练目标256点，CD/DCD匹配对照均128点；不能把不同目标采样预算当作只改变损失的消融。','',
           '| 运行 | 验证CD² ↓ | 验证F-score ↑ |','|---|---:|---:|']
    for row in selection['candidates']:
        lines.append(f'| {row["run"]} | {row["val_cd_squared"]:.8f} | {row["val_fscore"]:.6f} |')
    lines += ['',f'冻结部署模型：{selected}/best.pt，第20轮。交付model.pt仅保留推理配置与权重；续训使用原运行目录last.pt。','',
              '## 文献方法对照与负结果','',
              'DCD依据Wu等NeurIPS 2021论文第3.3、5.4节及作者calc_dcd，使用平方距离、邻居查询计数、训练alpha50与计数指数0.5。采用可见输入归一化坐标，不再除以区域R²。仅借鉴损失，非完整论文系统复现。',
              '[论文全文](https://arxiv.org/html/2111.12702v1) · [作者实现](https://github.com/wutong16/Density_aware_Chamfer_Distance/blob/main/utils_v2/model_utils.py)','',
              '| 等预算对照 | CD128 | DCD128 |','|---|---:|---:|']
    for label,key in [('等点数DCD ↓','dcd_equal_points_mean'),('近邻距离中位数/R','prediction_nn_median_over_radius'),
                      ('近邻距小于0.02R的点比例','prediction_close_fraction_0_02_radius'),
                      ('预测点到源网格距离/R ↓','prediction_mesh_mean_distance_over_radius')]:
        lines.append(f'| {label} | {quality["bronze_cd128_seed42"][key]:.6f} | {quality["bronze_dcd128_seed42"][key]:.6f} |')
    lines += ['',
              'DCD减少了本诊断定义下的点聚集，等点数DCD改善，但整体CD²和点到源网格距离变差。不能宣称全面改善，也未替换CD基线。排斥/分层生成等其他文献方案尚未实现。',
              'DCD评价在输入归一化坐标下alpha1000、指数1，固定抽取128目标点重复5次；目标不足128时有放回采样并在JSON标明。聚集统计是工程诊断，非论文标准指标。网格只用于评价，没有将预测投影回目标网格。','',
              '## 独立测试结果','',
              f'测试40个缺口、2个未参与训练器物。CD²={test["mean_local_cd_squared"]:.8f}，F-score={test["mean_local_fscore"]:.6f}（阈值0.02，输入归一化坐标）。',
              f'测试等点数DCD={qtest["dcd_equal_points_mean"]:.6f}；点到源网格距离/R={qtest["prediction_mesh_mean_distance_over_radius"]:.6f}。','',
              '| 测试器物 | 缺口数 | CD² ↓ | F-score ↑ |','|---|---:|---:|---:|']
    for mid,row in test['models'].items():
        lines.append(f'| {mid} | {row["samples"]} | {row["local_cd_squared"]:.8f} | {row["local_fscore"]:.6f} |')
    lines += ['', '## 预览与图片用途','',
              '- [测试器物bronze06补全](demo_results/bronze_06/comparison.html)：四羊方尊固定首个模拟缺口。',
              '- [测试器物bronze07补全](demo_results/bronze_07/comparison.html)：战国青铜鼎固定首个模拟缺口。',
              '- [训练曲线](figures/bronze_cd128_seed42_training_curves.png)：训练过程章节。',
              '- [CD与DCD验证对照](figures/bronze_dcd_seed42_validation_comparison.png)：方法消融章节。',
              '- [CD局部诊断](figures/cd128_local_views.png)：低/中/高验证误差案例，非随机抽样。',
              '- [DCD局部诊断](figures/dcd128_local_views.png)：同样按该方案误差排名选取，不能将同排名当同一缺口。',
              '- [数据划分与缺失比例](figures/青铜器数据集_划分与缺失比例_seed42.png)：数据准备章节。','',
              '- [固定测试缺口静态图](figures/bronze_prototype_test_examples.png)：测试结果章节。蓝色可见点、橙色预测、绿色目标，统一XZ投影与半径尺度，不是随机代表案例。','',
              '## 验收与限制','',
              '35项自动测试通过（34项全套检查及新增DCD续训检查），含坐标/颜色保持、选区边界、损失梯度、数据划分、断点恢复与模型加载。部署模型另外以无目标输入完成两个独立测试案例。','',
              '模型仍可能产生中心聚集、表面偏移；只输出点云，不是可直接3D打印的闭合网格。数据为网格表面采样加模拟缺失，不是真实损伤修复实验。单训练种子、验证和测试各2个器物，无法支撑稳定泛化或统计显著性结论。',
              '原型工程链路已跑通，不表示修复质量达到了文物应用要求。','',
              '## 文件与复现','',
              'model.pt为冻结推理模型；src/为源码快照；demo_inputs/为残缺演示输入；demo_results/为结果；evidence/为指标和数据划分记录；figures/为报告配图；source_hashes.json记录源文件与部署模型SHA256。',
              '训练数据与源OBJ保留在原项目outputs/datasets和models目录，未复制进轻量部署包。详细研究依据见src/LITERATURE_PLAN.md。','',
              '安装推理依赖：python -m pip install -r src/requirements-train.txt。随后python run_demo.py。','',
              '重跑匹配CD基线：','```powershell',
              r'.\.venv\Scripts\python.exe train.py --dataset outputs/datasets/bronze_20261013_seed42 --device cpu --epochs 20 --output-mode tanh --target-points 128 --output-dir outputs/runs/my_cd128_run',
              '```','',
              '重跑DCD时增加--loss-mode dcd --dcd-alpha 50 --dcd-exponent 0.5，并使用另一个新输出目录。测试集已在本报告使用，后续研究需避免据此反复调参。','']
    text='\n'.join(lines)
    (PACKAGE/'README.md').write_text(text,encoding='utf-8')
    # Small controlled Markdown renderer for this generated report; no external script or service.
    def inline(value):
        escaped=html.escape(value)
        return __import__('re').sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', escaped)
    content=[]
    code_block=False
    table_open=False
    list_open=False
    for line in lines:
        if line.startswith('|'):
            if not table_open:
                content.append('<table>'); table_open=True
            cells=[c.strip() for c in line.strip('|').split('|')]
            if all(set(c)<=set('-: ') for c in cells):
                continue
            content.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in cells)+'</tr>')
            continue
        if table_open:
            content.append('</table>'); table_open=False
        if line.startswith('```'):
            content.append('</code></pre>' if code_block else '<pre><code>'); code_block=not code_block
            continue
        if code_block:
            content.append(html.escape(line)+'\n'); continue
        if line.startswith('- '):
            if not list_open:
                content.append('<ul>'); list_open=True
            content.append('<li>'+inline(line[2:])+'</li>'); continue
        if list_open:
            content.append('</ul>'); list_open=False
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'))
            content.append(f'<h{level}>'+inline(line[level:].strip())+f'</h{level}>')
        elif line.strip():
            content.append('<p>'+inline(line)+'</p>')
    links=''.join(f'<li><a href="demo_results/{mid}/comparison.html">{mid} 补全预览</a></li>' for mid in test['models'])
    images=''.join(f'<figure><img src="figures/{name}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
                   for name in ('bronze_cd128_seed42_training_curves.png','bronze_dcd_seed42_validation_comparison.png','cd128_local_views.png','dcd128_local_views.png'))
    page=f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>青铜器补全原型实验报告</title><style>body{{max-width:1100px;margin:32px auto;padding:0 24px;font:16px/1.65 sans-serif;color:#243242}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:16px}}img{{max-width:100%}}a{{color:#176ba0}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td{{border:1px solid #d7e0e8;padding:8px}}tr:first-child{{background:#e8f0f5;font-weight:bold}}</style><nav><a href="README.md">Markdown报告</a> · <a href="model.pt">推理模型</a></nav>{"".join(content)}{images}</html>'
    (PACKAGE/'report.html').write_text(page,encoding='utf-8')
    print(PACKAGE/'README.md')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if not args.report_only:
        assemble()
    else:
        report()
