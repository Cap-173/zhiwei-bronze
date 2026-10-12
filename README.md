# 智维青铜

面向青铜器数字化修复的多模态三维局部可控编辑项目。
目前实现 PLY 点云读取、统计、可视化，以及球形选区和模拟局部缺失，尚未实现 AI 修复。

## 运行（Windows / PowerShell）

在项目目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

PyCharm 中选择项目 `.venv` 解释器，安装依赖后也可直接运行 `main.py`。
默认生成 12,000 点的器皿示例（并非真实扫描），预览位于 `outputs/point_cloud.html`。
用浏览器打开 HTML：拖动旋转、滚轮缩放，工具栏可切换平移。
HTML 包含绘图库，可离线使用；程序不会自动启动浏览器。

读取真实模型：

```powershell
.\.venv\Scripts\python.exe main.py "D:\models\bronze.ply"
.\.venv\Scripts\python.exe main.py "D:\models\bronze.ply" --max-points 20000 --output outputs/bronze.html
```

支持 ASCII/二进制 PLY 的 x/y/z 顶点。如有 red/green/blue 则按 RGB（0–255）显示，否则按高度着色。
网格文件只显示顶点，不显示三角面；坐标单位沿用输入文件。
报告总点数、无效坐标数与坐标范围。无效点不显示，空点云给出错误。
默认最多预览 50,000 点，固定种子采样仅影响显示，不修改原始文件。
完整点云仍会读入内存，大文件需考虑内存容量。生成文件位于 `outputs/`，不会提交到 Git。

## 球形选区与模拟缺失

在 PyCharm 中直接运行 `damage.py`，或执行：

```powershell
.\.venv\Scripts\python.exe damage.py
.\.venv\Scripts\python.exe damage.py --center 0.9 0 0.25 --radius 0.3
.\.venv\Scripts\python.exe damage.py "D:\models\bronze.ply" --center 10 20 30 --radius 5
```

此阶段通过参数设置球心和半径，尚不支持鼠标点选。真实模型必须提供球心和半径，单位沿用输入文件。
球内及球面上的点被删除，球外点的坐标、颜色、顺序保持不变。空选区和全部选中会报错。
每次运行生成新的 `outputs/samples/sample_.../`，用浏览器打开其中 `comparison.html`。
左侧橙色点为选区，右侧为残缺点云，透明球标出选区；两侧可独立旋转、缩放。
预览采样不影响保存的完整数据。

| 文件 | 含义 |
| --- | --- |
| `target.ply` | 完整有效点云，作为监督目标 |
| `input.ply` | 删除选区内点后的残缺点云 |
| `removed.ply` | 被删除的点，作为局部补全目标 |
| `mask.npy` | 布尔数组，与 **target.ply** 顶点顺序对应，True 表示被删除 |
| `region.json` | 球心、半径、点数、数据格式版本等 |
| `instruction.txt` | 此阶段固定补全指令，暂不用于证明文本控制能力 |
| `comparison.html` | 完整与残缺点云对比 |

注意：`mask.npy` 不对应 `input.ply` 的点数。缺口没有点，推理时用球心和半径表示待修复位置。
输出只保留 XYZ 和可选 RGB，不保留网格面、法向等其他属性。原始输入文件不会修改。
可用 `--output-dir outputs/samples/my_sample` 指定新的目录，已有目录不会覆盖。

## 批量训练数据准备

直接在 PyCharm 运行 `prepare_dataset.py`，或执行：

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
.\.venv\Scripts\python.exe prepare_dataset.py --samples-per-model 20 --points 4096 --seed 42
```

默认对示例器皿生成 10 个不同缺口，每个完整样本最多 4096 点。
半径随机取完整点云包围盒对角线的 8%–18%，要求删除 2%–20% 的点。
采样与缺口生成可由固定 seed 重现。每次创建新的 `outputs/datasets/dataset_.../`。
首个样本生成对比预览，其余只保存数据，避免重复保存大型 HTML。

每个样本除 PLY、Mask、区域参数外，增加 `training.npz`：

- `input`：归一化的残缺输入，点数可变。
- `local_target`：归一化的被删除点，点数可变。
- `target` / `mask`：完整采样点云与对应布尔选区。
- `center` / `radius`：归一化后的球心、半径。
- `offset` / `scale`：坐标还原参数，原坐标 = 归一化坐标 × scale + offset。

归一化的中心取可见输入包围盒中心，尺度取输入点到中心的最大距离。
变换只依据可见输入，完整目标和球心沿用相同变换；目标点不保证全在单位球内。
`input.ply` 等 PLY 仍为原始尺度下的采样坐标。
可通过 `prepare_dataset.load_training_sample(path)` 读取 NPZ；后续训练再增加固定点数采样/批处理。

使用真实模型时，准备 JSON 清单，例如 `models.json`：

```json
{
  "bronze_001": "data/bronze_001.ply",
  "bronze_002": "data/bronze_002.ply",
  "bronze_003": "data/bronze_003.ply"
}
```

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --models models.json
```

相对路径基于清单所在目录。每个 ID 必须代表不同的真实器物实例，不能把同一器物的不同扫描作为不同 ID。
至少 3 个模型才生成 train/val/test；按模型随机划分，验证和测试各约 10% 且至少 1 个。
少于 3 个模型全部放入 train，仅用于调试，不能评价泛化。三个实例也不足以支持可靠论文结论。
程序拒绝坐标完全相同的重复模型，但不同扫描或不同采样仍需人工确认器物身份。
`manifest.json` 记录划分、种子、参数、样本路径和生成状态；只使用 status 为 complete 的数据集。
目前要求源模型是完整监督目标；对已经残缺的真实扫描做二次删除，不能得到原有缺口的真实答案。

验证代码：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 基础局部补全训练

现已实现轻量 PointNet 风格编码器和 MLP 解码器，输入仅为可见点云与球形选区，不含文本。
模型预测球内 128 个点，拼接原始可见点形成结果；当前只预测 XYZ，不生成颜色或网格。
这是工程基线，不是文档中完整的多模态方法。

安装训练依赖（已在本项目虚拟环境安装 CPU 版）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

如果需要明确选择 CPU 版，可按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装 PyTorch。
训练和预测可直接在 PyCharm 运行 `train.py`、`predict.py`：

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py
```

训练默认读取最近创建的数据集，自动选择可用的 CUDA 或 CPU，运行 50 轮，batch size 为 8，输入采样 512 点（局部邻居与全局上下文各半）。
每步从缺失目标采样 256 点计算双向平均平方 Chamfer 损失，除以选区半径平方进行尺度调整。
评价使用全部缺失目标点与固定输入采样；CD² 越小越好，不能与未平方或不同归一化定义的 CD 直接比较。
每轮保存 `outputs/runs/run_.../last.pt`；有验证集时按验证 CD² 保存 `best.pt`，否则按训练 CD² 保存并明确标注。
`history.json` 保存初始分数与每轮记录。测试集不用于选择模型。
预测默认优先预览测试样本，再选验证样本、训练样本；只预览一个样本，不是完整测试集评价。

指定参数：

```powershell
.\.venv\Scripts\python.exe train.py --dataset outputs/datasets/dataset_具体目录 --epochs 100
.\.venv\Scripts\python.exe predict.py --checkpoint outputs/runs/run_具体目录/best.pt --sample outputs/datasets/dataset_具体目录/train/model_0000_sample_0000
```

真实残缺点云推理（不需要完整目标）：

```powershell
.\.venv\Scripts\python.exe predict.py --input "D:\models\damaged.ply" --center 10 20 30 --radius 5
```

输入应已经去掉选区内待替换的点，程序只添加预测点，不自动删除输入点。
推理归一化仅依据输入，预测后还原到原始尺度。结果目录包含 `generated.ply`、`completed.ply`、`metrics.json` 和 `comparison.html`。
完整参考与被删除点只在推理结束后用于评价和展示，不参与预测。
仅有示例器皿时，训练分数下降说明流程能学习，不代表真实青铜器修复能力或泛化性能。
保存加载遵循 [PyTorch 官方权重保存说明](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)，不加载外部完整 Python 模型对象。

## 后续步骤

GPU 安装、批量训练、断点续训和独立评价的命令见 [GPU 运行说明](GPU_TRAINING.md)。

1. 准备更多完整模型并检查缺口质量。
2. 多器物验证、完整测试集评价与失败案例分析。
3. 根据基线结果改进局部形状和边界质量。
4. 文本控制与交互系统。

完整方案见 [V1.0 文档](智维青铜_多模态3D局部可控编辑_V1.0_README.md)。
官方文档：[PLY 读写](https://python-plyfile.readthedocs.io/en/stable/usage.html)、[Plotly 三维散点](https://plotly.com/python/3d-scatter-plots/)。
