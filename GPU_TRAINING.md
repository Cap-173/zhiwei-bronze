# GPU 训练与迁移

当前代码支持单 GPU CUDA、CPU、batch 训练、可选 AMP 和完整轮次断点恢复。
本地安装的是 CPU 版 PyTorch；CUDA 与 AMP 路径尚未在真实 GPU 上验证。
不需要在本地安装 CUDA 版。租用机器后选已有 PyTorch/CUDA 环境，按 [官方安装页](https://pytorch.org/get-started/locally/)确认驱动与 PyTorch 匹配。

## 安装和设备检查（Linux GPU 机器）

```bash
git clone https://github.com/Cap-173/zhiwei-bronze.git
cd zhiwei-bronze
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

这些新增代码尚未推送时，需先上传代码或手动传输整个源码目录。
已有 CUDA 版 PyTorch 时无需重装；没有时按官方页安装，再安装其余依赖。
先运行 `nvidia-smi` 检查 GPU 和驱动，再运行上述 Python 检查。
`--device cuda` 在 CUDA 不可用时明确报错，不会悄悄转成 CPU。

## 数据准备与训练

传输完整数据集目录（含 manifest、所有 training.npz 和 PLY）。`outputs/` 被 Git 忽略，不会随克隆下载。
也可传输完整源模型和 JSON 清单，在服务器重新运行 `prepare_dataset.py`。
先确认多器物 train/val/test 划分，不把一个器物的多个缺口当作独立测试器物。

```bash
python train.py --dataset /workspace/dataset --device cuda --epochs 100 --batch-size 8 --workers 2 --amp --output-dir outputs/runs/gpu_run
```

输入和目标按需读取，不把整个数据集载入内存。默认目标采样 256 点，输出 128 点。
`--input-points`、`--target-points`、`--output-points` 可调整，但应保持各实验的评价条件一致。
显存不足先降低 batch size。`history.json` 记录每轮 CUDA 最大已分配显存（不代表全部 GPU 占用）。
AMP 使用 [PyTorch autocast/GradScaler](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)，网络可用半精度，Chamfer 距离保持 float32。
本地调试：`python train.py --device cpu --batch-size 8 --epochs 5`，不加 `--amp`。

## 断点续训

```bash
python train.py --dataset /workspace/dataset --resume outputs/runs/gpu_run/last.pt --device cuda --epochs 200 --batch-size 8 --workers 2 --amp
```

`--epochs 200` 表示累计训练到第 200 轮。恢复到最后完成的轮次；轮次中途停止，重启后重做该轮。
保存内容包括模型、Adam 状态、GradScaler、轮数、最佳分数、历史记录、随机数状态与配置。
按轮次/样本确定性生成采样，数据和训练参数变化会拒绝恢复。输入/输出/目标点数、batch size、种子、学习率与 AMP 必须沿用原值。
worker 数可以变化；跨设备迁移允许，但不同硬件的数值结果不保证完全一致。
迁移时传输整个运行目录，包括 `best.pt`、`last.pt`。数据路径变化通过 `--dataset` 指向新位置。
旧版训练权重只能用于预测，不能完整续训。
每个检查点通过临时文件替换保存；不能同时向同一运行目录启动两个训练进程。

## 独立测试集评价

```bash
python evaluate.py --checkpoint outputs/runs/gpu_run/best.pt --dataset /workspace/dataset --split test --device cuda
```

评价整个指定划分，输出每样本、每器物、样本平均与器物平均的局部 CD² 和 F-score。
CD² 是双向最近平方距离均值之和；F-score 阈值默认 `0.02`，单位为依据可见输入归一化后的坐标单位。
使用全部局部目标点，并分块计算距离以限制内存。训练损失仍使用采样目标。
默认报告 `evaluation_test.json`，已存在时拒绝覆盖，可用 `--output` 另指定报告路径。
没有 test 样本会明确报错，不使用 train 替代。
训练只依据 val 选择最佳模型；无 val 时按 train 选择，不能视作泛化评估。
`matches_training_dataset` 标记数据是否与检查点绑定的数据一致；评价外部数据时需自己确认器物不重叠。
测试集用于最终报告，不根据测试结果反复选择模型。

现有器皿示例仅用于调试；多器物实验、真实扫描缺失、边界质量和失败案例仍需进一步验证。

外缘实验可以增加 `--output-mode tanh --boundary-weight 0.5`；续训时必须沿用这两个参数。
保持输出方式、外缘权重、设备和训练预算的对照说明见 [ABLATIONS.md](ABLATIONS.md)。
