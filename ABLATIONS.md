# 外缘覆盖对照实验

使用相同数据、随机种子、20 轮训练、batch=8、输入512/目标256/输出128点和相同学习率。
仅比较验证集，不根据测试集选择输出方式或损失权重。

| 方案 | output-mode | boundary-weight |
| --- | --- | --- |
| baseline | soft | 0 |
| tanh | tanh | 0 |
| boundary | soft | 0.5 |
| combined | tanh | 0.5 |

`soft` 为旧版 `raw / (1 + norm(raw))`。
`tanh` 保持原始方向，将长度映射为 `tanh(norm(raw))`，相同原始幅度更容易到达球形选区外侧。
两种输出都限制在指定球内，选区外由原始输入点拼接保证不变。
旧权重缺少 output_mode 时按 soft 加载；两种方式不能混淆加载。

外缘覆盖项是距球心 >= 0.8R 的真实局部目标点到预测点的单向平均平方距离。
没有外缘目标的样本该项为0。与双向 CD 一起除以 R²：

```text
loss = mean_batch((local_CD_squared + lambda * outer_target_to_prediction_squared) / R_squared)
```

训练使用每样本随机采样的目标点，评价/诊断使用全部局部目标点。
该项不是法向、曲率或表面连续性损失，不能仅凭覆盖率提升宣称边界平滑。
lambda=0.5 是首轮预先指定的试验值，并非已验证最佳参数。

## 运行

```powershell
.\.venv\Scripts\python.exe train.py --dataset outputs/datasets/dataset_具体目录 --epochs 20 --output-mode tanh --output-dir outputs/runs/ablation_tanh_seed42
.\.venv\Scripts\python.exe train.py --dataset outputs/datasets/dataset_具体目录 --epochs 20 --boundary-weight 0.5 --output-dir outputs/runs/ablation_boundary_seed42
.\.venv\Scripts\python.exe train.py --dataset outputs/datasets/dataset_具体目录 --epochs 20 --output-mode tanh --boundary-weight 0.5 --output-dir outputs/runs/ablation_combined_seed42
```

租用 GPU 时再加 `--device cuda --amp`，所有对照须使用相同设备/精度条件。
续训时需沿用 `--output-mode`、`--boundary-weight` 及其他保存的训练参数。
旧版完整检查点可在 soft/0 的默认配置下续训；新模式必须重新训练，不伪装成恢复旧实验。

```powershell
.\.venv\Scripts\python.exe compare_runs.py --dataset outputs/datasets/dataset_具体目录 --runs baseline=outputs/runs/原基线目录 tanh=outputs/runs/ablation_tanh_seed42 boundary=outputs/runs/ablation_boundary_seed42 combined=outputs/runs/ablation_combined_seed42
```

比较器检查数据指纹、累计训练轮数、种子、采样/输出点数、批大小、学习率和 AMP 条件一致。
每个方案用验证 CD 最佳的权重，记录验证 CD、外缘覆盖与外缘距离诊断，以及案例预览。
结果写入新的 `outputs/comparisons/compare_.../`；每个子目录可用 render_diagnostics.py 生成静态图。

单种子、短程训练只能提供初步证据。需要增加种子、重复实验和真实扫描验证。
