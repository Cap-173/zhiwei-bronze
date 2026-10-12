# 公开数据：ModelNet40 器皿子集

来源：[Princeton ModelNet](https://modelnet.cs.princeton.edu/download.html)。
官方压缩包：https://modelnet.cs.princeton.edu/ModelNet40.zip ，约 1.90 GiB。
数据是 OFF CAD 网格，原始作者保留模型版权，官方提供用途是学术研究。
本项目只将代码提交 GitHub，不把下载的模型或转换的数据提交到公开仓库。
论文中需引用 ModelNet/3D ShapeNets；商用及再分发需另核对原始模型权利。

## 下载（PowerShell）

```powershell
New-Item -ItemType Directory -Force outputs/downloads
curl.exe --proxy http://127.0.0.1:7890 --location --fail --continue-at - --output outputs/downloads/ModelNet40.zip.part https://modelnet.cs.princeton.edu/ModelNet40.zip
```

下载成功并检查 ZIP 后，将文件名改为 `ModelNet40.zip`。`.part` 可续传，未完成的包不要导入。
原始文件大小为 2,039,180,837 字节（本次官方响应），该大小不是密码学完整性校验。
ZIP 文件读取时校验选中模型的 CRC；导入记录每个原始 OFF 文件的 SHA-256。

## 表面采样与官方划分

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe import_modelnet.py
```

默认类别：bottle、bowl、cup、flower_pot、vase。
每类随机选最多 40 个官方训练模型和 10 个官方测试模型，seed=42，总计最多 250 个。
从每类选中的官方训练模型划出约 10% 作为验证；官方测试模型不参与训练或验证。
无法解析的模型跳过并记录，之后尝试同类候选补足。
使用 [trimesh 表面采样](https://trimesh.org/trimesh.sample.html)，按三角面面积抽取 8192 点，不直接拿网格顶点当均匀点云。
输出 `outputs/public_data/modelnet_.../`，含 PLY、models.json、splits.json 和来源/采样记录 provenance.json。
需要更多数据可提高 `--train-per-category` / `--test-per-category`，上限为该类别可用数量。

## 构造补全样本

将下列 `具体目录` 替换为实际导入目录：

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --models outputs/public_data/modelnet_具体目录/models.json --splits outputs/public_data/modelnet_具体目录/splits.json --samples-per-model 3 --points 4096
.\.venv\Scripts\python.exe inspect_dataset.py
```

必须提供 `--splits` 保留官方测试模型。否则程序会重新随机划分，不能称为保留官方测试划分。
训练时按需扩大每模型的缺口数量，并保持验证/测试条件固定。
该子集与模拟球形缺口是我们构造的补全任务，不是官方 ModelNet 分类任务，也不是标准 Completion3D 分数。

## 质量与适用范围

器皿 CAD 用于学习一般几何形状，再用真实青铜器研究数据适配与验证。
这些数据不是青铜器扫描；薄壁、纹饰、扫描噪声与实际损坏可能有明显差别。
`watertight=false` 不一定代表损坏（杯口、薄片等也可能开放），不能自动据此筛除或认定完整。
抽样检查模型是否完整、是否有异常面片或不合理的内部结构。源文件精确重复会跳过，但相似模型仍需检查。
源模型少量元数据保存在本地 provenance.json，方便追踪器物 ID 和官方划分。
