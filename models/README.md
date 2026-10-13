# 三维模型文件

本地青铜器原始网格存放在 `青铜器模型_obj/`。转换命令：

```powershell
.\.venv\Scripts\python.exe import_local_models.py
```

脚本解压 ZIP 中的模型，按表面积为每个 OBJ 采样 20,000 点。新输出位于 `outputs/local_models/import_时间戳/`，含 `ply/`、`previews/`、`models.json` 和来源记录 `manifest.json`；不会覆盖原始网格。采样保留原始坐标，不包含纹理颜色。

输出的 `models.json` 可作为 `prepare_dataset.py --models` 的输入。模型数据的来源、授权和是否为扫描数据需另行记录。

将三维模型放在此文件夹，建议按物体建立子目录，例如 `models/bronze_vase/`。

- 文件名包含物体名称和版本，例如 `bronze_vase_v01.ply`。
- OBJ 的 MTL 文件和纹理图片一起保留在物体子目录中。
- 原始模型保留，清洗或采样后的文件另存并标注版本。
- 训练权重继续保存在 `outputs/runs/`。

预览 PLY 点云（替换为实际文件名，在项目目录执行）：

```powershell
.\.venv\Scripts\python.exe main.py "models\bronze_vase_v01.ply" --output outputs/bronze_vase.html
```
