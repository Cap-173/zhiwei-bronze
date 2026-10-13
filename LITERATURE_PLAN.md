# 青铜器局部补全：文献依据与下一轮实验

检索日期：2026-10-13。来源为会议论文、作者论文页面与作者代码，不使用博客或AI综述作为方法依据。以下是针对当前问题的定向调研，不是完整系统综述或最新方法排行榜。

## 当前证据与需要纠正的解释

当前模型在completion.py中将逐点MLP特征做全局max pooling，然后由全连接解码器一次生成128个坐标。它不是PF-Net、PCN或SnowflakeNet的复现。

验证图显示中心聚集、部分点偏离目标表面。权重0、0.1、0.5的边缘覆盖实验没有带来全部指标一致改善；只能作为经验消融，不能写成文献验证的方法。

此前定义“边缘”为距模拟球心至少0.8R的目标点。这是球形外缘带，不等于真实缺口边界，更不等于器物棱边。外缘覆盖提高也不能证明表面连续或接缝平滑。

## 别人如何处理相关问题

### 1. 密度失衡与CD盲区：DCD，NeurIPS 2021

Wu等讨论CD对局部密度失衡不敏感，提出Density-aware Chamfer Distance，既可评价也可训练。原文第3节、第5.4节和附录A.2分别给出方法、损失实验和点数不一致处理。

对本项目的推断：CD下降并不足以排除中心聚集，先补充密度诊断，再单独对照损失。不是已证明当前失败完全由CD造成。

- [会议页面（页面标题为Balanced Chamfer Distance，摘要方法为DCD）](https://papers.nips.cc/paper_files/paper/2021/hash/f3bd5ad57c8389a8a1a541a76be463bf-Abstract.html)
- [作者论文全文](https://arxiv.org/html/2111.12702v1)
- [作者实现calc_dcd](https://github.com/wutong16/Density_aware_Chamfer_Distance/blob/main/utils_v2/model_utils.py)

### 2. 点聚集与表面分布：PU-Net，CVPR 2018

Yu等在点云上采样中结合重建与局部排斥损失，改善表面点的分布；原文损失函数部分给出k近邻与距离衰减的排斥项，并评价点到网格距离和均匀性。

对本项目的推断：可借用局部排斥项作独立消融。PU-Net不是缺口补全方法；排斥只能抑制拥挤，不能保证点落在正确表面，也不能直接采用其尺度参数而不换算。

- [会议论文入口](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.html)
- [作者代码](https://github.com/yulequan/PU-Net)

### 3. 保留已知点、分层生成缺失区域：PF-Net，CVPR 2020

Huang等只预测缺失点，保留原输入空间位置；使用多尺度输入、Point Pyramid Decoder和多阶段监督，并包含对抗训练。作者生成器先输出64个中心，再通过相对偏移生成128个中心与最终点。

对本项目的推断：这是比全形状重建更接近当前局部修复目标的结构参考。优先试分层中心与子点生成；若省去其多尺度编码或对抗部分，必须称为借鉴PF-Net的简化消融，不能称完整复现。

- [会议论文入口](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.html)
- [作者生成器代码](https://github.com/zztianzz/PF-Net-Point-Fractal-Network/blob/master/model_PFNet.py)

### 4. 结构化局部细节：SnowflakeNet，ICCV 2021

Xiang等以逐级父点分裂和skip-transformer学习局部生成模式，处理无结构点生成难以表达细节的问题。

对本项目的推断：支持研究分层生成结构；不证明在14个青铜器上一定有效，完整复现应单独评估算力与数据需求。

- [会议论文入口](https://openaccess.thecvf.com/content/ICCV2021/html/Xiang_SnowflakeNet_Point_Cloud_Completion_by_Snowflake_Point_Deconvolution_With_Skip-Transformer_ICCV_2021_paper.html)

### 5. 局部几何信息：PoinTr，ICCV 2021

Yu等以局部点代理、位置编码及geometry-aware block表达邻域几何；用多尺度生成恢复缺失点。

对本项目的推断：当前单一全局池化可能丢失局部结构，需要实验检验；不能由现有结果直接认定必须使用Transformer。

- [论文全文](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_PoinTr_Diverse_Point_Cloud_Completion_With_Geometry-Aware_Transformers_ICCV_2021_paper.pdf)

### 6. 表面偏移与真实棱边：EC-Net，ECCV 2018

Yu等在点集整合任务中利用点到网格、点到几何边缘的监督，恢复表面与尖锐特征。其几何边缘不等于我们的球形外缘。

对本项目的推断：我们保留了源OBJ，可先用点到三角网格距离评价离面误差；若以后借用网格监督，只能访问训练器物目标网格。验证/测试网格不能用于推理后处理或拟合预测。

- [会议论文入口](https://openaccess.thecvf.com/content_ECCV_2018/html/Lequan_Yu_EC-Net_an_Edge-aware_ECCV_2018_paper.html)

## 推荐的实验顺序

1. **补齐诊断，先不改模型**：对已有三个方案增加等点数DCD、点聚集统计与点到网格距离。保留CD²/F-score/外缘覆盖指标。等点数评估从目标固定抽取128点，重复固定抽样并记录；全集CD仍使用全部目标点。原始坐标和归一化尺度明确记录。
2. **一个针对聚集的损失对照**：优先参考作者DCD公式；目标点数与尺度处理核对通过后，与原CD训练对照。保持架构、数据、点数与预算一致，不同时增加边缘项。另一个候选为CD+PU-Net排斥项，不在同一轮混加。
3. **一个结构对照**：借鉴PF-Net的粗到细缺失区域解码，显式记录修改部分，训练与各层监督分开消融。保持只生成缺失点、可见点坐标保留。
4. **再验证可靠性**：固定模型划分做多训练种子复验；当前验证只有2个器物，不能把40个相关缺口当作40个独立器物。之后再考虑器物级交叉验证。冻结方案后使用测试集作最终评价。

DCD公式带距离尺度参数，作者实现区分L1式CD和平方CD。现有训练损失还除以区域半径平方，因此不能把DCD直接替换进去并沿用这一步归一化。实现前需明确坐标体系、参数换算与点数处理，测试恒等点集、聚集点集及梯度。

## 后续实验记录要求

后续结构实验：已加入借鉴PF-Net的两级父点/残差子点解码和FPS粗层CD监督，另设参数量相差0.038%的宽MLP容量对照；seed42比较粗监督0/0.01，固定0.01后以seed43/44复验。40项代码检查通过。结果见outputs/repeats/bronze_hierarchy_confirmation/README.md。原方法的多尺度编码、三级分辨率和对抗模块仍未复现。子点偏移上限0.25R与球形裁剪属于工程改造，不是原论文公式。

2026-10-13执行记录：已完成等点数CD/DCD配对20轮训练，DCD采用alpha50、计数指数0.5和输入归一化坐标，省去旧CD损失的R²缩放；评价DCD采用alpha1000、指数1，目标固定128点抽样5次。已追加近邻聚集统计和到源OBJ的距离。DCD改善密度诊断，但CD²与离面误差变差，因此未替换CD基线。按验证CD²冻结CD128模型后完成40个独立测试缺口评价。详见outputs/prototype/bronze_completion/README.md。PF-Net分层结构与多种子研究仍未实施，不属于已完成的原型功能。

每次改进记录：现象证据、论文和具体章节/作者代码、原论文任务、采用部分、工程改动、验证指标与失败结果。区分“原方法复现”“借鉴后的改造”和“自行提出的假设”。不将文献在ShapeNet等数据上的效果转述为青铜器效果。

此前建议“先延长训练”的步骤是检查训练不足的工程对照，没有论文证明它能解决当前聚集。现在优先采用上述诊断和定向消融顺序，不继续盲目搜索边缘权重。
