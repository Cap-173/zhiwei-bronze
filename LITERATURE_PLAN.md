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

2026-10-13精确匹配借鉴对照：核对PCN tf_util.py等点数近似EMD，改用SciPy modified Jonker-Volgenant精确欧氏成本匹配、平均距离/R、离散对应detach，保留128缺失点和逐点网络。54项代码检查通过。固定缺口1000步CD²/R²0.010838→0.00006264，最近目标位置60→127/128，固定目标F-score1；同器物16缺口200轮训练CD²0.0009955→0.0007745，F-score0.6619→0.7177。但4新缺口CD²0.0058879→0.0060602、F-score0.14023→0.12175、离面/R0.08578→0.10448，换器物评价也差。不采用该损失，原型冻结。实验改变匹配与距离形式/半径缩放，不作匹配因素的单一因果解释。报告outputs/diagnostics/assignment_fit/README.md。下一步候选为固定优化更新次数的训练缺口数量/覆盖对照，而不是单凭拟合改善宣称泛化改善。

下一步候选匹配对照文献已核对：PCN（3DV2018）第4.3节给出等点数EMD双射对应，作者models/pcn_emd.py粗层用earth_mover、细层用chamfer。本项目128点缺口的一对一匹配对照为借鉴改造，需明确原论文近似EMD与拟采用实现、距离形式和半径单位的区别，不能把固定顺序MSE当作无序补全损失。原论文链接https://arxiv.org/pdf/1808.00671，作者代码https://github.com/wentaoyuan/pcn/blob/master/models/pcn_emd.py。

2026-10-13基础与单器物排查：开发12器物240样本的掩码、分区、可见归一化与同器物坐标一致性全部通过；不同OBJ语义朝向未统一，未将PCA轴等同语义朝向。19个目标少于128点，输入重复3.45%，FPS中心2R内29.69%，均只作为诊断。司母戊鼎固定一缺口1000步CD出现平台，随后同初始化/目标的有序MSE诊断可将CD²/R²降到约3e-7；固定CD预测128点只对应60/39个目标位置，有序诊断对应128。不是新的正式训练方法。单器物16缺口训练200轮，训练F-score逐点0.6619/邻域0.5811，同器物4新缺口0.1402/0.1394；另一个器物20缺口0.1165/0.1028。支持优先研究匹配优化与迁移，不能断言数据量是唯一根因。报告outputs/diagnostics/basic_fit/README.md，全部配图集中归档。检查流程参考[Stanford CS231n sanity checks](https://cs231n.github.io/neural-networks-3/#sanity-checks)。

2026-10-13器物级复验：以原训练/验证12个器物做四折分组CV，排除原测试06/07；每折9训练/3留出，复用240个原缺口，逐字节复制校验。两个架构每折从头训练60轮，固定训练seed42与分折seed20261013，主评价固定epoch60 last.pt而非留出选best。CD²器物宏平均pointwise_xyz0.00680672、local0.00913736（local高34.24%），local在12器物上均更差；F-score0.139844对0.117300，聚集8.71%对18.37%，离面/R0.104358对0.117687。当前邻域改造暂停。该CV仅开发池探索性复验、一个训练种子，不能称为全新盲测或统计显著性证据。报告outputs/repeats/bronze_object_cv4/README.md。分折/复制检查及480条数值有限性审查通过；固定案例图已集中归档。

分组评价参考[scikit-learn官方说明](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)，按器物而非单独缺口划分；实现使用固定等量器物分组，不依赖sklearn安装。

2026-10-13同步训练预算复验：邻域local与同参数pointwise_xyz各三个种子从20轮续到40/60轮，恢复优化器与随机状态，保留原20轮模型和40轮快照。60轮平均best CD²：local0.00718219、对照0.00511415，local高40.44%；20→60轮对照改善9.60%、local改善0.48%，local的三个best轮次35/21/32均在40轮内。训练误差持续下降而验证停滞，延长训练未解决当前改造问题。暂停该邻域改造，优先器物级复验和扩充独立器物。报告outputs/repeats/bronze_neighborhood_convergence/README.md。本轮为工程收敛检查，不是文献新方法，也不宣称无限预算收敛。

2026-10-13追加邻域编码实验：核对PointNet++（NeurIPS2017）第3.2节和作者pointnet_util.py，以单层FPS32+kNN16+相对坐标局部max改造补全编码器，无BN/MSG。同解码器、同参数615236的pointwise_xyz对照，三种子各20轮，输入/输出/目标和CD损失固定。邻域CD²均值0.00721678，对照0.00565752（高27.56%）；seed42聚集42.68%对25.35%、网格距离/R0.077847对0.069732。当前改造未获支持，原型保持冻结。全结果见outputs/repeats/bronze_neighborhood_confirmation/README.md，配套同缺口图见outputs/diagnostics/neighborhood_cases_seed42/README.md。分组、置换一致、参数/初始权重一致、有限梯度、旧模型兼容与续训一致性检查通过。

文献：[PointNet++论文](https://proceedings.neurips.cc/paper_files/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)；[作者代码](https://github.com/charlesq34/pointnet2/blob/master/utils/pointnet_util.py)。32/16是预先固定的工程预算选择，不是论文保证的最佳参数。

2026-10-13追加：完成PU-Net借鉴排斥项（权重0.01、带宽0.03、精确4个非自身近邻、缺口半径坐标）与宽MLP的三种子20轮配对。CD²均值下降约2.92%，F-score略降，seed42聚集和离面诊断未改善；不作为已解决聚集或替换原型的证据。详细公式改动、结果和图见outputs/repeats/bronze_repulsion_confirmation/README.md。新增排斥公式、有限梯度、尺度不变性及续训一致性检查通过。

后续结构实验：已加入借鉴PF-Net的两级父点/残差子点解码和FPS粗层CD监督，另设参数量相差0.038%的宽MLP容量对照；seed42比较粗监督0/0.01，固定0.01后以seed43/44复验。40项代码检查通过。结果见outputs/repeats/bronze_hierarchy_confirmation/README.md。原方法的多尺度编码、三级分辨率和对抗模块仍未复现。子点偏移上限0.25R与球形裁剪属于工程改造，不是原论文公式。

2026-10-13执行记录：已完成等点数CD/DCD配对20轮训练，DCD采用alpha50、计数指数0.5和输入归一化坐标，省去旧CD损失的R²缩放；评价DCD采用alpha1000、指数1，目标固定128点抽样5次。已追加近邻聚集统计和到源OBJ的距离。DCD改善密度诊断，但CD²与离面误差变差，因此未替换CD基线。按验证CD²冻结CD128模型后完成40个独立测试缺口评价。详见outputs/prototype/bronze_completion/README.md。PF-Net分层结构与多种子研究仍未实施，不属于已完成的原型功能。

2026-10-13缺口覆盖对照：16/64/128个同器物训练缺口，各固定800次更新，其余网络与CD损失不变，64为128的前缀；固定生成seed20261013并保留原16样本字节。该实验是验证覆盖不足假设的工程对照，64/128不是文献推荐最优数量。报告 `outputs/diagnostics/quantity_fit/README.md`；评价原4个新位置缺口和一个青铜觥，明确报告训练目标的表面覆盖与评价目标重叠，不能替代按独立器物划分的泛化实验。

2026-10-13覆盖候选复验：在seed42探索后固定64缺口，新增训练seed43/44配对16缺口基线，各800更新取last。两枚新增种子均降低同器物及青铜觥CD²，同器物均值下降26.76%；但seed44同器物F-score下降，聚集均值4.10%→18.16%。距离与离面改善可重复，点分布和薄壁仍未解决。报告 `outputs/repeats/bronze_quantity_confirmation/README.md`；只改变训练种子，仍固定同一开发样本且区域重叠，不是新器物盲测或显著性证明。

每次改进记录：现象证据、论文和具体章节/作者代码、原论文任务、采用部分、工程改动、验证指标与失败结果。区分“原方法复现”“借鉴后的改造”和“自行提出的假设”。不将文献在ShapeNet等数据上的效果转述为青铜器效果。

此前建议“先延长训练”的步骤是检查训练不足的工程对照，没有论文证明它能解决当前聚集。现在优先采用上述诊断和定向消融顺序，不继续盲目搜索边缘权重。
