# 模型类型与验证策略

## 目的

不同模型类型的"应该比什么"和"比到什么程度"完全不同。进入对齐前先判断模型类型和下游消费方式，确定验证边界和度量。

## 确定验证边界的方法

对每种模型类型，回答两个问题：

1. **模型直接输出是什么？**（forward 的返回值）
2. **直接输出是否确定性决定下游结果？**（输出到消费间是否有采样、随机过程、非线性放大）

- 是 → 验证边界 = 模型直接输出
- 否 → 验证边界 = 下游功能指标

## NLP / 序列模型（Transformer Encoder/Decoder）

**模型直接输出**：logits（连续向量）

**验证边界判断**：
- 贪心解码 / 非自回归：logits 确定性决定输出 → 比直接输出（logits cosine + max_abs）
- 采样解码：logits 确定性但 sampling 引入随机性 → 比 benchmark 分数
- 下游任务（分类、NER）：logits 确定性决定预测 → 比直接输出 + 任务指标（accuracy / F1）

**度量选择**：
- logits：cosine similarity + max_abs_diff
- token 序列（贪心）：完全匹配率
- benchmark：MMLU / HumanEval / BLEU / Rouge 等任务指标

## CV / 视觉模型（CNN、ViT、检测/分割）

**模型直接输出**：feature map / logits / bbox / mask

**验证边界判断**：
- 分类：logits 确定性决定预测 → 比直接输出 + Top-1 accuracy
- 检测/分割：输出经后处理（NMS、阈值化）产生最终结果 → 比后处理后结果（mAP / mIoU）

**度量选择**：
- feature map：cosine + max_abs_diff
- 分类 logits：cosine + max_abs_diff + Top-1 accuracy
- 检测 bbox/mask：mAP / mIoU（后处理后指标，非坐标 diff）

## 生成式模型（LLM、扩散模型、VAE）

**模型直接输出**：logits（LLM）/ 像素矩阵（diffusion）/ 潜在表示（VAE）

**验证边界判断**：
- LLM 贪心：比 logits
- LLM 采样：比 benchmark 分数（序列对比无意义——不同 seed 产生不同序列都合法）
- Diffusion：比图片语义（FID / LPIPS）——像素级 diff 无意义，两张"看起来一样"的图片像素可能完全不同
- VAE 重构：比 LPIPS 或固定 seed 下对比

**度量选择**：
- logits（贪心 LLM）：cosine + max_abs_diff
- benchmark（采样 LLM）：任务分数（MMLU / HumanEval 等）
- 图片（diffusion）：LPIPS（感知相似度）、FID（分布级，需多样本）、SSIM（结构相似度）
- 概率分布（attention weights）：KL 散度

## 图神经网络（GNN）

**模型直接输出**：节点/图级 embedding 或 logits

**验证边界判断**：
- 节点分类/图分类：logits 确定性决定预测 → 比直接输出 + accuracy
- 属性预测（如 MACE 的 energy/forces）：输出是物理量，下游直接消费 → 比直接输出

**度量选择**：
- embedding：cosine + max_abs_diff
- 物理量（energy、forces）：max_abs_diff + 相对误差（力的绝对值影响下游 MD 轨迹，不能只看方向）
- 图级指标：按任务选择

## 科学计算 / AI4S

AI4S 场景的特殊性：模型直接输出和"下游消费者真正关心的东西"可能不一致。必须判断用哪个作为验证边界。

### 分子动力学 / 力场（如 MACE）

**模型直接输出**：energy（标量）、forces（向量）、stress（矩阵）

**验证边界**：直接输出 → 下游 MD 模拟器直接消费，确定性决定 → 比直接输出

**度量选择**：
- energy：相对误差（标量）
- forces：max_abs_diff + cosine（向量，绝对值影响轨迹）
- stress：max_abs_diff（矩阵）

### 蛋白质结构预测

**模型直接输出**：3D 坐标

**验证边界**：下游功能指标 → 坐标 diff 不代表结构等价 → 比 RMSD / TM-score

**度量选择**：
- RMSD（均方根偏差，衡量结构整体偏差）
- TM-score（模板建模分数，衡量结构相似度，比 RMSD 更语义化）

### 天气/气候预测

**模型直接输出**：物理场（温度、湿度、风速等网格数据）

**验证边界**：下游功能指标 → 场的逐点 diff 不代表预报质量 → 比空间 RMSE / 异常相关系数

**度量选择**：
- 空间 RMSE（区域平均误差）
- 异常相关系数（ACC，预报技巧评分）

### 时序预测

**模型直接输出**：预测序列

**验证边界**：直接输出（确定性预测）或下游功能指标（多步滚动预测有累积误差）

**度量选择**：
- MAE / MSE / MAPE
- 多步滚动预测后半段是否漂移

## 使用原则

- 先按模型类型确定验证边界（直接输出 vs 下游功能指标），再选择度量
- 用户未说明模型类型时，先问清任务类型和最终评价指标
- 混合任务模型不要只看一个指标，同时保留任务指标和中间输出证据
- 当模型直接输出和下游功能指标都可获取时，Level 1 用直接输出（快），Level 2 加下游功能指标（准）
