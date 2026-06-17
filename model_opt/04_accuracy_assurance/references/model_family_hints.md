# 模型类型与精度对比策略

## 目的

不同模型类型的“应该比什么”和“重点看什么”完全不同。进入对齐前先判断模型类型，避免用错指标或遗漏关键检查点。

## NLP / 序列模型（Transformer Encoder/Decoder）

优先比较：
- 训练 loss 曲线和收敛行为
- 推理：logits cosine similarity、token 匹配率
- 下游任务指标（如 BLEU、Rouge、F1）

重点关注：
- tokenizer 一致性（不同版本可能结果不同）
- attention mask 和 padding 策略
- KV cache 在 decode 过程中的行为
- position encoding 的边界处理

## CV / 视觉模型（CNN、ViT、检测/分割）

优先比较：
- 特征图的逐像素 diff
- 任务指标（mAP、mIoU、Top-1 Acc）
- 后处理后的结果（bbox、mask）

重点关注：
- 图像预处理（归一化、resize、crop 顺序）
- NPU 上的 internal format 对卷积的影响
- batch norm 在训练/推理模式下的行为差异

## 生成式模型（LLM、扩散模型、VAE）

优先比较：
- 确定性设置下的 token/像素级输出
- logits 或概率分布
- 结果的语义合理性（不只看数值）

重点关注：
- sampling 参数（temperature、top_k、top_p）必须完全一致
- 随机数种子固定
- KV cache 和 decode step 的行为
- 扩散模型的采样循环中随机数相关开销

## 图神经网络（GNN）

优先比较：
- 图级或节点级指标
- logits 或中间 embedding

重点关注：
- 图构建顺序和邻接矩阵排序
- scatter / gather 相关算子在 NPU 上的行为
- batch 内 padding 或 mask 处理

## 科学计算 / AI4S

### PDE / CFD 类
优先比较：loss、验证集误差、物理量场的逐样本误差、长时滚动是否发散。
重点关注：网格预处理、边界条件编码、时间步累积误差。

### 分子/材料性质预测
优先比较：MAE、RMSE、R2、逐样本预测值偏差。
重点关注：原子/键特征构造、标准化与反标准化。

### 时序预测
优先比较：MAE、MSE、MAPE、多步滚动预测后半段是否漂移。
重点关注：滑窗切分、时间特征编码、teacher forcing 逻辑。

## 使用原则

- 先按模型类型选比较对象，再运行脚本
- 用户未说明模型类型时，先问清任务类型和最终指标
- 混合任务模型不要只看一个指标，同时保留任务指标和中间输出证据
