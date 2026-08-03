# 等价替换——用硬件更友好的等价实现

## 原理

对关键路径上的每一个操作，问：

**同样的数学结果，有没有 NPU 上物理代价更低的等价写法？**

等价替换不减少逻辑工作量（那是"去重"），不复用已有结果（那是"复用"），不并行化延迟（那是"掩盖"）——它做的是：**同样的计算，换一种硬件更亲和的表达方式**。

## 判断方法

以下现象提示可以用"等价替换"手段：
- kernel_details 中某算子 `Accelerator Core = AI_CPU` → 没有 AI Core 实现，寻找等价的 AI Core 友好 API
- kernel_details 中某算子组 mte 主导且调用次数极高 → 可能有融合算子可整体替换
- op_statistic 中某类算子占比高，但查源码后确认该计算是必要的（不是冗余）→ 不能去重，只能替换
- 一组连续算子构成已知 pattern（如 RMSNorm = Cast+Square+Mean+Add+Rsqrt+Mul+Cast）→ 有官方融合算子

## 方案示例

### 层 1: NPU 融合算子替换（最优先）

将一组连续的拆解算子替换为单个官方融合算子——N 个 kernel → 1 个，收益最大、风险最低。

**查询流程**：先查 [npu_operator_catalog.yaml](npu_operator_catalog.yaml)；表里没有再查库 `[x for x in dir(torch_npu) if 'npu_' in x.lower()]`，然后查昇腾官方 API 文档获取 dtype/constraint；查完更新 YAML 目录。

**可替换的典型 pattern**：
- 归一化拆解序列（Cast+Square+Mean+Add+Rsqrt+Mul+Cast）→ 对应 NPU 融合 norm 算子
- 注意力拆解序列（多步 MatMul+Softmax+Dropout）→ NPU 融合 attention 算子
- 激活函数拆解序列（如 gelu = Erf+Mul+Add）→ 框架内置激活函数（可能精度更好）
- 残差+归一化拆解（Add+LayerNorm）→ NPU 融合 add_norm 算子

**注意事项**：故意传错参数触发报错可查看完整签名 schema；Probe 必须加 `torch.npu.synchronize()`；多数融合算子返回多个值，用 `result, *_ =` 解包；融合算子可能有 dtype/shape 约束，需核对。

### 层 2: 换等价 API（同逻辑，不同 NPU kernel 映射）

同一数学运算在 PyTorch 中有多种 API 表达，映射到不同的 NPU kernel。

**方法**：明确当前算子的数学语义 → 搜索 PyTorch 中实现同一语义的其他 API → 对比两者在 kernel_details 中映射到什么 NPU 算子。

**已验证的替换对**：
- `prod(dim)` → `sum(dim) == shape[dim]`（ReduceProd 无 AI Core → ReduceSum 有）
- `tensor[row, col] = val` → `view(-1).scatter_()`（逐元素 host 下发 → 单次 device scatter）
- `F.one_hot(x).to(float)` → `zeros().scatter_(1, x.unsqueeze(-1), 1.0)`（sigmoid kernel → 直接写入）
- `4D matmul` → `reshape 3D + bmm`（避免运行时 Transpose）
- one-hot 向量 × 权重矩阵 → `index_select`/`gather`（矩阵乘法 → 直接索引）
- `einsum` / `einops.rearrange` → `bmm`/`matmul`/`view`+`permute`（字符串解析 + 多步拆解 → 原生操作，减少 Python 开销和 kernel 数）

### 层 3: 换算法（同结果，不同计算路径）

回到算法层面，同一功能用不同的数学方法实现。搜索空间最大、风险最高。

**典型替换**：
- 标准 attention → FlashAttention（分块重计算，减少 HBM 访存）
- sort-based TopK → heap-based / approximate TopK
- per-token sliding conv（unfold）→ pad+stack（backward 路径更 NPU 友好）
- 逐元素组合表达 → 矩阵化批量表达（减少 kernel 数，提升并行度）

## 约束

### 等价性验证

每次替换后必须验证等价性——替换是改实现不改语义，语义不变是铁约束。验证方法见 [equivalence_verification.md](../../04_accuracy_assurance/references/equivalence_verification.md)。

### NPU 浮点非结合律

NPU 浮点运算（尤其 float16）不满足结合律：改变计算顺序的等价变换可能引入精度差异，且差异随层数累积。这影响所有"改变计算顺序"的优化——权重折叠、算子融合等。对深层模型应优先尝试不改变计算顺序的替换（如用框架内置融合算子替代手动多步实现）。融合算子在 NPU 内部可能以更高精度处理中间计算，产出与手动实现等价甚至更精确的结果——不能因"手动实现中有类型转换"就判定该优化不可行，须实测验证。

### 融合算子适用性

NPU 融合算子的适用条件取决于模型架构特征。层 1 查询融合算子时，须核对当前模型的架构特征是否满足约束。已验证的适用性结论应记录到 evidence_db 的 platform_findings 中。常见约束维度：norm 位置（某些融合算子仅适用于 POST-norm）、attention 类型（不同 layout 和是否 GQA 的支持不同）、激活函数变体（须确认模型使用的是哪种变体）。

## 替换失败的记录

替换尝试失败同样有价值——记录到 `<workspace>/evidence_db/` 防止重复踩坑。
