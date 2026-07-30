# 等价替换——用硬件更友好的等价实现

## 原理

对关键路径上的每一个操作,问:

**同样的数学结果,有没有 NPU 上物理代价更低的等价写法?**

等价替换不减少逻辑工作量(那是"去重"),不复用已有结果(那是"复用"),不并行化延迟(那是"掩盖")——它做的是:**同样的计算,换一种硬件更亲和的表达方式**。

区分:
- "去重"发现 Transpose 是多余的(上游改布局可消除) → 删掉 Transpose
- "替换"发现 ReduceProd 没有 AI Core 实现 → 换成等价的 ReduceSum + Equal(逻辑等价,但后者有高效 kernel)

## 判断方法

以下现象提示可以用"等价替换"手段:
- kernel_details 中某算子 `Accelerator Core = AI_CPU` → 没有 AI Core 实现,寻找等价的 AI Core 友好 API
- kernel_details 中某算子组 mte 主导且调用次数极高 → 可能有融合算子可整体替换
- op_statistic 中某类算子占比高,但查源码后确认该计算是必要的(不是冗余) → 不能去重,只能替换
- 一组连续算子构成已知 pattern(如 RMSNorm = Cast+Square+Mean+Add+Rsqrt+Mul+Cast) → 有官方融合算子

## 三层搜索空间

### 层 1: NPU 融合算子替换(最优先)

检查当前的算子组合是否有官方融合算子覆盖。

**触发**: profiling 中看到一组已知 pattern 的拆解算子(如 RMSNorm 7 步、SwiGLU 2 步、Attention 多步)。

**查询流程**:
1. 先查 [npu_operator_catalog.yaml](npu_operator_catalog.yaml)，搜索模型中的计算模式是否已有对应的融合算子
2. 表里没有再查库：`[x for x in dir(torch_npu) if 'npu_' in x.lower()]`，然后查昇腾官方 API 文档（`site:hiascend.com torch_npu <算子名>`）获取 dtype/constraint
3. 查完更新 YAML 目录，避免重复查询

**注意事项**:
- 故意传错参数触发报错可查看完整签名 schema
- Probe 必须加 `torch.npu.synchronize()`，NPU 异步执行否则无法捕获错误
- 多数融合算子返回多个值，用 `result, *_ =` 解包更安全
- 融合算子可能有 dtype/shape 约束（如仅支持 fp16/bf16），需核对

**特点**: 收益最大(N 个 kernel → 1 个)、风险最低(官方实现)。但注意: 融合算子可能有 dtype/shape 约束(如仅支持 fp16/bf16),需核对。

### 层 2: 换等价 API(同逻辑,不同 NPU kernel 映射)

同一数学运算在 PyTorch 中有多种 API 表达,映射到不同的 NPU kernel。

**触发**: 某算子 AI_CPU fallback 或性能极差,且该计算是必要的。

**方法**:
1. 明确当前算子的数学语义(做了什么计算)
2. 搜索 PyTorch 中实现同一语义的其他 API
3. 对比两者在 kernel_details 中映射到什么 NPU 算子(跑一次 profiling 对比)

**已验证的替换对**:
- `prod(dim)` → `sum(dim) == shape[dim]` (ReduceProd 无 AI Core → ReduceSum 有)
- `tensor[row, col] = val` → `view(-1).scatter_()` (逐元素 host 下发 → 单次 device scatter)
- `F.one_hot(x).to(float)` → `zeros().scatter_(1, x.unsqueeze(-1), 1.0)` (sigmoid kernel → 直接写入)
- `4D matmul` → `reshape 3D + bmm` (避免运行时 Transpose)

**关键**: 替换后必须做等价性验证(见 `04_accuracy_assurance/references/equivalence_verification.md`)。

### 层 3: 换算法(同结果,不同计算路径)

回到算法层面,同一功能用不同的数学方法实现。

**触发**: 层 1/2 都找不到更好的替换,但该计算仍是瓶颈。

**方法**: 问"这个功能在数学/算法上还有什么别的实现方式",再对比各实现的 NPU kernel 链。

**例**:
- 标准 attention → FlashAttention(分块重计算,减少 HBM 访存)
- sort-based TopK → heap-based / approximate TopK
- per-token sliding conv(unfold) → pad+stack(backward 路径更 NPU 友好)

**注意**: 层 3 的搜索空间最大、风险最高——等价性验证尤其重要,且需要对算法有深入理解。

## 等价性约束

**每次替换后必须验证等价性**,因为替换是改实现不改语义——语义不变是铁约束。

验证方法见 `04_accuracy_assurance/references/equivalence_verification.md`。

## 替换失败的记录

替换尝试失败同样有价值——记录到 `<workspace>/evidence_db/` 防止重复踩坑。
