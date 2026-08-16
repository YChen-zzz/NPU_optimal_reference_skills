# Supernode 分析方法

## 什么是 Supernode

Supernode 是跨 compiler/backend 仍保留明确语义和数学合同的最小可优化区域。它可以对应：源码中的一个函数、多个 graph node、一个 GPU fused kernel、或多个 NPU operator。

## 从 GPU Compiled IR 推导划分

### 读取 GPU fusion groups

在 `ir_post_fusion.txt` 中：
- `FusedSchedulerNode(op_a, op_b, op_c)` → 3 个 op 被融合为 1 个 kernel
- `ExternKernelSchedulerNode` → 不可融合的外部 kernel（cuBLAS mm, flash_attn）
- `SchedulerNode(ComputedBuffer)` → 单独的 pointwise/reduction kernel

### 分组为 Supernode

**规则**：相邻 fusion groups 如果服务同一个语义目的，合为一个 Supernode。

**语义边界**（通用）：
- Input/Embedding 处理
- Normalization (RMSNorm, LayerNorm)
- Linear projection (QKV, output, MLP up/down)
- Position encoding (rotary, ALiBi, etc.)
- Attention compute (self-attn, cross-attn)
- Activation function (relu, gelu, swiglu, etc.)
- Residual connection + next norm
- Output + loss computation
- Optimizer state update
- Communication (reduce_scatter, all_gather, all_reduce)

不要把具体模型模块名写进通用划分 — 按功能分。

### 确定优先级

```
priority = (NPU 时间占比) × (GPU 融合比率)
```

- NPU 时间占比: 该 SN 占 step 时间的比例
- GPU 融合比率: GPU 用 K 个 kernel，NPU 用 N 个 kernel → N/K 越大越值得优化

## 三类 Gap 分析

对每个 Supernode，区分三类差异：

### 1. Source-Port Gap（移植差异）

NPU source 相对 GPU/共通 source **新增、遗漏或改变**的内容：
- dtype cast（多了 .float(), .type_as()）
- API 参数遗漏（GPU 传了 window_size，NPU 没传等价参数）
- Layout 差异（额外的 transpose, contiguous）
- Host 行为（.tolist() sync, Python loop vs vectorized）
- 训练/推理路径差异

### 2. Compile Delta（编译器优化差异）

GPU compiler 做了但 NPU 没有的优化：
- **删除**: GPU compile 消除了什么 op（dead code, 冗余 cast）
- **融合**: 哪些 ops 被合为 1 kernel（elementwise chain, norm+scale）
- **复用**: buffer reuse, saved tensor 精简
- **调度**: kernel launch 合并, latency hiding

### 3. Platform Extra（NPU 额外开销）

GPU post-compile 中不存在、但 NPU 当前执行的工作：
- 额外的 cast/transdata/copy
- 不必要的 materialization
- Graph break 导致的重复 dispatch
- 额外的 sync point

## 精度对齐检查

对每个 Supernode 检查：

| 维度 | 检查项 |
|------|--------|
| Input dtype | GPU 和 NPU 是否一致？NPU 是否多了 cast？ |
| Compute dtype | 中间计算精度是否对齐？(f32 accumulator 等) |
| Output dtype | 输出精度是否对齐？ |
| Saved for backward | GPU 保存什么？NPU autograd 保存什么？NPU 是否多保存了？ |
| Rounding boundary | 在哪里做 dtype 转换？边界位置是否和 GPU 一致？ |

## 匹配优先级

将 NPU ops 映射到 GPU Supernode 时，按以下顺序判断（不以名称为准）：

1. Source location + 语义功能
2. 依赖关系（上下游 tensor 流向）
3. Input/output shape + dtype
4. 调用频率（per-step 次数）
5. Operator/module 名（仅作弱证据）

## 何时细拆 Supernode

- SN 内有多个独立可优化子路径（如 Attention 的 prologue/core/epilogue）
- SN 内某部分已是 extern kernel 不可优化（如 flash_attn），周围的 elementwise 仍有空间
- Profile 显示 SN 内部有明显的 gap（kernel launch 间隙 > compute 时间的 10%）
