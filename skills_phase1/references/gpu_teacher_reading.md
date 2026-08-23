# GPU Teacher Evidence 读取指南

## Evidence Pack 结构

典型 GPU Teacher pack 包含：
```
gpu_teacher_data/
├── SUMMARY.txt                            ← 运行环境/配置概览
├── DATA_MANIFEST.md                       ← 详细数据清单 (给人看的)
├── compile_debug/torch_compile_debug/run_*/torchinductor/
│   ├── model__N_forward_M/
│   │   ├── fx_graph_readable.py           ← compile 前的 FX 图
│   │   ├── ir_post_fusion.txt             ← ★ compile 后的融合结果 (最重要)
│   │   └── output_code.py                 ← 生成的 Triton kernel 代码
│   └── model__N_backward_M/               ← backward 子图 (同结构)
├── aot_graphs/                            ← AOT Autograd 原始 op 图
│   ├── *_forward_readable.py              ← 不经融合的 forward op 序列
│   ├── *_forward_shapes.txt               ← ★ 每个中间 tensor 的 shape/dtype
│   ├── *_backward_readable.py             ← 不经融合的 backward op 序列
│   └── *_backward_shapes.txt              ← backward tensor 的 shape/dtype
├── inductor_kernels/                      ← 所有生成的 kernel 源码
│   └── *.py                               ← 头部 Source Nodes 标注融合清单
├── rank0/
│   ├── traces/                            ← Chrome trace timeline
│   ├── summaries/
│   │   ├── stage*_summary.txt             ← kernel 耗时统计
│   │   └── stage*_kernel_breakdown.txt    ← ★ 按类别分组统计
│   └── eager_vs_compiled/
│       ├── eager_trace.json               ← Eager 模式 timeline
│       └── eager_profile.txt              ← ★ Eager kernel 数 (对比基线)
└── rank1/ ... rankN/                      ← 多 rank 结构同上
```

## 读取顺序（按优先级）

### 1. ir_post_fusion.txt (最高优先级)

这是 compile 后的最终融合结果。每个 node 代表一个 kernel launch:

- `FusedSchedulerNode(op_a, op_b, ...)` → 多个 op 融合为 **1 个 kernel**
- `ExternKernelSchedulerNode` → 外部库 kernel (cuBLAS mm, flash_attn 等)
- `SchedulerNode(ComputedBuffer)` → 单独的 compute kernel

**关键信息**:
- 哪些 ops 被融合在一起 → NPU 上这些 ops 之间有优化空间
- 输入/输出 buffer 的 shape 和 dtype → 用于对齐精度
- extern kernel 的参数 → 用于对齐 API 调用

### 2. eager_vs_compiled/ — 融合空间量化

**第一个要看的数字**: Eager kernels/step vs Compiled kernels/step

```
Eager:    538 kernels/step
Compiled: 282 kernels/step
→ 融合减少了 47.6% 的 kernel launch
```

这个差值 = NPU 需要追赶的融合空间。如果 NPU eager 模式比 GPU eager 还多 kernel，
说明有移植遗留的冗余 op（多余的 cast / contiguous / copy）。

### 3. kernel_breakdown.txt — 优化优先级

各类 kernel 的时间占比:
```
GEMM/MatMul:     XX%   ← 通常占比最大，优化空间在 API 参数
Triton(fused):   XX%   ← compile 融合的 elementwise chain
FlashAttention:  XX%   ← 如有，通常是单个 extern kernel
NCCL:            XX%   ← 通信，优化方向是 overlap
Convolution:     XX%   ← 卷积，关注 algorithm selection
Reduction/Norm:  XX%   ← 可能的融合候选
```

用于确定 Supernode 优化优先级（时间占比大的 SN 优先）。

### 4. aot_graphs/ — 原始 op 链和精度路径

**`*_readable.py`**: 不经任何融合优化的完整 op 序列
- 每个 `call_function` 节点 = 一个原始算子调用
- 对比 NPU source 代码可以发现多余/缺少的 op

**`*_shapes.txt`**: 每个中间 tensor 的 shape 和 dtype
- 用于确认 GPU 训练的真实精度路径
- ⚡ **关键用途**: 确认哪些地方 GPU 实际用 fp16/bf16，避免 NPU 移植时错误插入 .float()

**`*_ops.txt`**: op 频次统计
- 快速了解模型使用了哪些算子类型
- 对比 NPU 侧是否有等价 API

### 5. inductor_kernels/ — 融合清单

每个 `.py` 文件头部的注释:
```python
# Source Nodes: [relu, square, mul_1]  ← 被融合的原始 op
# fused from: aten.relu, aten.pow, aten.mul
```

这是确定 NPU 需要用什么方式（API / compile / 手写 kernel）来对齐 GPU 融合效果的直接证据。

### 6. fx_graph_readable.py

Compile 前的 FX 图。用于：
- 理解每个 op 的原始 dtype/shape
- 对比 NPU source 发现 source-port gap
- 确认哪些 ops 在 compile 前就存在（vs compile 引入的）

## 从 IR 推导 Supernode

**示例**: 如果 ir_post_fusion.txt 显示:
```
op8_op9:    FusedSchedulerNode  (weight cast + scalar scale)
op10:       ExternKernel        (mm - QKV projection)
op11-op16:  FusedSchedulerNode  (Q-norm + K-norm + Q-rotary + K-rotary)
```

则 "QKV Projection + Q/K Norm + Rotary" 是一个 Attention Prologue Supernode。GPU 用 3 个 kernel launch 完成。NPU 侧对应代码可能用 7+ 个 kernel launch — 差距就是优化空间。

**CNN 示例** (cifar10-airbench):
```
op_N:       ExternKernel        (convolution - implicit GEMM)
op_N+1..4:  FusedSchedulerNode  (batch_norm + gelu + maxpool)
```
GPU 将 BN+GELU 融合为 1 个 Triton kernel。NPU 上如果是 3 个独立 kernel launch，即为优化目标。

## Source-Port Gap 检查清单

对比 GPU source 和 NPU source，逐项检查:

- [ ] dtype 差异: GPU 用 bf16/fp16 的地方 NPU 是否多了 .float()？ → 用 `aot_graphs/*_shapes.txt` 确认 GPU 真实 dtype
- [ ] API 参数逐项映射: 打印 GPU API 和 NPU API 的**完整签名**，逐参数对照。名称可能完全不同（如 GPU `window_size` 对应 NPU `pre_tockens`）。未传的参数通常走最慢默认路径。
- [ ] Layout: GPU 直接 matmul 的地方 NPU 是否多了 .T + F.linear？
- [ ] Saved tensors: GPU backward 保存了什么 vs NPU autograd 保存什么？ → 对比 `aot_graphs/backward_readable.py`
- [ ] Host sync: NPU 是否有 .tolist()/.item() 等 device→host sync？
- [ ] 冗余 ops: GPU compile 消除了什么 NPU 还保留着？(如 .contiguous() no-op) → 对比 `ir_pre_fusion.txt` vs `ir_post_fusion.txt`
- [ ] Kernel 数量: NPU eager 的 kernel/step 是否 > GPU eager？如果是，说明有移植遗留冗余

## 关键对比维度

| 对比项 | 文件 A | 文件 B | 看什么 |
|--------|--------|--------|--------|
| Eager vs Compiled | `eager_profile.txt` | `stage0_summary.txt` | kernel 数量差 = 融合空间 |
| 小 BS vs 大 BS | `stage0_kernel_breakdown.txt` | `stage2_kernel_breakdown.txt` | 各类 kernel 占比变化 |
| Forward vs Backward | `*_forward_*/output_code.py` | `*_backward_*/output_code.py` | Backward 是否有额外优化空间 |
| Rank 0 vs Rank N | `rank0/traces/stage0_trace.json` | `rankN/traces/stage0_trace.json` | 通信延迟差异 |
| 融合前 vs 融合后 | `ir_pre_fusion.txt` | `ir_post_fusion.txt` | 具体哪些 op 被合并 |
| AOT vs Compiled | `aot_graphs/*_ops.txt` | `ir_post_fusion.txt` | compile 消除/融合了什么 |
