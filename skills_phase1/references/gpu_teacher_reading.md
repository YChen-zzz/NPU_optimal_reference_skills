# GPU Teacher Evidence 读取指南

## Evidence Pack 结构

典型 GPU Teacher pack 包含：
```
gpu_teacher_data/
├── compile_debug/torch_compile_debug/run_*/torchinductor/
│   ├── model__N_forward_M/
│   │   ├── fx_graph_readable.py      ← compile 前的 FX 图
│   │   ├── ir_post_fusion.txt        ← compile 后的融合结果 (最重要)
│   │   └── output_code.py            ← 生成的 Triton kernel 代码
│   └── model__N_backward_M/          ← backward 子图
├── inductor_kernels/                  ← 所有生成的 kernel 源码
├── rank0/traces/                      ← Chrome trace timeline
└── rank0/summaries/                   ← kernel 耗时统计
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

### 2. fx_graph_readable.py

Compile 前的 FX 图。用于：
- 理解每个 op 的原始 dtype/shape
- 对比 NPU source 发现 source-port gap
- 确认哪些 ops 在 compile 前就存在（vs compile 引入的）

### 3. summaries/kernel_breakdown.txt

各类 kernel 的时间占比:
```
GEMM/MatMul:     XX%
FlashAttention:  XX%
Triton(fused):   XX%
NCCL:            XX%
Reduction/Norm:  XX%
```

用于确定优化优先级（时间占比大的 SN 优先）。

## 从 IR 推导 Supernode

**示例**: 如果 ir_post_fusion.txt 显示:
```
op8_op9:    FusedSchedulerNode  (weight cast + scalar scale)
op10:       ExternKernel        (mm - QKV projection)
op11-op16:  FusedSchedulerNode  (Q-norm + K-norm + Q-rotary + K-rotary)
```

则 "QKV Projection + Q/K Norm + Rotary" 是一个 Attention Prologue Supernode。GPU 用 3 个 kernel launch 完成。NPU 侧对应代码可能用 7+ 个 kernel launch — 差距就是优化空间。

## Source-Port Gap 检查清单

对比 GPU source 和 NPU source，逐项检查:

- [ ] dtype 差异: GPU 用 bf16 的地方 NPU 是否多了 .float()？
- [ ] API 参数逐项映射: 打印 GPU API 和 NPU API 的**完整签名**，逐参数对照。名称可能完全不同（如 GPU `window_size` 对应 NPU `pre_tockens`）。未传的参数通常走最慢默认路径。
- [ ] Layout: GPU 直接 matmul 的地方 NPU 是否多了 .T + F.linear？
- [ ] Saved tensors: GPU backward 保存了什么 vs NPU autograd 保存什么？
- [ ] Host sync: NPU 是否有 .tolist()/.item() 等 device→host sync？
- [ ] 冗余 ops: GPU compile 消除了什么 NPU 还保留着？(如 .contiguous() no-op)
