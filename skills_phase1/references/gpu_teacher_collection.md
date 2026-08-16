# GPU Teacher Evidence 采集指南

当没有现成的 GPU evidence pack 时，agent 需要在 GPU 机器上构造 Teacher 数据。

## 需要采集什么

| 产物 | 用途 | 采集方法 |
|------|------|---------|
| **Compile debug (FX graph + IR)** | 看 compile 前后做了什么优化 | `TORCH_COMPILE_DEBUG=1` |
| **Inductor kernels** | 看具体哪些 op 被融合 | `TORCH_COMPILE_DEBUG=1` 自动产出 |
| **Runtime profiling** | 各 kernel 时间、通信 overlap | `torch.profiler` |
| **Eager baseline** | 对比 compile 前后差异 | 同脚本不加 compile 跑一次 |

## 采集步骤

### 1. Compile Debug (最重要)

```bash
export TORCH_COMPILE_DEBUG=1
python train_script.py  # 用 torch.compile 的版本
```

产出目录: `torch_compile_debug/run_<timestamp>-pid_<N>/torchinductor/`

内含:
- `model__N_forward_M/fx_graph_readable.py` — compile 前 FX 图
- `model__N_forward_M/ir_post_fusion.txt` — **最重要**: compile 后融合结果
- `model__N_forward_M/output_code.py` — 生成的 kernel 代码
- `model__N_backward_M/` — backward 子图

**注意**: 多 rank 分布式训练会产生每个 rank 一个目录。通常只需 rank 0 的即可（同模型结构）。

### 2. Runtime Profiling

```python
import torch.profiler

# 按 regime 分阶段采集
profiling_stages = [
    {"name": "stage0", "steps": (5, 25), "bs": small_bs},
    {"name": "stage1", "steps": (mid, mid+20), "bs": medium_bs},
    {"name": "stage2", "steps": (late, late+20), "bs": large_bs},
]

for stage in profiling_stages:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1, warmup=2, active=stage["steps"][1]-stage["steps"][0]),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"profiling/{stage['name']}"),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for step in range(stage["steps"][0], stage["steps"][1]):
            train_step(...)
            prof.step()
```

### 3. 导出 Kernel Summary

```python
# 在 profiling 完成后
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

# 按类别分组
# GEMM, FlashAttention, NCCL, Triton, Elementwise, Reduction, Memory
```

### 4. 组织 Evidence Pack

推荐目录结构:
```
gpu_teacher_data/
├── SUMMARY.txt                 ← 环境/配置概览
├── compile_debug/              ← TORCH_COMPILE_DEBUG 产出
│   └── torch_compile_debug/
│       └── run_*-pid_*/torchinductor/
├── inductor_kernels/           ← 生成的 kernel 源码 (从 compile_debug 复制)
├── rank0/
│   ├── traces/                 ← Chrome trace JSON per stage
│   └── summaries/              ← Kernel 统计表 per stage
└── eager_baseline/             ← 不加 compile 的 profiling (可选)
```

## 采集覆盖要求

- **覆盖所有 regime**: 如果训练有不同 batch size / window size / precision 阶段，每阶段都要采
- **compile 后 warm 状态**: 必须越过首次编译（skip_first 几步），采 steady state
- **多 rank**: 至少保存 rank 0 的完整数据；如需分析通信不平衡，保存 2+ rank
- **eager 对比**: 可选但有价值 — 用于量化 compile 带来的 kernel 数量减少

## 最小可用 pack

如果资源有限，**最低限度**只需:
1. 一个 rank 的 `ir_post_fusion.txt` (forward + backward 各一个)
2. 一个 stage 的 kernel summary (top 50 by cuda time)

这足以确定 Supernode 划分和 fusion 机会。完整 pack 在需要精确对齐时才必要。

## 不采集时怎么办

如果完全无法访问 GPU 机器：
- 阅读 GPU 版本源码理解 `torch.compile` 装饰器位置
- 从 PyTorch/Triton 文档推断 Inductor 对常见 pattern 的 fusion 行为
- 使用 NPU 侧 profiling 定位瓶颈，不依赖 GPU 参照
- 标记 "GPU Teacher unavailable"，纯 Line A + Line B 分析
