# GPU Teacher Evidence 采集指南

当没有现成的 GPU evidence pack 时，agent 需要在 GPU 机器上构造 Teacher 数据。

## 需要采集什么

| 产物 | 用途 | 采集方法 | 必要性 |
|------|------|---------|--------|
| **Compile debug (FX graph + IR)** | 看 compile 前后做了什么优化 | `TORCH_COMPILE_DEBUG=1` | **必需** |
| **Inductor kernels** | 看每个 fused kernel 融合了哪些原始 op | 从 `/tmp/torchinductor_*` 复制 | **必需** |
| **Runtime profiling** | 各 kernel 时间、通信 overlap | `torch.profiler` Chrome trace + summary | **必需** |
| **Kernel breakdown** | 按类别统计时间占比 (GEMM/Triton/NCCL等) | 从 profiler 事件分类聚合 | **必需** |
| **Eager baseline** | 对比 compile 前后 kernel 数量差异 | 不加 compile 跑相同模型 profile | **必需** |
| **AOT Autograd graphs** | 看 compile 前原始的 forward/backward op 序列 | `torch.compile` + 自定义 aot_autograd backend | **必需** |

> **实战经验**: 之前将 eager baseline 和 kernel breakdown 标为"可选"，导致后续 Supernode 分析时缺少关键参照。
> eager vs compiled 的 kernel 数量对比是量化融合空间的最直接证据，kernel breakdown 是确定优化优先级的基础。
> AOT graphs 提供了不经任何融合的原始 op 链和每个中间 tensor 的 shape/dtype，对精度对齐至关重要。

## 采集步骤

### 1. Compile Debug (最重要)

```bash
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$OUTPUT_DIR/compile_debug"
python train_script.py  # 用 torch.compile 的版本
```

产出目录: `compile_debug/torch_compile_debug/run_<timestamp>-pid_<N>/torchinductor/`

内含:
- `model__N_forward_M/fx_graph_readable.py` — compile 前 FX 图
- `model__N_forward_M/ir_post_fusion.txt` — **最重要**: compile 后融合结果
- `model__N_forward_M/output_code.py` — 生成的 kernel 代码
- `model__N_backward_M/` — backward 子图（同样含 ir_post_fusion.txt）

同时启用 inductor debug 获取更详细的 trace:
```python
import torch._inductor.config as inductor_config
inductor_config.debug = True
inductor_config.trace.enabled = True
```

**注意**: 多 rank 分布式训练会产生每个 rank 一个目录。通常只需 rank 0 的即可（同模型结构）。

### 2. Eager Baseline Profiling

**在编译模型之前**，先创建一个不加 compile 的模型进行 profiling:

```python
from torch.profiler import profile as torch_profile, ProfilerActivity

# 创建 eager 模型（不编译）
model_eager = create_model()
model_eager.load_state_dict(model.state_dict())

# Warmup
for _ in range(5):
    loss = model_eager(inputs)
    loss.backward()
    model_eager.zero_grad()

# Profile
eager_steps = 10
with torch_profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
    profile_memory=True,
) as eager_prof:
    for _ in range(eager_steps):
        loss = model_eager(inputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 保存
eager_prof.export_chrome_trace("eager_vs_compiled/eager_trace.json")

# 统计 kernel 数
cuda_events = [e for e in eager_prof.key_averages() if e.device_time_total > 0]
total_calls = sum(e.count for e in cuda_events)
print(f"Eager: {total_calls / eager_steps:.0f} kernels/step")

# 保存详细 profile
with open("eager_vs_compiled/eager_profile.txt", "w") as f:
    f.write(f"=== EAGER MODE PROFILING ({eager_steps} steps) ===\n")
    f.write(eager_prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
    f.write(f"\n\nTotal CUDA kernel calls: {total_calls}\n")
    f.write(f"Average kernels per step: {total_calls / eager_steps:.0f}\n")
    # Self CUDA time distribution
    total_cuda_time = sum(e.self_device_time_total for e in cuda_events)
    for e in sorted(cuda_events, key=lambda x: -x.self_device_time_total)[:40]:
        pct = e.self_device_time_total / total_cuda_time * 100
        f.write(f"  {e.key:60s} {e.self_device_time_total/1000:10.2f}ms ({pct:5.1f}%) x{e.count}\n")
```

### 3. AOT Autograd Graph Export

导出 compile 前原始的 forward + backward 图（含 shape/dtype 信息）:

```python
from torch._dynamo.backends.common import aot_autograd

def _make_compiler(label, graphs_dir):
    def compiler_fn(gm, example_inputs):
        # Save readable graph
        with open(f"{graphs_dir}/{label}_readable.py", "w") as f:
            f.write(gm.print_readable(print_output=False))
        # Save op statistics
        op_counts = {}
        for node in gm.graph.nodes:
            if node.op == 'call_function':
                op_name = str(node.target).split('.')[-1]
                op_counts[op_name] = op_counts.get(op_name, 0) + 1
        with open(f"{graphs_dir}/{label}_ops.txt", "w") as f:
            for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {op:40s}: {cnt}\n")
        # Save shapes/dtypes
        with open(f"{graphs_dir}/{label}_shapes.txt", "w") as f:
            for node in gm.graph.nodes:
                meta = node.meta.get('val', None)
                if isinstance(meta, torch.Tensor):
                    f.write(f"  {node.name:40s}: shape={list(meta.shape)}, dtype={meta.dtype}\n")
        return gm.forward
    return compiler_fn

aot_backend = aot_autograd(
    fw_compiler=_make_compiler("forward", graphs_dir),
    bw_compiler=_make_compiler("backward", graphs_dir),
)

@torch.compile(backend=aot_backend, dynamic=False)
def traced_step(model, inputs, labels):
    return loss_fn(model(inputs), labels).sum()

loss = traced_step(model, inputs, labels)
loss.backward()
```

**兼容性注意**: 如果 `aot_autograd` backend 对某些模型失败，依次降级尝试:
1. `torch.compile(backend=aot_backend)` — 优先，可同时获取 fwd+bwd
2. `torch.fx.experimental.proxy_tensor.make_fx()` — 只能获取 forward
3. `torch.export.export()` — 最后手段

### 4. Compiled Model Runtime Profiling

**必须在 compile warmup 之后采集** (跳过首次编译的 overhead):

```python
# Warmup (触发编译)
model_compiled = torch.compile(model, mode='max-autotune')
for _ in range(10):
    loss = model_compiled(inputs)
    loss.backward()
    optimizer.step()

# Profile steady state
with torch_profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
) as prof:
    for _ in range(PROFILING_STEPS):
        loss = model_compiled(inputs)
        loss.backward()
        optimizer.step()

# 保存 Chrome trace
prof.export_chrome_trace("rank0/traces/stage0_trace.json")
```

**按 regime 分阶段采集** (如果训练有不同 batch size / window size):
```python
profiling_stages = [
    {"name": "stage0", "steps": (5, 25), "bs": small_bs},
    {"name": "stage1", "steps": (mid, mid+20), "bs": medium_bs},
    {"name": "stage2", "steps": (late, late+20), "bs": large_bs},
]
```

### 5. Kernel Summary + Breakdown

```python
# Summary (按 CUDA time 排序)
summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=60)

# Kernel breakdown (按类别分组) — 必须采集
cuda_events = [e for e in prof.key_averages() if e.device_time_total > 0]
total_cuda_time = sum(e.self_device_time_total for e in cuda_events)

kernel_groups = defaultdict(lambda: {"count": 0, "cuda_time_us": 0})
for e in cuda_events:
    key = e.key.lower()
    if "nccl" in key:
        group = "NCCL_Communication"
    elif "gemm" in key or "matmul" in key or "xmma" in key or "mm" in key:
        group = "GEMM/MatMul"
    elif "flash" in key or "attention" in key or "fmha" in key:
        group = "FlashAttention"
    elif "conv" in key or "winograd" in key:
        group = "Convolution"
    elif "triton" in key:
        group = "Triton_Kernels"
    elif "elementwise" in key or "vectorized" in key:
        group = "Elementwise"
    elif "reduce" in key or "norm" in key:
        group = "Reduction/Norm"
    elif "copy" in key or "memcpy" in key or "memset" in key:
        group = "Memory_Ops"
    else:
        group = "Other"
    kernel_groups[group]["count"] += e.count
    kernel_groups[group]["cuda_time_us"] += e.self_device_time_total
```

### 6. Inductor Kernels 独立提取

从 inductor cache 中复制生成的 Triton kernel 源码:

```python
import getpass, glob, shutil

username = getpass.getuser()
kernels_dir = os.path.join(output_dir, "inductor_kernels")
os.makedirs(kernels_dir, exist_ok=True)

patterns = [
    f"/tmp/torchinductor_{username}/**/*.py",
    f"/tmp/torchinductor_{username}/**/*.txt",
]
kernel_count = 0
for pattern in patterns:
    for fpath in glob.glob(pattern, recursive=True):
        if kernel_count >= 100:
            break
        dst = os.path.join(kernels_dir, f"{kernel_count:03d}_{os.path.basename(fpath)}")
        shutil.copy2(fpath, dst)
        kernel_count += 1
```

每个 kernel 文件头部的 `# Source Nodes` 注释标明了被融合的原始 op 清单，这是确定 NPU 融合目标的直接证据。

### 7. 组织 Evidence Pack

**必需的目录结构**:
```
gpu_teacher_data/
├── SUMMARY.txt                 ← 环境/配置/目录概览
├── DATA_MANIFEST.md            ← 详细数据清单 (给人看的)
├── profiling_run.log           ← 完整运行日志
│
├── compile_debug/              ← TORCH_COMPILE_DEBUG 产出
│   └── torch_compile_debug/
│       └── run_*-pid_*/torchinductor/
│           ├── model__N_forward_M/    ← forward 子图
│           └── model__N_backward_M/   ← backward 子图
│
├── aot_graphs/                 ← AOT Autograd forward+backward 原始图
│   ├── *_forward_readable.py   ← 不经融合的原始 op 序列
│   ├── *_forward_ops.txt       ← op 频次统计
│   ├── *_forward_shapes.txt    ← 每个中间 tensor 的 shape/dtype
│   ├── *_backward_readable.py
│   ├── *_backward_ops.txt
│   └── *_backward_shapes.txt
│
├── inductor_kernels/           ← 生成的 Triton kernel 源码 (头部含融合清单)
│   └── *.py
│
└── rank0/ (... rankN/)         ← 每个 rank 独立目录
    ├── traces/                 ← Chrome trace JSON per stage
    │   └── stage*_trace.json
    ├── summaries/              ← Kernel 统计表 per stage
    │   ├── stage*_summary.txt          ← 按耗时排序 + kernel 数 + 内存
    │   └── stage*_kernel_breakdown.txt ← 按类别分组统计 (GEMM/Triton/NCCL...)
    └── eager_vs_compiled/      ← Eager 基线对比
        ├── eager_trace.json    ← Eager Chrome trace
        └── eager_profile.txt   ← Eager kernel 统计 (kernel 数/step)
```

## 采集覆盖要求

- **覆盖所有 regime**: 如果训练有不同 batch size / window size / precision 阶段，每阶段都要采
- **compile 后 warm 状态**: 必须越过首次编译（skip_first 几步），采 steady state
- **多 rank**: 至少保存 rank 0 的完整数据；如需分析通信不平衡，保存 2+ rank
- **eager 对比**: **必需** — kernel 数量差 = 融合空间的直接证据
- **AOT graphs**: **必需** — 提供未融合的 op 链和精度路径 (shape/dtype)
- **kernel breakdown**: **必需** — 确定各类 kernel 时间占比，驱动优化优先级

## 最小可用 pack

如果资源有限，**最低限度**需要:
1. 一个 rank 的 `ir_post_fusion.txt` (forward + backward 各一个)
2. 一个 stage 的 kernel summary + kernel breakdown (top 50 by cuda time)
3. Eager baseline 的 kernel 数量统计 (至少一个数字: N kernels/step)

这足以确定 Supernode 划分和 fusion 机会。

**完整 pack** 在以下场景必要:
- 需要精确对齐 dtype/shape → 用 aot_graphs 的 shapes.txt
- 需要确定 NPU 融合目标 → 用 inductor_kernels 的 Source Nodes
- 需要看 compute/comm overlap → 用 Chrome trace
- 需要看 batch size 对 fusion 的影响 → 对比多 stage 的 breakdown

## 不采集时怎么办

如果完全无法访问 GPU 机器：
- 阅读 GPU 版本源码理解 `torch.compile` 装饰器位置
- 从 PyTorch/Triton 文档推断 Inductor 对常见 pattern 的 fusion 行为
- 使用 NPU 侧 profiling 定位瓶颈，不依赖 GPU 参照
- 标记 "GPU Teacher unavailable"，纯 Line A + Line B 分析

## 采集脚本模板

推荐使用 `run_collect_teacher.sh` + `collect_gpu_teacher.py` 的分离结构：

**`run_collect_teacher.sh`** (启动脚本):
```bash
#!/bin/bash
set -e
export ENABLE_COMPILE_DEBUG=${ENABLE_COMPILE_DEBUG:-1}
export SKIP_EAGER_PROFILING=${SKIP_EAGER_PROFILING:-0}
export SKIP_BACKWARD_GRAPH=${SKIP_BACKWARD_GRAPH:-0}
export PROFILING_STEPS=${PROFILING_STEPS:-20}
export PROFILING_OUTPUT_DIR="./gpu_teacher_data"

python collect_gpu_teacher.py 2>&1 | tee "$PROFILING_OUTPUT_DIR/profiling_run.log"
```

**`collect_gpu_teacher.py`** 按顺序执行:
1. Eager baseline profiling (不编译的模型)
2. AOT Autograd graph export (torch.compile + custom backend)
3. Compiled model warmup + profiling (per stage)
4. Inductor kernels 提取 (从 /tmp/torchinductor_*)
5. SUMMARY.txt 生成

参考实现: `cifar10-airbench/collect_gpu_teacher.py` 和 `nanogpt/train_gpt_profiling.py`。
