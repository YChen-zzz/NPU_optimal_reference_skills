# Profiling 采集

## 三种性能测量及其覆盖范围

优化过程中有三种性能测量，它们的采集方式不同但**必须覆盖完全相同的代码范围**——即模型的全部功能代码（包括安全检查器、后处理器等辅助组件，禁止禁用任何功能组件）：

| 测量 | 定义 | 采集方式 | 用途 |
|------|------|---------|------|
| **wall-clock** | 无 profiler 的真实端到端时间 | `torch.npu.synchronize()` → 计时 → `synchronize()` → 计时，取 ≥20 次中位数 | 真实性能基准；下界分析 Tier 3 |
| **L0** | 仅 NPU 活动的 profiling，不注入 CPU barrier | `torch_npu.profiler.profile(activities=[NPU])` + `tensorboard_trace_handler` | 设备执行时间；下界分析 Tier 2；L0/L1 交叉验证 |
| **L1** | CPU + NPU + 调用栈 + 内存 + AI Core 指标的 profiling | `torch_npu.profiler.profile(activities=[CPU,NPU], with_stack=True, ...)` + `tensorboard_trace_handler` | 瓶颈分析主力；交给 `run_analysis.py` |

**覆盖范围一致**：三者的"被测代码段"必须完全相同。例如若 wall-clock 框了 `input.to(device) + model(input)`，则 L0/L1 的 `with profiler.profile(...)` 块内也必须是 `input.to(device) + model(input)`。范围不一致会导致 wall-clock / L0_Computing 比值失真，进而导致下界分析的 gap B 计算错误。

> wall-clock 不使用 profiler，是唯一不受 profiler 开销影响的测量。L0 虽然最轻量但仍引入少量开销（L0 Free 会高估真实 host 开销）。L1 的 barrier 注入严重扭曲 host 时间（高估可达数十倍），但算子级数据（op_statistic、kernel_details 等）仍然有效。

## 采集规则

1. **环境变量**：必须在 `import torch_npu` 前设置 `TASK_QUEUE_ENABLE=2`（异步流水）和 `CPU_AFFINITY_CONF=1`（CPU 绑核）
2. **输出路径**：`<workspace>/profiling/<YYYYMMDD_HHMMSS>/`，禁止 `/tmp` 或无时间戳路径；每次更新 `profiling/latest` 软链接
3. **禁止 `export_chrome_trace`**：只产出 trace.json，不产出 `step_trace_time.csv`，无法做 L0/L1 交叉验证和下界分析。必须用 `tensorboard_trace_handler`
4. **采集与业务分离**：业务推理逻辑封装为函数（如 `run_inference(model, input_data)`），采集脚本只包围这个函数。优化改函数内部，采集脚本不改
5. **warmup**：推理场景采集前手动预热 3 次（触发编译/缓存），不使用 schedule；训练场景用 `schedule(skip_first=20)` 替代

## wall-clock benchmark 模板

```python
import time, torch_npu

# warmup
for _ in range(3):
    run_inference(model, input_data)
torch.npu.synchronize()

# benchmark（计时范围 = run_inference 的代码范围，须与 L0/L1 一致）
times = []
for _ in range(20):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    run_inference(model, input_data)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)  # ms

times.sort()
print(f"median={times[len(times)//2]:.2f}ms  p10={times[len(times)//10]:.2f}ms  p90={times[len(times)*9//10]:.2f}ms")
```

## L0 / L1 采集模板

L0 和 L1 的唯一区别是 profiler 参数——L0 仅 NPU 活动，L1 增加 CPU + 调用栈 + 内存 + AI Core 指标。共用路径构造和 warmup：

```python
import os, datetime, torch, torch_npu

# --- 路径构造（L0/L1 共用）---
PROFILING_BASE = os.path.join(os.getcwd(), "profiling")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
profiling_dir = os.path.join(PROFILING_BASE, timestamp)
os.makedirs(profiling_dir, exist_ok=True)
latest = os.path.join(PROFILING_BASE, "latest")
if os.path.islink(latest): os.remove(latest)
os.symlink(timestamp, latest)

# --- warmup（L0/L1 共用）---
for _ in range(3):
    run_inference(model, input_data)
torch.npu.synchronize()

# --- L0 采集 ---
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    run_inference(model, input_data)  # ← 与 wall-clock 同一函数，覆盖范围一致
    prof.step()

# --- L1 采集（替换上方 with 块）---
# with torch_npu.profiler.profile(
#     activities=[torch_npu.profiler.ProfilerActivity.CPU,
#                 torch_npu.profiler.ProfilerActivity.NPU],
#     with_stack=True, record_shapes=True, profile_memory=True,
#     experimental_config=torch_npu.profiler._ExperimentalConfig(
#         profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
#         aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
#     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
# ) as prof:
#     run_inference(model, input_data)
#     prof.step()
```

**L0 目录路径需记录**：`run_analysis.py --l0-dir <此目录>` 会用此数据与 L1 做交叉验证和下界分析。第 0 轮用 Phase 1 基线 L0；第 i 轮用第 i-1 轮 Phase 4 的 L0。

**L1 采集后**：将 L1 目录传给 `run_analysis.py <l1_dir> --l0-dir <l0_dir>` 即可进入 Phase 2 分析。L1 的 `with_stack=True` 和 `profile_memory=True` 会显著增大输出（可达数 GB），确认磁盘空间充足。

## 训练场景适配

训练与推理的差异仅在两点：用 `schedule(skip_first=20)` 替代手动 warmup，用 `for step, batch in enumerate(dataloader)` 替代单次调用。profiler 参数同上。

```python
with torch_npu.profiler.profile(
    activities=[...],  # L0: [NPU]；L1: [CPU, NPU] + with_stack 等
    schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=20),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

### 训练框架接入

三种框架的差异仅在 profiler 的启停接入点（哪个回调），profiler 参数构造完全一致：

- **PyTorch Lightning**：`on_train_start` 启动 profiler，`on_train_batch_end` 调 `prof.step()`，`on_train_end` 停止
- **HuggingFace Trainer**：`on_train_begin` 启动，`on_step_end` 调 `prof.step()`，`on_train_end` 停止
- **DeepSpeed 多卡**：每张卡独立采集，写入 `rank_<n>/` 子目录（`run_analysis.py --rank N` 定位）。只要涉及通信瓶颈分析就必须全卡采集

## GPU 对比采集

跨平台对比时，GPU 侧用 `torch.profiler`（而非 `torch_npu.profiler`）+ `ProfilerActivity.CUDA`（而非 `.NPU`），不使用 `experimental_config`。务必与 NPU 侧 L0 配对：相同输入数据、相同代码范围。

## 采集前检查

- 业务脚本可在不采集的情况下稳定跑通
- 已设置 `TASK_QUEUE_ENABLE=2` 和 `CPU_AFFINITY_CONF=1`
- 运行了 `scripts/validate_profiling_env.py` 确认环境就绪
- warmup 已完成（推理 ≥3 次，训练 `skip_first=20`）
- 磁盘空间充足（L1 可达数 GB）
- L0 目录路径已记录（供下一轮 Phase 2 交叉验证）
