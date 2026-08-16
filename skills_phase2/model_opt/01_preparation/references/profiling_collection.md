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

### 推理场景

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

### 训练场景

训练场景的 wall-clock **使用短跑脚本**（见下方「训练短跑策略」），取短跑全程的 `step_avg` 作为性能基准。不需要对每个 step 单独计时——短跑输出的 `step_avg`（= 短跑总时间 / 短跑总步数）即为 wall-clock 指标。

```python
# 训练 wall-clock = 短跑脚本的 step_avg 输出
# 短跑脚本自身输出格式示例:
# train_time=62.3s  total_steps=200  step_avg=311.5ms
```

**与 L0/L1 口径对齐**：短跑脚本的训练循环代码范围 = L0/L1 profiler 包围的代码范围。三种测量使用同一个短跑脚本，仅开关 profiler 不同。

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

### 训练短跑策略（强制）

训练场景的 profiling **禁止在完整训练流程上采集**——完整训练可能长达数小时，profiling 只需要每个 regime 的代表性 step。必须构造一个**等比压缩的短跑脚本**，在 ~60-120 秒内覆盖所有训练 regime。

**短跑脚本构造规则**：

1. **等比压缩 total steps**：将完整训练步数压缩到 1/10~1/20（如 2090 步 → 200 步），确保总运行时间在 60-120 秒
2. **按相同比例映射所有 step-dependent schedule**：lr scheduler、batch size 切换、window size 变化、warmup 阶段等所有基于 step 的 schedule，按相同压缩比例映射切换点
3. **覆盖所有 regime**：压缩后每个 regime 至少保留 10+ 步（确保 profiler 有足够的 active step 采样）
4. **保持相同 seed、数据、初始状态**：短跑与完整训练的唯一区别是步数，其他条件完全一致
5. **profiler schedule 对齐 regime**：在每个 regime 内分别采集，确保每个 regime 都有独立的 profiling 数据

**短跑脚本产出要求**：

```bash
# 短跑脚本命名约定
train_short.py          # 短跑训练脚本（从完整训练脚本派生）
run_short_profile.sh    # 短跑 + profiling 的启动脚本
```

**短跑脚本模板**（相对完整训练脚本的关键修改）：

```python
# === 短跑配置（相对完整训练的修改点）===
COMPRESS_RATIO = 10  # 压缩比，可调
num_scheduled_iterations = FULL_STEPS // COMPRESS_RATIO
num_extension_iterations = FULL_EXT // COMPRESS_RATIO

# 所有 step-dependent schedule 按相同比例压缩
# 例如：原 lr warmup 200 步 → 短跑 20 步
# 例如：原 batch_size 在 step 1000 切换 → 短跑 step 100 切换
# 例如：原 window_size 在 [500, 1000, 1500] 切换 → 短跑 [50, 100, 150] 切换

# 验证结束后输出 step_avg（用于后续 ablation 对比）
val_loss_every = num_scheduled_iterations + num_extension_iterations
```

**短跑脚本复用**：构造一次后，在整个优化迭代过程中复用：
- Phase 1 基线采集：用短跑脚本采 wall-clock + L0
- Phase 2 每轮 L1 采集：用短跑脚本采 L1
- Phase 4 收益确认：用短跑脚本采 wall-clock + L0
- 优化 ablation 验证：直接复用短跑脚本对比 step_avg

**regime 覆盖验证**：短跑脚本构造后，必须验证其输出覆盖了完整训练的所有 regime（通过 log 确认所有 schedule 切换点都被触发）。

### Profiler 接入

在短跑脚本基础上接入 profiler。训练与推理的差异：用 `schedule(skip_first=N)` 跳过初始化阶段，用 `for step, batch in enumerate(dataloader)` 替代单次调用。profiler 参数同上。

**采集策略**：对每个 regime 分别采集 1-3 个 active step，总 active step 控制在 3-10 个（避免 L1 输出过大）。

```python
# 在短跑脚本中接入 profiler
# skip_first 设为第一个 regime 稳定后的 step（跳过初始化 + warmup）
with torch_npu.profiler.profile(
    activities=[...],  # L0: [NPU]；L1: [CPU, NPU] + with_stack 等
    schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1, skip_first=20),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
        if step >= total_short_steps:
            break
```

**多 regime 采集**：如果训练有多个 regime（不同 batch size / sequence length），需要为每个 regime 设置独立的 profiler schedule，或分多次短跑采集（每次只采一个 regime 的 step）。

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
