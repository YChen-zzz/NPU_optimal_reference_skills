# Profiling 采集代码模板

阅前须知：
- **路径规范**：所有 profiling 输出必须保存到 `<workspace>/profiling/<timestamp>/`，禁止使用 `/tmp` 或固定路径。以下模板使用 `profiling_dir` 变量，调用前按下方 §0 构造。
- **一致性要求**：agent 为项目编写采集脚本时，必须遵循主 SKILL.md「标准化操作规范」中的约束（环境变量、时间戳目录、运行日志、可复现性）。以下为模板示例，需按项目实际适配。
- **采集与业务分离原则**：采集脚本只负责"包围"业务调用（`with profiler.profile(...): run_inference(...)`），不嵌入业务逻辑。业务推理逻辑用函数封装（如 `run_inference(model, input_data)`），优化改这个函数，采集脚本不改。这样多次优化对比时用同一个采集脚本跑 before/after，确保采集条件一致。

```python
# business.py — 业务代码，被优化的是这里
def run_inference(model, input_data):
    """推理入口，优化改这里，采集脚本不改"""
    return model(input_data)

# profile.py — 采集脚本，只调用 run_inference，不改
def profile_run(profiling_dir, model, input_data):
    with torch_npu.profiler.profile(
        activities=[...],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
    ) as prof:
        run_inference(model, input_data)
        prof.step()
```

## 0. 通用前置

### 路径构造

在所有采集代码前执行一次，各场景共用：
```python
import os, datetime
PROFILING_BASE = os.path.join(os.getcwd(), "profiling")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
profiling_dir = os.path.join(PROFILING_BASE, timestamp)
os.makedirs(profiling_dir, exist_ok=True)
# 更新 latest 软链接
latest_link = os.path.join(PROFILING_BASE, "latest")
if os.path.islink(latest_link):
    os.remove(latest_link)
os.symlink(timestamp, latest_link)
```

### 推理 warmup

推理场景**不使用 schedule**（schedule 是训练场景控制采集步范围的机制）。推理前手动预热，触发 JIT 编译、算子缓存、内存池预热，确保采集到的数据是稳态性能：
```python
# 预热：3 次推理触发编译/缓存，不记录
for _ in range(3):
    model(input_data)
torch.npu.synchronize()

# 正式采集：1 次推理，prof.step() 触发导出
with torch_npu.profiler.profile(...) as prof:
    model(input_data)
    prof.step()
```

> 若需多次推理取平均（减少单次抖动），在 `with` 块内循环调用 `model(input_data)` 多次，最后调用一次 `prof.step()`。注意：多次推理的 profiling 数据会合并，`step_trace` 显示的是多次累计值。

### schedule 参数说明（仅训练场景）

训练场景用 schedule 控制采集范围，避免全量采集膨胀。推理场景不用 schedule，用手动 warmup 替代（见上方）：
```python
schedule = torch_npu.profiler.schedule(
    wait=1, warmup=1, active=1, repeat=1, skip_first=20
)  # 跳过前 20 步 → 等待 1 步 → 预热 1 步 → 采集 1 步
```

---

## 1. 分级采集模板（推理为主）

推理场景直接用 §1.1–1.3 的模板（无 schedule，手动 warmup）；训练场景见 §1.4 适配。

| 级别 | 内容 | 数据量 | 使用时机 |
|------|------|--------|----------|
| **L0** | 仅采集 NPU 活动，最小膨胀 | 小 | **性能基线 + L0/L1 交叉验证**：项目最开始采集一次作 baseline；每个优化阶段结束后采集一次做收益比对；下一轮 Phase 2 用此 L0 与 L1 做交叉验证 |
| **L1** | CPU + NPU + 算子详情 + 调用栈 + 内存 + AI Core 指标 + CANN 运行时 API 统计（覆盖全部 8 个解析脚本） | 大 | **优化分析主力**：每个优化阶段开始前采集一次，交给 Phase 2 分析 |

### 1.1 L0 — 基线 / 快速比对 / L0/L1 交叉验证（NPU Only）

用途：项目开始时采一次作 baseline；每个优化阶段结束后再采一次做收益比对；Phase 2 的 L0/L1 交叉验证用此数据（对比 L1 的 step_trace，排除 profiler 伪影）。不注入 CPU 侧 barrier，Host/Device 比例更接近真实。

```python
import torch, torch_npu

# 预热
for _ in range(3):
    model(input_data)
torch.npu.synchronize()

# 采集（NPU Only，无 schedule）
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    model(input_data)
    prof.step()
```

> **L0 目录路径需记录**：Phase 2 的 `run_analysis.py --l0-dir <此目录>` 会用此数据与 L1 做交叉验证。第 0 轮用 Phase 1 基线 L0；第 i 轮用第 i-1 轮 Phase 4 采集的 L0。

### 1.2 L1 — 优化分析主力（CPU + NPU，覆盖全部解析脚本）

用途：每个优化阶段开始前采集一次，交给 Phase 2 的 `run_analysis.py` 定位优化点。此档覆盖全部 8 个解析脚本所需文件：`op_statistic.csv`、`step_trace_time.csv`、`kernel_details.csv`（含硬件单元占比列）、`memory_record.csv`、`operator_details.csv`（含 Call Stack）、`operator_memory.csv`、`api_statistic.csv`、`trace_view.json`。

```python
import torch, torch_npu

# 预热
for _ in range(3):
    model(input_data)
torch.npu.synchronize()

# 采集（CPU + NPU，无 schedule）
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    with_stack=True,        # operator_details.csv 的 Call Stack 列
    record_shapes=True,     # kernel_details / operator_details 的 Input Shapes 列
    profile_memory=True,    # memory_record.csv 与 operator_memory.csv
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,  # kernel_details 的 mac/mte/vec 占比列
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    model(input_data)
    prof.step()
```

> `with_stack=True` 和 `profile_memory=True` 会显著增大输出（可达数 GB），确认磁盘空间充足。
> 采集后将 L1 目录传给 `run_analysis.py <l1_dir> --l0-dir <l0_dir>` 即可进入 Phase 2 分析。

### 1.3 训练场景适配

训练场景与推理的差异：用 `schedule` 控制采集步范围（替代手动 warmup），用 `for step, batch in enumerate(dataloader)` 循环（替代单次调用）。`activities` / `experimental_config` 按 §1.1–1.3 对应档位填写：

```python
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],  # 按档位调整
    schedule=torch_npu.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

---

## 2. 训练框架适配

三种训练框架的差异仅在**采集器的接入点**（在哪个回调启停 profiler），profiler 参数构造逻辑完全一致——统一用下方 `build_profile_kwargs`，各框架复用。

**共用参数构造**（level 分支对应 §1 各档模板）：
```python
import os, torch_npu

def build_profile_kwargs(output_dir, level="L0",
                         skip_first=20, wait=1, warmup=1, active=1, repeat=1):
    kwargs = {
        "activities": [torch_npu.profiler.ProfilerActivity.NPU],
        "schedule": torch_npu.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
        "on_trace_ready": torch_npu.profiler.tensorboard_trace_handler(output_dir),
    }
    if level == "L1":
        kwargs["activities"].insert(0, torch_npu.profiler.ProfilerActivity.CPU)
        kwargs["record_shapes"] = True
        kwargs["with_stack"] = True
        kwargs["profile_memory"] = True
        level_enum = (torch_npu.profiler.ProfilerLevel.Level1)
        kwargs["experimental_config"] = torch_npu.profiler._ExperimentalConfig(
            profiler_level=level_enum,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
    return kwargs
```

### 2.1 PyTorch Lightning（Callback）

```python
import pytorch_lightning as pl

class NPUProfilingCallback(pl.Callback):
    def __init__(self, output_dir, level="L0", **sched):
        super().__init__()
        self.kwargs = build_profile_kwargs(output_dir, level, **sched)
        self.prof = None

    def on_train_start(self, trainer, pl_module):
        self.prof = torch_npu.profiler.profile(**self.kwargs)
        self.prof.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.prof:
            torch.npu.synchronize()
            self.prof.step()

    def on_train_end(self, trainer, pl_module):
        if self.prof:
            self.prof.stop()
```

### 2.2 HuggingFace Trainer（TrainerCallback）

接入点不同（`on_train_begin` / `on_step_end` / `on_train_end`），参数构造复用 `build_profile_kwargs`：

```python
from transformers import TrainerCallback

class NPUProfilingTrainerCallback(TrainerCallback):
    def __init__(self, output_dir, level="L0", **sched):
        self.kwargs = build_profile_kwargs(output_dir, level, **sched)
        self.prof = None

    def on_train_begin(self, args, state, control, **kw):
        self.prof = torch_npu.profiler.profile(**self.kwargs)
        self.prof.start()

    def on_step_end(self, args, state, control, **kw):
        if self.prof:
            torch.npu.synchronize()
            self.prof.step()

    def on_train_end(self, args, state, control, **kw):
        if self.prof:
            self.prof.stop()
```

### 2.3 DeepSpeed 多卡

**多卡必须每张卡都采集**，不能只采 rank 0。原因：通信类文件（`communication.json`、`communication_matrix.json`，L1 采集）依赖各 rank 的通信记录，只采 rank 0 会丢失 all-reduce/all-gather 等集合通信画像，也无法发现 rank 间负载不均（straggler）。每个 rank 写入独立子目录 `rank_<n>/`，与解析脚本的 `--rank N` 约定一致。

```python
import os, torch_npu

rank = int(os.environ.get("RANK", local_rank))
rank_dir = os.path.join(profiling_dir, f"rank_{rank}")
os.makedirs(rank_dir, exist_ok=True)

prof = torch_npu.profiler.profile(**build_profile_kwargs(rank_dir, level="L1"))
prof.start()
for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
    prof.step()
prof.stop()
```

> **数据量控制**：全卡 L1 数据量随卡数线性增长。若仅需 kernel/算子/内存等单卡即可代表的分析，可只对 rank 0 采 L1、其余 rank 采 L0（或缩短 `active`）以省空间；但**只要涉及通信瓶颈分析，就必须全卡采集**。
>
> **分析入口**：`run_analysis.py` 通过 `--rank N` 定位到 `profiling_dir/rank_N/`。跨 rank 对比需分别运行后比对，识别 straggler。

---

## 3. GPU 对比采集

跨平台对比时，**GPU 侧与 NPU 侧的 L0（§1.1）配对采集**——两端都用最小档、相同输入数据，只对比整体耗时与 Host/Device 分布，避免高档位 profiler 注入开销干扰跨平台可比性。GPU 端用 `torch.profiler`：

```python
# GPU 侧同样先 warmup
for _ in range(3):
    model(input_data)
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_dir),
) as prof:
    model(input_data)
    prof.step()
```

关键差异（相对 NPU L0）：
- 用 `torch.profiler.profile`（而非 `torch_npu.profiler.profile`）
- 用 `ProfilerActivity.CUDA`（而非 `.NPU`）
- 不使用 `experimental_config` 参数
- **务必与 NPU 侧 L0 配对**：输入数据、batch 保持一致，否则两端不可比

---

## 4. 采集前检查清单

- 业务脚本可在不采集的情况下稳定跑通
- 推理场景：输入固定、batch 稳定，已完成 warmup 预热
- 训练场景：`skip_first` 跳过编译/数据预热 step
- 磁盘空间充足
- 已设置 `TASK_QUEUE_ENABLE=2` 和 `CPU_AFFINITY_CONF=1`
- 运行了本 skill 的 `scripts/validate_profiling_env.py` 确认环境就绪
- L0 采集目录路径已记录（供下一轮 Phase 2 交叉验证使用）
