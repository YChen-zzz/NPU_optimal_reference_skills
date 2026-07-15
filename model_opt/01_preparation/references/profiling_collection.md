# Profiling 采集代码模板

> **路径规范**：所有 profiling 输出必须保存到 `<workspace>/profiling/<timestamp>/`，禁止使用 `/tmp` 或固定路径。详见 01_preparation/SKILL.md「Profiling 输出路径规范」。以下模板使用 `profiling_dir` 变量，调用前按下方 §0 构造。
>
> **一致性要求**：agent 为项目编写采集脚本时，必须遵循主 SKILL.md「标准化操作规范」中的约束（环境变量、时间戳目录、运行日志、可复现性）。以下为模板示例，需按项目实际适配。

## 0. 通用前置：路径构造

**路径构造**（在所有采集代码前执行一次，各场景共用）：
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

**schedule 参数说明**（训练场景控制采集范围，避免全量采集膨胀；各档模板中已内联，此处解释参数含义）：
```python
schedule = torch_npu.profiler.schedule(
    wait=1,          # 跳过后等待的步数
    warmup=1,        # 预热步数（采集但不记录）
    active=1,        # 实际记录的步数
    repeat=1,        # 重复轮数
    skip_first=20    # 跳过初始迭代（编译、数据加载等非稳态开销）
)
```
含义：跳过前 20 步 → 等待 1 步 → 预热 1 步 → 采集 1 步。推理场景通常不用 schedule，推理结束后直接 `prof.step()` 触发导出。

---

## 1. 分级采集模板

三档对应不同用途：**L0 用于性能判定基线与快速比对，L1 是优化分析主力，L2 在 L1 信息不足时深度下探**。使用时机详见 SKILL.md「采集级别选择」。定位到所需档位后直接复制对应代码块即可。

### 1.1 L0 — 基线判定 / 快速比对（NPU Only）

用途：项目开始时采一次作 baseline；每个优化阶段结束后再采一次，与 baseline/上一轮快速比对判定收益。不注入 CPU 侧 barrier，Host/Device 比例更接近真实。也用作 GPU/NPU 跨平台对比的 NPU 侧（见 §3）。

```python
import torch, torch_npu

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    for step, batch in enumerate(dataloader):
        forward_step(batch)
        prof.step()
```

### 1.2 L1 — 优化分析主力（CPU + NPU，覆盖全部解析脚本）

用途：每个优化阶段开始前采集一次，交给 Phase 2 瓶颈分析模块定位优化点。此档覆盖 `02_bottleneck_analysis` 全部 7 个解析脚本所需文件：`op_statistic.csv`、`step_trace_time.csv`、`kernel_details.csv`（含硬件单元占比列）、`memory_record.csv`、`operator_details.csv`（含 Call Stack）、`operator_memory.csv`。

```python
import torch, torch_npu

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    with_stack=True,        # operator_details.csv 的 Call Stack 列
    record_shapes=True,     # kernel_details / operator_details 的 Input Shapes 列
    profile_memory=True,    # memory_record.csv 与 operator_memory.csv
    schedule=torch_npu.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,  # kernel_details 的 mac/mte/vec 占比列
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    for step, batch in enumerate(dataloader):
        forward_step(batch)
        prof.step()
```

> `with_stack=True` 和 `profile_memory=True` 会显著增大输出（可达数 GB），确认磁盘空间充足。

### 1.3 L2 — 深度下探（CANN Runtime/GE + AI CPU）

用途：与 L1 用在同一阶段（优化分析前），仅当 L1 信息不足以定位优化点时启用。相比 L1 仅改 `profiler_level=Level2`，额外采集 CANN 层 Runtime/GE 数据和 AI CPU 数据（生成 `data_preprocess.csv`），用于排查 Runtime 底层调度开销或算子 fallback 到 AI CPU。

```python
import torch, torch_npu

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    with_stack=True,
    record_shapes=True,
    profile_memory=True,
    schedule=torch_npu.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    for step, batch in enumerate(dataloader):
        forward_step(batch)
        prof.step()
```

> L2 数据量最大（可达 10GB+），仅在需要时启用。

### 1.4 推理场景

推理场景去掉 `schedule`，在推理结束后调用 `prof.step()` 触发导出（`activities` / `experimental_config` 按 §1.1–1.3 对应档位填写）：

```python
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],  # 按档位调整
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    model(input_data)
    prof.step()
```

---

## 2. 框架适配

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
    if level in ("L1", "L2"):
        kwargs["activities"].insert(0, torch_npu.profiler.ProfilerActivity.CPU)
        kwargs["record_shapes"] = True
        kwargs["with_stack"] = True
        kwargs["profile_memory"] = True
        level_enum = (torch_npu.profiler.ProfilerLevel.Level2 if level == "L2"
                      else torch_npu.profiler.ProfilerLevel.Level1)
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

**多卡必须每张卡都采集**，不能只采 rank 0。原因：通信类文件（`communication.json`、`communication_matrix.json`，L1/L2 采集）依赖各 rank 的通信记录，只采 rank 0 会丢失 all-reduce/all-gather 等集合通信画像，也无法发现 rank 间负载不均（straggler）。每个 rank 写入独立子目录 `rank_<n>/`，与解析脚本的 `--rank N` 约定一致。

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

> **数据量控制**：全卡 L1/L2 数据量随卡数线性增长。若仅需 kernel/算子/内存等单卡即可代表的分析，可只对 rank 0 采 L1、其余 rank 采 L0（或缩短 `active`）以省空间；但**只要涉及通信瓶颈分析，就必须全卡采集**。
>
> **分析入口**：解析脚本通过 `--rank N` 定位到 `profiling_dir/rank_N/`。跨 rank 对比（各 rank Computing/Communication 时间）需分别解析后比对，识别 straggler。

---

## 3. GPU 对比采集

跨平台对比时，**GPU 侧与 NPU 侧的 L0（§1.1）配对采集**——两端都用最小档、相同 schedule、相同输入数据，只对比整体耗时与 Host/Device 分布，避免高档位 profiler 注入开销干扰跨平台可比性。GPU 端用 `torch.profiler`：

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_dir),
) as prof:
    for step, batch in enumerate(dataloader):
        forward_step(batch)
        prof.step()
```

关键差异（相对 NPU L0）：
- 用 `torch.profiler.profile`（而非 `torch_npu.profiler.profile`）
- 用 `ProfilerActivity.CUDA`（而非 `.NPU`）
- 不使用 `experimental_config` 参数
- **务必与 NPU 侧 L0 配对**：schedule、输入数据、batch 保持一致，否则两端不可比

---

## 4. 采集前检查清单

- 业务脚本可在不采集的情况下稳定跑通
- 训练场景：`skip_first` 跳过编译/数据预热 step
- 推理场景：输入固定、batch 稳定，避免多轮结果不可比
- 磁盘空间充足（L2 可达 10GB+）
- 已设置 `TASK_QUEUE_ENABLE=2` 和 `CPU_AFFINITY_CONF=1`
- 运行了本 skill 的 `scripts/validate_profiling_env.py` 确认环境就绪
