# Profiling 采集代码模板

> **路径规范**：所有 profiling 输出必须保存到 `<workspace>/profiling/<timestamp>/`，禁止使用 `/tmp` 或固定路径。详见 01_preparation/SKILL.md「Profiling 输出路径规范」。以下模板中使用 `profiling_dir` 变量，调用前需按规范构造。
>
> **一致性要求**：agent 为项目编写采集脚本时，必须遵循主 SKILL.md「标准化操作规范」中的约束（环境变量、时间戳目录、运行日志、可复现性）。以下代码为模板示例，需根据项目实际情况适配。

```python
# 路径构造（在所有采集代码前执行）
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

## 1. 训练场景 schedule 配置

训练场景使用 `schedule` 控制采集范围，避免全量采集导致数据膨胀：

```python
schedule = torch_npu.profiler.schedule(
    wait=1,          # 跳过后等待的步数
    warmup=1,        # 预热步数（采集但不记录）
    active=1,        # 实际记录的步数
    repeat=1,        # 重复轮数
    skip_first=20    # 跳过初始迭代（编译、数据加载等非稳态开销）
)
```

此配置表示：跳过前 20 步 → 等待 1 步 → 预热 1 步 → 采集 1 步。

## 2. 推理场景

推理场景在推理结束后调用 `prof.step()` 触发 trace 导出：

```python
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    model(input_data)
    prof.step()
```

---

## 3. 分级采集模板

### 3.1 L0 — 最小膨胀（NPU Only）

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

### 3.2 L1 — 算子级（CPU + NPU）

```python
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    with_stack=False,
    record_shapes=True,
    profile_memory=False,
    schedule=torch_npu.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    for step, batch in enumerate(dataloader):
        forward_step(batch)
        prof.step()
```

### 3.3 L2 — 完整调用栈 + 内存

```python
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
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    for step, batch in enumerate(dataloader):
        forward_step(batch)
        prof.step()
```

> `with_stack=True` 和 `profile_memory=True` 显著增大输出，仅在需要时开启。

---

## 4. 框架适配

### 4.1 PyTorch Lightning

通过 Callback 机制接入：

```python
import torch_npu
import pytorch_lightning as pl

class NPUProfilingCallback(pl.Callback):
    def __init__(self, output_dir=None, level="L0",
                 skip_first=20, wait=1, warmup=1, active=1, repeat=1):
        super().__init__()
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "profiling",
                                      datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(output_dir, exist_ok=True)
        self.output_dir, self.level = output_dir, level
        self.skip_first = skip_first
        self.wait, self.warmup, self.active, self.repeat = wait, warmup, active, repeat
        self.prof = None

    def _build_kwargs(self):
        kwargs = {
            "activities": [torch_npu.profiler.ProfilerActivity.NPU],
            "schedule": torch_npu.profiler.schedule(
                wait=self.wait, warmup=self.warmup,
                active=self.active, repeat=self.repeat, skip_first=self.skip_first),
            "on_trace_ready": torch_npu.profiler.tensorboard_trace_handler(self.output_dir),
        }
        if self.level in ("L1", "L2"):
            kwargs["activities"].insert(0, torch_npu.profiler.ProfilerActivity.CPU)
            kwargs["record_shapes"] = True
            kwargs["experimental_config"] = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
        if self.level == "L2":
            kwargs["with_stack"] = True
            kwargs["profile_memory"] = True
        return kwargs

    def on_train_start(self, trainer, pl_module):
        self.prof = torch_npu.profiler.profile(**self._build_kwargs())
        self.prof.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.prof:
            torch.npu.synchronize()
            self.prof.step()

    def on_train_end(self, trainer, pl_module):
        if self.prof:
            self.prof.stop()
```

### 4.2 HuggingFace Trainer

```python
from transformers import TrainerCallback
import torch_npu

class NPUProfilingTrainerCallback(TrainerCallback):
    def __init__(self, output_dir=None, skip_first=20):
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "profiling",
                                      datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(output_dir, exist_ok=True)
        self.output_dir, self.skip_first = output_dir, skip_first
        self.prof = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=1, warmup=1, active=1, repeat=1, skip_first=self.skip_first),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.output_dir),
        )
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof:
            torch.npu.synchronize()
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.prof:
            self.prof.stop()
```

### 4.3 DeepSpeed 多卡

仅在 rank 0 采集以减少数据量：

```python
if local_rank == 0:
    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=1, warmup=1, active=1, repeat=1, skip_first=20),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir),
    )
    prof.start()

for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
    if local_rank == 0:
        prof.step()

if local_rank == 0:
    prof.stop()
```

---

## 5. GPU 对比采集

GPU 端使用 `torch.profiler`，配置与 NPU 对称：

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1, skip_first=20
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_dir)
) as prof:
    for step, batch in enumerate(dataloader):
        forward_step(batch)
        prof.step()
```

关键差异：
- 使用 `torch.profiler.profile`（而非 `torch_npu.profiler.profile`）
- 使用 `ProfilerActivity.CUDA`（而非 `.NPU`）
- 不使用 `experimental_config` 参数

---

## 6. 采集前检查清单

- 业务脚本可在不采集的情况下稳定跑通
- 训练场景：`skip_first` 跳过编译/数据预热 step
- 推理场景：输入固定、batch 稳定，避免多轮结果不可比
- 磁盘空间充足（L2 可达 10GB+）
- 已设置 `TASK_QUEUE_ENABLE=2` 和 `CPU_AFFINITY_CONF=1`
- 运行了本 skill 的 `scripts/validate_profiling_env.py` 确认环境就绪
