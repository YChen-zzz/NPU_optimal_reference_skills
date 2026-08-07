# 掩盖——用并行让延迟不可见

## 原理

有些延迟无法消除（通信的物理传输时间、内存分配的系统调用时间），但可以让它与其他有用工作**在时间上重叠**，使其对端到端延迟的贡献为零。

核心条件：**找到两段工作之间没有数据依赖**，把它们放到不同的执行单元上并行。

## 判断方法

以下现象提示可以用"掩盖"手段优化：
- 通信和计算串行排列（通信结束后计算才开始）→ 检查两者之间是否有数据依赖，无依赖则可重叠
- 计算流中有空泡且空泡前后分别是通信和计算 → 可用多 stream 重叠
- 某段延迟占总时间比例大但本身无法压缩（如物理带宽限制）→ 寻找可并行的独立计算来填充

## 方案示例

### 通信-计算重叠

NPU 设备可以同时做 DMA 传输（通信）和 AI Core 计算。让通信在通信流执行，计算在默认流执行，两者物理并行。

**寻找可重叠对**：对每个通信操作，问"在等待通信完成的这段时间内，有没有不依赖通信结果的计算可以做？"典型模式：通信的数据和计算 B 的输入来自不同上游 → 可重叠；通信发出后、结果被使用前，存在其他独立计算 → 将通信提前发出。

**约束**：启动 comm_stream 前必须 `wait_stream` 确保输入就绪；同一 communicator 的两个集合操作不能在不同流并发（跨 rank 顺序不一致）；`all_to_all_single` 不支持 `async_op`，同步不可消除。

### 双 buffer 流水线

当一个操作需要依次处理 P 个块，且每个块需要"获取数据 + 计算"两步时，用两个 buffer 交替：第 k 轮计算用 buf[0]，同时第 k+1 轮获取数据写 buf[1]。当获取时间 ≈ 计算时间时，加速比接近 2×。

**典型场景**：分块计算中每块需要先 gather 数据再计算——无流水线时 P × (gather + compute)，有流水线时 gather₀ + P × max(gather, compute)。

### 前置分配与计算流水重叠

如果输出 buffer 的大小是编译期已知的常量，可以在前序计算执行期间提前分配 buffer 并填入部分数据，使分配/拷贝的延迟被前序计算掩盖。

**典型场景**：输出由多段拼接而成，且第一段在前序计算之前就可用——先分配 buffer 并写入第一段（异步执行），同时执行前序计算产出后续段，最后写入剩余部分。

### 图编译（掩盖的极端形态）

图编译把一段 eager 算子捕获为子图，交给 NPU 编译后端做融合、消除、重排和设备侧调度。它可能同时减少 Host dispatch、临时张量和 Device kernel；因此当 profile 出现大量短 kernel、明显 launch gap 或 graph break 时，应优先检查图编译机会，而不是只优化单个算子。

`torch.compile` 是图捕获入口，实际 lowering 与代码生成由 `backend` 决定。NPU 上必须使用当前软件栈已注册且验证过的 NPU backend；不要假设默认 Inductor/Triton 路径可用。TorchAir 等 NPU 编译器可能作为 backend 或框架集成的一部分接入，按项目版本核实。

以下最小实验把 `add` 和 `relu` 两个算子放进一个完整编译区：

```python
import torch


def add_relu(x, bias):
    y = torch.add(x, bias)   # 算子 1
    return torch.relu(y)     # 算子 2


def compile_add_relu(npu_backend):
    return torch.compile(
        add_relu,
        backend=npu_backend,  # 已注册的名称或 backend callable
        fullgraph=True,
    )


compiled_add_relu = compile_add_relu(project_npu_backend)
compiled_add_relu(x, bias)  # 首次调用触发编译；不要计入稳态耗时
```

`fullgraph=True` 用于小范围诊断：若两个算子无法捕获为一个 graph，会直接报 graph break。捕获成一个 graph 不保证后端一定生成一个 kernel；必须检查编译后 graph/IR 和 profile，确认融合、kernel 数、launch gap、重编译次数及稳态耗时。

**编译范围策略**：从纯计算、重复执行、形状稳定且“碎而密”的 Supernode 开始；先编译最小闭合子图，再逐步扩大边界。避免直接编译整个模型，因为控制流、side effect、动态 shape 或不支持算子可能切图或频繁重编译。

**验证要求**：编译与 warmup 放在计时区外；先比 eager/compiled 输出与精度，再比较同一 execution regime 下的 graph break、kernel/launch 数、Host/Device 时间、内存和端到端稳态耗时。只有 profile 证实 gap 被消除或隐藏，才登记为有效优化。

**放弃条件**：算子不兼容且无法缩小边界绕过、频繁重编译、编译期 OOM、后端错误、精度不达标、稳态无收益，或编译成本无法缓存。回退 eager，并把失败边界、backend、shape 和原因写入 evidence_db。

## 环境配置

`TASK_QUEUE_ENABLE=2` 使 Host 下发和 Device 执行在时间上重叠，是异步流水线的基础。注意：`ASCEND_LAUNCH_BLOCKING=1` 时 task_queue 关闭不生效；可能导致 NPU 内存峰值上升，遇 OOM 可回退到 `=1`。
