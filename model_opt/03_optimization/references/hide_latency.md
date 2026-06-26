# 掩盖——用并行让延迟不可见

## 原理

有些延迟无法消除（通信的物理传输时间、内存分配的系统调用时间），但可以让它与其他有用工作**在时间上重叠**，使其对端到端延迟的贡献为零。

核心条件：**找到两段工作之间没有数据依赖**，把它们放到不同的执行单元上并行。

## 判断方法

以下现象提示可以用"掩盖"手段优化：
- 通信和计算串行排列（通信结束后计算才开始）→ 检查两者之间是否有数据依赖，无依赖则可重叠
- 计算流中有空泡且空泡前后分别是通信和计算 → 可用多 stream 重叠
- 某段延迟占总时间比例大但本身无法压缩（如物理带宽限制）→ 寻找可并行的独立计算来填充

## 通信-计算重叠

### 原理

NPU 设备可以同时做 DMA 传输（通信）和 AI_CORE 计算。让通信在通信流执行，计算在默认流执行，两者物理并行。

### 模式

```python
comm_stream = torch.npu.Stream()

# 1. 通信流上发起通信
with torch.npu.stream(comm_stream):
    gathered = all_gather(local_tensor)

# 2. 默认流上做不依赖通信结果的计算
compute_result = independent_compute(other_input)

# 3. 同步后使用通信结果
torch.npu.current_stream().wait_stream(comm_stream)
output = use_both(compute_result, gathered)
```

### 寻找可重叠对的方法

对每个通信操作，问：**在等待通信完成的这段时间内，有没有不依赖通信结果的计算可以做？**

典型模式：
- 通信 A 的数据和计算 B 的输入来自不同的上游 → 可重叠
- 通信发出后、结果被使用前，存在其他独立计算（如投影、norm）→ 将通信提前发出

### 约束

- 启动 comm_stream 前必须 `comm_stream.wait_stream(current_stream)` 确保输入就绪
- **同一 communicator 的两个集合操作不能在不同流并发**（跨 rank 顺序不一致）
- 全局复用单个 comm_stream
- `all_to_all_single` 不支持 `async_op`，同步不可消除

## 双 buffer 流水线

### 原理

当一个操作需要依次处理 P 个块，且每个块需要"获取数据 + 计算"两步时，用两个 buffer 交替：第 k 轮计算用 buf[0]，同时第 k+1 轮获取数据写 buf[1]。

### 时序

```
无流水线：P × (获取 + 计算)
有流水线：获取₀ + P × max(获取, 计算)
```

当获取时间 ≈ 计算时间时，加速比接近 2×。

### 适用场景

- Ring 通信：P 轮 broadcast + einsum → 双 buffer 让通信和计算重叠
- 分块加载：从 HBM 读取大数据分块 → 一块在计算另一块在传输

## 前置分配与计算流水重叠

### 原理

如果输出 buffer 的大小是编译期已知的常量，可以在**前序计算执行期间**提前分配 buffer 并填入部分数据，使分配/拷贝的延迟被前序计算掩盖：

```python
# 分配 + 拷贝发生在 compute_B 执行之前
out = torch.zeros(shape)
out.narrow(-1, 0, d1).copy_(part1)     # NPU 异步执行

part2 = compute_B(...)                 # 与上面的 copy_ 流水重叠

out.narrow(-1, d1, d2).copy_(part2)    # 两部分都就绪
```

## 图编译的掩盖本质

图编译将多个算子合成一个子图，设备侧一次性调度执行。从"掩盖"角度理解：逐算子 dispatch 时每个 kernel 之间有 host 调度的间隙（延迟），编译后这些间隙被设备内部流水线掩盖——设备不再等待 host 逐个下发。

## Host-Device 异步流水（TASK_QUEUE_ENABLE）

将部分算子适配工作从 Host 侧迁移至 Device 侧，使 Host 下发和 Device 执行在时间上重叠。训练场景（大量密集算子下发）收益显著。

详细配置见 [training_tuning.md](training_tuning.md)。
