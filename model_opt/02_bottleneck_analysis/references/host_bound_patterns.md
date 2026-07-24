# Host-Bound 深度诊断

当 `parse_step_trace` 判定设备利用率低（瓶颈在 host 侧）时，按以下步骤定位根因。

## 步骤 1：查看 host 时间分布

运行 `parse_operator_details.py` 默认模式。输出的 "Pure Host Ops" 部分展示了哪些操作占用了 host 时间但没有触发 device 计算。

按以下类别聚合判断方向：

| 类别 | 典型算子 | 问题特征 |
|------|---------|---------|
| tensor metadata ops | empty_tensor, view, as_strided | 大量临时 tensor 创建，host 时间被 metadata 操作占据 |
| Python dispatch wrapper | aten::matmul, aten::dropout | 框架调度开销，设备 idle 但 host 在 Python 层忙碌 |
| D→H sync ops | aten::item, aten::_local_scalar_dense | 每次调用 drain 整个 NPU pipeline，强制 host 等待 device |
| ACL kernel launch | aclnnMm, aclnnRmsNorm | 逐算子下发，kernel 间存在 host dispatch 间隙 |
| format/sync | format_cast, SynchronizeStream | 运行时格式转换或显式同步打断流水 |

**特别注意 `.item()` 同步**：HuggingFace Trainer 的 gradient clipping 和 NaN detection 会在每步调用 `.item()`，每次强制 D→H 同步。在训练场景中这可能占据 >40% 的 host 时间。检查 `parse_operator_details --filter item` 或 `--filter _local_scalar` 确认。

## 步骤 2：定位 host2device bound 区段

`parse_operator_details` 给的是 host 时间聚合，看不出"哪段时间设备在空等 host"。运行 `parse_trace_view.py`，看第 4 节 **Host2Device Bound Regions**。

原理：CANN stream 里的 `Node@launch` 事件经 `connection_id` 与设备算子一一配对，`gap = device_start - launch_ts`。流水排布好时 host 远远跑在前面（gap 大，p50 常达数十~百 ms）；若一串相邻算子 gap 极小（<50us，含因 host/device 时钟 skew 出现的小负值），说明设备一 launch 就立刻执行、队列空转——这就是 host2device bound。

每段区段输出：起止时间、算子链、设备空闲占比、以及经 `async_npu(torch_to_npu)` flow 回连到 `cpu_op` 的 Call stack。直接定位到"哪段代码、哪个 forward 在 host 侧喂不动设备"。

典型成因：
- 碎片化的 host 计算（大量小 op 串行 dispatch，如逐元素/广播类小算子链）
- host 侧 Python 逻辑重（条件分支、循环里构造 tensor）
- 与 §3 Dispatch Latency 互补：§3 看全局下发延迟是否健康，§4 看具体哪段代码 host-bound

## 步骤 3：确认 bubble 是否真实

查看 `parse_kernel_details.py` 的 Wait Time Distribution。

**L0 vs L1**：L1 在每个 kernel 前后插入 barrier，破坏 TASK_QUEUE 异步流水。L1 的 bubble 大部分是 profiler 注入的。

确认方法：
- L0 重新采集
- 对比两次的 Computing/Free 时间
- wall-clock 延迟交叉验证

## 步骤 4：判断 TASK_QUEUE 状态

`parse_kernel_details.py` 高等待上下文中：
- 前几个 kernel wait 大、后续稳定 → 正常 ramp-up
- 所有 kernel wait 均匀大 → TASK_QUEUE 未生效，检查环境变量 `TASK_QUEUE_ENABLE=2`

## 步骤 5：排除核间切换影响

`parse_kernel_details.py` 输出的 Accelerator Core Distribution 如果 AI_CORE 和 AI_VECTOR_CORE 比例接近，可能频繁切换。用 `--filter` 对比切换点和非切换点的 wait time 差异，< 10μs 则不是主因。

## 特殊场景：自回归 Decode

Decode 天然 host-bound——每 token 只有少量计算但需完整 dispatch。

判断依据：`parse_kernel_details.py` 全局模式如果显示 avg kernel duration 极小（<20us）且 kernel 数极多 → decode 场景。

此时 eager 模式下 dispatch 开销无法根本消除，根因是每步计算量太小而 dispatch 次数不减。
