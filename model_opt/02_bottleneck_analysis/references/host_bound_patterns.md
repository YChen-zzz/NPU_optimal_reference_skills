# Host-Bound 深度诊断

当 `parse_step_trace` 判定设备利用率低（瓶颈在 host 侧）时，按以下步骤定位根因。

## 步骤 1：查看 host 时间分布

运行 `parse_operator_details.py` 默认模式。输出的 "Pure Host Ops" 部分展示了哪些操作占用了 host 时间但没有触发 device 计算。

按以下类别聚合判断方向：

| 类别 | 典型算子 | 优化方向 |
|------|---------|---------|
| tensor metadata ops | empty_tensor, view, as_strided | 预分配 buffer、减少临时 tensor |
| Python dispatch wrapper | aten::matmul, aten::dropout | flat forward、去除冗余调用 |
| D→H sync ops | aten::item, aten::_local_scalar_dense | 消除 .item()/.numpy()，缓存或延迟到 batch 结束 |
| ACL kernel launch | aclnnMm, aclnnRmsNorm | 图编译（通常不可单独压缩） |
| format/sync | format_cast, SynchronizeStream | 统一 layout、消除运行时 transpose |

**特别注意 `.item()` 同步**：HuggingFace Trainer 的 gradient clipping 和 NaN detection 会在每步调用 `.item()`，每次强制 D→H 同步。在训练场景中这可能占据 >40% 的 host 时间。检查 `parse_operator_details --filter item` 或 `--filter _local_scalar` 确认。

## 步骤 2：确认 bubble 是否真实

查看 `parse_kernel_details.py` 的 Wait Time Distribution。

**Level0 vs Level1**：Level1 在每个 kernel 前后插入 barrier，破坏 TASK_QUEUE 异步流水。Level1 的 bubble 大部分是 profiler 注入的。

确认方法：
- Level0 重新采集
- 对比两次的 Computing/Free 时间
- wall-clock 延迟交叉验证

## 步骤 3：判断 TASK_QUEUE 状态

`parse_kernel_details.py` 高等待上下文中：
- 前几个 kernel wait 大、后续稳定 → 正常 ramp-up
- 所有 kernel wait 均匀大 → TASK_QUEUE 未生效，检查环境变量 `TASK_QUEUE_ENABLE=2`

## 步骤 4：排除核间切换影响

`parse_kernel_details.py` 输出的 Accelerator Core Distribution 如果 AI_CORE 和 AI_VECTOR_CORE 比例接近，可能频繁切换。用 `--filter` 对比切换点和非切换点的 wait time 差异，< 10μs 则不是主因。

## 特殊场景：自回归 Decode

Decode 天然 host-bound——每 token 只有少量计算但需完整 dispatch。

判断依据：`parse_kernel_details.py` 全局模式如果显示 avg kernel duration 极小（<20us）且 kernel 数极多 → decode 场景。

此时 eager 模式下 dispatch 开销无法根本消除，需要：
- fp16/bf16 启用融合算子（减少 kernel 数）
- 图编译（消除逐算子 dispatch）
