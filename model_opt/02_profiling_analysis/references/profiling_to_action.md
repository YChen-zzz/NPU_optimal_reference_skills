# Profiling 分析推理指南

脚本给出了数据和疑点，但从疑点到优化决策之间还需要推理。本文提供：
- 信号组合的联合推理方法
- 脚本信息不够时如何深入原始数据
- 具体的代码级行动模式

## 从疑点到优化决策的推理

脚本标记的每个 Suspect Signal 只是起点。做优化决策还需回答：
1. **根因是什么**：需要结合源码理解为什么这个操作会出现
2. **是否值得优化**：占总耗时多少、优化后其他瓶颈是否会暴露
3. **怎么优化**：具体用什么手段，参见下方行动模式

## 信号组合的联合推理

单一信号往往含义模糊，多个信号组合才能确定方向。

### Host-Bound + empty_tensor 占主导

step_trace 利用率低 + operator_details 中 `empty_tensor` Host Duration 占比 > 30%

→ 不是 Python 调度开销，是 **allocator 同步瓶颈**。每次 `empty()` 都可能等 device 完成才能分配。
→ 进一步确认：`parse_operator_memory` 是否有重复同尺寸分配
→ 行动：预分配 buffer + `out=` 写入

### Host-Bound + dispatch wrapper 占主导

step_trace 利用率低 + operator_details 中 `aten::matmul` / `aten::linear` 等 pure host ops 占主导

→ **Module.__call__ 调度栈**是瓶颈。每个算子都要走完整的 hook → check → dispatch 流程。
→ 进一步确认：`parse_kernel_details` 的 wait time 是否均匀分布（每 kernel 都等 50-200μs）
→ 行动：flat forward 绕过 Module，或图编译

### 利用率高 + mte_ratio >> mac_ratio

step_trace 利用率高 + kernel_details 硬件单元中 mte（搬运）远大于 mac（计算）

→ **Memory-Bound**：kernel 在忙但大部分时间在搬数据而非计算
→ 进一步确认：`parse_kernel_details --filter <Top算子>` 看是否所有 shape 都 mte 高，还是只有特定 shape
→ 如果所有 shape 都高：可能同时有过多大 tensor 导致 HBM 带宽竞争
→ 如果只有特定 shape：该 shape 计算密度太低，考虑 padding 或换 shape

### 小算子 > 50% + Block Dim=1 多 + 利用率低

→ **Decode 场景的碎片化**：每 token shape 太小导致并行度不足 + dispatch 开销占比高
→ 确认方法：kernel_details avg duration 是否极小（<20us）
→ 行动：fp16/bf16 启用融合算子减少 kernel 数，或图编译

### memory_record 高频抖动 + operator_memory 重复同尺寸分配

→ **缺少 buffer 复用**：同一个计算每次都分配释放相同大小的 tensor
→ 进一步：`parse_operator_details --filter <op>` 看 Call Stack 确认是哪行代码
→ 行动：在 `__init__` 中预分配，forward 中通过 `out=` 复用

### kernel_details 少数 kernel wait 极大（>10ms）

→ 该 kernel 前有**显式同步**操作阻塞了 pipeline
→ 看 wait 上下文中前面的 kernel 类型
→ 常见原因：`.item()`、`.numpy()`、`.to(device)`、`empty_cache()`
→ 行动：消除同步点，缓存结果或延迟到 batch 结束

## 常见代码级行动模式

| 诊断结论 | 代码行动 |
|---------|---------|
| Transpose 过多（layout 不一致） | 初始化时 `weight = weight.t().contiguous()`，forward 中直接用转置后的权重 |
| 4D matmul 触发运行时 Transpose | reshape 为 3D `(B*H, S, D)` + `torch.bmm` 替代 4D `matmul`，K 存储为 `(B*H, D, S)` 省去运行时 transpose |
| empty_tensor 过多（每次分配） | `self.buf = torch.empty(size, device=dev)` 在 init 预分配，forward 中 `torch.matmul(a, b, out=self.buf)` |
| Module.__call__ 开销大 | 提取权重到 dict/list，写纯函数 forward 绕过 nn.Module 调度链 |
| Mul + Sigmoid 成对高频 | `x.sigmoid_()` 原地化，或 `torch_npu.npu_swiglu(x)` 融合 |
| torch.cat 导致 MemSet（如 KV cache） | 预分配 `(B, H, max_len, D)` buffer，每步 `cache[:,:,step,:] = new_kv` 替代 cat |
| dropout p=0 仍有开销 | `model.layer.dropout = nn.Identity()` 或 monkey-patch 跳过 |
| .item() / .numpy() 触发 D→H 同步 | 缓存到 list 或 device tensor，batch 结束后统一取回；禁用 Trainer 中的 nan_filter/grad_clip logging |
| einsum 拆成多个 bmm | 改为显式 `torch.matmul` 减少算子数量 |
| RmsNorm 手动实现碎片化 | 替换为 `torch_npu.npu_rms_norm` 或 `torch_npu.npu_add_rms_norm`（后者同时融合 residual add） |
| 多个独立 Linear 可合并 | Q/K/V 三个 Linear 合并为一个大 MatMul + split，减少 kernel 数 |

## 脚本不够时的深入方法

当脚本输出信息不足以做判断时，直接读原始 CSV：

| 想了解什么 | 去哪里 | 看什么 |
|-----------|--------|--------|
| 某算子的实际 input shape | `kernel_details.csv` | Input Shapes 列 |
| 某次分配时系统内存有多满 | `operator_memory.csv` | Allocation Total Allocated(MB) |
| 某算子在整个 forward 中出现的位置序列 | `kernel_details.csv` | 按 Start Time 排序，搜目标算子看它在序列中的分布 |
| 完整的 Python 调用链 | `operator_details.csv` | Call Stack 列（分号分隔帧列表） |
| 两个 kernel 之间的真实 gap | `kernel_details.csv` | 当前 kernel 的 Start Time - 上一个 kernel 的 (Start Time + Duration) |
| 某个 step 的独立数据 | `kernel_details.csv` | 按 Step Id 列过滤 |
| Level0 vs Level1 对比 | 两份 profiling | 分别运行脚本对比 Computing/Free 差异，Level1 bubble 可能被夸大 |

## 多种问题并存时的优先级

按收益确定性和实施风险排序：

```
1. 显式同步（.item / .numpy / H2D / empty_cache）→ 消除（单点修复，收益确定，零风险）
2. 图编译可行？→ 尝试（收益上限最高，但可能不兼容）
3. allocator 同步（empty_tensor 占比高）→ 预分配（收益确定，改动较大）
4. 框架 dispatch 开销 → flat forward（收益大，改动大）
5. 碎片算子融合（Pow+Mean+Rsqrt / SiLU+Mul / QKV 合并）→ 融合算子或合并 Linear
6. 通信串行（hcom 占比高）→ 通信-计算重叠（收益取决于可重叠的计算量）
7. 数据布局（Transpose 多）→ 预转置 + 3D bmm（累积收益可观）
8. kernel 本身慢（Compute-Bound）→ 降精度启用硬件加速（需精度验证）
```
