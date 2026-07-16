# 多卡并行方案设计

## 何时需要多卡并行

**前提**: 多卡并行不是一种"优化手段",而是资源扩展手段。只有当单卡资源(内存/计算)真正不够、且单卡优化(去重/复用/替换)已无空间时才考虑。

进入此文档的触发条件(profiling 提供 trigger,源码决定方案):

1. **内存放不下**: `parse_operator_memory` 的 Parallelism Trigger 显示"消除 waste 后投影峰值仍 > 80% HBM" → 短命大 tensor 已无法解释内存压力,说明必要数据(模型参数/激活/梯度)本身超过单卡容量
2. **计算到顶仍不够快**: `parse_kernel_details` 显示 compute-bound(mac_ratio 高 + cube_util 高 + Block Dim 满),单卡已优化到硬件上限但性能仍不满足
3. **已多卡但效率低**: `parse_step_trace` 显示已有多卡通信(comm 列有值)但 device 利用率 < 50% → 当前并行策略需改进

⚠ **本文档的切分方案需要源码分析,profiling 只提供 trigger。** 进入后:
1. 用 operator_details 的 Call Stack 定位大 tensor 的源码位置
2. 阅读该处的计算结构,找出可切分的大维度
3. 按下方"切分维度选择"原则评估

| 场景 | 方案方向 |
|------|---------|
| 参数量大到单卡放不下 | 参数切分（Tensor Parallel / Pipeline Parallel） |
| 参数能放下但激活值 OOM | 激活值维度切分（本文重点） |
| batch 维度可并行 | 数据并行（最简单，无需额外通信） |

## 切分维度选择

### 分析步骤

1. 定位峰值瞬间所有存活张量的 shape（见 [memory_profiling.md](../../02_bottleneck_analysis/references/memory_profiling.md)）
2. 找到这些张量**共享的大维度**
3. 对候选维度逐一评估：切分后哪些操作仍可本地完成，哪些需要通信

### 选择原则

| 原则 | 原因 |
|------|------|
| 切分后大多数操作仍本地完成 | 通信次数最少 |
| 收缩维（被求和的维度）不切 | 否则需要 ReduceScatter 做部分和汇总 |
| 切分维度在多个张量中对称出现 | 统一约定，避免频繁行列转换 |

### 通信原语选择

每个需要跨卡数据的操作，选通信量最小的原语：

| 原语 | 语义 | 通信量 |
|------|------|--------|
| AllGather | 收集所有分片拼成完整张量 | `(P-1)/P × 数据量` |
| ReduceScatter | 各卡部分和汇总后分发 | `(P-1)/P × 数据量` |
| AllToAll | 维度间重分布（行切↔列切） | `(P-1)/P × 数据量` |
| Broadcast | 一卡广播给所有卡 | `数据量` |

通信时间 = 数据量 / 链路带宽。多层累积的通信总量可能很大——必须依赖通信-计算重叠来隐藏（见 `hide_latency.md`）。

## 并行区域设计

### 进入

- 方式 A：每卡从完整张量取本地分片（零拷贝 slice）
- 方式 B：rank 0 scatter 到各卡（节省初始内存）

### 退出

关键决策：**尽量全程保持切分**，避免在中间重建 GB 级全局张量。

- 下游模块能在切分数据上运行 → 不退出，继续传递本地分片
- 个别操作必须使用全局数据（如需要完整二维索引的 gather）→ 临时 AllGather + 用完立即释放
- 仅在最终输出时 AllGather 回全量

### 任意长度支持

切分要求维度大小能被卡数整除。不整除时在数据入口 padding 到 `ceil(N/P) × P`，用 mask 屏蔽 pad 位置，输出时按原始长度裁剪。开销 < 1%。

## 精度保证

- AllGather / AllToAll / Broadcast 只搬运数据，不改变数值
- fp32 下分块累加顺序改变引入的误差可忽略；bf16 下需实测
- 建议用小输入做多卡 vs 单卡 bit-exact 对比验证

## 常见陷阱

### 全局并行开关误触发子模块

全局 `is_parallel()` flag 被不需要并行的内部子模块检测到，对未切分数据调用通信算子。

**解法**：并行分支加模块级 guard，或进入非并行子模块前临时 disable。

### HCCL P2P 不可靠

Ascend HCCL 的 `dist.isend` / `dist.irecv` 存在 tag 不匹配等问题。

**解法**：用 broadcast 循环替代 ring P2P，性能可通过双 buffer 异步补回。

### AllToAll 要求 contiguous

`tensor.chunk()` 返回 view（非连续），HCCL AllToAll 报错。

**解法**：chunk 后加 `.contiguous()`，或用 reshape + permute 后调 `all_to_all_single`。

### 跨迭代 shape 变化

循环中第一轮输入可能是全局 `[N, ...]`，后续轮已变为本地 `[N/P, ...]`。

**解法**：用 `shape[0] == N` 显式判断，不假设输入总是全局形状。

### 负索引维度歧义

对低维张量 `[N/P, c]` 使用 `.unsqueeze(-3)` 结果可能不符预期。

**解法**：并行路径中用正整数维度索引。
