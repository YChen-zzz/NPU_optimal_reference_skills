# Decode 路径优化思路

## 问题本质

自回归推理的特点是 **每步只生成一个 token**，导致每步的计算量很小但 host-device 交互次数不减。根本矛盾是：

- 每步计算 ~微秒级，但 host 准备 + dispatch ~毫秒级
- 每步都需要 host-device 同步（取 token、检查 EOS）
- 推理框架（如 HF generate）的通用性设计带来大量冗余逻辑

## 思考路径

### 思路 1：减少框架开销

**问题**：通用 generate 函数包含采样策略、beam search、流程控制等大量逻辑，纯 greedy 推理并不需要。

**解法**：自定义精简的 decode 循环，只保留 forward + argmax + EOS 检测。

**何时值得做**：profiling 显示每步中非计算开销（Python 逻辑）占比 > 50%。

### 思路 2：减少内存操作

**问题**：KV cache 每步动态扩容（`torch.cat`）触发内存分配和拷贝。

**解法**：预分配最大长度的 cache，每步用 index write 填充。

**关键洞察：GPU 经验在 NPU 上可能反转**：

GPU 上 StaticCache（预分配）通常最快，因为避免了动态内存分配。但 NPU 上 DynamicCache（动态拼接）反而可能更快，因为：
- NPU 的动态拼接算子高度优化
- 预分配方案在 NPU 上可能展开为更多子 kernel

**必须实测**，不能假设。

### 思路 3：减少同步次数

**问题**：每步检查 EOS 需要 `.item()` 把结果从 device 读回 host，触发同步等待。

**解法**：每 N 步检查一次 EOS，而不是每步。代价是可能多跑 N-1 步。

**确定最优 N**：用微基准遍历候选值，找净收益 = 节省的 sync 次数 × sync 单位开销 - 多跑步数 × 单步开销。

### 思路 4：消除冗余计算

**问题**：某些计算在 decode 循环中每步重复但结果不变。

**典型场景**：
- encoder-decoder 架构的 cross-attention KV：只依赖 encoder 输出，在 decode 循环外一次性计算
- position bias：只依赖序列长度，按长度缓存复用

**判断方法**：某个计算的输入是否只依赖模型参数或序列形状，而不依赖输入内容 → 可以预计算或缓存。

### 思路 5：统一数据格式

**问题**：decode 过程中频繁的 squeeze/unsqueeze/reshape 引入额外的格式转换开销。

**解法**：
- 确定一套维度约定并全程贯彻（如全程 2D `(B*S, hidden)` 或 3D `(B*H, S, D)`）
- 注意 NPU 上连续写入和非连续写入的性能差异可能很大，决定了 cache 的维度顺序

## 适用性判断

| 思路 | 适用条件 | 不适用条件 |
|------|---------|----------|
| 精简 decode 循环 | 纯 greedy 推理 | 需要 beam search / sampling 策略 |
| KV cache 预分配 | max_len 可预知且不太大 | max_len 非常大或不可知 |
| EOS interval check | 平均生成长度 >> N | 大多数序列很短（多跑代价高） |
| cross-attn 预计算 | encoder-decoder 架构 | decoder-only 架构 |

## 核心教训

- **不要假设 GPU 上的最优方案在 NPU 上也最优**：每种 KV cache 策略都必须在 NPU 上实测
- **微基准先行**：先用单独的小脚本量化每个操作的开销（torch.cat vs index write、.item() 同步开销等），再决定优化方向
- **dispatch/compute ratio > 1 时是理论瓶颈**：eager 模式下每个 kernel 的 Python dispatch 开销是固定的，当计算量很小时 dispatch 占比会超过 compute，此时只能通过减少 kernel 数量（融合、dtype 降精度启用更多融合算子）或图编译来解决
