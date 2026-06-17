---
name: NPU 优化实施
description: 在昇腾 NPU 上实施模型调优，包括算子融合、Host 开销消减、布局优化、Decode 优化、图编译和训练调优。当用户要求优化性能或实施优化方案时触发。
---

# NPU 优化实施

## 总体策略

根据 Profiling 分析结果选择优化方向，每次优化后重新 Profiling 确认瓶颈是否转移。

**图编译优先尝试**（收益上限最高），失败后走 eager 模式优化。

## 优化方向选择

| Profiling 特征 | 优化方向 | 参考文档 |
|---------------|---------|----------|
| Host-Bound 且形状固定 | 图编译（优先尝试） | 见下文 |
| 大量小 kernel，timeline 密集碎片 | 算子融合 | [operator_fusion.md](references/operator_fusion.md) |
| Host 利用率低，NPU 等待 CPU 调度 | Host 开销消减 | [host_overhead_reduction.md](references/host_overhead_reduction.md) |
| 非计算 kernel（format_cast、Transpose）占比高 | 数据布局优化 | 见下文 |
| Decode Free time 高，逐步延迟抖动 | Decode 路径优化 | [decode_optimization.md](references/decode_optimization.md) |
| 训练场景 step time 偶发抖动 | 训练调优 | [training_tuning.md](references/training_tuning.md) |

## 图编译

应**优先尝试**，可彻底消除 Host dispatch 开销。

**步骤**：最简子模块验证 → 逐步扩大范围 → 确认精度和性能。

| 模式 | 适用场景 |
|------|----------|
| `reduce-overhead`（ACLGraph） | 形状固定、Host 调度开销显著 |
| `max-autotune`（GE） | 可接受较长编译时间，追求极致吞吐 |

**放弃条件**（满足任一即回退 eager）：
- 算子不兼容且无替代方案
- 图太大导致编译期 OOM
- 触发无法绕过的框架 bug

**失败诊断思路**：
- 先简后繁：单层 Linear → 加 Norm → 加 Attention → 完整模型，定位是哪个算子导致失败
- 区分失败类型：
  - 算子不支持（GE converter 缺失）→ 查替代算子或 fallback
  - shape 推导失败（tiling illegal）→ dump GE graph 检查节点 shape 是否传播
  - 编译期 OOM → 缩小编译范围（只编译热点子图）
  - multi-stream 不兼容 → 替换为兼容算子（如 FA → FIA）
- 环境排查：确认 torch_npu / CANN / torchair 版本配套；torchair API 有新旧两版（`get_config` vs `CompilerConfig`）

## 算子融合

将多个小 kernel 合并为单个融合 kernel，消除 dispatch gap。

思路：从 profiling 识别连续小算子组 → 查找对应融合算子 → probe 验证可用性 → profiling 确认收益。
注意融合不一定更快（小 shape 时固定开销可能超过收益）。

详见 [operator_fusion.md](references/operator_fusion.md)，NPU 算子 API 细节见 [npu_operator_reference.md](references/npu_operator_reference.md)

## Host 开销消减

减少 Host 在两个 kernel 之间做的事，让 device 流水线保持忙碌。

思路分层：减少调用次数 → 消除运行时分配 → 预计算/prefetch → 消除格式转换 → 清理冗余逻辑。

详见 [host_overhead_reduction.md](references/host_overhead_reduction.md)

## 数据布局优化

选择对 NPU 友好的 tensor 布局，消除算子内部的 format_cast / transpose。

核心原则：
- 理论上更优的布局在 NPU 上可能更慢 — 必须微基准验证
- NPU 上连续写入与非连续写入的性能差异可能很大，决定了 cache 维度顺序
- 统一推理过程的维度约定，避免频繁 squeeze/unsqueeze/reshape

## Decode 路径优化

自回归推理每步计算量小但 host-device 交互不减。

思路：精简 decode 循环 → 优化 KV cache 策略 → 减少同步次数 → 消除冗余计算 → 统一数据格式。
注意 GPU 上的 KV cache 最优策略在 NPU 上可能反转，必须实测。

详见 [decode_optimization.md](references/decode_optimization.md)

## 训练调优

训练场景侧重 OS 级手段：流水优化（TASK_QUEUE_ENABLE）→ CPU 绑核 → tcmalloc 替换。
每次只变更一个配置，观察后再叠加。

详见 [training_tuning.md](references/training_tuning.md)

## CANN 环境调优

- `TASK_QUEUE_ENABLE=2`：Host-Device 异步流水，训练收益显著，推理需实测
- 去除多余全局 hook（如 `transfer_to_npu`）的额外包装开销
- 环境变量必须在 `import torch_npu` **之前**设置

## 通用原则

- 每次优化后重新 Profiling，确认瓶颈是否转移
- GPU 最优实践在 NPU 可能反效果，必须实测验证
- 保留原始实现供 fallback
- 每次验证通过后 git commit
