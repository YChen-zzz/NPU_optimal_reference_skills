---
name: NPU 优化实施
description: 在昇腾 NPU 上实施模型调优。当用户要求优化性能或实施优化方案时触发。
---

# NPU 优化实施

## 总体策略

根据 Profiling 分析结果选择优化方向，每次优化后重新 Profiling 确认瓶颈是否转移。

**图编译优先尝试**（收益上限最高），失败后走 eager 模式优化。

### 优化三原语

所有具体的性能优化手段，本质上只做三件事：

| 原语 | 核心问题 | 识别线索 |
|------|---------|---------|
| **去重** | "这个工作是必要的吗？能和相邻工作合并吗？" | 同类算子调用次数异常多；代码中存在可合并的独立调用 |
| **复用** | "这个结果之后还会被需要吗？" | MemSet / empty_tensor 次数多；同一结果被反复计算/分配 |
| **掩盖** | "这段延迟能和其他工作并行吗？" | timeline 中通信/计算串行；Host 等待 Device 或反之 |

详见各原语文档中的原理推导和示例。

## 参考文档索引

| 文档 | 内容 | 加载时机 |
|------|------|---------|
| [eliminate_redundancy.md](references/eliminate_redundancy.md) | 去重原理：合并、消除、清理 | 需要减少 kernel 数、清理冗余操作时 |
| [reuse_and_precompute.md](references/reuse_and_precompute.md) | 复用原理：缓存、预分配、原地操作 | 需要减少内存分配、消除重复计算时 |
| [hide_latency.md](references/hide_latency.md) | 掩盖原理：通信-计算重叠、流水线 | 需要隐藏通信延迟、减少 pipeline bubble 时 |
| [parallel_design.md](references/parallel_design.md) | 多卡并行方案设计方法论 | 单卡 OOM、需要设计张量切分方案时 |
| [npu_checklist.md](references/npu_checklist.md) | NPU 已知问题主动扫描清单 | 每次接到优化任务时，不依赖 profiling 即可扫描 |
| [npu_operator_reference.md](references/npu_operator_reference.md) | NPU 融合算子 API 速查 | 需要查找具体算子签名和注意事项时 |
| [decode_optimization.md](references/decode_optimization.md) | 自回归 Decode 路径专题 | Decode 场景性能问题时 |
| [training_tuning.md](references/training_tuning.md) | 训练场景 OS 级调优配置 | 训练 step time 抖动、需要开环境变量时 |

## 图编译

应**优先尝试**，可彻底消除 Host dispatch 开销。

**步骤**：最简子模块验证 → 逐步扩大范围 → 确认精度和性能。

| 模式 | 适用场景 |
|------|----------|
| `reduce-overhead`（ACLGraph） | 形状固定、Host 调度开销显著 |
| `max-autotune`（GE） | 可接受较长编译时间，追求极致吞吐 |

**放弃条件**（满足任一即回退 eager）：算子不兼容、图太大编译期 OOM、触发框架 bug。

**编译范围策略**：
- 不建议直接 `torch.compile(model)`——通信、控制流、side-effect 都会切图
- **核心原则**：挑"碎而密"的地方编，不挑"大而炸"的地方编
- 从纯计算子模块开始（Transition、LayerNorm + Linear 链路）
- 大 attention / 大 matmul 主体留在图外
- 含 HCCL 通信的 block 不编译
- 编译后必须验证精度

## CANN 环境配置

- `TASK_QUEUE_ENABLE=2`：Host-Device 异步流水
- `CPU_AFFINITY_CONF=1`：CPU 绑核
- `LD_PRELOAD=libtcmalloc.so`：高性能 malloc
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`：减少碎片
- `HCCL_BUFFSIZE=32`：通信缓冲区大小
- 以上环境变量必须在 `import torch_npu` **之前**设置

## 通用原则

- 每次优化后重新 Profiling，确认瓶颈是否转移
- GPU 最优实践在 NPU 可能反效果，必须实测验证
- 保留原始实现供 fallback
- 每次验证通过后 git commit
- 权重修改须保持 `state_dict` key 不变，确保 checkpoint 加载兼容
- 优化尝试失败也要记录（what + why + 实际效果），避免重复踩坑
