# 图编译与 CANN 环境配置

## 图编译

应**优先尝试**，可彻底消除 Host dispatch 开销（属于"掩盖"原语的极端形态）。

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

以下环境变量必须在 `import torch_npu` **之前**设置：

| 变量 | 值 | 作用 |
|------|---|------|
| `TASK_QUEUE_ENABLE` | `2` | Host-Device 异步流水，消除逐算子同步等待 |
| `CPU_AFFINITY_CONF` | `1` | CPU 绑核，减少调度抖动 |
| `LD_PRELOAD` | `libtcmalloc.so` | 高性能 malloc，减少 Python 内存分配开销 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 减少 NPU 内存碎片 |
| `HCCL_BUFFSIZE` | `32` | 通信缓冲区大小（MB），多卡场景 |

**注意**：`TASK_QUEUE_ENABLE=2` 是所有 eager 优化的前提——不开启则每个 kernel 都要等 host 确认，profiling 中表现为 wait time 均匀分布在所有 kernel 上。
