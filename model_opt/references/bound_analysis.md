# 下界分析：优化空间在哪、最多能改善多少

> 跨阶段分析模块。Phase 2 用它确定优化方向，residual audit 用它判断是否终局。回答两个问题：优化空间在哪一层？最多能改善多少？

## 三档下界

| 下界 | 含义 | 数据来源 |
|------|------|---------|
| **Tier 1: Roofline** | 硬件物理极限——纯计算或纯带宽限制下的最快时间 | 模型结构（权重大小、FLOPs 估算）+ 硬件规格（peak compute, HBM bandwidth） |
| **Tier 2: L0 Computing** | 设备执行下界——实际所有 kernel 执行时间之和 | L0 profiling 的 step_trace Computing 值 |
| **Tier 3: 对齐 wall-clock** | 实际端到端时间——无 profiler 的真实性能 | wall-clock benchmark（计时范围与 L0/L1 对齐，见 [profiling_collection.md](../01_preparation/references/profiling_collection.md) §三种性能测量及其覆盖范围） |

Tier 1 不需要精确——目的是给出物理极限的量级，不是精确预测。对主要计算操作取 `max(FLOPs / peak_compute, data_bytes / HBM_bandwidth)` 即可。

## gap 分解

两个 gap 各自指向不同的优化层：

- **gap A = Tier 2 − Tier 1**：kernel 实现效率 gap（tiling、occupancy、小 kernel 无法饱和带宽等）。不在 Python 应用层可直接优化的范围内——需 CANN/OPP 算子级优化、图编译或量化。
- **gap B = Tier 3 − Tier 2**：host 开销 gap（Python dispatch、同步、内存分配等未被异步流水线重叠的部分）。Python 层可优化（flat forward、合并调用、融合算子等）。

**gap B 是 Python 层优化的收益上限。** 当 wall-clock ≈ L0_Computing（gap B → 0）时，Python 层优化空间耗尽。

## 方向判定

| gap B / Tier 3 | 判定 | 优化方向 |
|----------------|------|---------|
| > 15% | host 开销显著 | 优先 host 侧优化：减少 dispatch 次数（合并/flat forward）、消除冗余调用、融合算子 |
| < 5% | host 开销已极小 | 只能缩小 gap A：融合算子（减少 kernel 数量提升带宽利用率）、换算法、图编译、量化 |
| 5%~15% | 两条路都有空间 | 按具体候选的反事实收益上限排序 |

## 停滞与路由

单个下界或失败信号只能关闭某类搜索空间，不能证明 workload 已经最优：

| 信号 | 可得结论 | 强制动作 |
|---|---|---|
| `wall_clock / L0_Computing < 1.1` | Host gap 接近耗尽 | 转向 graph/compiler/kernel/算法层，继续分析 gap A |
| gap A 主导且 gap B / Tier 3 < 5% | kernel/compiler efficiency 主导 | 检查融合、图编译、官方算子、work-domain、layout、精度与 custom kernel 上限 |
| 连续 wave 收益低于阈值 | 当前观察或 backlog 停滞 | 采 current best NPU L1，重新生成候选 |
| 当前候选全部失败或处理完 | 已有候选耗尽 | 做候选覆盖审计，不得等同全局空间耗尽 |
| Teacher backlog 用完 | 当前 Teacher 翻译已处理 | 采新 NPU profile，并用原 GPU pack 做 residual Teacher alignment |
| 环境无法执行候选 | 当前环境阻塞 | 标记 `blocked_environment` 和恢复 predicate，不得声称最优 |

融合算子可同时缩小 gap A 和 gap B；“compute-bound”判断前仍须检查 D2H、格式转换、编译、work-domain 和碎片化 kernel，不能仅看 utilization。

## 终局证明

只有 [执行协议](execution_protocol.md) 的停止门全部通过，且独立 Stop Auditor 返回 `stop_allowed`，才能结束优化。`host_layer_exhausted`、`stalled`、`coverage_audit_required`、`teacher_residual_required` 和 `blocked_environment` 都不是终局。
