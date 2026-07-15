---
name: NPU 瓶颈分析
description: 分析昇腾 NPU 性能瓶颈——包含源码结构分析(主动发现结构性冗余)和 Profiling 数据分析(定位可见瓶颈)两条线。当用户要求分析性能、定位瓶颈、查看 profiling 数据、分析源码优化机会时触发。
---

# NPU 瓶颈分析

## 双线分析模型

Phase 2 的分析由两条线驱动,顺序执行:

| | Line B: Profiling 分析 | Line A: 源码分析 |
|---|---|---|
| 起点 | profiling 数据中的异常 | 源码的计算结构 |
| 发现 | 可见瓶颈(算子慢/空闲/等待) | 结构性冗余(可合并/可复用/可预计算) |
| 方法 | 五种分析模式推理 | 四维度审视源码 |

**执行顺序**:

```
1. 采集 profiling(同时可开始通读源码)
2. Line B:跑脚本 → 五种模式推理 → 定位可见瓶颈 → 产出候选
3. Line A:通读源码(穿透框架) → 四维度审视 → 用 Line B 数据量化 → 产出候选
4. 合并两条线的候选 → ★A 用户确认
```

**关键约束**:
- Line B 和 Line A **都必须执行**,不设跳过条件——Line B 覆盖可见瓶颈,Line A 覆盖 profiling 盲区
- Line B 先做(脚本快,秒出结果),其数据供 Line A 做量化
- 两条线的产出都是"问题定位"(问题 + 位置 + 影响范围),不是方案

## Line A: 源码分析

**流程**:
1. 穿透框架层,定位真实的模型实现代码(跳过 generate/pipeline/Module.__call__ 等 wrapper)
2. 通读推理路径,建立对三层的认知:模型结构(架构组成) → 实现逻辑(数据流/生命周期/控制流) → 算法(计算方法)
3. 对每个计算逻辑块用四维度提问:去重(有没有多余)?复用(有没有浪费)?掩盖(能不能并行)?替换(有没有更好的写法)?
4. 命中的优化机会用 profiling 数据量化(op_statistic/operator_memory 确认占比),占比 <1% 的跳过
5. 产出"源码问题候选清单"(每条含:问题描述 + 源码位置 + 影响范围)

详见 [proactive_source_analysis.md](references/proactive_source_analysis.md)。

## Line B: Profiling 分析

**流程**:
1. 采集 profiling(L1,不足时 L2)
2. 跑脚本提取结构化数据(按下方典型工作流顺序)
3. 用五种分析模式推理:从脚本输出的信号组合中定位瓶颈类型 → 详见 [profiling_to_action.md](references/profiling_to_action.md)
4. 用桥梁字段从 profiling 跨到源码:定位根因的具体代码位置 → 详见 [profiling_to_source.md](references/profiling_to_source.md) + [source_code_analysis.md](references/source_code_analysis.md)
5. 确认根因后,归入优化维度(去重/复用/掩盖/替换)

**核心原则**:
- 不要停在"什么慢"——必须追到"为什么慢"(源码根因)
- 向上追溯调用链(谁调了它、循环几次)
- 向下深入实现(内部有没有浪费)
- 区分必要 vs 冗余(同一操作可能既有必要调用也有冗余调用)

**深入参考**(按需加载):
- [host_bound_patterns.md](references/host_bound_patterns.md) — Host-Bound 深度诊断
- [memory_profiling.md](references/memory_profiling.md) — 显存峰值分析
- [profiling_scripts_guide.md](references/profiling_scripts_guide.md) — 脚本详细使用指南

### 瓶颈分类

| 类型 | Profiling 表现 | 核心问题 |
|------|---------------|---------|
| **Host-Bound** | 设备利用率低（Free >> Computing） | host dispatch、Python 开销、同步 |
| **Compute-Bound** | 利用率高，kernel 耗时大，mac_ratio 高 | 算子本身计算密集 |
| **Memory-Bound** | 利用率高但 mte_ratio >> mac_ratio | HBM 带宽瓶颈 |
| **Allocator-Bound** | 类似 Host-Bound 但 empty_tensor 占比高 | allocator 同步阻塞 |

### 典型工作流

> 脚本位于本 skill 的 `scripts/` 目录。以下 `$S` 代表该目录。

```
$S/parse_step_trace.py <dir>         → 判断瓶颈侧(host or device)
$S/parse_op_statistic.py <dir>       → 哪类算子最耗时
$S/parse_kernel_details.py <dir>     → 硬件单元、小算子、流水 stall
$S/parse_trace_view.py <dir>         → host→device 下发链、device 空隙、在线编译
$S/parse_communication.py <dir>      → 多卡通信:时间分解、带宽、同步等待
$S/parse_memory_record.py <dir>      → 内存峰值、碎片化
$S/parse_operator_memory.py <dir>    → tensor 生命周期、parallelism trigger
```

## 下一步

分析完成后,整理优化建议清单(Line A + Line B 合并),进入主 SKILL.md 的**确认节点 A**。
