---
name: npu-bottleneck-analysis
description: 瓶颈分析：profiling 数据分析 + 源码结构分析双线定位性能瓶颈。当用户需要分析性能瓶颈、查看 profiling 数据、定位慢的根因、或分析源码优化机会时触发。
---

# NPU 瓶颈分析

## 双线分析模型

Phase 2 的分析由两条线驱动,顺序执行:

| | Line B: Profiling 分析 | Line A: 源码分析 |
|---|---|---|
| 起点 | profiling 数据中的异常 | 源码的计算结构 |
| 发现 | 可见瓶颈(算子慢/空闲/等待) | 结构性问题(可合并/可复用/可预计算/可替换) |
| 方法 | 分析推理 | 多维度审视源码 |

**执行顺序**:

1. 采集 profiling：训练场景使用短跑脚本（见 [profiling_collection.md](../01_preparation/references/profiling_collection.md) §训练短跑策略），推理场景直接采集（同时可开始通读源码）
2. Line B:跑脚本 → 基于脚本的信息分析进行推理 → 定位可见瓶颈 → 产出候选
3. Line A:通读源码(穿透框架) → 多维度审视 → 用 Line B 数据量化 → 产出候选
4. 合并两条线的候选 → ★A 用户确认

**关键约束**:
- Line B 和 Line A **都必须执行**,不设跳过条件——Line B 覆盖可见瓶颈,Line A 覆盖 profiling 盲区
- Line B 先做(脚本快,秒出结果),其数据供 Line A 做量化
- 两条线的产出都是"问题定位"(问题 + 位置 + 影响范围),不是方案

## Line A: 源码分析

**流程**:
1. 穿透框架层,定位真实的模型实现代码(跳过 generate/pipeline/Module.__call__ 等 wrapper)
2. 通读推理路径,建立对三层的认知:模型结构(架构组成) → 实现逻辑(数据流/生命周期/控制流) → 算法(计算方法)
3. 对每个计算逻辑块用四维度提问:去重(有没有多余)?复用(有没有浪费)?掩盖(能不能并行)?替换(有没有更好的写法)?
4. 命中的优化机会用 profiling 解析的数据进行量化
5. 产出"源码问题候选清单"(每条含:问题描述 + 源码位置 + 影响范围)

详见 [proactive_source_analysis.md](references/proactive_source_analysis.md)。

## Line B: Profiling 分析

**流程**:
1. 采集 L1 profiling
2. **下界分析（新增前置步骤）**：在跑脚本之前，先计算三档下界，确定本轮优化方向。详见 [bound_analysis.md](../references/bound_analysis.md)。

   - **Tier 1 (Roofline)**：估算模型关键操作的物理极限下界——对主要计算操作，取 `max(FLOPs / peak_compute, data_bytes / HBM_bandwidth)`。目的是给出物理极限的量级，不需要精确。
   - **Tier 2 (L0 Computing)**：从当前轮 L0 profiling 的 step_trace 读取 Computing 值——实际所有 kernel 的执行时间之和。
   - **Tier 3 (对齐 wall-clock)**：训练场景取短跑脚本的 step_avg（见 [profiling_collection.md](../01_preparation/references/profiling_collection.md) §训练短跑策略）；推理场景从 benchmark 读取——计时范围必须与 L0/L1 覆盖的代码范围完全一致（见 [profiling_collection.md](../01_preparation/references/profiling_collection.md) §三种性能测量及其覆盖范围）。
   - **gap 分析**：gap A = Tier2 − Tier1（kernel 实现效率 gap，Python 层不可直接优化）；gap B = Tier3 − Tier2（host 开销 gap，Python 层可优化）。
   - **方向判定**：gap B / Tier3 > 15% → host 侧优化优先（减少 dispatch/同步/冗余调用）；gap B / Tier3 < 5% → host 开销已极小，只能缩小 gap A（融合算子/换算法/图编译）；中间区间两者都有空间。

3. 运行 `run_analysis.py`（统一入口）提取结构化数据，输出一份按 A~H 节归类的完整报告。报告 A 节自动包含 L0/L1 交叉验证（传入 `--l0-dir` 时对比 L0 和 L1 的 step_trace，未传入时标注"未经交叉验证，须谨慎"）。L0 来源：第 0 轮用 Phase 1 基线 L0；第 i 轮用第 i-1 轮 Phase 4 的 L0。各脚本输出含义详见 [profiling_scripts_guide.md](references/profiling_scripts_guide.md)
4. **推理与根因追踪（强制，覆盖所有显著发现，不可跳过）**：阅读完整报告后，用 [profiling_to_action.md](references/profiling_to_action.md) 的两种分析模式（横向关联 + 纵向深入）从信号组合定位瓶颈类型（现象→归因），再通过三座桥（Call Stack、Input Shapes、下发时序）从 profiling 数据定位到**源码中的具体代码位置**，沿调用链追溯根因。定位到源码后回答：**这段代码为什么导致了这个 profiling 现象？**

   "显著"的判定标准 = 脚本自身输出的 DEFINITE 信号 / WARNING 警告，或占比超过脚本定义的阈值。

   **参考示例**（非穷举，仅为说明不同发现类型可能需要不同的追踪路径）：

   - 算子开销高（来自 op_statistic / operator_details）→ `--filter <op>` 获取 Call Stack → 追溯到调用该算子的 Python 函数 → 判断是必要计算还是框架内部操作
   - 设备空闲 / 流间隙（来自 step_trace / trace_view）→ trace_view 的 Host2Device Bound Regions + async_npu flow 回连 cpu_op Call Stack → 定位哪段 Python 代码导致设备等待
   - Host 开销分类中的"other"占比高（来自 operator_details）→ 该类别是未归类的 host 操作聚合 → 按 host self time 排序找到具体算子 → `--filter` 追 Call Stack
   - 内存高频抖动（来自 memory_record / operator_memory）→ 重复同尺寸分配列表 → 对应算子的 Call Stack → 定位哪个操作在反复分配/释放
   - AI_CPU 回退（来自 kernel_details Accelerator Core）→ `--filter <op>` 获取 Input Shapes → 判断 dtype/shape 是否不匹配导致 fallback

   **执行规则**：
   - 每个脚本运行后，先记录该脚本产出的所有 DEFINITE/WARNING 信号
   - 对每个信号，选择能将其连接到源码的桥梁（或多桥梁组合），执行根因追踪
   - 追踪产出格式：`发现来源 | 发现内容 | 使用的桥梁 | 源码位置 | 根因 | 候选方案`
   - 如果追踪过程中发现了**之前未识别的优化机会**，必须加入候选清单

   **门禁规则**：
   - 所有脚本的 DEFINITE 信号和 WARNING 警告全部完成根因追踪后才能进入候选合并
   - 不得用"这个信号看起来不重要"跳过追踪——脚本的 DEFINITE/WARNING 标记是脚本自身定义的显著性判断，agent 不可覆盖
   - 追踪到的根因如果是"框架内部操作"，必须进一步追到"是框架的哪段代码导致的"

5. 确认根因后,产出候选清单(每条含:问题 + 位置 + 影响范围 + 反事实收益上限)。候选评估方法见 [profiling_to_action.md](references/profiling_to_action.md) §候选评估：反事实收益上限。

如需对单个脚本做 `--filter` 深入查询（如 `parse_operator_details --filter Transpose` 获取 Call Stack），可单独调用对应脚本。

**门禁规则**：
- 报告中任何 **DEFINITE** 信号或 **WARNING 警告**（由脚本自身定义，如"SEVERE Host-Bound"、"AI_CPU detected"、"高频抖动"）**必须**在确认节点 A 中产生对应候选，或附 profiling 数据依据显式排除

## 下一步

分析完成后,整理优化建议清单(Line A + Line B 合并),进入主 SKILL.md 的**确认节点 A**。
