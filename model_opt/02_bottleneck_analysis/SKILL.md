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

1. 采集 profiling(同时可开始通读源码)
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
1. 采集 profiling(L1,不足时 L2)
2. 跑脚本提取结构化数据(按下方脚本检查清单顺序)。脚本输出含义详见 [profiling_scripts_guide.md](references/profiling_scripts_guide.md)
3. 用五种分析模式推理:从脚本输出的信号组合中定位瓶颈类型 → 详见 [profiling_to_action.md](references/profiling_to_action.md)
4. **根因追踪（强制，覆盖所有显著发现，不可跳过）**：

   8 个脚本产出的每一类信息中，每个**显著发现**都必须追溯到源码根因。"显著"的判定标准 = 脚本自身输出的 DEFINITE 信号 / WARNING 警告，或占比超过脚本定义的阈值。

   **方法论**：每个显著发现都是 profiling 层面的"症状"——一个可观测的现象（如某算子开销高、设备空闲、内存抖动）。根因追踪分两步：
   - **定位**：通过 [profiling_to_source.md](references/profiling_to_source.md) 定义的桥梁（Call Stack、Input Shapes、AI Core 指标、Accelerator Core、下发时序）从 profiling 数据定位到**源码中的具体代码位置**。
   - **分析**：定位到源码后，按 [profiling_to_action.md](references/profiling_to_action.md) 模式 2（纵向深入）的方法沿调用链追溯（向上找谁调用、向下看内部实现），通过归因层判断根因属于哪类浪费，回答：**这段代码为什么导致了这个 profiling 现象？**

   桥梁的选择取决于发现类型——agent 需要自行判断哪个桥梁（或哪些桥梁的组合）能将当前发现连接到源码。[profiling_to_source.md](references/profiling_to_source.md) 定义了五座桥及其适用条件，断桥时的降级路径也有说明。

   **参考示例**（非穷举，仅为说明不同发现类型可能需要不同的追踪路径）：

   - 算子开销高（来自 op_statistic / operator_details）→ `--filter <op>` 获取 Call Stack → 追溯到调用该算子的 Python 函数 → 判断是必要计算还是框架内部操作
   - 设备空闲 / 流间隙（来自 step_trace / trace_view）→ trace_view 的 Host2Device Bound Regions + async_npu flow 回连 cpu_op Call Stack → 定位哪段 Python 代码导致设备等待
   - Host 开销分类中的"other"占比高（来自 operator_details）→ 该类别是未归类的 host 操作聚合 → 按 host self time 排序找到具体算子 → `--filter` 追 Call Stack
   - 内存高频抖动（来自 memory_record / operator_memory）→ 重复同尺寸分配列表 → 对应算子的 Call Stack → 定位哪个操作在反复分配/释放
   - AI_CPU 回退（来自 kernel_details Accelerator Core）→ `--filter <op>` 获取 Input Shapes → 判断 dtype/shape 是否不匹配导致 fallback

   **执行规则**：
   - 每个脚本运行后，先记录该脚本产出的所有 DEFINITE/SIGNAL 信号
   - 对每个信号，选择能将其连接到源码的桥梁（或多桥梁组合），执行根因追踪
   - 追踪产出格式：`发现来源 | 发现内容 | 使用的桥梁 | 源码位置 | 根因 | 候选方案`
   - 如果追踪过程中发现了**之前未识别的优化机会**，必须加入候选清单

   **门禁规则**：
   - 所有脚本的 DEFINITE 信号和 WARNING 警告全部完成根因追踪后才能进入候选合并
   - 不得用"这个信号看起来不重要"跳过追踪——脚本的 DEFINITE/WARNING 标记是脚本自身定义的显著性判断，agent 不可覆盖
   - 追踪到的根因如果是"框架内部操作"，必须进一步追到"是框架的哪段代码导致的"

5. 确认根因后,产出候选清单(每条含:问题 + 位置 + 影响范围 + 量化上限)

### 强制脚本检查清单（确认节点 A 前必须完成）

> 脚本位于本 skill 的 `scripts/` 目录。以下 `$S` 代表该目录。
> **每个脚本必须运行**。跳过任何脚本需在确认节点 A 中说明理由。
> **必读参考**列的文件在运行对应脚本后**必须加载**——不是"按需"，是"绑定"。

| # | 脚本 | 作用 | 必读参考（运行后加载） |
|---|------|------|---------------------|
| 1 | `$S/parse_step_trace.py <dir>` | 判断瓶颈侧(host or device) | — |
| 2 | `$S/parse_op_statistic.py <dir>` | 哪类算子最耗时 | — |
| 3 | `$S/parse_kernel_details.py <dir>` | 硬件单元、小算子、流水 stall | — |
| 4 | `$S/parse_trace_view.py <dir>` | host→device 下发链、device 空隙、host2device bound 区段、资源利用率 counter 时间线、流并发/掩盖、idle 成因分解、在线编译 | [host_bound_patterns.md](references/host_bound_patterns.md) |
| 5 | `$S/parse_operator_details.py <dir>` | Call Stack 定位源码、host self duration | [profiling_to_source.md](references/profiling_to_source.md) |
| 6 | `$S/parse_memory_record.py <dir>` | 内存峰值、碎片化、高频抖动 | [memory_profiling.md](references/memory_profiling.md) |
| 7 | `$S/parse_operator_memory.py <dir>` | tensor 生命周期、重复同尺寸分配 | [memory_profiling.md](references/memory_profiling.md) |
| 8 | `$S/parse_api_statistic.py <dir>` | CANN 运行时 API 开销（memory-mgmt/sync/tiling/launch 分解） | — |

> 多卡场景额外运行 `$S/parse_communication.py <dir>` 分析通信开销（HCCL all-reduce/all-gather 等）。

**门禁规则**：
- 每个脚本运行后，写一行发现摘要（如"step_trace: Host-Bound, 利用率 8%"）
- 脚本输出中的任何 **DEFINITE** 信号或 **WARNING 警告**（由脚本自身定义，如"SEVERE Host-Bound"、"AI_CPU detected"、"高频抖动"）**必须**在确认节点 A 中产生对应候选，或附 profiling 数据依据显式排除
- 未运行脚本 6-7 时，禁止在候选中包含"内存管理阻塞"类问题（无数据支撑）；运行后若 `memory_profiling.md` 中的模式匹配到脚本输出，则对应问题为必选候选

## 下一步

分析完成后,整理优化建议清单(Line A + Line B 合并),进入主 SKILL.md 的**确认节点 A**。
