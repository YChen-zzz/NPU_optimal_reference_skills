---
name: npu-bottleneck-analysis
description: 自动生成 NPU 性能优化候选；始终执行源码结构分析和 NPU Profiling 分析，在用户要求 GPU Teacher 或检测到可用的编译后 GPU evidence pack 时，自动路由到 GPU Teacher Supernode 对齐并合并三条证据线。用于定位训练、推理或科学计算负载的关键路径、源码根因和高价值 Action。
---

# Phase 2：瓶颈分析与候选路由

本阶段负责“发现什么值得改、GPU/编译器已采用什么方法、该方法怎样迁移到 NPU、依据是什么、收益上限多大”。具体实现进入 Phase 3，正确性和性能验收进入 Phase 4。

## 路由

先校验 Phase 1 产物：workload/precision contract、regime map、NPU baseline、当前 NPU source/profile、Git baseline 和 GPU Teacher pack 状态。

| 路由 | 条件 | 执行 |
|---|---|---|
| profiling_only | 没有可用 Teacher，且用户未要求 | Line A + Line B |
| teacher_auto | 检测到已编译、已 warmup、可比的 Teacher pack | Line A + Line B + Line T |
| teacher_required | 用户明确要求使用 GPU Teacher | Line A + Line B + Line T；缺证据时生成精确 capture request |
| teacher_hybrid | Teacher 只有部分 regime/provenance，但存在可用信号 | A/B 全量执行，T 只覆盖有证据的范围 |

用户明确说“teacher-only”时，关键 Teacher 证据缺失则停止 Line T 并报告缺口；否则不因 Teacher 不可用而阻塞 Line A/B。

Line T 的资格、信号强度和降级规则见 [GPU Teacher 分析](gpu_teacher/SKILL.md)。

## Line A：源码结构分析

读取 [proactive_source_analysis.md](references/proactive_source_analysis.md)，穿透 wrapper 到真实计算路径。

1. 建立模型结构、实现逻辑、算法和 state mutation 图。
2. 按每个 regime 审计热路径操作。
3. 从去重、复用、掩盖、替换四个维度发现结构性机会。
4. 使用 Line B 的 NPU 数据量化调用频率、设备时间、host 暴露和内存影响。
5. 输出源码候选；候选必须包含文件/函数/行、适用 regime 和影响范围。

Line A 不因 Profiling 未显示单个大热点而停止；同源独立调用、重复计算、错误 work-domain、dtype/reporting 混用和跨步可复用结果常需要源码语义才能发现。

## Line B：NPU Profiling 分析

1. 校验 regime/rank/step coverage 和采集口径。
2. 读取 [bound_analysis.md](../references/bound_analysis.md)，计算 NPU 三档下界与 gap A/B。GPU 时间不能作为 NPU floor。
3. 对当前 L1 运行 scripts/run_analysis.py <l1_dir> --l0-dir <l0_dir>。
4. 使用 [profiling_to_action.md](references/profiling_to_action.md) 的横向关联和纵向深入，从显著信号追到源码根因。
5. 对所有 DEFINITE/WARNING 信号生成候选，或附具体数据做排除。
6. 将 device 时间转为 critical-path exposed time；host self time、collective 总时长和 kernel duration 不直接等于收益。

解析器说明见 [profiling_scripts_guide.md](references/profiling_scripts_guide.md)。阈值是工作负载相关默认值，不能代替 Agent 判断。

## Line T：GPU Teacher 监督信号

当路由启用时，读取 [gpu_teacher/SKILL.md](gpu_teacher/SKILL.md)。

Line T 额外回答：

- NPU port 是否丢失或改变了 GPU/common source 已有的 dtype、API 参数、work-domain、layout、state 或训练/报告分支；
- GPU compiler 从 pre 到 post 删除、融合、折叠、去 materialization、复用或隐藏了什么；
- GPU/common source 或 compiler 使用了哪种优化方法，该方法依赖哪些成立条件；
- 当前 NPU 是否仍在为对应工作付出关键路径时间；
- 哪些机制可直接迁移，哪些只适用于 GPU；同一意图在 NPU 上应走哪一级实现路径。

Line T 分别产出 `source_direct`、`compile_method` 和 `runtime_gap` claim。强 source-port gap 可在精确 NPU exposed time 未完成时生成 provisional candidate；Line B 的 timing 用于排序、收益置信度和最终采纳，不用于抹掉已由语义证据成立的候选。

Line T 必须输出 Supernode 表和带 method guideline 的 Action Sheet，不直接修改代码。

## 合并三条线

按 semantic role、source location、tensor contract 和 regime 合并候选，不按名称拼接。

- 同一根因被多条线发现：合并为一个候选，提高 evidence grade，并保留各来源。
- Teacher 证明 compile 意图/已用方法、Line B 证明 NPU 暴露、Line A 证明源码位置和适用条件：优先形成强候选，并优先测试其 NPU 翻译方案。
- Line T 没覆盖的 NPU 独有问题继续由 A/B 处理。
- 三条线结论冲突：降低置信度，先补最小证据或 microbenchmark，不直接改代码。

所有候选遵循 [Candidate Contract](references/candidate_contract.md)。先完成每个 Supernode/根因的合理 Action Sheet，再做全局排序。

## Phase 2 完成条件

进入 Phase 3 前必须满足：

- A/B 已执行；T 的路由决定和证据状态已记录；
- 每个显著发现都有候选或证据化排除；
- 每个候选包含目标 gap、regime、证据、实现路径、成本、风险和验证门；直接收益已测量/保守估计，或对 source-direct 明确标记 unmeasured 并给出最小计时动作；
- 候选已去重、处理依赖/冲突，并完成全局排序；
- evidence_db/candidates.csv 或等价 JSONL 已写入。

一次 Phase 2 产出的候选集合构成一个 wave backlog。Phase 3 沿 backlog 持续执行；不因接受一个 Action 就重新运行本阶段。
