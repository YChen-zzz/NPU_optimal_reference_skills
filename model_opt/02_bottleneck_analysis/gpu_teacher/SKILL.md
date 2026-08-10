---
name: npu-gpu-teacher-analysis
description: Phase 2 的条件分析子技能；读取 GPU/common/NPU source、已编译 GPU graph/IR/code/profile 和当前 NPU evidence，按 execution regime 与 semantic Supernode 恢复 source-direct gap、compile method guideline 和 runtime gap，再生成可审计的 NPU adaptation 候选。当用户要求 GPU Teacher、提供 GPU profiling/compiled graph，或主 npu-model-optimization 路由启用 Line T 时使用。
---

# Line T：GPU Teacher 分析

本子技能增强 Phase 2 的观察与方法选择。它输出证据、Supernode、GPU optimization method guideline 及 NPU 适配候选；不直接实施代码，也不定义另一套精度流程。Phase 3 使用 [NPU 优化实施](../../03_optimization/SKILL.md)，Phase 4 使用 [精度保证](../../04_accuracy_assurance/SKILL.md)。

## 1. 运行门控

读取 [teacher_gate.md](references/teacher_gate.md)，记录 teacher_required、teacher_auto、teacher_hybrid 或 teacher_unavailable，并分别判断 `source_direct`、`compile_method` 和 `runtime_gap`。

GPU evidence pack 默认由 GPU 机器离线提供。缺失时生成 capture request，不假设 NPU 机器能连接 GPU 机器。证据合同见 [teacher_evidence.md](references/teacher_evidence.md)，实际读取顺序与 claim 账本见 [evidence-reading.md](references/evidence-reading.md)。

## 2. 锁定可比范围

对每个 regime 记录：

- common/GPU/NPU source revision；
- shape、dtype、mask/window/sparsity、work-domain；
- training/eval、forward/backward/optimizer/state；
- batch/累积、world size、rank 和 step window；
- GPU compiled/warm 状态；
- NPU 匹配 profile 与出现频率。

语义关键字段未知时不得把该 regime 提升为强 compile/runtime Teacher 信号；但已定位、正确性合同明确的 source-port gap 可作为 `source_direct` provisional candidate。

## 3. 恢复三类监督信号

先比较 GPU/common source 与 NPU port source，再读 GPU compile 证据。严格区分：

- source_port_gap：NPU 迁移新增、遗漏或改变的 op、dtype、API 参数、work-domain、layout、host 行为、state 或训练/报告分支。
- teacher_compile_delta：GPU compiler 完成的删除、融合、常量折叠、去 materialization、layout canonicalization、buffer/saved-tensor reuse 或 latency hiding。
- teacher_method_guideline：GPU/common source 或 compiler 实际采用的方法、作用对象、成立条件、依赖和精度语义；例如 window/work-domain 缩减、fp32 accumulation boundary、epilogue fusion、buffer reuse、graph capture 或 compute/communication overlap。

沿以下链路记录 provenance：

~~~text
source
→ readable/pre-compile graph
→ transformed graph
→ pre/post-fusion IR
→ generated code
→ compiled runtime
~~~

文件存在不等于已读。实际读取范围、支持/反驳事实、尚未读取的必要证据和不能证明的内容写入 evidence ledger 与 claim map。

## 4. 建立 Supernode

读取 [supernode_alignment.md](references/supernode_alignment.md)。按语义与 state、tensor contract、work-domain/API、依赖与 lifetime、调用频率对齐，kernel 名只作弱证据。

主表：

| Supernode | 原始语义与 regime | 精度合同 | GPU 编译前 | 编译前 GPU↔NPU Gap | GPU 编译后 | GPU Compile 做了什么 | NPU 当前（source + runtime） | 剩余 Gap | 证据/置信度 |
|---|---|---|---|---|---|---|---|---|---|

每个 NPU extra cast/copy/transdata/materialization/sync/graph break 单独记录为 `npu_extra`，不因 GPU 侧无同名 kernel 而忽略。

使用 [runtime-alignment.md](references/runtime-alignment.md) 将 Line B 的 all-rank/regime NPU evidence 映射到 Supernode，计算 exposed gain、rank imbalance 和 direct/enabling gain。Profiling 决定排序与收益置信度，不决定强语义 source-direct 候选能否存在。

## 5. 翻译方法并生成 Action Sheet

对每个有 Gap 的 Supernode，先写明 Teacher 已采用的方法及其成立条件，再拆成“可迁移机制”和“GPU-specific 实现”。针对可迁移机制列出合理 NPU 实现、成立条件和限制，并映射到主流程 [Candidate Contract](../references/candidate_contract.md)。无需机械列满所有路径；跳过更低阶方法必须记录原因。

优先搜索：

1. 修复 source porting、删除或缓存；
2. 官方 NPU API、融合算子、稀疏/work-domain 参数；
3. layout、storage、精度边界或 API 改写；
4. selective compile/graph boundary；
5. buffer、saved tensor、autograd、stream 或 collective schedule；
6. custom kernel。

若 NPU 有同语义官方实现，优先验证；否则选择最接近 Teacher guideline 的 manual/API/layout/compile/schedule 方案。GPU compile delta 说明优化机制，不自动把 compile 提升为 NPU 第一实现。GPU kernel 的 tiling、指令和绝对时间不得直接迁移，但 Teacher 的算法、work-domain、融合边界、精度边界、复用与调度方法本身是强先验。

高价值 Supernode 可忠实隔离且一轮对照成本不超过一次短跑时，Action Sheet 默认要求 Phase 3 运行 [Supernode Lab](../../03_optimization/references/supernode_lab.md)；不可运行时写明精确原因。

## 6. 与 Line A/B 合并

Line T 候选必须引用当前 NPU source 和 runtime：

- Teacher 证明差异，但 NPU 未显示关键路径暴露：保留为低优先级或补测候选。
- NPU 显示热点，但 Teacher 无对应信息：交给 Line A/B，不强行构造 Teacher 解释。
- 三线共同证明相同根因：合并并提高 evidence grade。
- 同名 fused op 但 work-domain、dtype、saved tensor 或 schedule 不同：不得判等价。

Line T 结束时写入 teacher_gate.json、coverage.csv、supernodes.csv、evidence ledger、claim map 和统一 candidates。

## 7. 重新对齐

一次 Teacher 对齐应支撑多个 Action。当前 backlog 仍有显著收益时不重新读取全部 GPU pack。

仅在主流程判定收益停滞、当前 Supernode map 无法解释新热点，或 graph/dtype/layout/state/communication 变化使旧证据失效时，采新 NPU profile 并做 residual Teacher alignment。GPU source/compiler/regime 未改变时复用原 Teacher pack。
