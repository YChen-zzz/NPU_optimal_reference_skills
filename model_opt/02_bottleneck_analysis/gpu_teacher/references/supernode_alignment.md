# Semantic Supernode 对齐

Supernode 是跨 compiler/backend 仍保持明确语义和可实施边界的最小区域。它可能对应一个源码表达式、多个 graph node、一个 GPU fused kernel、多个 NPU operator，或一段 compute/collective schedule。

## 构造

从 workload 语义边界开始，不从 kernel 名开始。先比较 common/GPU source 与 NPU port source，再追踪 GPU pre/post compile。

NPU 匹配优先级：

1. source location、semantic range、state mutation；
2. dependency neighborhood；
3. input/output shape、dtype、stride、layout；
4. regime-specific 调用次数；
5. operator/module/call stack；
6. correlated kernel/collective；
7. 名称相似性。

显式保留一对多和多对一映射。

## 主表字段

| 字段组 | 内容 |
|---|---|
| 语义 | formula、state、mask/window/sparse、regime |
| 精度 | storage、compute、accumulator、output、saved tensor、rounding、reporting |
| Porting | GPU/common source、NPU source、source_port_gap、API/work-domain 差异 |
| Teacher pre | op、node、materialization、saved tensor |
| Teacher post | kernel/fusion、layout、buffer/saved-tensor reuse、lifetime |
| Compile delta | eliminated、fused、folded、work reduction、removed boundary、hidden latency |
| Method prior | GPU 已用方法、作用对象、成立条件、依赖、可迁移机制、GPU-specific 部分 |
| NPU current | source、op/kernel、materialization、graph break、time、exposed time |
| Frequency | calls/step、regime occurrences、rank |
| Evidence | artifact、read scope、grade、confidence |
| Decision | candidate、status、trial、failure signature |

人可读表固定使用：

| Supernode | 原始语义与 regime | 精度合同 | GPU 编译前 | 编译前 GPU↔NPU Gap | GPU 编译后 | GPU Compile 做了什么 | NPU 当前（source + runtime） | 剩余 Gap | 证据/置信度 |
|---|---|---|---|---|---|---|---|---|---|

## Gap 分类

- porting_artifact
- semantic_diff
- elimination
- fusion
- work_domain
- precision
- layout
- reuse
- dispatch
- synchronization
- communication
- memory_lifetime
- compiler_boundary
- hardware_reference

其他类别未审计完前，不得用 hardware_reference 结束分析。

## 证据等级

- A：source/provenance、tensor contract、相关 regime/rank 和 measured interval 均匹配；
- B：语义与 tensor contract 匹配，但 timing、provenance 或 coverage 不完整；
- C：只有名称、shape 或架构假设。

A 与强 B 可进入第一批高收益低风险 bundle；C 先补证或 microbenchmark。

## Action Sheet

每个 Gap 允许多个 NPU 实现。记录：

- target gap/compile intent；
- teacher method guideline、成立条件与 method confidence；
- transferable mechanism 与 GPU-specific exclusions；
- mechanism 与 implementation path；
- regime scope；
- weighted exposed gain；
- platform support；
- difficulty；
- precision/memory/multi-rank risk；
- Phase 4 validation gate；
- dependencies/conflicts。

完成各 Supernode Action Sheet 后再全局排序，不能看到第一个 GPU fusion 就直接实施。
