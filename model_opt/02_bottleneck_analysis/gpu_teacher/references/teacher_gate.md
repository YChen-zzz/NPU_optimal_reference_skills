# GPU Teacher 路由与信号门控

门控分开判断：Teacher pack 能否使用、某个 claim 是否成立、候选优先级是否已有 NPU runtime 支撑。不要用 timing 缺失否定强语义候选。

## 路由

1. 用户明确要求 GPU Teacher：`teacher_required`。
2. 未明确要求但发现 evidence pack：自动评估。
3. 有部分可用 regime/provenance：`teacher_hybrid`，只在已证明范围使用。
4. 无可用 pack：`teacher_unavailable`，继续 Line A/B。
5. 用户要求 `teacher-only` 且关键证据缺失：停止 Line T 并生成 capture request；不伪造 Teacher 结论。

## Pack 资格

强 compile/runtime Teacher 应满足：

- GPU 程序语义匹配、已 compile、active window 在 warmup 后；
- source revision、environment、regime、rank/window 可追溯；
- shape、dtype、work-domain、训练状态和关键 state 可比；
- claim 能区分 `source_port_gap`、`teacher_compile_delta` 与推断；
- 关键结论引用实际读取范围，而非文件存在或二手 summary。

任一语义关键字段未知时，只降低对应 regime/claim，不必丢弃整个 pack。

## 三类 Teacher 信号

### source_direct

用于 NPU port 相对 common/GPU source 的直接差异，例如 dtype/rounding、API 参数、work-domain、layout、state、host read、training/reporting 路径。

成立条件：

- common/GPU 与 NPU source revision 可比；
- 差异定位到 source/semantic range；
- correctness contract 已知；
- 不是 validation/cold path 被误当 steady path。

不要求先有精确 NPU exposed time。Timing 缺失时仍生成候选，但：`gain=unmeasured`、优先级受限、携带最小计时动作，只有 Phase 4 正确性与性能通过后才接受。

### compile_method

用于 GPU pre→post 证明的删除、融合、work reduction、precision/layout boundary、buffer/saved-tensor reuse 或 latency hiding guideline。

成立条件：

- 至少有可追溯的 pre/post chain；
- method、作用对象和成立条件能从 artifact 恢复；
- 已区分 transferable mechanism 与 GPU-specific 实现；
- NPU source/runtime 存在对应语义或 residual work，或明确安排最小补测。

NPU exposed time 不是 claim 存在条件，但缺失时不能给高收益置信度。

### runtime_gap

用于当前 NPU profile 证明的 residual work、fragmentation、sync、communication、memory 或 rank imbalance。

成立条件：

- regime/rank/window 与单位有效；
- source/operator/kernel/collective 映射可追溯；
- 使用 critical-path exposure，而非简单 duration；
- 至少存在一条 NPU adaptation option 或最小补证动作。

runtime gap 可以没有 Teacher 对应方法，继续交给 Line A/B。

## 信号强度与优先级

候选存在性由语义/compile/runtime 证据分别决定；排序再合并：

~~~text
priority ∝ exposed_gain × evidence × transferability × feasibility × correctness / cost
~~~

- 多类信号共同指向同一根因：合并并提高 evidence grade。
- `source_direct` 为 A/强 B、低风险且试验成本低时，可以先做低成本 trial；不等待完美 timing。
- `compile_method` 缺 NPU residual/exposure 时保留为 provisional，并先做 instrumentation/microbenchmark。
- 只有名称相似、GPU kernel 更少或 GPU 更快：不是有效 Teacher 信号。
- GPU 绝对时间不作为 NPU floor。

## Route 判定

### teacher_auto

Evidence pack 合格，且至少存在一个有效 `source_direct`、`compile_method` 或跨线强候选。允许部分 candidate 尚未完成 timing；用 coverage 和 evidence grade 控制范围。

### teacher_hybrid

存在有效局部信号，但 regime/rank/provenance/runtime mapping 不完整。Line T 只覆盖有证据范围，A/B 继续全量分析；每个缺口写下一项最小读取/采集动作。

### teacher_unavailable

无语义可比 source、compiled/warm 状态无法证明，或不存在任何可追溯 claim。记录原因并继续 Line A/B。

## 输出

`teacher_gate.json` 记录：

- route 与 user requirement；
- pack hard conditions；
- eligible/partial/unavailable regime；
- signal IDs、signal class、证据等级；
- measured/unmeasured gain 与 next minimum evidence；
- coverage/provenance gap；
- fallback 与 capture request。
