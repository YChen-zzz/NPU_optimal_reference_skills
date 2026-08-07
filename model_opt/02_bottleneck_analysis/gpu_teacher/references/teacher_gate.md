# GPU Teacher 路由门控

## 路由优先级

1. 用户明确要求使用 GPU Teacher：teacher_required。
2. 未明确要求，但发现可用 evidence pack：自动评估。
3. 没有 pack：teacher_unavailable，继续 Line A/B。
4. 用户明确要求 teacher-only 且关键证据缺失：生成 capture request 并停止 Line T，不静默降级。

## 硬条件

Teacher 强信号必须满足：

- GPU 程序已 compile 且 active window 在 warmup 后；
- GPU/common source 与 NPU 目标语义一致；
- regime 的 shape、dtype、work-domain、训练状态和关键 state 可比；
- 有当前 NPU source 与匹配 regime profile；
- 结论能区分 source_port_gap 与 teacher_compile_delta；
- 关键 claim 可回溯到 artifact 和实际读取范围。

任一语义关键字段未知，该 regime 最多为 partial。

## 有效 Teacher 信号

只有同时满足以下条件才算一个 Teacher signal：

1. GPU source 或 pre/post compile 证据证明了具体差异/意图；
2. 当前 NPU source/runtime 仍保留对应额外工作或缺失合同；
3. 该工作在 NPU critical path 上有可测 exposed time；
4. 存在至少一条 NPU implementation path；
5. correctness contract 已知。

“GPU kernel 更少”或“GPU 时间更短”本身不是信号。

## 三档判定

### teacher_auto

同时满足：

- 所有受影响 regime 有匹配证据；
- 无 semantic-critical unknown；
- 至少一个 A 或强 B 级 Action；
- 保守总收益上限大于 max(3 × timing_noise, configured_min_gain)。

configured_min_gain 未配置时，可用完整任务时间的 5% 作为初始筛选值；它只控制优先级，不是停止标准。

### teacher_hybrid

Teacher 已编译且存在有效局部信号，但 regime coverage、provenance 或 runtime mapping 不完整。只在有证据范围内生成 Line T 候选，其余由 Line A/B 覆盖。

### teacher_unavailable

未编译/未 warmup、语义不可比、没有 NPU 匹配 profile，或不存在能映射到 NPU exposed time 的 Action。记录原因并继续 Line A/B。

## 输出

写入 evidence_db/teacher_gate.json：

- route；
- user_requirement；
- eligible/unavailable regime；
- hard-condition 结果；
- signal IDs；
- conservative gain；
- evidence gaps；
- fallback 行为；
- 下一项最小采集动作。
