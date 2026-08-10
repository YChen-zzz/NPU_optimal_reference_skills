# Candidate Contract

Line A、Line B 和 Line T 必须输出同一种候选。Phase 3 不根据候选来源选择流程，只根据机制、收益、风险和依赖实施。

## 必填字段

| 字段 | 含义 |
|---|---|
| candidate_id | 稳定 ID |
| source_lines | line_a、line_b、line_t 的一个或多个 |
| semantic_role | 对应源码逻辑或 Supernode |
| source_location | 文件、函数、行或可追溯 graph/source ref |
| mapping_ref | Supernode/mapping variant；无 Teacher 时可为 not_applicable |
| regime_scope | 适用和不适用的 regime |
| problem | 当前执行了什么不必要或低效工作 |
| root_cause | 为什么源码/后端产生该现象 |
| gap_class | porting、elimination、fusion、work_domain、precision、layout、reuse、dispatch、sync、communication、memory、compiler_boundary 等 |
| optimization_dimension | 去重、复用、掩盖、替换；允许多值 |
| teacher_signal_class | source_direct、compile_method、runtime_gap；无 Teacher 时为 not_applicable |
| teacher_method_guideline | GPU/common source 或 compiler 已采用的方法、作用对象和成立条件；无 Teacher 时为 not_applicable |
| transferable_mechanism | 可迁移的机制、成立条件，以及不可直接迁移的 GPU-specific 部分 |
| transfer_class | direct_port_fix、algorithmic、compiler_intent、schedule、hardware_specific 或 not_applicable |
| teacher_method_confidence | 方法及成立条件的证据置信度；无 Teacher 时为 not_applicable |
| npu_adaptation_options | 按实现阶梯排列的适用方案；每项写成立条件、主要限制和所需证据。无需列出明显不适用的方法 |
| implementation_path | 下方实现阶梯中的一级 |
| supernode_lab_ref | 高价值/多路径候选的对照测试计划或结果；不需要时写 `lab_not_required:<具体理由>` |
| platform_support_evidence | NPU API/version、microbenchmark、graph/profile 或环境事实 |
| weighted_exposed_gain | 按 regime 频率和关键路径暴露量计算的保守直接收益；允许 unmeasured |
| enabling_gain | 内存、compile、fusion、batch 或 overlap 的间接使能价值；不得直接加到 wall-clock |
| evidence_grade | A/B/C |
| evidence_refs | source/profile/graph/IR/code/artifact ID |
| contradicting_evidence_refs | 反证或不支持该方法的 artifact/claim |
| difficulty | low/medium/high |
| correctness_risk | 语义、数值、state、memory、multi-rank 风险 |
| validation_gate | Phase 4 必须运行的最低门禁 |
| negative_control | 最低成本的反例/消融，用于否证机制或错误归因 |
| next_minimum_evidence | 当前证据不足时最小读取、instrumentation 或计时动作 |
| dependencies、conflicts | 前置和互斥候选 |
| status | proposed/blocked/accepted/rejected/inconclusive |

## 实现阶梯

Ascend 默认 API-first。按以下顺序寻找常规 NPU 实现；只有前一级不适用、失败或收益不足时才升级：

1. remove_or_cache：删除、缓存、预计算、buffer 复用、避免 host read；
2. official_npu_api：官方 NPU native/fused/sparse API 或参数；
3. manual_rewrite：代数、layout、storage、precision boundary、向量化或 API 表达改写；
4. schedule_or_autograd：stream/collective schedule、custom autograd、saved-tensor policy；
5. custom_kernel：剩余 exposed gain 足以覆盖实现与维护成本时才使用。

`selective_compile` 不参与常规抢占排序，是最后解锁的 fallback。只有官方 NPU API 已完成发现与 Lab 验证，且所有适用的非 compile 路径均不适用、失败或收益不足时才能进入 trial；GPU Teacher 的 compile guideline 只提供方法线索，不能跳过该门禁。不得因 compile 或框架实现失败就强制编写自定义 kernel。

## 收益与排序

每个 regime 计算：

~~~text
weighted_exposed_gain =
  Σ regime_occurrences × min(removable_time, critical_path_exposed_time)
~~~

GPU 时间不是 NPU floor。缺少精确 interval dependency 时使用保守 non-overlapped union。

强语义 `source_direct` 在 NPU exposed time 未精确映射时仍可存在：`weighted_exposed_gain=unmeasured`，保留调用频率/粗上限和 `next_minimum_evidence`。它可以进入低成本、低风险 exploratory trial，但不能在无 NPU 性能证据时宣称收益或进入最终 accepted 状态。

全局优先级：

~~~text
priority =
  weighted_exposed_gain
  × evidence_confidence
  × feasibility
  × correctness_confidence
  / implementation_cost
~~~

所有因子和估算依据必须落盘；不要只保存最终分数。`enabling_gain` 单独用于解锁关系和依赖排序，不与直接 wall-clock gain 相加。

## 合并与 Bundle

同一 semantic role、source location、regime 和 root cause 的候选合并，保留多条证据、Teacher method guideline 与适用的 NPU adaptation path。先排序 Supernode，再在节点内选择最低的有效阶梯。

多个候选可组成 bundle，条件是：

- 互不冲突且依赖已满足；
- 不共同改变尚未独立验证的 dtype/rounding、state、memory lifetime 或 collective ordering；
- 各自保留独立 commit、开关或 patch；
- bundle 有明确消融方案。

删除冗余、缓存、静态 buffer、host sync 和官方 API 参数通常可合并。loss/backward、optimizer、communication、custom kernel 和改变数值顺序的候选默认隔离。
