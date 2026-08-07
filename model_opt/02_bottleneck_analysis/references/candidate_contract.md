# Candidate Contract

Line A、Line B 和 Line T 必须输出同一种候选。Phase 3 不根据候选来源选择流程，只根据机制、收益、风险和依赖实施。

## 必填字段

| 字段 | 含义 |
|---|---|
| candidate_id | 稳定 ID |
| source_lines | line_a、line_b、line_t 的一个或多个 |
| semantic_role | 对应源码逻辑或 Supernode |
| source_location | 文件、函数、行或可追溯 graph/source ref |
| regime_scope | 适用和不适用的 regime |
| problem | 当前执行了什么不必要或低效工作 |
| root_cause | 为什么源码/后端产生该现象 |
| gap_class | porting、elimination、fusion、work_domain、precision、layout、reuse、dispatch、sync、communication、memory、compiler_boundary 等 |
| optimization_dimension | 去重、复用、掩盖、替换；允许多值 |
| teacher_method_guideline | GPU/common source 或 compiler 已采用的方法、作用对象和成立条件；无 Teacher 时为 not_applicable |
| transferable_mechanism | 可迁移的机制、成立条件，以及不可直接迁移的 GPU-specific 部分 |
| npu_adaptation_options | 按实现阶梯排列的 NPU 适配方案；可包含多个备选 |
| implementation_path | 下方实现阶梯中的一级 |
| weighted_exposed_gain | 按 regime 频率和关键路径暴露量计算的保守收益区间 |
| evidence_grade | A/B/C |
| evidence_refs | source/profile/graph/IR/code/artifact ID |
| difficulty | low/medium/high |
| correctness_risk | 语义、数值、state、memory、multi-rank 风险 |
| validation_gate | Phase 4 必须运行的最低门禁 |
| dependencies、conflicts | 前置和互斥候选 |
| status | proposed/blocked/accepted/rejected/inconclusive |

## 实现阶梯

按以下顺序寻找 NPU 实现；只有前一级不适用、失败或收益不足时才升级：

1. remove_or_cache：删除、缓存、预计算、避免 host read；
2. official_npu_api：官方 NPU native/fused/sparse API 或参数；
3. api_layout_rewrite：代数、layout、storage、API 表达改写；
4. selective_compile：调整 graph boundary、functionalization、局部编译；
5. schedule_or_autograd：buffer/stream/collective schedule、custom autograd、saved-tensor policy；
6. custom_kernel：剩余 exposed gain 足以覆盖实现与维护成本时才使用。

不得因一个框架实现失败就强制编写自定义 kernel。是否升级由剩余收益上限、成功概率、实现成本和维护风险共同决定。

## 收益与排序

每个 regime 计算：

~~~text
weighted_exposed_gain =
  Σ regime_occurrences × min(removable_time, critical_path_exposed_time)
~~~

GPU 时间不是 NPU floor。缺少精确 interval dependency 时使用保守 non-overlapped union。

全局优先级：

~~~text
priority =
  weighted_exposed_gain
  × evidence_confidence
  × feasibility
  × correctness_confidence
  / implementation_cost
~~~

所有因子和估算依据必须落盘；不要只保存最终分数。

## 合并与 Bundle

同一 semantic role、source location、regime 和 root cause 的候选合并，保留多条证据、Teacher method guideline 与多个 NPU adaptation path。

多个候选可组成 bundle，条件是：

- 互不冲突且依赖已满足；
- 不共同改变尚未独立验证的 dtype/rounding、state、memory lifetime 或 collective ordering；
- 各自保留独立 commit、开关或 patch；
- bundle 有明确消融方案。

删除冗余、缓存、静态 buffer、host sync 和官方 API 参数通常可合并。loss/backward、optimizer、communication、custom kernel 和改变数值顺序的候选默认隔离。
