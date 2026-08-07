# Semantic Supernode 对齐

Supernode 是跨 compiler/backend 仍保留明确语义、正确性合同和可实施优化边界的最小区域。它可以对应源码表达式、多个 graph node、一个 GPU fused kernel、多个 NPU operator，或一段 compute/collective schedule。

## 目录

1. 构造与映射
2. 磁盘事实 Schema
3. 人类可读主表
4. 方法翻译卡
5. 强制审计问题
6. Gap、证据与收益
7. 上下文管理

## 1. 构造与映射

从 workload 语义边界开始，不从 kernel 名开始。候选边界包括输入/embedding、normalization、projection、局部或全局交互、activation、residual、output/loss、state update、parameter gather/reduce；不要把特定模型模块名写进通用 schema。

先比较 common/GPU source 与 NPU port source，再沿 GPU provenance 追踪：

~~~text
common semantic contract
  ├─ GPU source/pre-compile → transformed graph → pre/post-fusion IR → generated code → runtime
  └─ NPU port source       → NPU graph/operator/kernel/collective → runtime
~~~

严格区分：

- `source_port_gap`：NPU port 相对 common/GPU source 新增、遗漏或改变的 dtype、API 参数、work-domain、layout、host 行为、state 或训练/报告路径；
- `teacher_compile_delta`：GPU compiler 从 pre 到 post 完成的删除、融合、折叠、去 materialization、复用、精度边界移动或 latency hiding；
- `teacher_method_guideline`：由 source 或 compile delta 证明的方法、作用对象、成立条件、依赖和精度语义。

NPU 匹配优先级：

1. source location、semantic range、state mutation；
2. dependency neighborhood；
3. input/output shape、dtype、stride、layout、alias；
4. regime-specific 调用次数；
5. operator/module/call stack；
6. correlated kernel/collective；
7. 名称相似性，仅作弱证据。

显式保留一对多和多对一映射。若多个 mapping 都合理，为同一 Supernode 写不同 `mapping_variant`，不要过早强制双射。

GPU post 中不存在、但 NPU 当前执行的 cast、transdata、copy、materialization、重复 mask、sync、graph break 或额外 optimizer work，分别建立 `mapping_status=npu_extra` 的行。

## 2. 磁盘事实 Schema

在 `evidence_db/supernodes.csv` 或等价 JSONL 中，每个 `(regime_id, supernode_id, mapping_variant)` 一行。数组使用 JSON string；raw event/interval 单独保存并引用。

| 分组 | 字段 |
|---|---|
| 标识 | `regime_id`, `supernode_id`, `semantic_role`, `mapping_variant`, `mapping_status` |
| 引用 | `contract_ref`, `regime_ref`, `precision_ref`, `source_refs` |
| 公共语义 | `semantic_formula`, `state_mutation`, `mask_window_sparse_semantics`, `downstream_consumer` |
| Tensor | `input/output_shapes`, `input/output_dtypes`, `strides`, `layouts`, `storage_aliases`, `implicit_api_transforms`, `dynamic_constraints` |
| 精度 | `semantic_input_dtype`, `storage_dtype`, `compute_dtype`, `accumulator_dtype`, `output_dtype`, `saved_tensor_dtype`, `rounding_boundary`, `reporting_dtype` |
| Work-domain/API | `logical_work_domain`, `actual_kernel_work_domain`, `mask_window_sparse_contract`, `teacher_api_parameters`, `npu_api_parameters` |
| Porting | `gpu_common_source`, `npu_source`, `source_port_gap`, `added_or_missing_ops`, `api_contract_diff`, `training_vs_reporting_path` |
| Teacher pre | `pre_ops`, `pre_node_count`, `pre_materializations`, `pre_saved_tensors` |
| Teacher post | `post_kernels`, `post_kernel_count`, `fusion_group`, `post_layout`, `buffer_reuse`, `post_saved_tensors`, `saved_tensor_lifetime` |
| Compile delta | `eliminated_ops`, `folded_or_cached`, `fused_nodes`, `removed_materializations`, `work_domain_reduction`, `removed_dtype_layout_boundary`, `buffer_saved_tensor_reuse`, `hidden_latency`, `launch_count_delta` |
| Method guideline | `teacher_method`, `target_object`, `preconditions`, `dependencies`, `precision_semantics`, `transferable_mechanism`, `gpu_specific_exclusions`, `method_confidence` |
| NPU current | `npu_ops`, `npu_kernels`, `kernel_count`, `collectives`, `materializations`, `saved_tensors`, `lifetime`, `layout`, `graph_breaks` |
| 频率 | `calls_per_step_teacher`, `calls_per_step_npu`, `regime_occurrences`, `rank_scope` |
| 时间 | `npu_device_ms`, `npu_exposed_ms`, `npu_p95_ms`, `rank_imbalance_ms`, `teacher_device_ms` |
| 额外工作 | `extra_cast_ms`, `extra_transdata_ms`, `extra_copy_ms`, `extra_dispatch_ms`, `extra_sync_ms`, `extra_comm_wait_ms`, `extra_compute_ms` |
| 机会 | `gap_class`, `teacher_signal_class`, `transfer_class`, `direct_gain`, `enabling_gain`, `upper_bound`, `npu_adaptation_options` |
| 风险/门禁 | `semantic_invariants`, `numerical_risk`, `memory_risk`, `multi_rank_risk`, `correctness_tests`, `negative_control` |
| 证据 | `evidence_grade`, `confidence`, `supporting_claim_ids`, `contradicting_claim_ids`, `next_minimum_evidence` |
| 决策 | `candidate_ids`, `status`, `last_trial_id`, `failure_signature`, `notes` |

`mapping_status` 至少使用 `matched`、`teacher_only`、`npu_extra`、`ambiguous`、`unmapped`。未知字段写 `unknown`；不适用写 `not_applicable`，不要静默省略。

## 3. 人类可读主表

首次 gap 审计、每个 Teacher wave 结束和最终报告必须生成完整主表：

| Supernode | 原始语义/regime | 精度合同 | GPU 编译前 | Source-port Gap | GPU 编译后 | Compile Delta | Teacher 方法/成立条件 | NPU 当前 | Exposed Gap | 证据/置信度 |
|---|---|---|---|---|---|---|---|---|---|---|

- 一行对应一个可实施语义边界；不同 regime 有实质差异时使用子行或 override。
- `GPU 编译前` 与 `Source-port Gap` 必须先于 compile delta，避免漏掉 port 时丢失的精度、window、work-domain 或 API 合同。
- `GPU 编译后` 写实际执行结构；`Compile Delta` 写 pre→post 变换；`Teacher 方法`写优化 guideline 与成立条件，三者不能合并为“更融合”。
- `精度合同` 必须覆盖 accumulator、saved tensor、rounding 和 training/reporting 路径，不只写 dtype 名。
- `NPU 当前` 同时写 source、operator/kernel、次数、时间与关键路径暴露；同名 fused op 不能证明等价。
- 证据不足时保留 `unknown`，并写下一项最小读取或采集动作。

## 4. 方法翻译卡

为每个 unresolved Gap 生成方法翻译卡，再映射到共享 [Candidate Contract](../../references/candidate_contract.md)：

~~~text
Supernode / regime:
Target gap:
Teacher method guideline:
成立条件与依赖:
可迁移机制:
GPU-specific exclusions:
NPU adaptation options:
  1. remove/cache
  2. official NPU API/parameter
  3. algebra/layout/precision/API rewrite
  4. selective compile/graph boundary
  5. autograd/buffer/stream/collective schedule
  6. custom kernel
Platform support evidence:
Direct gain / enabling gain / upper bound:
Correctness and resource risks:
Minimum validation gate:
Negative control:
Evidence gaps:
~~~

同一 Gap 允许多个 NPU adaptation option。Teacher 提供优化方法 guideline，Phase 3 负责 NPU adaptation；方法相同不代表 GPU kernel 实现可直接复制。

先完成所有高价值 Supernode 的方法翻译卡，再全局排序。不要发现第一个 fusion 就立刻实施。

## 5. 强制审计问题

每个 `yes` 或 `unknown` 都产生独立候选或补证动作，不能全部折叠成 `fusion`：

1. NPU 是否执行 common/GPU source 不需要的工作？
2. NPU kernel/API 是否知道最小、可证明正确的 logical work-domain？
3. NPU 是否 materialize 更宽 dtype/storage，或保留大 tensor 更久？
4. 显式 view/transpose 是否与 API 隐式 layout contract 冲突？
5. backward 是否重算或保存 Teacher 已复用、alias 或删除的内容？
6. Teacher fused intent 是否能由官方 NPU op/参数表达？
7. custom kernel 前，remove/cache、selective compile 或 custom autograd 是否足够？
8. 额外工作是否暴露在各 regime 的最慢 rank 关键路径？
9. 收益或正确性是否随 regime 改变，需要 regime-specific dispatch？
10. 哪个 correctness predicate 与 negative control 能最低成本否证候选？

## 6. Gap、证据与收益

Gap 分类：`porting_artifact`、`semantic_diff`、`elimination`、`fusion`、`work_domain`、`sparsity`、`precision`、`layout`、`reuse`、`dispatch`、`synchronization`、`communication`、`memory_lifetime`、`compiler_boundary`、`hardware_reference`。其他类别未审计完前，不得用 `hardware_reference` 结束分析。

证据等级：

- A：source/provenance、tensor contract、相关 regime/rank 和 measured interval 均匹配；
- B：语义与 tensor contract 匹配，但 timing、provenance 或 coverage 不完整；
- C：仅名称、shape 或架构假设。

强语义 `source_direct` 可以在 NPU exposed time 未知时生成候选，但其收益状态为 `unmeasured`，必须携带最小计时动作。Compile/runtime 候选若缺 NPU 对应工作或关键路径证据，不得给高优先级。

每个 regime：

~~~text
direct_upper_bound = min(removable_or_reducible_time, npu_exposed_time)
weighted_direct_gain = Σ regime_occurrences × direct_upper_bound
~~~

GPU 时间不是 NPU floor。通信使用最慢 rank 的 non-overlapped interval；无法建立 dependency 时使用保守 interval union。

`enabling_gain` 与直接收益分开记录。例如降低峰值内存可能暂时不减少 step time，但能解锁更大 fusion、compile、batch 或 overlap；不得把使能收益直接加到 wall-clock 预测。

## 7. 上下文管理

完整 schema 保存在磁盘。上下文中的单个 Supernode 使用 compact card：

~~~text
SN:<id> [regime refs]
source_gap=<非空摘要>; compile_delta=<非空摘要>; method=<guideline>
npu_residual=<热点>; precision=<仅风险>; exposed=<上限/未测>
actions=<IDs>; evidence=<claim/artifact IDs>; next=<最小补证>
~~~

工作集只加载 unresolved 高价值候选、当前关键路径、Action 依赖和回归 Supernode。按累计 exposed gap 与 token budget 分批读取；不以固定 Supernode 数量截断。

公共 workload/regime/precision contract 只保存一次并引用。完全相同的 regime 共享 base record，只为差异写 override；无 transformation 的维度不生成空自然语言。
