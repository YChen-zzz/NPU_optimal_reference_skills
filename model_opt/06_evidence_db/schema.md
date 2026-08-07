# evidence_db Schema v2

evidence_db 是 model_opt 的统一状态与证据层。Profiling-only、GPU Teacher 和 hybrid 路线使用同一目录；未启用 Teacher 时对应字段写 not_applicable。

## 目录

~~~text
<workspace>/evidence_db/
├── campaign.yaml
├── regimes.yaml
├── artifacts.jsonl
├── evidence_read_ledger.csv
├── teacher_gate.json
├── supernodes.csv
├── candidates.csv
├── trials/
│   └── <trial_id>.yaml
├── findings.jsonl
└── stop_audit.yaml
~~~

raw profiling、graph、IR 和生成代码不复制进 evidence_db，只登记不可变路径与 identity。

## campaign.yaml

至少包含：

- schema_version、campaign_id；
- workload、semantic revision、training/inference mode；
- accuracy/performance/iterative baseline；
- correctness metric、方向、阈值、自然波动；
- NPU/GPU hardware、software、world size 和关键环境变量；
- source repository、优化分支、baseline commit；
- objective 和 full-run command。

## regimes.yaml

每个 regime：

- id、触发条件和 transition；
- shape/dtype/layout/work-domain；
- batch/累积、forward/backward/optimizer/state；
- expected occurrences 或完整任务权重；
- rank/world size；
- accuracy/performance coverage；
- GPU/NPU 匹配状态。

只有 batch 相同不代表同一 regime；work-domain、精度、控制流、state 或 communication 改变时拆分。

## artifacts.jsonl 与读取账本

artifact 字段：

artifact_id, backend, kind, path, identity, source_revision, environment_id, regime, rank, steps, notes

读取账本字段：

artifact_id, read_status, read_scope, read_method, extracted_facts, claim_ids, limitations, read_at

文件存在、索引存在和实际读取必须区分。

## teacher_gate.json

未启用 Teacher 时记录 route=profiling_only。启用时记录：

- user requirement；
- route 和 hard-condition；
- eligible/unavailable regime；
- signal、gain 和 evidence gap；
- fallback 与 capture request。

## supernodes.csv

仅 teacher_auto/teacher_required/teacher_hybrid 强制。每行保存 semantic contract、source_port_gap、teacher pre/post、compile delta、NPU current、regime frequency、exposed gain、evidence 和 Candidate refs。

完整字段见 Phase 2 GPU Teacher 的 supernode_alignment.md。

## candidates.csv

遵循 Phase 2 Candidate Contract。必须能从 candidate 回到 source/profile/teacher artifact，并能映射到 trial。Teacher 候选额外保存 `teacher_method_guideline`、`transferable_mechanism`、成立条件、GPU-specific exclusions 和按优先级排列的 `npu_adaptation_options`。

## trials/<trial_id>.yaml

~~~yaml
trial_id:
parent_baseline:
candidate_ids: []
git:
  branch:
  parent_commit:
  trial_commit:
  diff_hash:
environment_id:
regime_scope: []
prediction:
  weighted_exposed_gain:
  assumptions: []
implementation:
  files_modified: []
  summary:
validation:
  static:
  operator:
  layers:
  gradient_state:
  weighted_short_run:
  full_run:
performance:
  baseline_ref:
  distribution:
  per_regime:
  memory:
decision:
  status:
  reason:
  failure_predicate:
  reopen_when:
evidence_refs: []
~~~

accepted、rejected、inconclusive 和 blocked 均必须记录。

## findings.jsonl

存放可迁移或版本特定发现：

- mechanism；
- workload/tensor/regime signature；
- hardware/software validity；
- source/graph/profile evidence；
- accepted/rejected result；
- failure signature；
- transferable、workload-specific 或 platform-version-specific；
- confidence 和重新验证条件。

算子参数、CANN 行为和版本限制不得脱离环境传播为通用规则。

## stop_audit.yaml

包含：

- regime/rank/correctness/full-run coverage；
- remaining gap 与 exposed-gain 上限；
- 未测候选和原因；
- 最近两次 wave/profile；
- best commit 与复现命令；
- 停止或阻塞结论。
