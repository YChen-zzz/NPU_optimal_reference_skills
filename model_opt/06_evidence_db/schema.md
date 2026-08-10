# evidence_db Schema v3

`evidence_db` 是 Profiling-only、GPU Teacher 和 hybrid 路线共用的状态与证据层。未启用 Teacher 时，对应字段写 `not_applicable`；不要维护两套 campaign 数据。

## 内容索引

- 目录
- Campaign 与 Regime
- Artifact、Coverage 与读取
- Claim Evidence Map
- Teacher Gate、Semantic Diff、Supernodes 与 Candidates
- Normalized Runtime、Trial、Findings 与停止

## 目录

~~~text
<workspace>/evidence_db/
├── campaign.yaml
├── regimes.yaml
├── artifacts.jsonl
├── coverage.csv
├── evidence_read_ledger.csv
├── claim_evidence_map.csv
├── teacher_gate.json
├── semantic_diff.md
├── supernodes.csv
├── candidates.csv
├── phase3_preflight.json
├── preflight_receipts/
│   └── <wave>-<stage>-<candidate>.json
├── normalized/
│   ├── phase_summary.csv
│   ├── rank_imbalance.csv
│   ├── operator_summary.csv
│   ├── kernel_summary.csv
│   ├── communication_summary.csv
│   └── memory_summary.csv
├── supernode_labs/
│   └── <supernode_id>/<lab_id>.yaml
├── short_runs/
│   └── <short_run_id>.yaml
├── trials/
│   └── <trial_id>.yaml
├── findings.jsonl
└── stop_audit.yaml
~~~

raw profiling、graph、IR 和 generated code 不复制进 evidence_db，只登记不可变路径、identity 和实际读取范围。

## Campaign 与 Regime

`campaign.yaml`：

- schema_version、campaign_id、workload、semantic revision；
- training/inference mode、objective、full-run command；
- accuracy/performance/iterative baseline、阈值与自然波动；
- NPU/GPU hardware、software、world size、environment ID；
- source repository、优化分支、baseline commit；
- performance goal、已完成的 20% goal-progress full-run 档位。

`regimes.yaml` 每个 regime：

- id、触发条件、transition；
- shape/dtype/layout/work-domain；
- batch/累积、forward/backward/optimizer/state；
- expected occurrences 或完整任务权重；
- rank/world size；
- accuracy/performance coverage；
- GPU/NPU mapping 状态。

batch 相同不代表 regime 相同；work-domain、precision、control flow、state 或 communication 改变时拆分。

## Artifact、Coverage 与读取

`artifacts.jsonl`：

`artifact_id, backend, kind, path, identity, source_revision, environment_id, regime, rank, steps, producer, notes`

`coverage.csv` 每个 `(backend, regime, rank)`：

`compiled_warm, light_steps, deep_steps, trace, ops, kernels, communication, shapes, stack, memory, graph, selection_reason, artifact_refs, limitations`

`evidence_read_ledger.csv`：

`artifact_id, read_status, read_scope, read_method, parser_version, extracted_facts, claim_ids, limitations, read_at`

文件存在、索引存在、实际读取和 claim 使用必须区分。

## Claim Evidence Map

`claim_evidence_map.csv`：

`claim_id, supernode_id, claim_type, claim, regime_scope, rank_scope, supporting_artifact_ids, contradicting_artifact_ids, required_but_unread_ids, source_port_or_compile_delta, evidence_grade, confidence, next_minimum_read`

`claim_type` 至少支持：

- `source_direct`：source-port semantic/API/precision/work-domain gap；
- `compile_method`：GPU pre/post 证明的方法 guideline；
- `runtime_gap`：NPU current profile 证明的 residual/critical-path gap；
- `hardware_residual`：其他类别排除后的硬件残差。

一个候选可引用多个 claim。反证必须进入结构化字段，不能只写 notes。

## Teacher Gate 与 Semantic Diff

`teacher_gate.json`：

- route、user requirement、pack hard conditions；
- eligible/partial/unavailable regime；
- signal IDs/classes、measured/unmeasured gain；
- coverage/provenance gap；
- fallback、capture request、next minimum evidence。

`semantic_diff.md` 保存 common/GPU/NPU source 对比，并把每项分类为 `semantic`、`porting_artifact`、`backend_requirement` 或 `unknown`。

## Supernodes 与 Candidates

`supernodes.csv` 每个 `(regime_id, supernode_id, mapping_variant)` 一行，保存：

- semantic/tensor/precision/work-domain contract；
- source_port_gap、Teacher pre/post、compile delta、method guideline；
- NPU current、`npu_extra`、frequency、rank 与 exposed gain；
- supporting/contradicting claim、risk、Candidate refs 和 failure signature。

完整字段见 [Supernode 对齐](../02_bottleneck_analysis/gpu_teacher/references/supernode_alignment.md)。

`candidates.csv` 遵循 [Candidate Contract](../02_bottleneck_analysis/references/candidate_contract.md)，必须能回到 mapping、claim、artifact 和 trial。Teacher 候选保存 signal class、method guideline、transferable mechanism、GPU-specific exclusions、NPU adaptation options、direct/enabling gain、negative control 和 next minimum evidence。

`phase3_preflight.json` 登记有效 NPU L1、约 60 秒 baseline、高价值 Supernode Lab、候选与 API-first 状态。Phase 3 脚本校验后把 manifest hash 写入 `preflight_receipts/`；trial 和 accepted commit 必须引用对应 action receipt。

## Normalized Runtime

`normalized/` 保存可重建的派生表；每行包含 artifact ID、capture identity 和 generator/version：

- phase_summary：regime/rank median/p95 与 step accounting；
- rank_imbalance：critical rank、spread 与因果 interval；
- operator/kernel summary：semantic mapping、count/time、tensor/runtime 字段；
- communication summary：payload、total/non-overlap/skew；
- memory summary：peak/churn/materialization/saved-tensor lifetime。

## Trial

`trials/<trial_id>.yaml`：

~~~yaml
trial_id:
parent_baseline:
candidate_ids: []
claim_ids: []
mapping_refs: []
git:
  branch:
  parent_commit:
  trial_commit:
  diff_hash:
environment_id:
regime_scope: []
prediction:
  weighted_exposed_gain:
  enabling_gain:
  assumptions: []
  next_minimum_evidence:
implementation:
  files_modified: []
  summary:
  npu_adaptation_option:
  supernode_lab_ref:
  cumulative_winner_ref:
validation:
  static:
  operator:
  layers:
  gradient_state:
  negative_control:
  weighted_short_run:
    accuracy_baseline_short_run_ref:
    performance_baseline_short_run_ref:
    same_seed_state_data_steps:
    shortened_total_steps:
    scheduler_regime_transition_mapping:
    duration_seconds:
    loss_curve_diff:
    validation_loss_diff:
    precision_or_resource_replay_reason:
    replay_checkpoint_refs: []
    goal_progress:
  full_run:
performance:
  baseline_ref:
  distribution:
  per_regime:
  per_rank:
  memory:
decision:
  status:
  reason:
  failure_predicate:
  reopen_when:
evidence_refs: []
~~~

accepted、rejected、inconclusive 和 blocked 均必须记录。

## Findings 与停止

`findings.jsonl` 保存 mechanism、workload/tensor/regime signature、environment validity、source/graph/profile evidence、accepted/rejected result、failure signature、transferability、confidence 和重新验证条件。

`stop_audit.yaml` 保存：

- `stop_proposed` revision、提出原因和证据时间；
- Stop Auditor 的 `continue/stop_allowed/blocked` 决定与 failed gates；
- regime/rank/correctness/full-run coverage；
- remaining gap 与 exposed-gain 上限；
- 未测高置信候选、source-direct unmeasured 候选和原因；
- uncovered gap、next candidate/minimum evidence 和 blocking predicate；
- 最近两次 wave/profile；
- best commit、复现命令与停止/阻塞结论。

算子参数、compiler 行为和版本限制不得脱离 environment 传播为通用事实。
