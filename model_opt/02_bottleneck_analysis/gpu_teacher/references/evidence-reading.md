# GPU Teacher 证据读取与 Claim 账本

Artifact inventory 只证明文件存在；read ledger 证明 Agent 实际读取了什么；claim map 证明结论由什么支持。三者不得混用。

## 强制产物

`evidence_read_ledger.csv` 每项 artifact 记录：

`artifact_id, backend, kind, path, content_identity, read_status, read_scope, read_method, extracted_facts, claim_ids, limitations, read_at`

`read_status`：

- `fully_read`：完整阅读适合人工消费的文本；
- `targeted_read`：读取已登记的行、graph、kernel、rank/regime 或 event window；
- `parsed`：确定性 parser 处理了完整文件或声明范围，并记录 parser/version/query；
- `index_only`：只读取 metadata/signature/source refs，不能证明 transformation/timing；
- `inventory_only`：只知道存在、大小或 checksum；
- `unread` / `unavailable`。

`claim_evidence_map.csv` 每个 claim 记录：

`claim_id, supernode_id, claim_type, claim, regime_scope, rank_scope, supporting_artifact_ids, contradicting_artifact_ids, required_but_unread_ids, source_port_or_compile_delta, evidence_grade, confidence, next_minimum_read`

高优先级 Supernode 至少检查适用的 source contract、GPU pre/post chain、compiled runtime、NPU source/runtime。缺一类就显式降级，不用 summary 或二手报告补成 A 级。

## 规定读取顺序

| 顺序 | 文件族 | 必须提取 | 不能据此声称 |
|---:|---|---|---|
| 1 | workload/common/GPU/NPU source、config | 数学、train/reporting、dtype、mask/window、layout/API、state、regime schedule | compiler 已融合或 runtime 很快 |
| 2 | manifest、run log、environment | revision、world size、compile/warmup、active window、rank/regime coverage | 计划采集即数据有效 |
| 3 | compiled summary/kernel breakdown | kernel family、count、time、regime/rank 缩放 | overlap、source ancestry、独占 Supernode 时间 |
| 4 | raw compiled timeline | logical step、launch、kernel/collective interval、overlap、critical rank | 无 marker/rank 对齐时外推全程 |
| 5 | graph inventory/index | graph bundle、signature、source refs、coverage | `index_only` 证明删除或融合 |
| 6 | readable→transformed→pre/post IR | node ancestry、fusion、删除、materialization、layout、reuse | 缺 provenance 时用相似名称判同一语义 |
| 7 | generated code/kernel metadata | 实际参数、work-domain、dtype、load/store、buffer、saved tensor | GPU tiling/指令可直接复制到 NPU |
| 8 | eager vs compiled（可选） | compile delta 的方向与规模 | eager GPU 是最终 Teacher 或 NPU baseline |
| 9 | all-rank communication timeline | non-overlap、wait、skew、imbalance | collective 总 duration/bytes 等于收益 |
| 10 | NPU source/profile | residual work、kernel chain、critical-path exposure | Teacher 方法在 NPU 必然有效 |

先读 source/contract，再解释 graph；先确认 compiled runtime，再谈方法价值；最后用 NPU runtime 决定优先级。

## Claim 类型

- `source_direct`：由 source-port diff 证明的语义/API/precision/work-domain 候选；可先于精确 NPU timing 存在。
- `compile_method`：由 GPU pre/post chain 证明的方法 guideline；需要 NPU 对应工作才能获得高优先级。
- `runtime_gap`：由 NPU current profile 证明的 residual/critical-path gap；Teacher 可有或没有对应方法。
- `hardware_residual`：其他类别审计后仍存在的硬件差异；不得作为早期默认解释。

一个候选可以引用多种 claim。冲突证据不得只写在 notes，必须进入 `contradicting_artifact_ids`。

## Human report

每轮 Teacher 审计输出：

| Artifact/文件族 | 读取状态与范围 | 提取事实 | 支持/反驳的 Supernode/Claim | 不能证明或尚未读取 |
|---|---|---|---|---|

先报告 raw/direct evidence，再报告派生 summary、历史报告或 PPT。PPT 和历史结论用于导航与回归，不能单独把证据提升为 A 级。
