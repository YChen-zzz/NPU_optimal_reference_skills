# GPU Teacher 证据合同

文件存在不等于证据可用。每项 claim 必须能追溯到 execution regime、rank、source revision、environment 和 logical-step/capture window。

## 目录

1. Campaign 与不可变身份
2. GPU Teacher evidence pack
3. Regime 与 rank coverage
4. 静态 semantic diff
5. 时间归一化
6. 缺失与补采

## 1. Campaign 与不可变身份

在 `campaign.yaml`、`regimes.yaml` 和 `artifacts.jsonl` 中至少记录：

- workload semantic revision 与完整命令；
- GPU/NPU source revision、hardware/software environment、world size；
- accuracy/performance baseline、目标与自然波动；
- regime ID、触发条件、transition、shape/dtype/layout/work-domain/state、出现频率；
- rank、logical steps/capture window、compile/warmup 状态；
- artifact path、kind、identity/checksum、生成工具与版本。

注册后的 raw evidence 只读。内容变化必须产生新 artifact ID；路径相同不能视为身份相同。

## 2. GPU Teacher evidence pack

只把语义匹配、已 compile、已 warmup 的 GPU 稳态窗口作为最终 Teacher。Eager GPU 只用于解释 compile delta。

| 证据 | 提取内容 |
|---|---|
| source、config、command、environment | 数学、训练/报告路径、precision、API、regime 和版本 |
| run log/manifest | compile、warmup、active window、成功状态和完整性 |
| compiled graph inventory | distinct graph、signature、rank/regime coverage |
| readable/transformed graph | pre/post-grad 结构、decomposition、canonicalization |
| pre/post-fusion IR | 删除、fusion boundary、materialization、layout、reuse、dependency |
| generated code/kernel metadata | 实际 dtype、work-domain、load/store、saved tensor、buffer、launch |
| compiled summary/timeline | kernel/collective、调用数、overlap、rank、critical interval |
| eager profile（可选） | compile delta 方向与规模；不作为 NPU floor |
| memory snapshot/stats | peak/live bytes、allocation churn、saved-tensor pressure |

其他 compiler stack 使用语义等价 artifact，不要求特定文件名。尽量建立：

~~~text
source/module
→ pre-compile node
→ transformed/post-grad node
→ fusion group
→ generated kernel
→ runtime interval
~~~

链路断裂时明确标记 `unknown/unavailable`，不用相似 kernel 名填补。

## 3. Regime 与 rank coverage

先从 source 和 schedule 发现会改变 graph、kernel、precision、work-domain、state 或 communication 的 regime。batch 相同不代表 regime 相同。

- steady、transition、validation、checkpoint 和 cold compile 分开；
- distributed workload 优先保存 all-rank light trace，用于找关键 rank、wait 和 imbalance；
- deep trace 选择代表、最慢和异常 rank，打开 shape/stack/module/memory；
- 每个窗口包含多个 warm 后 steady step，并保留 regime 与 logical-step marker；
- GPU graph 只有在 signature/config/hash 等价时才能跨 rank/regime 复用；runtime/collective 结论仍需 rank 证据。

默认建议 light ≥5 steady step、deep ≥3 steady step；Agent 可根据任务成本与方差调整，但必须在 coverage 记录中说明。缺少 marker、字段为空或 profiler flag 未开启时，写 `unavailable`，不能由文件名推断。

`coverage.csv` 每个 `(backend, regime_id, rank)` 至少记录：compiled/warm、light/deep step count、trace/op/kernel/communication/shape/stack/memory/graph 状态、selection reason 和 artifact refs。

## 4. 静态 semantic diff

在依赖 Profiling 排序前，比较 common/GPU source 与 NPU port source，审计：

- 新增、删除或重排的计算；
- semantic/storage/compute/accumulator/output/saved/reporting dtype；
- cast、scalar promotion、rounding/materialization boundary；
- shape、stride、view、transpose、contiguous、layout、alias；
- mask、window、sparsity、有效 work-domain 和完整 API 参数；
- training loss、reporting loss、saved tensor、custom autograd；
- optimizer grouping、state、collective payload/order；
- host read、shape-dependent Python、graph break、mutation、functionalization。

每项分类为 `semantic`、`porting_artifact`、`backend_requirement` 或 `unknown`。

强语义 `porting_artifact` 可以在首次或精确 NPU exposed-time 映射前生成 `source_direct` 候选，但仍必须：

- 声明 correctness contract；
- 标记收益为 `unmeasured` 或保守范围；
- 指定最小 NPU 计时动作；
- 通过 Phase 4 后才可接受。

这条规则防止 GPU Teacher 被退化为“只有 Profiling 找到热点后才提供解释”。

## 5. 时间归一化

对每个 regime/rank 计算：

- steady step median/p95；
- compute 与 communication interval union；
- compute/communication overlap；
- non-overlapped communication；
- host/launch/synchronization gap；
- slowest-rank critical span；
- Supernode device time 与 exposed time。

存在 overlap 时不能直接相加 duration。compile、warmup、validation、checkpoint 和 data input 等 fixed overhead 分开记录。

完整任务投影：

~~~text
projected_total = fixed_overhead + Σ regime_occurrences × median_step_time(regime)
~~~

GPU 绝对时间不能作为 NPU floor。详细 NPU runtime 映射见 [runtime-alignment.md](runtime-alignment.md)。

## 6. 缺失与补采

标准模式读取 GPU 机器离线提供的 evidence pack，不假设 NPU Agent 能跨机器采 GPU。

缺失时生成 capture request，列出：

- source revision/config/environment；
- 缺失 regime、rank、step window；
- 缺失 graph/IR/code/runtime/memory family；
- 每项缺失阻塞的 claim、Supernode 和证据等级；
- 最小 capture 范围与推荐开关。

GPU source/compiler/regime 未变化时复用原 pack。NPU 优化停滞通常只重采 NPU，再做 residual Teacher alignment；只有 GPU coverage/provenance 不足时才补采 GPU。

实际读取与 claim 规则见 [evidence-reading.md](evidence-reading.md)。
