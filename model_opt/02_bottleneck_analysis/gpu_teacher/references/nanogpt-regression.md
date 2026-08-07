# NanoGPT GPU→Ascend 条件回归

仅在用户提供该历史 evidence pack，或 forward-test GPU Teacher 能力时读取。本文件是机会召回测试，不是新 workload 的优化配方。

## 回归目的

验证 Agent 在未被直接告知具体 NPU API/patch 的条件下，能否从 source、GPU compile evidence 和 NPU profile 自行恢复高价值方法 family，并正确处理 regime、precision、critical path 与负结果。

## 历史事实

- 初始 NPU 约 500 秒，compiled GPU reference 约 127 秒；
- GPU-teacher-guided 早期试验约 13 次达到约 349 秒；
- 后续 regime/all-rank、work-domain、layout/reuse 与 custom-kernel 工作达到约 257.5 秒；
- 这些结果不是完全独立可加的 ablation，不得直接相加归因。

历史任务有三种 batch size，但至少四个 execution regime：small、medium、scheduled-large，以及 batch 与 large 相同但 attention window 不同的 extension。Agent 必须从执行合同恢复 regime，不得按 batch 数量硬编码。

## 必须自行发现的机会 family

1. training loss 路径的 source-port dtype/storage mismatch，并区分 reporting `.float()`；
2. Teacher 已缩小、NPU 未显式声明的 attention logical work-domain；
3. softcap/cross-entropy forward/backward 的 native/fusion/lowering opportunity；
4. projection transpose/layout 与 backward saved-tensor/recompute mismatch；
5. rotary/pointwise/cast/scalar region 的 native lowering 或融合机会；
6. host read/sync、重复 allocation 和静态 buffer 复用；
7. optimizer/collective grouping，但必须按 non-overlapped exposure 降权；
8. rank imbalance、regime-specific gain/regression 和 extension coverage gap。

允许发现不同 NPU 实现，但不能靠硬编码历史 winner 通过。

## 必须保留的正确性与负例

- attention work-domain hint 不等于删除 causal/document mask；必须保留公共语义。
- 单独改写 `.T` 曾无显著收益，不能泛化为所有 transpose 都应删除。
- collective payload 曾明显下降，但因通信大多 overlap，端到端收益很小；不能按 bytes 排序。
- 过宽 compile 可能更慢或 OOM；compile 是 lowering option，不是默认政策。
- custom fused loss 曾通过 scalar/宽松 similarity，却在完整 tensor、tile/pipeline、BF16 rounding 与 backward 验证失败。
- extension regime 缺 Teacher/runtime coverage 时必须降级，不能并入 large 假装已验证。

失败结论保存精确 predicate：environment、regime、shape、dtype、API/实现、错误位置和重新打开条件。

## 回放要求

1. 先建立 evidence inventory、read ledger、claim map 和 regime/rank coverage。
2. 先做 common/GPU/NPU source semantic diff，再读 GPU pre/post compile chain。
3. 把 source-direct、compile-method 与 runtime-gap claim 分开。
4. 为每个高价值 Supernode 生成完整方法翻译卡，而非只写“融合”。
5. 用 NPU all-rank/regime evidence计算 exposed upper bound；缺失时写最小补采动作。
6. 按 Candidate Contract 排序，并说明为什么通信候选被降权。
7. 不执行历史 patch 时也要产出预期正确性门、negative control 与 NPU adaptation ladder。

## 召回检查点

以约 495.6 秒 baseline、257.5 秒最终结果作为历史校准：

| 召回范围 | 历史里程碑 | 约占总历史 gain |
|---|---:|---:|
| porting debt、native op、大 tensor selective fusion | 354.6s | 约 59% |
| 再加 host sync、native CE、通信调度、loss dtype/storage | 298.5s | 约 83% |
| 再加 attention logical work-domain/API hint | 269.6s | 约 95% |
| 再加 layout 与 saved-tensor reuse | 267.4s | 约 96% |
| 再加 custom-kernel ABI/pipeline/lifetime | 257.5s | 100% |

“历史 gain 90% 召回”要求 Agent 未获具体 API 配方时仍把 attention work-domain 排到前列。约 349 秒的早期结果只召回约 63% 总历史增益，不能作为 90% 通过线。

## 通过标准

- 召回上述前四类高价值机制，尤其是 source-direct loss mismatch 和 attention work-domain；
- 正确识别 extension 为独立 regime 或 evidence gap；
- 不把 GPU 时间当 NPU floor，不把 kernel 名当 semantic mapping；
- 不把 payload、raw duration 或峰值内存变化直接等同 wall-time gain；
- 为 custom kernel 给出完整 tensor、边界 tile、重复 launch、buffer lifetime、gradient 与所有 regime 的门禁；
- 报告 opportunity-retrieval coverage、实际/预测 gain 与不确定性，而不是只报告最终时间。
