# 自动执行协议

本协议定义 Phase 转换、trial 状态和重新 Profiling 条件。正常优化不设置人工确认节点。

## Phase 2 → Phase 3

必须满足：

- Line A/B 已完成；Line T 的路由和证据状态已记录；
- 所有显著发现已有候选或证据化排除；
- 候选符合 Candidate Contract；
- 候选已去重并标注依赖、冲突、风险和验证门；
- priority 的输入因子与依据已落盘。

若 top candidate 的排序主要来自不确定性，先补最小证据或 microbenchmark，不直接改代码。

## Trial 状态机

~~~text
proposed
→ instrumented
→ local_correctness_passed
→ model_scope_passed
→ weighted_short_run_passed
→ accepted_for_iteration
→ release_verified
~~~

任一门禁可转入：

- rejected_correctness
- rejected_performance
- rejected_memory
- rejected_instability
- inconclusive_noise
- blocked_environment
- quarantined_interaction

失败 trial 保留 diff、命令、结果和精确失败 predicate，但不合入 current best。

## Phase 3 → Phase 4

进入验证前确认：

- 改动范围与 Candidate 声明一致；
- bundle 内每项可独立消融；
- semantic、precision、state、memory 和 multi-rank 风险已映射到门禁；
- 受影响 regime 和边界 shape 已知；
- 没有覆盖用户未提交修改。

## Phase 4 → Phase 5

自动接受必须同时满足：

1. 原始 accuracy baseline 的适用门禁通过；
2. 性能相对 iterative baseline 的改善超过噪声；
3. 无禁止的 per-regime/rank/memory 回退；
4. trial artifacts 和 evidence 记录完整；
5. 代码对应唯一 diff/commit。

不满足则拒绝、隔离或标为 inconclusive，不通过调整阈值事后放行。

## Full run

Agent 根据以下因素自动决定是否提前运行完整任务：

- full_run 成本；
- 剩余 wave 的试验成本；
- latent correctness risk；
- 错误 iterative baseline 导致的 downstream waste；
- 短跑是否覆盖训练动态和全部受影响 regime。

wave 里程碑、重大数值/optimizer/communication 改动和最终交付必须 full run。完整任务通过后标记 release_verified。

## Git

accepted_for_iteration 自动提交到优化分支并更新 iterative baseline。rejected trial 保留独立 commit、patch 或 diff，不进入 best branch。

不得自动覆盖用户未提交改动、重写历史、force push 或移动 release tag。是否合并稳定主分支由任务授权决定，不作为内部优化确认节点。

## 重新 Profiling

当前 wave 有高价值未测候选时继续 backlog。满足以下任一条件才采新的高开销 NPU profile：

- backlog 已接受、拒绝或阻塞；
- 优先候选通过正确性但收益均在噪声内；
- 当前 map 无法解释新热点；
- 旧 graph/profile 被改动失效；
- 环境、compiler、world size 或 regime 改变。

Line T 已启用时，重新 Profiling 后做 residual Teacher alignment；GPU pack 未改变则复用，不重复采 GPU。

## 停止审计

停止前生成 stop audit：

- regime/rank coverage；
- accuracy、training dynamics 和 full-run 状态；
- 各 gap family 的剩余 exposed gain；
- 未测高置信候选和原因；
- 最近 profile/wave 摘要；
- 当前 best commit 与可复现命令。

预算耗尽、实现困难或单次失败只是暂停/阻塞理由，不等同于达到最优。
