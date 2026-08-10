# 自动执行协议

本协议定义 Phase 转换、trial 状态和重新 Profiling 条件。正常优化不设置人工确认节点。

## Phase 2 → Phase 3

必须满足：

- Line A/B 已完成；Line T 的路由和证据状态已记录；
- 所有显著发现已有候选或证据化排除；
- 候选符合 Candidate Contract；
- 候选已去重并标注依赖、冲突、风险和验证门；
- priority 的输入因子与依据已落盘。

进入生产代码 trial 前必须运行 `<skill-root>/model_opt/03_optimization/scripts/check_phase3_preflight.py --stage wave`；非零退出时停留在 Phase 1/2 补齐产物。选择 Action 时运行 `--stage action --candidate-id <id>`。selective compile 未通过 API-first unlock 时必须被脚本拒绝；Phase 5 使用同一命令追加 `--verify-receipt` 校验 action receipt。

若高价值 Supernode 可忠实隔离且一轮对照成本不超过一次短跑，默认运行 [Supernode Lab](../03_optimization/references/supernode_lab.md)；否则记录精确例外，不直接跳过局部补证。

## Trial 状态机

~~~text
proposed
→ instrumented
→ supernode_lab_passed | lab_not_required
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

- 改动范围与 Candidate 声明一致，Lab cumulative winner 可回退；
- 已从原 setting 生成约 60 秒的 total-step 与 scheduler/regime 映射；
- 直接修改 backward、optimizer、communication 或 state 时已指定对应专项检查；
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

训练候选默认只对比 current iterative baseline 与 Lab cumulative winner。保留原 setting，只缩短 total step，并按原比例压缩 scheduler、regime 和 transition，使 baseline 在约 60 秒完成 validation；candidate 使用相同 seed、初始状态、数据顺序和 step，比较同一步的 val loss 与时间。

Lab 收益只作为局部排序上限；短跑收益更低不触发回放。只有精度或资源结果偏离 Lab 时，才运行 cumulative checkpoint 或 `winner-minus-one` 定向消融。

有明确 `baseline_time` 和 `goal_time` 时计算：

~~~text
goal_progress =
  (baseline_time - current_projected_time)
  / (baseline_time - goal_time)
~~~

首次跨过新的 20% 档位时运行 full run，同一档位不重复。Agent 还根据以下因素决定是否提前 full run：

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

## 停滞恢复

以下信号不得直接解释为“没有优化空间”：

| 信号 | 状态与下一步 |
|---|---|
| 连续 wave 收益落在阈值或噪声内 | `stalled`：采当前 best 的 NPU L1，重新运行 Phase 2 |
| Host gap 接近零 | `host_layer_exhausted`：关闭 Host 候选族，转向 graph/compiler/kernel/算法层 |
| 当前候选全部失败或处理完 | `coverage_audit_required`：检查遗漏 Supernode、gap family、regime 和未走完的实现阶梯 |
| Teacher backlog 用完 | `teacher_residual_required`：采新 NPU profile，复用 GPU pack 做 residual Teacher alignment |
| 环境、权限或依赖无法继续 | `blocked_environment`：保存精确 blocker 与恢复条件，不得标记最优 |

恢复流程必须产生新候选、最小补证动作或可审计的排除。不能只把旧候选重新排序。

## Stop Proposal 与独立审计

主 Agent 只能写入 `stop_proposed`，不得自行转为完成。可调用 subagent 时，必须启动一个独立、只读的 Stop Auditor；向它提供 current best revision、最新 NPU profile、Teacher manifest/pack、coverage、Supernode、candidate、trial、gap bound 和 full-run artifacts，不提供期望结论。

Stop Auditor 的任务是反证停止主张：检查证据新鲜度、regime/rank/Supernode 覆盖、Teacher residual、未测候选、实现阶梯、失败 predicate 和剩余 exposed gain。输出固定为：

~~~yaml
decision: continue | stop_allowed | blocked
failed_gates: []
uncovered_gaps: []
next_candidates: []
next_minimum_evidence: []
blocking_predicates: []
~~~

- `continue`：至少给出一个符合 Candidate Contract 的候选或最小补证动作；主 Agent 返回 Phase 2/3。
- `blocked`：记录精确外部 blocker 与重新打开条件；不得写“已最优”。
- `stop_allowed`：仅在以下停止门全部通过时允许。

## 停止门

停止前全部满足：

- 最新 NPU profile 对应 current best revision；
- Line T 启用时已完成 residual Teacher alignment；
- regime/rank/Supernode 和 gap family coverage 完整；
- 没有未处理的高置信或 `unmeasured source_direct` 候选；
- 高价值候选已走完适用实现阶梯，或有精确失败/阻塞 predicate；
- 各 gap family 的剩余加权 exposed gain 低于计时噪声或明确成本阈值；
- accuracy、training dynamics、最终 full run 和复现命令通过；
- Stop Auditor 返回 `stop_allowed`。

预算耗尽、实现困难或单次失败只是暂停/阻塞理由，不等同于达到最优。
