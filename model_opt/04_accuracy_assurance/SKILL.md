---
name: npu-accuracy-assurance
description: 为 NPU 优化执行轻量正确性与性能门禁；默认比较受影响中间层和最后一层，再保持原训练设置、按比例缩短 step 与 scheduler 到约 60 秒，比较同一步的 val loss 和时间；仅对直接修改 backward、optimizer、通信或 state 的候选追加专项检查。由 npu-model-optimization Phase 4 调用，也适用于独立的 GPU/NPU 精度对齐。
---

# Phase 4：轻量门禁

GPU Teacher 提供语义和方法证据，不能代替原始 NPU accuracy baseline。原始 baseline 是固定精度参考；current iterative baseline 是性能参考。seed、数据、代码和验证合同未变时复用已有短跑结果。

## 1. 默认训练门禁

1. **模型传播**：使用相同输入比较受影响中间层、最后一层和模型输出。局部算子正确性已由 Supernode Lab 验证时不重复。
2. **60 秒缩短训练**：保留原 setting，只修改总 step 和所有 step-dependent schedule：
   - 选择使 baseline 从首个计时 step 到 validation 约 60 秒的总 step；
   - 按完整任务比例压缩 scheduler、batch/shape regime、transition 和 extension；
   - baseline 与 candidate 使用相同 seed、初始状态、数据顺序和 step；
   - 比较同一步的 val loss、必要的 loss 曲线和 wall-clock。
3. **条件专项检查**：只有候选直接修改 backward、saved tensor、optimizer/state 或 communication/collective 时，才追加对应 gradient、state 或多卡顺序检查。

val loss 容差必须在运行 candidate 前确定，优先使用原始短跑的自然波动或任务 precision contract。非训练负载使用等价的、约 60 秒内可完成的代表性任务质量门。

## 2. Lab winner 与消融

短跑默认只对比 current iterative baseline 与 Lab cumulative winner，不逐级重放 Lab。Lab 收益是局部上限；短跑收益更低不触发消融。

只有精度或资源结果偏离 Lab 时，才运行 cumulative checkpoint 或 `winner-minus-one`。资源包括显存、graph/compile 行为、稳定性和多卡状态。

启动或一次性编译使 60 秒不可达时，记录原因并优先复用进程或缓存；不要直接跳到 full run。编译时间是否计入收益遵循正式 benchmark 口径。

## 3. Full run

有明确时间目标时：

~~~text
goal_progress =
  (baseline_time - current_projected_time)
  / (baseline_time - goal_time)
~~~

`current_projected_time` 由同口径的缩短训练外推。首次跨过新的 20% 档位时运行 full run，同一档位不重复；重大数值、optimizer、communication/state 修改、wave 里程碑和最终版本也必须 full run。

## 4. 决策

| 结果 | 动作 |
|---|---|
| 中间层/输出和 val loss 通过，收益超过噪声 | `accepted_for_iteration` |
| val loss 或模型传播失败 | `rejected_correctness`，按 [debugging_guide.md](references/debugging_guide.md) 定位首个分歧 |
| 正确性通过但无显著收益 | `rejected_performance` |
| 显存、编译、稳定性或多卡状态失败 | `rejected_constraint`；需要时做定向消融 |
| 只通过短跑 | provisional；最终交付仍需 full run |

所有结果记录输入、命令、阈值、时间、val loss、diff/commit 和失败 predicate。baseline 选择见 [baseline_policy.md](references/baseline_policy.md)，局部等价验证见 [equivalence_verification.md](references/equivalence_verification.md)。
