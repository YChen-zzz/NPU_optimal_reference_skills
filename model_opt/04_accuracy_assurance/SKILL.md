---
name: npu-accuracy-assurance
description: 为 NPU 优化自动建立正确性门禁；按输出语义、精度合同和候选风险，验证算子输出、中间层、梯度、训练短跑及最终任务质量，并在失败时定位首个分歧点。由 npu-model-optimization Phase 4 调用，也适用于独立的 GPU/NPU 精度对齐与精度调试。
---

# Phase 4：自动正确性与性能门禁

本阶段接受统一 Candidate Contract，不区分候选来自源码、NPU Profiling 或 GPU Teacher。GPU 结果可提供语义和编译意图证据，但不能代替原始精度 baseline。

## 1. 冻结验证合同

修改前记录：

- 原始 accuracy baseline 与当前 iterative performance baseline；
- 被修改产物的 shape、dtype、layout、mask/work-domain、state mutation 和下游消费者；
- 距离函数、阈值、代表输入、边界输入和受影响 regime；
- 训练候选是否影响 backward、saved tensor、optimizer 或 collective ordering；
- 计时区间、warmup、同步方式和噪声阈值。

阈值必须在看到优化结果前声明。baseline 选择见 [baseline_policy.md](references/baseline_policy.md)。

## 2. 选择距离函数

距离函数必须匹配下游语义：

| 产物 | 最低比较集合 |
|---|---|
| 连续张量、logits、embedding | cosine + max absolute/relative error |
| loss、energy、score | absolute + relative error |
| 概率或 attention 分布 | max error + KL/JS，按消费方式补充 |
| token、标签、离散索引 | 完全匹配率或任务允许的等价规则 |
| 图像、结构、物理场 | LPIPS/FID、RMSD/TM-score、空间 RMSE 等领域指标 |

模型输出不能确定性决定最终任务时，必须验证下游功能指标。模型家族提示见 [model_family_hints.md](references/model_family_hints.md)。

## 3. 自动验证阶梯

只执行 Candidate Contract 要求和风险传播所需的层级；失败立即停止性能采纳并进入定位。

1. **静态合同**：源码路径、shape、dtype、layout、mask/work-domain、alias、state 和随机性。
2. **局部数值**：同输入下比较修改区域的完整输出与边界 case；融合候选同时比较融合区入口和出口。
3. **模型传播**：比较受影响中间层、最后一层和模型最终输出，定位误差是否被放大。
4. **训练语义**：受影响时比较 loss、gradient、saved tensor、参数更新、optimizer state 与 collective ordering。
5. **加权短跑**：按真实 regime 出现次数构造短程训练；覆盖所有受影响 regime，比较 loss/validation loss、grad norm、稳定性和 wall-clock。
6. **完整任务**：在 wave 里程碑、重大高风险修改和最终版本运行，验证最终任务指标、稳定性与端到端时间。

短跑步数不是固定常数。Agent 根据完整运行时长、状态覆盖、编译摊销、噪声和候选风险选择最小充分步数；约 100 step 仅是常见起点。多 batch/shape 训练按实际出现比例采样，并显式覆盖 transition step。

完整任务也不是每个 Action 都运行。若 full run 很便宜可提高频率；否则在 compatible bundle/wave 通过短跑后运行。最终采纳必须至少有一次完整任务结果。

## 4. 性能门

正确性通过后才判性能：

- 使用无 profiler wall-clock 作为采纳依据，比较当前 iterative baseline；
- 计时覆盖相同代码范围、regime 混合与同步边界；
- 收益必须超过测量噪声，且内存、编译时间和稳定性未越界；
- L0/L1 用于解释收益，不要求每个 Action 重采；重新 Profiling 由主流程的 stall 条件触发。

## 5. 自动决策

| 结果 | 动作 |
|---|---|
| 正确性通过且收益显著 | accept，更新 iterative baseline |
| 正确性通过但收益在噪声内 | reject_no_gain；若优先候选连续如此，触发 stall 审计 |
| 局部正确性失败 | reject_accuracy，回退并按 [debugging_guide.md](references/debugging_guide.md) 定位 |
| 仅短跑通过 | provisional；不得作为最终完成状态 |
| 某 regime/rank 未覆盖 | incomplete，补最小覆盖后再判定 |
| 内存、编译或稳定性退化 | reject_constraint，记录触发条件 |

失败试验必须保留候选、补丁、环境、输入、阈值、差异首发位置和失败 predicate，避免后续重复。

## 6. 精度调试

验证失败时，从最终产物逆向定位：

1. 复现相同输入、权重、seed 和 state；
2. 找到第一个产生显著差异的 regime/step；
3. 逐模块或二分比较，定位首个分歧的 Supernode/算子；
4. 检查 dtype 收窄、`.float()` 边界、mask/window/work-domain、广播、layout、in-place/alias、随机数、saved tensor 与 collective 顺序；
5. 修正后从最低失败层级重新进入验证阶梯。

等价替换的局部门禁见 [equivalence_verification.md](references/equivalence_verification.md)。
