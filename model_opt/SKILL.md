---
name: npu-model-optimization
description: 自动优化昇腾 NPU 上的训练、推理或科学计算负载；建立可复现基线，通过源码分析与 NPU Profiling 定位瓶颈，在提供或检测到编译后 GPU Teacher evidence pack 时自动启用跨后端 Supernode 对齐，随后实施优化、验证精度和性能、记录证据并用 Git 管理试验。用于 NPU 性能调优、GPU→NPU 迁移、Profiling 分析、训练加速和跨后端优化。
---

# NPU 模型自动优化

在授权工作区内自动完成分析、试验、验证和 Git 记录。正常候选不设置人工确认节点；只有外部授权、破坏性操作或任务范围扩张需要询问。

## 核心规则

1. 先锁定语义、精度、评测合同和 execution regime，再解释性能。
2. Line A 源码分析和 Line B NPU Profiling 始终执行；满足路由条件时增加 Line T GPU Teacher。
3. GPU Teacher 同时提供 source-port gap、compile method guideline 和 runtime gap；迁移优化机制，不照搬 GPU kernel，并用 NPU 实测决定优先级与采纳。
4. 所有路线输出同一 Candidate Contract；Phase 3、4、5 不区分候选来源。
5. 候选按加权关键路径收益、证据、可行性、正确性风险和实现成本排序。
6. 使用可回滚 iterative baseline；失败试验保留证据，但不进入当前最佳分支。
7. 沿当前 backlog 持续优化；收益停滞或证据失效后才重新采集高开销 Profiling。
8. 主 Agent 不得直接宣布“没有优化空间”；只能提交 `stop_proposed`，并由独立 Stop Auditor 审计。

## Phase 1：准备

读取 [01_preparation/SKILL.md](01_preparation/SKILL.md)，完成：

- 工作负载语义、训练/推理路径和 precision contract；
- regime 触发条件、shape/dtype/work-domain、出现频率和 transition；
- 一次可比的原始 NPU wall-clock + L0 baseline；已有可信 baseline 直接复用；
- 精度 baseline、验证阈值和自然波动；
- Git 安全点、工作分支和未提交改动保护；
- GPU Teacher pack 的路径、manifest 或缺失状态。标准模式不跨机器自动采 GPU。

## Phase 2：三线候选生成

读取 [02_bottleneck_analysis/SKILL.md](02_bottleneck_analysis/SKILL.md)。

```text
Line A  源码结构：发现冗余、复用、并行和等价实现机会
Line B  NPU Profiling：量化当前关键路径、算子、kernel、内存和通信
Line T  GPU Teacher：恢复 source-port gap、GPU 已用方法与 compile 意图（条件启用）
   ↓
合并、去重、补证
   ↓
统一 candidates.csv / action sheets
```

当用户明确要求使用 GPU Teacher，Line T 为必选；当 evidence pack 可用且门控通过时自动启用；其余情况运行 Line A+B。Line T 细则位于 [GPU Teacher 分析](02_bottleneck_analysis/gpu_teacher/SKILL.md)。

强 source-port semantic gap 不等待精确 Profiling 才能成为候选；它以 unmeasured/provisional 状态进入低成本补证或试验。NPU Profiling 负责关键路径定量、全局排序和最终性能验收。

Phase 2 结束条件不是“列出想法”，而是每个候选均满足 [Candidate Contract](02_bottleneck_analysis/references/candidate_contract.md)，并完成显著信号的候选或证据化排除。

## Phase 3：自动实施

读取 [03_optimization/SKILL.md](03_optimization/SKILL.md)。

- 先执行高收益、高置信、低/中风险 Action；兼容 Action 可组成 bundle。
- 实现顺序：删除/缓存 → 官方 NPU API/参数 → layout/API 改写 → selective compile → schedule/custom autograd → custom kernel。
- 每项保留独立 commit、开关或 patch，能够消融和回退。
- 每个 Action 后运行最低成本正确性门和低开销计时；不默认重采 L1。

## Phase 4：自动门禁

读取 [04_accuracy_assurance/SKILL.md](04_accuracy_assurance/SKILL.md)。按风险逐级执行：

1. 静态语义、shape、dtype、mask/work-domain 和 state；
2. 修改区域完整输出、边界 case、受影响中间层和最后一层；
3. 受影响时验证 gradient、saved tensor、optimizer state 和 collective ordering；
4. 训练负载执行按真实 regime 比例构造的短跑；推理负载覆盖代表和边界输入；
5. 比较原始 accuracy baseline，并用当前 iterative baseline 判断性能收益；
6. wave 里程碑和最终版本执行完整任务。

正确性、性能、内存或稳定性任一失败，自动拒绝或隔离该 trial。

## Phase 5：证据与 Git

读取 [05_engineering/SKILL.md](05_engineering/SKILL.md) 和 [06_evidence_db/schema.md](06_evidence_db/schema.md)。通过门禁后自动：

- 写入 artifacts、候选、trial、失败 predicate 和平台发现；
- 提交到优化分支并更新 iterative baseline；
- 不覆盖用户未提交修改；
- 不自动合并稳定主分支，除非当前任务已明确授权。

## Wave、重新 Profiling 与停止

一次 Phase 2 分析及其 backlog 构成一个 optimization wave。候选仍产生超过计时噪声的收益时继续当前 wave。

满足任一条件时采集新的 NPU L1，并重新运行 Phase 2：

- 高置信低/中风险 backlog 已接受、拒绝或阻塞；
- 优先 Action 通过正确性门但收益落在计时噪声内，且没有更高收益未测候选；
- 新热点无法由当前 candidate/supernode map 解释；
- graph、dtype、layout、memory、communication 或环境变化使旧证据失效。

停滞信号只触发恢复流程，不直接终止：Host gap 接近零时转向 graph/compiler/kernel/算法层；候选全部失败时重做候选覆盖审计；Teacher backlog 用完时采当前 best 的 NPU profile 并做 residual Teacher alignment；环境不可用时标记 `blocked_environment`，不得声称已经最优。

停止前必须完成 residual audit，并获得 Stop Auditor 的 `stop_allowed`。详细状态机与 subagent 合同见 [执行协议](references/execution_protocol.md)。
