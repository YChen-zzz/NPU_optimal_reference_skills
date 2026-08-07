---
name: npu-optimization-preparation
description: 为自动 NPU 优化建立可复现合同、execution regime、一次原始基线、精度基线、Profiling 入口、GPU Teacher evidence pack 状态和 Git 安全点。由 npu-model-optimization Phase 1 使用，也适用于独立的 NPU 性能采集准备。
---

# Phase 1：准备

目标是取得“足够开始分析”的可信状态，不重复已经存在且可比的证据。

## 1. 恢复工作负载合同

从源码、配置、命令、日志和评价脚本自动恢复：

- 任务语义、输入到 loss/输出的完整路径；
- training/eval、forward/backward/optimizer/validation/checkpoint 范围；
- shape、dtype、layout、mask/window/sparsity、work-domain 和 state；
- batch、gradient accumulation、world size、rank 与通信；
- 正确性指标、性能指标、优化方向和约束。

不把“能运行”视为合同完整。关键验收语义存在多个互斥解释且无法从现有证据判断时，才请求外部输入。

## 2. 建立 Regime Map

execution regime 是会改变 graph、kernel、精度、工作域、状态或通信结构的运行区间，不只是 batch size。

每个 regime 记录：触发条件、transition、shape/dtype/layout/work-domain、batch/累积、训练状态、rank、完整任务出现次数和采样 step。多 batch/shape 任务分别采样 warm 后 steady step，并覆盖切换边界。

产物写入 `evidence_db/regimes.yaml`。字段见 [evidence schema](../06_evidence_db/schema.md)。

## 3. 恢复或采集一次 NPU Baseline

先检索已有 run log、benchmark、L0/L1 和 manifest。只有在缺失、不可比或来源不明时才运行一次原始 NPU baseline；不默认重复多次完整任务。

至少保存：

- 无 profiler 的端到端 wall-clock 与计时分布；
- 各 regime 的轻量 step timing；
- 一份覆盖当前关键 regime 的 NPU L0；
- Phase 2 所需的 NPU L1，若已有有效 L1 直接复用；
- 命令、source revision、environment、warmup、计时范围和 artifact identity。

重复次数由 Agent 根据噪声与任务时长确定：先做最小采样，置信区间不足再增加；不使用固定次数代替判断。采集方式见 [profiling_collection.md](references/profiling_collection.md)。

## 4. 建立 Accuracy Baseline

按 [baseline policy](../04_accuracy_assurance/references/baseline_policy.md) 自动发现可信结果；缺失上游基线时保存优化前 NPU 输出、loss/validation loss、任务指标和自然波动。

在看到优化结果前声明距离函数、阈值、代表/边界输入和训练状态。比较脚本应离线可运行并保存机器可读结论。

## 5. 发现 GPU Teacher Pack

扫描用户提供路径、项目 manifest 和 evidence_db，只登记已存在的 GPU evidence pack：

- source/config/command/environment；
- compiled graph/IR/code 与 compiled runtime profile；
- regime、rank、step window、compile/warmup 状态；
- artifact identity 与可读范围。

标准模式不从 NPU 机器跨机采 GPU。这里只记录 pack path 或缺失状态；资格与信号强度由 Phase 2 的 [Teacher gate](../02_bottleneck_analysis/gpu_teacher/references/teacher_gate.md) 判断。

## 6. Git 与产物安全

- 记录当前 repository、branch、HEAD 和 dirty files；保护用户未提交修改。
- 优先创建专用 optimize branch/worktree；无 Git 时在工作副本初始化。
- source、脚本、配置和证据摘要入 Git；raw profiling、权重、compiler cache 和大型 graph/code 只登记路径与 identity。
- 不自动重写历史、force push 或合并稳定主分支。

模板见 [工程实践](../05_engineering/SKILL.md)。

## 完成条件

进入 Phase 2 前必须具备：workload/precision contract、regime map、一次可信 NPU baseline、accuracy baseline、NPU profile 状态、Teacher pack 状态、Git 安全点和可复现命令。缺少非关键证据时记录 limitation 并继续，不用人工确认替代可恢复的信息。
