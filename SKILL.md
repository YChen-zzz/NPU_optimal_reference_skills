---
name: npu-two-stage-tuning
description: NPU 训练性能两阶段优化。Stage 1 以 GPU compiled evidence 为参照做 Supernode 逐节点调优；Stage 2 切换到 Profiling 驱动的系统性瓶颈分析与四维度优化。当用户需要优化 NPU 训练性能、有 GPU baseline 可参照时触发。
---

# NPU 两阶段训练性能优化

## 核心思路

GPU compile 提供了"哪些 ops 可以融合/消除"的 ground truth，但它无法覆盖所有优化空间（host 开销、通信、内存管理等）。两阶段策略：先用 GPU Teacher 快速拿到对齐增益，再用 Profiling 深挖剩余空间。

## Stage 1: GPU Teacher Supernode 调优

**目标**：对齐 GPU compile 已证明可行的优化（fusion、消除冗余 cast、API 参数对齐等）。

**方法**：从 GPU `ir_post_fusion.txt` 提取 fusion groups → 划分 Supernode → 逐 SN 在 Lab 中穷举 L0-L6 方案 → 多卡 ablation 验证 → 组合进 full training。

**详见**：[skills_phase1/SKILL.md](skills_phase1/SKILL.md)

## 阶段切换判定

Agent 在满足以下**任一**条件时，可以判定 Stage 1 结束并切换到 Stage 2：

1. **所有高优先级 Supernode 的 L0-L4 已测完**：progress.md 中高优先级 SN 的所有格子都已填满（✅ 或 ❌+原因）
2. **GPU Teacher 不可用/不充分**：无 IR 数据、或 IR 覆盖不全（如只有 forward 无 backward），无法继续 Supernode 分析

**切换时必须产出**：
- 当前累计优化结果（step_avg 改善量）
- 各 SN 的最终状态（已优化 / 平台限制 / 未覆盖）
- 明确声明："Stage 1 结束，切换到 Stage 2 Profiling 驱动优化"

## Stage 2: Profiling 驱动优化

**目标**：用 Profiling 数据定位 Stage 1 未覆盖的瓶颈（host 开销、通信、内存、小算子碎片等），做系统性深层优化。

**方法**：构造短跑脚本 → 采集 L1 → 双线分析（Line A 源码 + Line B Profiling）→ 下界分析确定方向 → 四维度优化实施 → 精度验证 → 收益确认 → 迭代。

**详见**：[skills_phase2/SKILL.md](skills_phase2/SKILL.md)

**Stage 2 的起点特殊性**：
- Stage 1 已完成的优化是 Stage 2 的 baseline（不重做）
- Stage 2 的 Phase 1（准备）可跳过已完成的部分（环境已搭好、短跑脚本可能已有），只需补充未完成的产出
- Stage 2 的首轮 L1 profiling 采集的是 Stage 1 优化后的代码——分析的是"Stage 1 之后还剩什么瓶颈"

## 全局规则

1. **不回退 Stage 1 的成果**：Stage 2 的优化建立在 Stage 1 之上，不允许撤销已通过 ablation 的 Stage 1 优化（除非 Stage 2 发现了效果更好的实现）
2. **短跑脚本共享**：Stage 1 的多卡短跑脚本可直接作为 Stage 2 的短跑脚本（或在其基础上调整压缩比）
