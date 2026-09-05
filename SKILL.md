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

### 切换的唯一正当条件

Agent **必须**满足以下条件之一，才能判定 Stage 1 结束：

1. **正常完成**：`progress.md` 中每个 Supernode 要么 L0-L4 格子都已填满（✅ / ❌+具体原因 / skip:具体原因），要么有基于 step 占比的跳过理由（如 `skip: 占比 X%，L0-L2 无明显优化点`）。没有任何 `-`（未开始）或 🔄（进行中）状态的格子。
2. **GPU Teacher 完全不可用**（仅限以下客观事实，不接受主观判断）：
   - 用户明确声明没有 GPU 环境且无法提供 IR 数据，**或**
   - 经过实际尝试采集后，`ir_post_fusion.txt` 文件确实不存在或为空

**以下理由不构成提前切换的正当条件**：
- ❌ "IR 覆盖不全（只有 forward 没有 backward）"——有 forward IR 就继续用 forward IR 分析，backward 部分标注为 "无 IR 参照" 即可，不影响 forward 相关 SN 的完整分析
- ❌ "剩余 SN 优化空间看起来不大"——必须实际测试后用数据证明，不能凭直觉跳过
- ❌ "已经达到目标性能"——Stage 1 的目标是穷尽 GPU Teacher 能教的所有优化，不是达到某个性能数字就停
- ❌ "时间不够 / 对话太长"——记录进度到 progress.md，下次继续，不切换

### 切换前强制自检（Stage 1 完成门禁）

切换前 agent **必须**执行以下自检，并将结果输出给用户：

**Step A: 读取并展示 progress.md 全表**
```
读取 benchmarks/supernodes/progress.md，原样输出完整表格
```

**Step B: 逐行扫描未完成项**
```
遍历 progress.md 每一行、每一个格子：
- 如果存在任何 `-`（未开始）→ 阻止切换，列出未完成项
- 如果存在任何 🔄（进行中）→ 阻止切换，列出进行中项
- 每个 ❌ 必须附有具体原因（不能只写 ❌）
- 每个 skip 必须附有具体原因（如占比和 L0-L2 结论）
```

**Step C: 统计完成度**
```
输出：
- 总 SN 数: X
- 已完成 SN 数: X (所有格子已填满)
- 未完成 SN 数: X (列出每个及其缺失的格子)
- Stage 1 完成率: XX%
```

**只有 Step B 无阻止项（完成率 100%）时，才允许切换到 Stage 2。**

### 切换时必须产出

- 上述 Step A/B/C 的完整输出（作为完成证明）
- 当前累计优化结果（step_avg 改善量）
- 各 SN 的最终状态（已优化 / 平台限制 / 未覆盖）
- 明确声明："Stage 1 结束，切换到 Stage 2 Profiling 驱动优化"
- **建议运行一次 full training 确认 Stage 1 的绝对性能**（train_time + val_loss），作为 Stage 2 的 baseline。此建议独立于最终优化目标——不应因"距离目标还远"而跳过

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
