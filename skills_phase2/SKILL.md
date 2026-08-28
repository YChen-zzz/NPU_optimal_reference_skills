---
name: npu-model-optimization
description: 模型性能优化全流程（profiling 驱动的瓶颈分析 → 四维度优化 → 精度验证 → 工程化提交）。当用户想优化模型性能、推理/训练速度、分析 profiling 数据、定位性能瓶颈、或在昇腾 NPU 上做模型适配与调优时触发。
---

# NPU 模型适配优化

## 启动协议

无论从哪个子技能进入，执行前必须：
1. 确认当前在哪个 Phase（参见下方「全流程」）
2. 确认上一个 Phase 的产出已完成
3. 按全流程顺序执行，不跳步
4. 进入每个phase后，所有reference文件需要按需加载，确认已读/跳过状态，并在 evidence_db 中记录

## 核心原则

- **Profiling 驱动**：所有优化决策必须有 profiling 数据支撑
- **源码定位根因**：Profiling 只给出"哪里慢"，必须深入功能性源码的具体实现来定位"为什么慢"
- **精度优先**：每次改动后必须跑精度验证，未得到用户允许前，禁止对模型做量化操作
- **记录一切**：成功和失败方案都要记录
- **checkpoint 兼容**：旧 checkpoint 必须仍可加载且数值等价。默认保持 state_dict key/结构不变；当优化涉及结构性融合必须改变结构时，须提供与模型同处的确定性重映射函数，并通过等价性验证（remap 加载后输出与未优化模型对齐）
- **禁止未仔细分析就动手**：禁止直接用"量化/图编译"来解决性能问题。要先分析瓶颈、定位根因、再定制优化方案。量化/图编译是直觉上的"万能钥匙"，但并不一定能解决问题
- **不要放弃**：无论是分析定位，还是优化实施，不要只想着简单尝试就结束，而是要穷尽探索（多种实现、完整验证）

## 标准化操作规范

Profiling 采集、精度对比等操作必须遵循统一规范，确保一致性和可复现。详见 [references/standardized_operations.md](references/standardized_operations.md)。

## 优化四维度

所有具体的性能优化手段，本质上只做四件事：**去重**（消除重复/无效的工作）、**复用**（对已有资源重复利用而非重新创建）、**掩盖**（用并行让延迟不可见）、**替换**（用硬件更友好的等价实现）。详见 [03_optimization/SKILL.md](03_optimization/SKILL.md)。

## 全流程

> **术语：一个「优化阶段」= Phase 2 → Phase 4 的一轮迭代**（profiling 分析 + 优化实施 + 精度/收益确认）。下方 Phase 2–4 构成一轮，Phase 5 提交后若瓶颈转移则回到 Phase 2 开启下一轮。

```
Phase 1  前期准备
         ├─ 项目 git 初始化 + .gitignore（权重/profiling 等大文件）
         └─ 采集 L0 基线 + wall-clock benchmark（全程仅一次，作为收益判定基准）
   ↓
┌─────────────────── 一个「优化阶段」（可迭代多轮）───────────────────┐
│ Phase 2  瓶颈分析                                                  │
│          ├─ Line B: Profiling 分析（采集 L1 → 脚本 → 定位可见瓶颈） │
│          └─ Line A: 源码分析（必做,四维度审视源码,发现结构性冗余）    │
│    ↓                                                              │
│  ★ A  用户确认优化方案（展示方案 → 确认/裁剪）                       │
│    ↓                                                              │
│ Phase 3  优化实施（每条改动后 Level 1 快速精度验证）                 │
│    ↓                                                              │
│ Phase 4  门禁验证                                                  │
│          ├─ ① 全量精度验证（Level 2）                              │
│          └─ ② 阶段末重新采集 L0，与基线/上一轮快速比对确认收益        │
└──────────────────────────────────────────────────────────────────┘
   ↓
 ★ B  用户确认提交（展示本批总结 + evidence_db 已记录 → 确认/回退）
   ↓
Phase 5  工程化提交（git commit + evidence_db 记录）
   ↓
 ★ C  用户确认是否继续（展示本轮总结 + 剩余瓶颈 → 继续/停止）
   ↓
 ├─ 继续 → 回到 Phase 2（必须：重新采集 L1 → 重新分析 → 重新估算优化空间，禁止沿用上轮结论）
 └─ 停止 → 结束
```

> **三种性能测量在流程中的落点**：
> - **wall-clock**：Phase 1 采一次作基线；每个优化阶段 Phase 4 再采一次做收益确认（无 profiler 开销，是收益判定的可靠依据）。训练场景均基于短跑脚本的 step_avg。
> - **L0**：Phase 1 采一次作基线；每个优化阶段 Phase 4 采一次做收益比对和下一轮 Phase 2 的 L0/L1 交叉验证。
> - **L1**：每个优化阶段的 Phase 2 开始前采集，交分析模块定位优化点（迭代回环时每轮都重新采）。
> - **full training**：仅在最终确认时运行（训练场景），验证绝对性能（train_time + val_loss）。
> 训练场景下 wall-clock、L0、L1 均基于同一个短跑脚本采集（见 [profiling_collection.md](01_preparation/references/profiling_collection.md) §训练短跑策略）。三者必须覆盖相同代码范围。定义与模板见 [profiling_collection.md](01_preparation/references/profiling_collection.md)。

## 执行协议（agent 程序约束）

三个用户确认节点控制迭代流程，详细门禁（优先级覆盖表、Line A 完整性门禁、提交/继续审核）详见 [execution_protocol.md](references/execution_protocol.md)：

- **★A 方案审核**（Phase 2→3）：展示候选清单（按反事实收益上限降序），须完成优先级覆盖门禁 + Line A 完整性门禁。仅实施用户确认的条目。
- **★B 提交审核**（Phase 4→5）：展示本批总结（性能+精度），evidence_db 已记录才允许 git commit。
- **★C 继续确认**（Phase 5 后）：展示本轮总结 + 剩余瓶颈，询问是否开启下一轮。

## 子技能索引

| 阶段 | 子技能 | 触发时机 |
|------|--------|----------|
| Phase 1 | [01_preparation/SKILL.md](01_preparation/SKILL.md) | 项目启动、环境搭建、基线采集、脚本构建 |
| Phase 2 | [02_bottleneck_analysis/SKILL.md](02_bottleneck_analysis/SKILL.md) | 瓶颈分析:源码结构线 + Profiling 数据线 |
| Phase 3 | [03_optimization/SKILL.md](03_optimization/SKILL.md) | 实施具体优化手段 |
| Phase 4 | [04_accuracy_assurance/SKILL.md](04_accuracy_assurance/SKILL.md) | 验证精度、调试精度问题 |
| Phase 5 | [05_engineering/SKILL.md](05_engineering/SKILL.md) | 代码管理、日志、版本控制 |
| 案例库 | [06_evidence_db/schema.md](06_evidence_db/schema.md) | 优化案例的记录格式(schema 定义；案例数据存在项目工作目录 `evidence_db/` 下) |

## 各阶段要点

**Phase 1 前期准备**：理解模型代码、搭建 NPU 环境、准备测试数据、构建 profiling 采集脚本和精度验证脚本。训练场景须构造**短跑脚本**（等比压缩 total steps 到 ~60-120 秒，覆盖所有 regime，详见 [profiling_collection.md](01_preparation/references/profiling_collection.md) §训练短跑策略）。关键产出：可复现的**基线性能数据（L0 + wall-clock，全程仅一次，作为后续每轮收益判定的固定基准）** + 可一键运行的验证脚本 + 短跑脚本（训练场景）。

**Phase 2 瓶颈分析**：
- **新轮次强制重置**：每轮 Phase 2 必须从零开始——重新采集 L1（训练场景使用短跑脚本）、重新运行分析脚本、重新计算下界和 gap。上一轮的分析结论/未实施方案/优化方向全部失效（性能 profile 已变）。
- **下界分析（先做）**：计算三档下界（Roofline / L0 Computing / 对齐 wall-clock），分解 gap A/B，确定优化方向。详见 [bound_analysis.md](references/bound_analysis.md)。
- **Line B**:采集 **L1**,跑脚本,用两种分析模式定位可见瓶颈。
- **Line A (必做)**:通读源码(穿透框架),用四维度审视,发现结构性冗余。用 Line B 的数据量化收益。
- 两条线**都必须执行**,产出合并后进入**确认节点 A**。

**★ 确认节点 A**：向用户展示优化方案清单（每条含内容、预期收益、风险等级），询问方案是否合适、有无需要跳过的优化点。仅实施用户确认的条目。

**Phase 3 优化实施**：根据用户确认的优化清单，用四维度（去重、复用、掩盖、替换）框架选择具体手段。每条优化后进行Level 1 快速精度验证。实施完成后，必须回溯 Phase 2 的所有结构化产出（热路径审计表、归因层 10 类浪费、根因追踪发现），逐行验证每个 actionable 条目是否已实施或附依据排除。详见 [execution_protocol.md](references/execution_protocol.md)「Phase 3 → Phase 4 门禁」。**任何未关闭的条目阻止进入 Phase 4。**

**Phase 4 精度验证 + Profiling 确认**：本批（本轮优化阶段）所有优化完成后，**必须依次完成**：
1. 全量精度验证 —— 与原始 baseline 对比，确认精度无退化
2. 重新采集 **wall-clock + L0**（训练场景使用短跑脚本）—— wall-clock 确认真实收益，L0 与基线/上一轮比对确认收益来源（gap A/B 变化）
3. 两项均通过后才可进入提交流程；任一不通过则回退或调整
4. **阶段性 full training 确认（训练场景，建议启用）**：当自上一次 full training 以来，短跑 step_avg 累计改善 ≥ 2% 时，**建议**在当前 Phase 4 运行一次 full training 确认绝对性能（train_time + val_loss）。此建议**独立于最终优化目标**——不应因"距离目标还远"而跳过。目的是为 reviewer 提供每个有实质收益的阶段的绝对性能快照。Full training 参数从 baseline git commit 的原始脚本获取，不从短跑反推
5. **最终性能确认（训练场景）**：所有迭代轮次完成、用户确认停止后，无论累计改善是否达到 2%，都运行一次 full training 作为最终验证

**★ 确认节点 B**：向用户展示本批总结（优化点、性能收益、精度数据、未采纳方案），询问是否确认提交。用户确认后才执行 git commit。

**★ 确认节点 C（下一轮决策）**：提交完成后，向用户展示当前状态（累计收益、剩余 gap）并询问是否继续。若继续，**必须作为全新一轮执行**——上一轮的分析结论、未实施方案、优化方向全部作废，不可沿用。原因：优化改变了性能 profile，上一轮的瓶颈可能已不存在或已转移。

**Phase 5 工程化提交**：全部工作在 optimize/ 分支进行，每批一个 commit，用户确认后合入 main。**必须先按 [06_evidence_db/schema.md](06_evidence_db/schema.md) 将本轮优化案例记录到项目工作目录的 `evidence_db/` 下（与 `profiling/` 同级），再执行 git commit**——案例未记录不允许提交（见确认节点 B 第 3 步）。

## 迭代退出条件

由用户在确认节点 C 中决定是否继续。agent 应基于三档下界的 gap 分析（定义见 [bound_analysis.md](references/bound_analysis.md)）提供量化建议。满足以下**任一**条件时建议停止：

1. `wall_clock / L0_Computing < 1.1`——host 开销（gap B）已极小，Python 层优化空间耗尽
2. gap A 主导（kernel 效率差距大）且 gap B / Tier 3 < 5%——Python 层无法改善，需图编译/量化/换 CANN
3. 连续 10 轮优化均 < 0.5% wall-clock 改进
4. 所有候选被拒绝且无新候选产生

终局判断前必须穷尽 NPU 融合算子库，不能仅看 utilization 数字下结论。
