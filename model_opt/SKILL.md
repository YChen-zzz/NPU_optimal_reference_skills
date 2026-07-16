---
name: NPU 模型适配优化
description: 指导将深度学习模型适配到昇腾 NPU 并进行性能优化的全流程技能。当用户提到模型适配、NPU 优化、profiling 分析等任务时触发。
---

# NPU 模型适配优化

## 核心原则

- **Profiling 驱动**：所有优化决策必须有 profiling 数据支撑
- **源码定位根因**：Profiling 只给出"哪里慢"，必须深入功能性源码的具体实现来定位"为什么慢"
- **微基准先行**：大改动前先写独立 benchmark 验证方向
- **精度优先**：每次改动后必须跑精度验证
- **NPU ≠ GPU**：GPU 经验不能直接迁移，必须实测
- **记录一切**：成功和失败方案都要记录
- **checkpoint 兼容**：权重优化不改变 `state_dict` key，确保加载兼容
- **峰值决定 OOM**：显存优化关注同时存活张量的总量，而非累计分配量

## 标准化操作规范

Profiling 采集、精度对比等操作必须遵循统一规范，确保一致性和可复现。详见 [references/standardized_operations.md](references/standardized_operations.md)。

## 优化四维度

所有具体的性能优化手段，本质上只做四件事：**去重**（消除重复/无效的工作）、**复用**（对已有资源重复利用而非重新创建）、**掩盖**（用并行让延迟不可见）、**替换**（用硬件更友好的等价实现）。详见 [03_optimization/SKILL.md](03_optimization/SKILL.md)。

## 全流程

> **术语：一个「优化阶段」= Phase 2 → Phase 4 的一轮迭代**（profiling 分析 + 优化实施 + 精度/收益确认）。下方 Phase 2–4 构成一轮，Phase 5 提交后若瓶颈转移则回到 Phase 2 开启下一轮。

```
Phase 1  前期准备
         └─ 采集 L0 基线（全程仅一次，作为最终收益判定基准）
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
 ★ B  用户确认提交（展示本批总结 → 确认/回退）
   ↓
Phase 5  工程化提交（git commit + 更新优化日志）
   ↓
   └──→ 瓶颈转移？──→ 回到 Phase 2 开启下一轮（重新采集 L1 分析新瓶颈）
```

> **三个采集级别在流程中的落点**（L0/L1/L2 为 profiling 采集级别，与上文精度验证的 Level 1/Level 2 是不同概念）：
> - **L0**：Phase 1 采一次作基线；每个优化阶段的 Phase 4 采一次做收益快速比对。
> - **L1**：每个优化阶段的 Phase 2 开始前采集，交分析模块定位优化点（迭代回环时每轮都重新采）。
> - **L2**：仅当某轮 L1 信息不足以定位优化点时，在同一 Phase 2 改采。
> 级别定义与代码模板见 [01_preparation/SKILL.md](01_preparation/SKILL.md)「采集级别选择」及 [profiling_collection.md](01_preparation/references/profiling_collection.md) §1。

## 用户确认节点

### 确认节点 A：优化方案审核（Phase 2 → Phase 3 之间）

Phase 2 分析完成后、进入实施前，**必须**向用户展示优化方案并等待确认：

1. 列出所有建议的优化点，每条包含：优化内容、预期收益、风险等级
2. 使用 `ask_user_question` 询问用户：
   - 方案整体是否合适
   - 是否有需要跳过/不实施的优化点
   - 是否有额外想尝试的方向
3. 根据用户反馈调整优化清单，仅实施用户确认的条目

#### 优先级覆盖门禁（展示候选前必须完成）

`profiling_to_action.md` 的优先级列表定义了优化方向的理论收益排序。向用户展示候选前，必须为**每个优先级**填写下表。

瓶颈类型由 `parse_step_trace.py` 的输出判定（Host-Bound / Compute-Bound / Memory-Bound / Allocator-Bound）。`profiling_to_action.md` 定义了每种瓶颈类型最相关的优先级。

| 优先级 | 类型 | 状态 | 备注 |
|-------|------|------|------|
| 1 | 显式同步 | □有候选 / □已排除(附依据) / □不适用 | |
| 2 | 在线编译 | □有候选 / □已排除(附依据) / □不适用 | |
| 3 | 图编译 | □有候选 / □已排除(附依据) / □不适用 | |
| 4 | allocator 同步 | □有候选 / □已排除(附依据) / □不适用 | |
| 5 | 框架 dispatch | □有候选 / □已排除(附依据) / □不适用 | |
| 6 | 碎片算子融合 | □有候选 / □已排除(附依据) / □不适用 | |
| 7 | 数据布局 | □有候选 / □已排除(附依据) / □不适用 | |
| 8 | kernel 本身慢 | □有候选 / □已排除(附依据) / □不适用 | |

**门禁规则**：
- 对于 `parse_step_trace.py` 判定的瓶颈类型，`profiling_to_action.md` 映射到该类型的优先级**不可**标记"不适用"——必须有候选或附 profiling 数据依据的"已排除"
- "已排除"必须引用具体脚本输出作为依据（如"`parse_operator_memory.py` 显示无高频分配"），不可凭空判断

#### Line A 产出完整性门禁

Line A（源码分析）的产出必须包含以下两项分析结果，且其中的发现必须对应候选：

1. **穿透层级量化**（见 [proactive_source_analysis.md](02_bottleneck_analysis/references/proactive_source_analysis.md)「穿透层级量化」）：
   - 调用链中任何层贡献 > 10% 的 total host time → 该层**必须**有对应候选

2. **热路径操作审计表**（见 [proactive_source_analysis.md](02_bottleneck_analysis/references/proactive_source_analysis.md)「热路径操作审计表」）：
   - 审计表中任何维度标记为"是"或"否(不随输入变)"的操作 → 该操作**必须**有对应候选
   - 用 `parse_op_statistic.py` / `parse_operator_memory.py` 数据量化影响范围，占比 <1% 的可跳过

### 确认节点 B：提交前审核（Phase 4 → Phase 5 之间）

本批优化的精度验证和 profiling 确认均通过后、git commit 前，**必须**向用户展示本批总结并等待确认：

1. 总结本批实施的所有优化点及实际效果（性能数据 + 精度数据）
2. 列出未采纳的方案及原因
3. 使用 `ask_user_question` 询问用户：
   - 是否确认提交本批优化
   - 是否需要回退某些改动
4. 用户确认后才执行 git commit

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

**Phase 1 前期准备**：理解模型代码、搭建 NPU 环境、准备测试数据、构建 profiling 采集脚本和精度验证脚本。关键产出：可复现的**基线性能数据（L0 采集，全程仅一次，作为后续每轮收益判定的固定基准）** + 可一键运行的验证脚本。

> L0 基线与各优化阶段 Phase 2 的 L1 是**两次目的不同的采集**：L0 基线只采一次、贯穿全程用于收益对比；L1 每轮迭代前都重新采集、用于定位当轮优化点。级别定义与代码模板见 [01_preparation/SKILL.md](01_preparation/SKILL.md)「采集级别选择」及 [profiling_collection.md](01_preparation/references/profiling_collection.md) §1。

**Phase 2 瓶颈分析**：
- **Line B (先做)**:采集 **L1**(信息不足时改采 **L2**),跑脚本,用五种分析模式定位可见瓶颈。
- **Line A (必做)**:通读源码(穿透框架),用四维度审视,发现结构性冗余。用 Line B 的数据量化收益。
- 两条线**都必须执行**,产出合并后进入**确认节点 A**。

**★ 确认节点 A**：向用户展示优化方案清单（每条含内容、预期收益、风险等级），询问方案是否合适、有无需要跳过的优化点。仅实施用户确认的条目。

**Phase 3 优化实施**：根据用户确认的优化清单，用四维度（去重、复用、掩盖、替换）框架选择具体手段。图编译优先尝试，失败后走 eager 路线。每条优化后做 Level 1 快速精度验证。

**Phase 4 精度验证 + Profiling 确认**：本批（本轮优化阶段）所有优化完成后，**必须依次完成**：
1. 全量精度验证（Level 2）——与原始 baseline 对比，确认精度无退化
2. 阶段末重新采集 Profiling（**L0** 快速比对）——与 L0 基线/上一轮对比，确认本批优化确实带来性能收益
3. 两项均通过后才可进入提交流程；任一不通过则回退或调整

**★ 确认节点 B**：向用户展示本批总结（优化点、性能收益、精度数据、未采纳方案），询问是否确认提交。用户确认后才执行 git commit。

**Phase 5 工程化提交**：全部工作在 optimize/ 分支进行，每批一个 commit，用户确认后合入 main。维护优化日志记录每批的优化点、修改、效果和未采纳方案。提交前按 [06_evidence_db/schema.md](06_evidence_db/schema.md) 将本轮优化案例记录到项目工作目录的 `evidence_db/` 下（与 `profiling/` 同级）。

## 迭代退出条件

- 性能达到预设目标
- Profiling 显示剩余瓶颈已无优化空间
- 进一步优化的边际收益低于工程维护成本
