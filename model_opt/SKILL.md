---
name: npu-model-optimization
description: 模型性能优化全流程（profiling 驱动的瓶颈分析 → 四维度优化 → 精度验证 → 工程化提交）。当用户想优化模型性能、推理/训练速度、分析 profiling 数据、定位性能瓶颈、或在昇腾 NPU 上做模型适配与调优时触发。
---

# NPU 模型适配优化

## ⚠ 启动协议

无论从哪个子技能进入，执行前必须：
1. 确认当前在哪个 Phase（参见下方「全流程」）
2. 确认上一个 Phase 的产出已完成
3. 按全流程顺序执行，不跳步

## 核心原则

- **Profiling 驱动**：所有优化决策必须有 profiling 数据支撑
- **源码定位根因**：Profiling 只给出"哪里慢"，必须深入功能性源码的具体实现来定位"为什么慢"
- **微基准先行**：大改动前先写独立 benchmark 验证方向
- **精度优先**：每次改动后必须跑精度验证
- **NPU ≠ GPU**：GPU 经验不能直接迁移，必须实测
- **记录一切**：成功和失败方案都要记录
- **checkpoint 兼容**：旧 checkpoint 必须仍可加载且数值等价。默认保持 state_dict key/结构不变；当优化涉及结构性融合（如 QKV 合并、layout 重排）必须改变结构时，须提供与模型同处的确定性重映射函数，并通过等价性验证（remap 加载后输出与未优化模型对齐）
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
 ★ B  用户确认提交（展示本批总结 + evidence_db 已记录 → 确认/回退）
   ↓
Phase 5  工程化提交（git commit + 更新优化日志）
   ↓
 ★ C  用户确认是否继续（展示本轮总结 + 剩余瓶颈 → 继续/停止）
   ↓
 ├─ 继续 → 回到 Phase 2 开启下一轮（重新采集 L1 分析新瓶颈）
 └─ 停止 → 结束
```

> **三个采集级别在流程中的落点**（L0/L1/L2 为 profiling 采集级别，与上文精度验证的 Level 1/Level 2 是不同概念）：
> - **L0**：Phase 1 采一次作基线；每个优化阶段的 Phase 4 采一次做收益快速比对。
> - **L1**：每个优化阶段的 Phase 2 开始前采集，交分析模块定位优化点（迭代回环时每轮都重新采）。
> - **L2**：仅当某轮 L1 信息不足以定位优化点时，在同一 Phase 2 改采。
> 级别定义与代码模板见 [01_preparation/SKILL.md](01_preparation/SKILL.md)「采集级别选择」及 [profiling_collection.md](01_preparation/references/profiling_collection.md) §1。

## 执行协议（agent 程序约束）

三个用户确认节点控制迭代流程，详细门禁（优先级覆盖表、Line A 完整性门禁、提交/继续审核）详见 [execution_protocol.md](references/execution_protocol.md)：

- **★A 方案审核**（Phase 2→3）：展示候选清单（按收益上限降序），须完成优先级覆盖门禁 + Line A 完整性门禁。仅实施用户确认的条目。
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

**Phase 5 工程化提交**：全部工作在 optimize/ 分支进行，每批一个 commit，用户确认后合入 main。维护优化日志记录每批的优化点、修改、效果和未采纳方案。**必须先按 [06_evidence_db/schema.md](06_evidence_db/schema.md) 将本轮优化案例记录到项目工作目录的 `evidence_db/` 下（与 `profiling/` 同级），再执行 git commit**——案例未记录不允许提交（见确认节点 B 第 3 步）。

## 迭代退出条件

由用户在确认节点 C 中决定是否继续。以下信息供 agent 在展示时参考：

- 性能是否达到预设目标
- Profiling 显示剩余瓶颈是否还有优化空间
- 进一步优化的边际收益是否低于工程维护成本
