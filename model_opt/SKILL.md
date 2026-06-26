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

## 优化三原语

所有具体的性能优化手段，本质上只做三件事：**去重**（消除重复/无效的工作）、**复用**（对已有资源重复利用而非重新创建）、**掩盖**（用并行让延迟不可见）。详见 [03_optimization/SKILL.md](03_optimization/SKILL.md)。

## 全流程

```
Phase 1  前期准备
   ↓
Phase 2  Profiling 分析
   ↓
 ★ A  用户确认优化方案（展示方案 → 确认/裁剪）
   ↓
Phase 3  优化实施（每条改动后 Level 1 快速验证）
   ↓
Phase 4  门禁验证
         ├─ ① 全量精度验证（Level 2）
         └─ ② 重新 Profiling 确认收益
   ↓
 ★ B  用户确认提交（展示本批总结 → 确认/回退）
   ↓
Phase 5  工程化提交（git commit + 更新优化日志）
   ↓
   └──→ 瓶颈转移？──→ 回到 Phase 2 继续迭代
```

## 用户确认节点

### 确认节点 A：优化方案审核（Phase 2 → Phase 3 之间）

Phase 2 分析完成后、进入实施前，**必须**向用户展示优化方案并等待确认：

1. 列出所有建议的优化点，每条包含：优化内容、预期收益、风险等级
2. 使用 `ask_user_question` 询问用户：
   - 方案整体是否合适
   - 是否有需要跳过/不实施的优化点
   - 是否有额外想尝试的方向
3. 根据用户反馈调整优化清单，仅实施用户确认的条目

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
| Phase 2 | [02_profiling_analysis/SKILL.md](02_profiling_analysis/SKILL.md) | 定位瓶颈、解读 profiling 数据 |
| Phase 3 | [03_optimization/SKILL.md](03_optimization/SKILL.md) | 实施具体优化手段 |
| Phase 4 | [04_accuracy_assurance/SKILL.md](04_accuracy_assurance/SKILL.md) | 验证精度、调试精度问题 |
| Phase 5 | [05_engineering/SKILL.md](05_engineering/SKILL.md) | 代码管理、日志、版本控制 |

## 各阶段要点

**Phase 1 前期准备**：理解模型代码、搭建 NPU 环境、准备测试数据、构建 profiling 采集脚本和精度验证脚本。关键产出：可复现的基线性能数据 + 可一键运行的验证脚本。

**Phase 2 Profiling 分析**：采集性能数据，定位瓶颈类型（Compute-Bound / Host-Bound / Memory-Bound / Allocator-Bound），输出优先级排序的优化清单。含显存峰值分析方法。分析完成后进入**确认节点 A**。

**★ 确认节点 A**：向用户展示优化方案清单（每条含内容、预期收益、风险等级），询问方案是否合适、有无需要跳过的优化点。仅实施用户确认的条目。

**Phase 3 优化实施**：根据用户确认的优化清单，用三原语（去重、复用、掩盖）框架选择具体手段。图编译优先尝试，失败后走 eager 路线。每条优化后做 Level 1 快速精度验证。

**Phase 4 精度验证 + Profiling 确认**：本批所有优化完成后，**必须依次完成**：
1. 全量精度验证（Level 2）——与原始 baseline 对比，确认精度无退化
2. 重新采集 Profiling——确认本批优化确实带来性能收益
3. 两项均通过后才可进入提交流程；任一不通过则回退或调整

**★ 确认节点 B**：向用户展示本批总结（优化点、性能收益、精度数据、未采纳方案），询问是否确认提交。用户确认后才执行 git commit。

**Phase 5 工程化提交**：全部工作在 optimize/ 分支进行，每批一个 commit，用户确认后合入 main。维护优化日志记录每批的优化点、修改、效果和未采纳方案。

## 迭代退出条件

- 性能达到预设目标
- Profiling 显示剩余瓶颈已无优化空间
- 进一步优化的边际收益低于工程维护成本
