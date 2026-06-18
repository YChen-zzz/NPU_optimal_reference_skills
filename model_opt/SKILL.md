---
name: NPU 模型适配优化
description: 指导将深度学习模型适配到昇腾 NPU 并进行性能优化的全流程技能。当用户提到模型适配、NPU 优化、profiling 分析等任务时触发。
---

# NPU 模型适配优化

## 核心原则

- **Profiling 驱动**：所有优化决策必须有 profiling 数据支撑
- **微基准先行**：大改动前先写独立 benchmark 验证方向
- **精度优先**：每次改动后必须跑精度验证
- **NPU ≠ GPU**：GPU 经验不能直接迁移，必须实测
- **记录一切**：成功和失败方案都要记录
- **checkpoint 兼容**：权重优化不改变 `state_dict` key，确保加载兼容
- **峰值决定 OOM**：显存优化关注同时存活张量的总量，而非累计分配量

## 优化三原语

所有具体的性能优化手段，本质上只做三件事：**去重**（消除重复/无效的工作）、**复用**（对已有资源重复利用而非重新创建）、**掩盖**（用并行让延迟不可见）。详见 [03_optimization/SKILL.md](03_optimization/SKILL.md)。

## 全流程

```
Phase 1 前期准备
    ↓
Phase 2 Profiling 分析  ←─────────────────┐
    ↓                                    │
Phase 3 优化实施 → Phase 4 精度保证    │ 瓶颈转移后重新 profiling
    ↓                                    │
Phase 5 工程化 ──────────────────┘
```

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

**Phase 2 Profiling 分析**：采集性能数据，定位瓶颈类型（Compute-Bound / Host-Bound / Memory-Bound / Allocator-Bound），输出优先级排序的优化清单。含显存峰值分析方法。

**Phase 3 优化实施**：根据 profiling 结果，用三原语（去重、复用、掩盖）框架选择具体手段。图编译优先尝试，失败后走 eager 路线。每条优化后验证精度，批次完成后全量验证 + git commit，然后重新 profiling 确认瓶颈转移。

**Phase 4 精度保证**：嵌入 Phase 3 循环，每条优化后快速验证，复杂改动全量验证，批次结束后必须全量验证。验证未通过时进入精度调试流程。

**Phase 5 工程化**：全部工作在 optimize/ 分支进行，每批一个 commit，用户确认后合入 main。维护优化日志记录每批的优化点、修改、效果和未采纳方案。

## 迭代退出条件

- 性能达到预设目标
- Profiling 显示剩余瓶颈已无优化空间
- 进一步优化的边际收益低于工程维护成本
