# NPU 两阶段训练性能优化 Skill

面向昇腾 NPU 训练负载的两阶段自动性能优化。Stage 1 利用 GPU compiled evidence 快速对齐已知优化；Stage 2 切换到 Profiling 驱动做深层瓶颈分析和系统性优化。

## 工作流

```text
Stage 1: GPU Teacher Supernode 调优
         ├─ 提取 GPU compile IR → 划分 Supernode
         ├─ 逐 SN 创建 Lab → L0-L6 穷举 → 多卡 ablation
         └─ 组合优化 → full run 确认
   ↓
 [阶段切换判定]
   ↓
Stage 2: Profiling 驱动优化
         ├─ 构造短跑脚本 → 采集 L1
         ├─ 双线分析 (Line A 源码 + Line B Profiling)
         ├─ 四维度优化实施 → 精度验证 → 收益确认
         └─ 迭代直至终局
```

## 目录

```text
NPU_two_stage_tuning/
├── SKILL.md                    ← 两阶段流程编排
├── README.md                   ← 本文件
├── skills_phase1/              ← Stage 1: GPU Teacher Supernode 调优
│   └── tuning_v2/
│       ├── SKILL.md
│       └── references/
└── skills_phase2/              ← Stage 2: Profiling 驱动优化
    └── model_opt/
        ├── SKILL.md
        ├── references/
        ├── 01_preparation/
        ├── 02_bottleneck_analysis/
        ├── 03_optimization/
        ├── 04_accuracy_assurance/
        ├── 05_engineering/
        └── 06_evidence_db/
```

## 使用

将本目录放入 Agent 的 skills 目录。Agent 根据 `SKILL.md` 的 description 自动触发。

```bash
ln -sfn $(pwd)/NPU_two_stage_tuning ~/.kerminal/skills/npu_two_stage_tuning
```
