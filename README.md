# NPU Optimal Reference Skills

### 面向 Coding Agent 的两阶段 NPU 训练性能优化 Skills


---

> **一套可复用的 Skills，引导 Coding Agent 自动完成 NPU 训练负载的两阶段性能优化——先对齐 GPU 已验证的优化，再通过 Profiling 驱动深层调优。**
>
> 使用 [Codex](https://openai.com/index/codex/) + GPT-5.6 配合本 Skills，对 [modded-nanogpt-npu](https://github.com/YChen-zzz/modded-nanogpt-npu/tree/pr/submit-v1-optimized) speedrun 排行榜中的 27 个 record 进行了 NPU 专项优化（16×Ascend 910C），平均加速 **2.0x**（已上传 20 个，剩余近期上传）。本 Skills 同时兼容 [Kerminal](https://kerminal.cn/)。

---

## ✨ 亮点

- 🤖 **Agent 驱动优化** — Coding Agent 读取 Skills 后，自动分析 GPU compiled evidence、采集 NPU Profiling、实施优化（主要使用 [Codex](https://openai.com/index/codex/) + GPT-5.6 验证，同时兼容 [Kerminal](https://kerminal.cn/)）
- 🔄 **两阶段流水线** — Stage 1 利用 GPU compile IR 作为 teacher signal 快速对齐；Stage 2 切换到 Profiling 驱动的深层瓶颈分析
- 🚀 **平均 2.0x 加速** — 在 modded-nanogpt-npu 排行榜 27 个 record 上验证（16×Ascend 910C）
- 🛠️ **NPU 原生优化** — 算子融合、自定义 Ascend C 算子、bfloat16 权重存储、buffer 复用、schedule 调优

---

## 📊 实践案例：modded-nanogpt-npu

使用 Codex + GPT-5.6 配合本 Skills，对 [modded-nanogpt-npu](https://github.com/YChen-zzz/modded-nanogpt-npu/tree/pr/submit-v1-optimized) speedrun 排行榜进行优化——该项目是 [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) 在 16×Ascend 910C 上的移植版本。

| 指标 | 数值 |
|---|---|
| 优化 record 数 | 27（已上传 20 个，剩余 7 个近期上传） |
| 平均加速 | **2.0x** |
| 单 record 最高加速 | **2.58x**（record_017: 1183s → 458s） |
| 最新 record_050 | 563s → 257s（**2.19x**） |
| 硬件 | 16×Ascend 910C NPU |

> 完整结果与日志 → [modded-nanogpt-npu (pr/submit-v1-optimized 分支)](https://github.com/YChen-zzz/modded-nanogpt-npu/tree/pr/submit-v1-optimized)

---

## 🗺️ 工作流

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
         ├─ 双线分析（Line A 源码 + Line B Profiling）
         ├─ 四维度优化实施 → 精度验证 → 收益确认
         └─ 迭代直至终局
```

---

## 📁 目录结构

```text
NPU_optimal_reference_skills/
├── SKILL.md                    ← 两阶段流程编排入口
├── README.md                   ← 本文件
├── skills_phase1/              ← Stage 1: GPU Teacher Supernode 调优
│   ├── SKILL.md
│   └── references/
└── skills_phase2/              ← Stage 2: Profiling 驱动优化
    ├── SKILL.md
    ├── references/
    ├── 01_preparation/         ← 环境搭建、短跑脚本、Profiling 采集
    ├── 02_bottleneck_analysis/ ← Profiling 解析 + 源码双线瓶颈定位
    ├── 03_optimization/        ← 四维度优化实施（去重/复用/掩盖/替换）
    ├── 04_accuracy_assurance/  ← 精度基线管理、分层验证、调试
    ├── 05_engineering/         ← 目录规划、版本管理、文档维护
    └── 06_evidence_db/         ← 优化证据归档 schema
```

---

## 🚀 快速开始

将 Skills 目录安装到 Agent 的 skills 目录。Kerminal 示例：

```bash
ln -sfn $(pwd)/NPU_optimal_reference_skills ~/.kerminal/skills/npu_two_stage_tuning
```

Codex 用户将 Skills 目录加入 workspace 即可，Agent 会自动识别。

当任务涉及 NPU 训练性能优化且有 GPU baseline 可参照时，Agent 自动触发本 Skills。

---

## ⚙️ 硬件 / 软件要求

| 项目 | 要求 |
|---|---|
| NPU | Ascend 910C |
| NPU 数量 | 16×Ascend 910C（完整训练） |
| CANN | 与 Ascend 910C 兼容的版本 |
| PyTorch | ≥ 2.1 + torch_npu |
| Python | ≥ 3.10 |
| Agent | [Codex](https://openai.com/index/codex/) + GPT-5.6 / [Kerminal](https://kerminal.cn/) |

---

## 许可证

Apache License 2.0

---

## 致谢

- [Codex](https://openai.com/index/codex/) (OpenAI) — 主要使用的 Coding Agent 平台
- [Kerminal](https://kerminal.cn/) (Autokernel / 智子芯元) — 兼容的 Coding Agent
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — 上游 NanoGPT speedrun 排行榜
- [modded-nanogpt-npu](https://github.com/zhaoyiyidan/modded-nanogpt-npu) — 昇腾 NPU 移植版
- [CANN](https://www.hiascend.com/) (华为) — 昇腾计算平台
- [wamcs/skills](https://github.com/wamcs/skills) — Skills 设计参考
