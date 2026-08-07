# NPU 自动性能优化 Skill

面向昇腾 NPU 训练、推理和科学计算负载的自动性能优化 Skill。主流程以 NPU Profiling 和源码分析为基础；存在可用的编译后 GPU Teacher evidence pack 时，在 Phase 2 自动增加跨后端监督信号与优化方法先验。

## 使用

将 `model_opt/` 放入 Agent 的 skills 目录。Agent 根据 `SKILL.md` 的 description 自动触发。

```bash
git clone https://github.com/YChen-zzz/NPU_optimal_reference_skills.git
ln -sfn $(pwd)/NPU_optimal_reference_skills/model_opt ~/.kerminal/skills/model_opt
```

GPU Teacher evidence pack 通常由 GPU 机器离线生成并作为输入提供；Skill 不要求 NPU 机器能够访问 GPU 机器。

## 工作流

```text
Phase 1  合同、regime、基线和 Git 安全点
   ↓
Phase 2  候选生成
         ├─ Line A：源码结构分析
         ├─ Line B：NPU Profiling 分析
         └─ Line T：GPU Teacher 对齐（条件启用）
   ↓
Phase 3  按收益、证据、风险和成本自动实施 Action
   ↓
Phase 4  自动精度/训练/性能门禁
   ↓
Phase 5  evidence_db + Git commit
   ↓
继续当前 backlog；收益停滞或证据失效后重新 Profiling
```

## 目录

```text
model_opt/
├── SKILL.md
├── references/
├── 01_preparation/
├── 02_bottleneck_analysis/
│   ├── SKILL.md
│   ├── references/
│   ├── scripts/
│   └── gpu_teacher/
│       ├── SKILL.md
│       └── references/
├── 03_optimization/
├── 04_accuracy_assurance/
├── 05_engineering/
└── 06_evidence_db/
```

## 设计原则

- GPU Teacher 迁移编译优化意图，不复制 CUDA/Triton 实现，也不把 GPU 时间当作 NPU floor。
- Line T 同时提供 Gap 定位和 GPU 已验证的优化方法先验；Phase 3 将其翻译为 NPU 原生实现，实施、验证、Git 和证据记录仍与普通 Profiling 路线共用。
- 正常优化在授权工作区内自动完成，不设置逐轮人工确认节点。
- 每个 trial 可回滚；正确性失败、收益不足或内存回退的修改不进入当前最佳分支。
- 不因接受一个 Action 就重采高开销 Profiling；仅在收益停滞、热点无法解释或旧证据失效时重采。

## Profiling 工具

`02_bottleneck_analysis/scripts/run_analysis.py` 是 CANN Profiling 的统一入口。各阈值集中在 `thresholds.py`，应按负载和芯片校准，不视为普适判据。
