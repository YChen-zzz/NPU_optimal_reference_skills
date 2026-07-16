# Skill 流程强制化改进

> 基于 Ankh3-large NPU 推理优化实战中暴露的根因，将 skill 从"知识丰富但流程松散"改进为"知识+流程门禁"。
> 核心设计原则：**门禁的结构固定（流程层面），触发条件和内容由 profiling 数据与模型代码动态驱动，不硬编码任何 case-specific 的阈值或模式。**

## 问题背景

在一次完整的 6 轮 NPU 推理优化中，agent 遗漏了多个高价值优化点（buffer 预分配、flat forward、自定义 KV cache、npu_add_rms_norm 等）。skill 中已有全部所需知识，但 agent 没有调用。

根因不是知识缺失，而是**执行流程没有强制调用知识的路径**。

## 设计原则：泛化门禁 vs 定制门禁

### 什么不是泛化的（不采用）

- "H/D > 3x 时 level 4-5 不可跳过" — 阈值和 level 编号来自一个 case
- "empty_tensor > 10K 时 buffer 预分配必选" — 阈值和具体优化来自一个 case
- 预填的 Transformer 模式→融合算子匹配表 — 只适用于 Transformer

### 什么是泛化的（采用）

门禁**结构**固定，**触发条件**来自 skill 自身的数据和框架：

| 门禁 | 结构（固定） | 触发条件（动态） | 数据来源 |
|------|------------|----------------|---------|
| 脚本完整性 | 每个脚本必须运行+写发现摘要 | 脚本自身的 DEFINITE/warning 信号 | 脚本输出 |
| 优先级覆盖 | 每个优先级必须显式标注状态 | step_trace 瓶颈类型 → profiling_to_action.md 映射 | 数据驱动 |
| 操作审计 | 热路径每个操作填四维度表 | 表中任何"是"→必须产候选 | 模型代码驱动 |
| 方向放弃 | 放弃前测框架实现+自定义实现 | 失败原因为"框架 overhead"→必须测自定义 | 实测结果驱动 |
| 融合算子匹配 | 列出模型计算模式→查可用算子→填表 | `dir(torch_npu)` 动态发现 + 模型代码动态提取 | 运行时驱动 |
| 穿透深度 | 列出调用链每层→量化 host 开销占比 | 任何层 > 10% total host time → 必须有候选 | profiling 数据驱动 |

## 修改文件清单

| 文件 | 改动 |
|------|------|
| `model_opt/SKILL.md` | 确认节点 A 增加优先级覆盖门禁 |
| `model_opt/02_bottleneck_analysis/SKILL.md` | "典型工作流"→强制脚本检查清单 |
| `model_opt/02_bottleneck_analysis/references/proactive_source_analysis.md` | 穿透层级量化 + 操作审计表 |
| `model_opt/03_optimization/references/decode_optimization.md` | 方向放弃标准 |
| `model_opt/03_optimization/references/npu_operator_reference.md` | 动态匹配模板 |
