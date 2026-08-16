# 执行协议（agent 程序约束）

> 本文件是迭代流程中三个用户确认节点的详细门禁。方法论（分析思路 + 方案设计思路）在 [SKILL.md](../SKILL.md) 和 [profiling_to_action.md](../02_bottleneck_analysis/references/profiling_to_action.md)，本文件只管"执行时必须完成的程序约束"。

## 确认节点 A：优化方案审核（Phase 2 → Phase 3 之间）

Phase 2 分析完成后、进入实施前，**必须**向用户展示优化方案并等待确认：

1. 列出所有建议的优化点，每条包含：优化内容、预期收益、风险等级
2. 每条方案必须标注"消除的开销类别"和"反事实收益上限"（该类别占总时间的比例，引用 parse 脚本输出）
3. 方案按"反事实收益上限"降序排列，而非实现难度
4. 使用 `ask_user_question` 询问用户：
   - 方案整体是否合适
   - 是否有需要跳过/不实施的优化点
   - 是否有额外想尝试的方向
5. 根据用户反馈调整优化清单，仅实施用户确认的条目

### 优先级覆盖门禁（展示候选前必须完成）

[profiling_to_action.md](../02_bottleneck_analysis/references/profiling_to_action.md) 的**归因层**定义了 10 类浪费（优化方向）。向用户展示候选前，为**每类浪费**填写下表。

瓶颈类型由 `parse_step_trace.py` 的输出判定（Host-Bound / Compute-Bound / Memory-Bound / Allocator-Bound）。归因层每类浪费映射到最相关的瓶颈类型。

| 浪费类别 | 瓶颈类型 | 状态 | 备注 |
|---|---------|------|------|
| 显式同步 | Host-Bound | □有候选 / □已排除(附依据) / □不适用 | |
| dispatch 调度 | Host-Bound | □有候选 / □已排除(附依据) / □不适用 | |
| 内存管理阻塞 | Host/Allocator-Bound | □有候选 / □已排除(附依据) / □不适用 | |
| 在线编译/重编译 | Host-Bound | □有候选 / □已排除(附依据) / □不适用 | |
| 内存带宽受限 | Memory-Bound | □有候选 / □已排除(附依据) / □不适用 | |
| compute 饱和 | Compute-Bound | □有候选 / □已排除(附依据) / □不适用 | |
| 布局/格式转换 | Memory-Bound | □有候选 / □已排除(附依据) / □不适用 | |
| 通信同步等待 | Comm-Bound | □有候选 / □已排除(附依据) / □不适用 | |
| 小算子碎片 | 通用 | □有候选 / □已排除(附依据) / □不适用 | |
| 延迟未掩盖 | 通用 | □有候选 / □已排除(附依据) / □不适用 | |


**门禁规则**：
- 对于 `parse_step_trace.py` 判定的瓶颈类型，归因层映射到该类型的浪费类别**不可**标记"不适用"——必须有候选或附 profiling 数据依据的"已排除"
- "已排除"必须引用具体脚本输出作为依据（如"`parse_operator_memory.py` 显示无高频分配"），不可凭空判断

### Line A 产出完整性门禁

Line A（源码分析）的产出必须包含以下两项分析结果，且其中的发现必须对应候选：

1. **穿透层级量化**（见 [proactive_source_analysis.md](../02_bottleneck_analysis/references/proactive_source_analysis.md)「穿透层级量化」）：
   - 调用链中任何层贡献 > 10% 的 total host time → 该层**必须**有对应候选

2. **热路径操作审计表**（见 [proactive_source_analysis.md](../02_bottleneck_analysis/references/proactive_source_analysis.md)「热路径操作审计表」）：
   - 审计表中任何维度标记为"是"或"否(不随输入变)"的操作 → 该操作**必须**有对应候选
   - 用 `parse_op_statistic.py` / `parse_operator_memory.py` 数据量化影响范围，占比 <1% 的可跳过

### Line B 产出完整性门禁

Line B（profiling 分析）的产出必须包含以下内容，且其中的发现必须对应候选：

1. **L0/L1 交叉验证结论**（见 [02_bottleneck_analysis/SKILL.md](../02_bottleneck_analysis/SKILL.md) Line B 流程 step 2，`run_analysis.py` 报告 A 节自动完成）：
   - 记录 L0 和 L1 的 Computing%/Free%/Utilization
   - 若 L1 Utilization 显著低于 L0（差 >20pp），标注"profiler 伪影警告"——瓶颈类型判定以 L0 为准，L1 的算子级数据仍然有效但 step_trace 的 host/Free time 不可直接作为瓶颈判据
   - 若 L0 不可用，标注"L1 未交叉验证"后方可继续，但后续判断须谨慎

2. **根因追踪记录**（见 [02_bottleneck_analysis/SKILL.md](../02_bottleneck_analysis/SKILL.md) Line B 流程 step 3 + [profiling_to_action.md](../02_bottleneck_analysis/references/profiling_to_action.md)）：
   - `run_analysis.py` 报告中所有 DEFINITE 信号和 WARNING 警告必须有对应的根因追踪记录
   - 追踪产出格式：`发现来源 | 发现内容 | 使用的桥梁 | 源码位置 | 根因 | 候选方案`
   - Call Stack 断桥（"(no stack)"）的信号，桥梁列标注"断桥 → Line A 穿透框架层"，按 Line A 方法论追溯 codegen 生成路径
   - 所有 DEFINITE/WARNING 的追踪未完成 = 不得填写优先级覆盖表

3. **候选清单**（见 [profiling_to_action.md](../02_bottleneck_analysis/references/profiling_to_action.md) §候选评估：反事实收益上限）：
   - 每条候选含：问题 + 位置 + 影响范围 + 反事实收益上限（引用报告输出数值）
   - 每条候选须标注"消除的开销类别"（对应优先级覆盖表的浪费类别）和"反事实收益上限"（对应该类浪费的量化上限）
   - 候选按反事实收益上限降序排列

## 确认节点 B：提交前审核（Phase 4 → Phase 5 之间）

### Phase 3 → Phase 4 门禁：分析产出闭环检查

Phase 3 实施完成后、Phase 4 验证前，**必须**回溯 Phase 2 产出的所有结构化分析，确认每个"actionable"发现都有对应的实施或排除。产出闭环表：

| 发现来源 | 发现内容 | 实施状态 | 证据 |
|----------|----------|----------|------|
| 热路径审计表 - 行N | (审计表中的原始内容) | 已实施 / 已排除 | 代码位置 或 profiling 依据 |
| 归因层 - 浪费类别N | (归因表中的原始内容) | 已实施 / 已排除 | 代码位置 或 profiling 依据 |
| 根因追踪 - 发现N | (根因追踪产出) | 已实施 / 已排除 | 代码位置 或 profiling 依据 |

**门禁规则**：
- "已排除"必须引用具体 profiling 数据或兼容性约束作为依据，不可凭空判断
- 任何行处于"未处理"状态 = 不得进入 Phase 4

### 方向放弃检查

对每个在 Phase 3 中标记为"已放弃"的优化方向，必须按 [03_optimization/SKILL.md](../03_optimization/SKILL.md)「方向放弃标准（分级）」中对应级别的条件填写检查记录。任何未满足对应级别条件的方向 = 不允许标记为"已放弃"。

"未做"的合理原因仅限：方向在当前环境不可行（如需要特定硬件不支持）、或被用户明确要求跳过。

---

本批优化的精度验证和 profiling 确认均通过后、git commit 前，**必须**向用户展示本批总结并等待确认：

1. 总结本批实施的所有优化点及实际效果（性能数据 + 精度数据）
2. 列出未采纳的方案及原因
3. 确认 evidence_db 案例已按 [schema](../06_evidence_db/schema.md) 记录到项目工作目录的 `evidence_db/` 下（展示案例文件路径）
4. 使用 `ask_user_question` 询问用户：
   - 是否确认提交本批优化
   - 是否需要回退某些改动
5. 用户确认后才执行 git commit

> 案例未记录 = 不允许提交。与"精度未通过 = 不允许提交"同等约束力。

## 确认节点 C：继续优化确认（Phase 5 之后）

git commit 完成后，**必须**向用户展示本轮总结并询问是否继续：

1. 展示本轮优化总结（性能提升 + 精度状态）
2. 展示当前剩余瓶颈（最新 profiling 数据）
3. 使用 `ask_user_question` 询问用户：是否继续下一轮优化？
4. 用户确认继续 → 回到 Phase 2 开启下一轮（重新采集 L1，并基于新的采集数据进行优化而不是沿着原来的路径继续做）
5. 用户确认停止 → 结束
