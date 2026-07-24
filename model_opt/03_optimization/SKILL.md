---
name: npu-optimization-implementation
description: 优化实施：用去重/复用/掩盖/替换四维度框架实施性能优化。当用户需要实施优化方案、融合算子、预分配 buffer、图编译、flat forward、或换等价实现时触发。
---

# NPU 优化实施

## 定位

本阶段承接 Profiling 分析（02）的结论，将定位到的瓶颈点转化为具体的优化方案并实施。

核心流程：Profiling 定位瓶颈 → 基于三原语选择优化手段 → 方案经用户确认后实施 → 验证精度和性能 → 用户确认后提交。

> 方案需用户确认后才可实施，每批优化须通过精度验证 + Profiling 确认收益后才可提交。详见主 SKILL.md 的确认节点 A 和 B。

## 实施前的源码理解

在设计和实施优化方案前，必须对要修改的代码有充分理解：

**全局认知**：
- 理解模型整体结构（层级、循环次数、计算路径），才能判断修改影响的范围
- 理解框架交互（HF generate / Trainer / DataLoader），才能知道哪些是框架行为、哪些是模型行为
- 了解数据流（tensor 在各模块间如何传递、shape 如何变化），才能设计正确的 buffer 预分配

**修改评估**：
- 修改点的上下游依赖：改这里会影响哪些调用方
- 正确性约束：哪些是不能改的（如 attention mask 方向、位置编码公式）
- 回退方案：改动失败时如何干净回退到原始实现

**NPU 特异性**：
- 同一段 PyTorch 代码在 NPU 和 GPU 上行为可能不同（如 4D matmul 触发物理 Transpose）
- 修改前查阅 [npu_operator_reference.md](references/npu_operator_reference.md) 确认替代算子的 dtype/shape 约束
- 用微基准验证修改方向，再做全量改动

## 优化四维度

所有性能优化手段本质上只做四件事：

| 维度 | 核心问题 | 典型现象 |
|------|---------|---------|
| **去重** | "这个工作是必要的吗？能和相邻工作合并吗？" | 同类算子调用次数异常多；存在可合并的独立调用 |
| **复用** | "这个结果/资源之后还会被需要吗？" | 相同尺寸 tensor 反复分配释放；同一计算结果被重复计算 |
| **掩盖** | "这段延迟能和其他工作并行吗？" | 通信和计算串行排列；计算流中有可填充的空泡 |
| **替换** | "同样的结果有没有硬件更便宜的等价写法？" | 某算子落 AI_CPU；一组拆解算子有官方融合算子；某 API 有 NPU 更友好的等价表达 |

前三者改变工作量/工作方式，第四者改变同一工作的物理执行路径——它们正交，可组合。

每个维度的详细原理、具体手段和代码模式见对应 reference：

| 维度 | Reference | 核心内容 |
|------|-----------|---------|
| 去重 | [eliminate_redundancy.md](references/eliminate_redundancy.md) | 合并调用、消除冗余、清理框架开销 |
| 复用 | [reuse_and_precompute.md](references/reuse_and_precompute.md) | 预计算缓存、预分配 buffer、原地操作 |
| 掩盖 | [hide_latency.md](references/hide_latency.md) | 通信-计算重叠、双 buffer 流水、图编译 |
| 替换 | [equivalent_substitution.md](references/equivalent_substitution.md) | NPU 融合算子、换等价 API、换算法 |

## 其他 Reference 索引

以下文件**不是"按需加载"**——Phase 3 开始前必须对每个文件确认读取状态：

| Reference | 读取状态 | 可执行 insight（一句话） |
|-----------|----------|------------------------|
| [eliminate_redundancy.md](references/eliminate_redundancy.md) | 已读 / 跳过(原因) | (从此文件中提取的一个可执行手段) |
| [reuse_and_precompute.md](references/reuse_and_precompute.md) | 已读 / 跳过(原因) | ... |
| [hide_latency.md](references/hide_latency.md) | 已读 / 跳过(原因) | ... |
| [equivalent_substitution.md](references/equivalent_substitution.md) | 已读 / 跳过(原因) | ... |
| [graph_compile_and_cann.md](references/graph_compile_and_cann.md) | 已读 / 跳过(原因) | ... |
| [npu_checklist.md](references/npu_checklist.md) | 已读 / 跳过(原因) | ... |
| [npu_operator_reference.md](references/npu_operator_reference.md) | 已读 / 跳过(原因) | ... |
| [decode_optimization.md](references/decode_optimization.md) | 已读 / 跳过(原因) | ... |
| [parallel_design.md](references/parallel_design.md) | 已读 / 跳过(原因) | ... |
| [training_tuning.md](references/training_tuning.md) | 已读 / 跳过(原因) | ... |

**规则**：
- "跳过"必须有具体原因（如"当前场景无自回归 decode"），不可写"按需"
- 任何文件如果被已读的文件引用（含递归引用），且自身未读，必须补充读取
- "可执行 insight"列强制 agent 从每个文件中提取至少一个可在当前 case 上执行的手段，而非泛泛读完就忘
- 原表中"加载触发条件"仅作参考——条件满足时文件状态必须为"已读"，不满足时可标"跳过"但需附原因

## 通用原则

- 每次优化后重新 Profiling，确认瓶颈是否转移
- GPU 最优实践在 NPU 可能反效果，必须实测验证
- 保留原始实现供 fallback
- 权重修改须保持 checkpoint 可加载且数值等价：默认保持 state_dict key/结构不变；当结构性融合必须改变结构时，须提供与模型同处的确定性重映射函数并通过等价性验证（详见 [reuse_and_precompute.md](references/reuse_and_precompute.md)「加载时权重重映射」）
- 优化尝试失败也要记录（what + why + 实际效果），避免重复踩坑
- **优化方向正交性**：四维度（去重/复用/掩盖/替换）的优化方向是正交的。一个方向的失败不影响其他方向。例如"融合算子替换失败"不影响"权重合并（去重）"的实施——两者独立。禁止因一个方向失败而连带放弃独立的方向
- **深度优先于广度**：对每个优化方向，穷尽探索（多种实现、完整验证）比浅尝多个方向更有价值。如果环境支持子 agent，建议对独立的方向/算子 spawn 子 agent 逐个深挖，避免因同时处理太多方向而浅尝辄止

### 方向放弃标准

在放弃一个优化方向前，**必须**满足以下全部条件：

1. **至少测试 2 种实现**：至少 1 种框架提供的实现 + 至少 1 种自定义实现（绕过框架的 bare 实现）
2. **每种实现记录**：实现描述、耗时、与基线对比、慢的原因（需 profiling 数据支撑，如 `parse_operator_details` 证实框架 overhead）
3. **失败原因归类**：概念错误 / 框架实现 overhead / 硬件不友好 / 其他
4. **若失败原因为"框架实现 overhead"**：自定义实现是**强制要求**——只有自定义实现也失败后才能放弃该方向
5. **验证步骤完整性**：每种实现必须经过"微基准 → 小样本 → 全量"三步验证。在某一步放弃时，必须填写：
   - 放弃于哪一步（微基准/小样本/全量）
   - 为什么不能继续下一步（具体数值，如"微基准 diff=X，超过阈值 Y"）
   - 是否尝试过调整参数后重试（如换 dtype、换 shape、换输入顺序）

   不合理放弃原因（自动驳回）："我觉得不会通过"（未实测）、"diff 看起来有点大"（未量化）、"和另一个方向耦合"（独立方向不应耦合，见通用原则"优化方向正交性"）

> 核心原则：框架实现 ≠ 概念验证。框架的 Cache 类、generate 函数等带有 Python 调度开销，其失败不代表优化概念本身无效。
