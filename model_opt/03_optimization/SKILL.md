---
name: npu-optimization-implementation
description: 自动实施 Phase 2 产生的统一 NPU 优化候选；按去重、复用、掩盖、替换四维度和分级实现阶梯选择方案，执行可回滚 trial、低成本门禁、消融和性能测量。用于官方 NPU 算子替换、融合、work-domain/精度/layout 修复、缓存、图编译、调度和自定义 kernel。
---

# Phase 3：自动优化实施

输入必须符合 [Candidate Contract](../02_bottleneck_analysis/references/candidate_contract.md)。候选来自 Line A/B/T 使用同一验证与采纳流程；Line T 候选额外携带 Teacher method guideline，用于缩小 NPU adaptation 的搜索空间。

## 0. 强制 Preflight

首次修改 workload 前创建 `evidence_db/phase3_preflight.json`，然后运行：

~~~text
python <skill-root>/model_opt/03_optimization/scripts/check_phase3_preflight.py --workspace <workload-root> --stage wave
~~~

脚本非零退出时不得开始生产代码 trial。修复缺失产物；环境无法完成时标记 `blocked_environment`。选择 Action 前再运行 `--stage action --candidate-id <id>`；Phase 5 必须引用脚本生成的 receipt。不得只在记录中声称门禁通过。字段模板由脚本的 `--print-template` 输出。

## 1. 选择 Action 或 Bundle

按以下顺序处理：

1. 过滤依赖未满足、证据为 C 且无法低成本补证、或已测收益上限低于噪声的候选；
2. 按 NPU weighted exposed gain 选择最高价值 Supernode；
3. 在节点内先搜索官方 NPU API，再处理其他适用的非 compile 路径；
4. 可忠实隔离且一轮对照成本不超过一次短跑时，默认运行 Supernode Lab；
5. 只有 Lab/独立门禁通过的高收益、低风险、机制兼容 Action 才可组成 bundle；
6. dtype/rounding、loss/backward、optimizer、communication、state/lifetime 和 custom kernel 默认隔离。

每个 Action 建立独立 trial ID、父 iterative baseline、代码 diff、命令、环境、预测收益和失败 predicate。

A/强 B 的 `source_direct` 若正确性风险和试验成本低，即使 exposed gain 尚未精确测量，也可进入独立 exploratory trial；必须先声明 negative control 和最小计时动作，不与高风险 bundle 合并，且无 NPU 性能证据时不得 accepted。

## 2. 四维度选择机制

四个维度是问题分类，不是固定实现：

| 维度 | 核心问题 | 按需读取 |
|---|---|---|
| 去重 | 工作是否必要，能否合并或缩小 work-domain | [eliminate_redundancy.md](references/eliminate_redundancy.md) |
| 复用 | 结果、buffer、weight、saved tensor 能否跨调用复用 | [reuse_and_precompute.md](references/reuse_and_precompute.md) |
| 掩盖 | 不可消除延迟能否与计算/通信重叠 | [hide_latency.md](references/hide_latency.md) |
| 替换 | 是否有 NPU 更友好的等价实现 | [equivalent_substitution.md](references/equivalent_substitution.md) |

始终扫描 [npu_checklist.md](references/npu_checklist.md) 中与当前热路径相关的已知问题。

GPU Teacher 的算法、eliminated/fused/work-domain/layout/reuse/precision/schedule 方法映射到上述维度。先验证其成立条件，再优先尝试语义等价的 NPU 翻译；不得复制 GPU-specific kernel 指令或未经证明的硬件参数。

## 3. 实现阶梯

Ascend 默认 API-first：

1. 删除、缓存、预计算、buffer 复用、修复 porting gap；
2. 官方 NPU native/fused/sparse API 或参数；
3. manual：代数、layout、storage、precision boundary、向量化或 API 改写；
4. stream、collective schedule、custom autograd/saved tensor；
5. custom kernel。

Teacher 已给出 guideline 时，不从空白重新发散：先测试 Candidate 中的 NPU adaptation options；只有不适用、失败或收益不足时才扩展搜索。方法相同不代表实现相同，NPU 侧仍按本阶梯选择官方 API、图编译、调度或 kernel。

`selective_compile` 是最后解锁的 fallback，不因 GPU Teacher 使用 compile、图上小算子多或理论上可融合而提前。进入 compile trial 前必须由 action preflight 证明：官方 NPU API 已发现并实测，适用的非 compile 路径已不适用、失败或收益不足，当前非 compile 高价值 backlog 已处理。官方实现失败不自动触发 compile 或 custom kernel。

查询官方算子时先读 [npu_operator_catalog.yaml](references/npu_operator_catalog.yaml)，再用当前环境和官方文档验证签名、dtype、shape、layout、版本和训练/反向支持。目录中的经验不得脱离版本直接当事实。

## 4. Supernode Lab

对每个可忠实隔离、且一轮对照成本不超过一次 60 秒短跑的高价值 Supernode，默认读取并执行 [supernode_lab.md](references/supernode_lab.md)。测试代码与结果必须持久化；临时命令不能作为路径已比较的证据。

只比较适用方法。至少保留当前 NPU 实现作为 control，沿实现阶梯记录单项边际收益和累计收益，并覆盖受影响 regime 的真实 shape、dtype、layout/stride 和 forward/backward。Lab 获胜只允许累计 winner 进入训练短跑，不等于最终接受。

## 5. Trial 执行

每个 trial：

1. 从 accepted iterative commit 派生；
2. 先写静态 semantic/precision/work-domain 断言；
3. 实施最小可验证改动；
4. 运行 Phase 4 指定的最低成本正确性门；
5. 使用 Supernode Lab、匹配 regime 的 microbenchmark 或低开销 step timing 测分布；
6. 根据结果标记 accepted、rejected、inconclusive 或 blocked；
7. accepted 才能更新 iterative baseline。

代码能运行不代表接受。性能改善必须超过测量噪声，且无实质性 precision、memory、regime 或 rank 回退。

## 6. Bundle 规则

可合并：

- 删除稳定冗余；
- 缓存/预计算常量；
- 修复官方 API 参数和有效 work-domain；
- 不改变数值顺序的 layout/view；
- 独立的 host sync 清理。

默认不合并：

- 改变 accumulator/rounding；
- loss/backward/optimizer；
- in-place alias 和 saved-tensor lifetime；
- collective ordering；
- selective compile 边界；
- custom kernel。

每项保留独立 commit、开关或 patch，bundle 必须能做消融。`enabling_gain` 只用于依赖和解锁排序，不与直接 wall-clock gain 相加。

## 7. 放弃与失败知识

放弃方向需要可复现依据，但不强制穷举自定义实现。

合理依据包括：

- 正确性合同证明不可等价；
- 当前平台/API 明确不支持；
- microbenchmark 与真实负载均无收益或回退；
- 保守 exposed gain 小于噪声/成本；
- 与更高价值候选冲突；
- 内存、编译或维护风险超过收益。

记录准确 predicate：环境、regime、shape、dtype、API 参数、实现、性能和重新打开条件。不要把局部失败泛化成“该方向永远不可用”。

## 8. 与重新 Profiling 的关系

每个 Action 后执行低开销计时，但不默认采新 L1。当前 backlog 仍有显著候选时继续 Phase 3。

以下情况返回 Phase 2：

- backlog 已处理；
- 优先候选收益均落在噪声内；
- 新热点无法解释；
- graph/dtype/layout/state/memory/communication 变化使旧证据失效；
- 实测收益或回退显著超出预测区间。
