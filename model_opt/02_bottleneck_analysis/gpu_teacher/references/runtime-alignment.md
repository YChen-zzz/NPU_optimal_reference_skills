# NPU Runtime 与 Teacher Supernode 映射

本文件扩展 Line B 的 NPU Profiling：不替代现有 parser，而是规定如何把 parsed NPU evidence 与 Teacher Supernode、regime、rank 和关键路径连接起来。

## 两级覆盖

- `light_all_rank`：覆盖每个 execution regime 和 rank，保留 step、operator、kernel、communication 与 raw interval，用于找最慢/异常 rank和 wall-time accounting。
- `deep_selected_rank`：覆盖代表、最慢和异常 rank，打开 shape、stack/module、memory，用于解释源码、tensor contract 和 materialization。

CSV 只有列名或 profiler flag 声明但字段为空时，不算证据。默认 light ≥5、deep ≥3 steady step；任务成本较高时可调整并记录理由。

## 五类视图

| 视图 | 必须回答 |
|---|---|
| Step/流水 | 各 regime/rank 的 step span、compute/communication union、overlap、free/launch gap、critical rank |
| Operator | operator count/self time、shape/dtype/layout、call stack、host/sync、关联 kernel |
| Kernel | interval、stream/task、family、fragmentation、一个 Supernode 被拆成多少 kernel |
| Memory | peak/live/reserved、allocation churn、materialization、saved tensor lifetime、headroom |
| Communication | collective/payload/dtype/group、rank interval、skew、wait、non-overlap、producer/consumer |

先建立 step accounting 和关键 rank，再下钻 operator/kernel。不要从全局 top-k kernel 直接决定候选，它会丢失 regime、rank、overlap 和语义。

## 映射步骤

1. 校验 regime、rank、logical-step window、单位和非空字段。
2. 计算 interval union、overlap、non-overlap 和 slowest-rank critical span。
3. 用 call stack/module/source marker 把 operator 映射到 semantic range。
4. 用 correlation/flow、shape/dtype/layout 和 dependency neighborhood 展开 NPU kernel chain。
5. 将 NPU chain 与 Teacher pre/post Supernode 对齐；名称只作弱证据。
6. 分解 `npu_extra`：cast、transdata、copy、dispatch、sync、communication wait、extra compute、materialization。
7. 计算每个 Supernode 的 device time、exposed upper bound、p95 和 rank imbalance。
8. 按 regime 出现次数聚合，并写回 Supernode/候选。

## Source-direct 的处理

强 source-port gap 不要求先有精确 Supernode timing 才能生成候选。运行时映射负责：

- 证明对应路径是否在当前任务执行；
- 估算调用频率和收益上限；
- 决定它进入高优先级 bundle、低成本探测或仅保留；
- 验证修改后 residual work 是否消失。

因此 Profiling 控制排序和收益置信度，不否定由语义证据成立的候选存在性。

## 标准派生表

- `phase_summary.csv`：regime/rank median/p95 与 step accounting；
- `rank_imbalance.csv`：最慢/最快 rank、spread 与因果 interval；
- `operator_summary.csv`：semantic marker、count/time、shape/dtype/source；
- `kernel_summary.csv`：Supernode、kernel family、count/time/interval union；
- `communication_summary.csv`：payload、total/non-overlap/skew；
- `memory_summary.csv`：peak/churn/materialization/saved-tensor lifetime。

每行携带 artifact ID 和 capture identity，能回到 raw evidence。

## 收益规则

- compute：只计算未被独立工作覆盖的 interval union；
- communication：使用最慢 rank 的 non-overlapped interval；
- host/dispatch：只计算能因果归于该操作的 device-free gap；
- memory：直接 allocation/copy/transdata 时间与间接 enabling gain 分开；
- GPU 绝对时间不作为 NPU floor。

若无法做精确反事实调度，使用保守 non-overlapped union。实测 gain 明显超出预测区间时，标记 mapping/accounting 可能不完整并触发补证。

## 常见错误

- 最长 kernel 不一定是最佳候选，可能已接近 native floor 或被 overlap。
- rank0 不一定决定全局 wall time。
- 相同 batch 不代表相同 work-domain 或 graph regime。
- GPU fused kernel 名不要求在 NPU 存在；迁移的是方法与语义。
- collective bytes 或 raw duration 不等于端到端收益。
- 没有单个大 op 不代表没有 fragmentation、dispatch、cast 或 graph-break 机会。
