# Supernode Lab

用最小、可复现的局部负载比较同一 Supernode 的 NPU 实现。Lab 用于选择方法和支撑 aggressive update，不能代替训练短跑。

## 默认执行与例外

对每个高价值 Supernode，只要能忠实隔离且一轮候选对照成本不超过一次 60 秒短跑，就默认运行 Lab。以下情况才写 `lab_not_required:<具体理由>`：

- 无法在局部环境忠实复现语义、state、通信或 graph boundary；
- 已有环境、版本、shape、dtype 和合同完全匹配的可复用 Lab；
- 单一低风险参数修复已有强语义与性能证据；
- 保守收益上限低于测量噪声。

## 对照方法

始终保留 `current_npu` control。按实现阶梯加入适用方法：official native/fused、manual/layout/API、selective compile、schedule/autograd、custom kernel。不要为凑齐类别实现明显不适用的方法。

每条路径记录：成立条件、主要限制、版本/API 证据、预期机制和跳过条件。

不能在第一个可运行实现后停止。适用的低/中成本路径必须完成对照或留下精确跳过证据；custom kernel 仅在剩余收益上限足够时进入 Lab。

使用累计搜索：令 `B0=current_npu`；同一级的适用方案从相同 parent 出发，分别记录相对 parent 的 `marginal_gain` 和相对 B0 的 `cumulative_gain`。正确性通过且目标函数最优的方案成为下一层 parent；存在明显交互时只保留少量分支，避免组合爆炸。最终输出 cumulative winner 和可回退的中间 checkpoint。

## 输入与覆盖

- 从真实 workload 恢复 shape、dtype、layout、stride、mask/window/work-domain、state 和调用频率；随机张量必须满足这些合同。
- 覆盖全部受影响 regime 和边界/transition；只在行为相同时共享结果。
- 训练路径至少覆盖 forward、backward、输出和输入/参数 gradient；涉及 optimizer、通信或 state 时使用相应最小多卡/状态化 harness。

## 测量

在相同输入和同步边界下记录：

- output/gradient diff 与预先声明的阈值；
- warmup、compile time、first iteration、steady-state median/p95；
- 峰值显存、kernel 数、总 kernel/device time 和可见间隙；
- 每 regime 单次收益 × 真实调用频率的保守 step 上限。

compile 必须分别报告编译开销和稳态收益；不能只报告编译后的单次最快值。

## 落盘与决策

测试代码保存为 `benchmarks/supernodes/<supernode_id>.*` 或 workload 等价位置；结果保存为 `evidence_db/supernode_labs/<supernode_id>/<lab_id>.yaml`，并由 Candidate/Trial 引用。

只有正确性通过、收益超过噪声且资源约束可接受的方法才能进入训练短跑。Lab 收益是局部上限，只用于方法排序；短跑收益更低不构成 Lab 偏离，也不触发 checkpoint 回放。无明确胜者时标记 `inconclusive`，记录下一项最小证据。
