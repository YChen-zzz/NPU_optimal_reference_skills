# 局部等价验证

用于 Phase 3 中 API 替换、融合、代数改写、work-domain 缩减和 custom kernel 的最低成本门禁。它不能替代模型传播、训练短跑或最终任务验证。

## 1. 冻结合同

在实现前保存原始实现，并声明：

- 输入/输出 shape、dtype、layout、device 和 alias；
- mask/window/sparsity、有效与无效位置；
- state mutation、随机数、exception/empty-input 行为；
- forward、backward、saved tensor 和 accumulator 精度；
- 距离函数与阈值。

## 2. 输入覆盖

从每个受影响 regime 选择正常、边界和 transition 输入。覆盖最小/中位/最大 shape、batch、padding/mask 边界、空域（若合法）和 dtype 极值。固定输入、权重、seed 和初始 state 后复用。

## 3. 比较范围

按候选风险选择：

1. 修改区入口与完整出口；
2. 所有返回值、辅助输出和被修改 state；
3. 训练候选的 loss、输入/参数 gradient、saved tensor 与参数更新；
4. 下游首个消费者，确认 layout/alias/lifetime 合同未破坏。

连续张量至少比较 cosine、max absolute 和 relative error；标量比较 absolute/relative error；离散结果按任务规则比较。无效位置只有在下游确实不会读取且合同明确允许时才可排除。

## 4. 判定

- 阈值来自 dtype、原始实现自然波动和下游容忍度，必须在看到新结果前声明。
- bit-exact 只在任务合同要求且实现确定时使用；不能把 bit-exact 当作所有 NPU 训练的通用要求。
- `.float()`、accumulator dtype、reduction 顺序或 fused loss 改变时，即使 forward 接近，也必须验证 gradient 与短跑 validation loss。
- compile/internal format 等设置应与目标生产路径一致；不得为了通过局部测试切换成不会交付的执行模式。

## 5. 失败定位

先找首个失败的 regime/step，再按修改区二分。重点检查广播、mask/window、work-domain、dtype 收窄、reduction 顺序、in-place/alias、saved tensor、随机数和状态更新。

局部门禁通过后，返回 Phase 4 的模型传播与加权短跑；只有后续门禁也通过才能采纳。
