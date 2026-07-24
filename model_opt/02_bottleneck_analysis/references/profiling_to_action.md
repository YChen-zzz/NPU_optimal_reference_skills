# Line B 分析路线

> 方法论脊柱：**现象 → 归因**。
## 方法论脊柱

把"观察到什么"推到"属于哪类浪费"：

```
现象层  观察到什么（表象 + 瓶颈类型）
  ↓ 为什么慢 / 属于哪类开销
归因层  属于哪类浪费（识别信号 + 量化上限）
```

定位到根因后，产出候选清单（问题 + 位置 + 影响范围），进入 ★A 用户确认。优化维度选择和手段设计是 Phase 3 的工作。

### 1. 现象层：观察到什么

只回答"看起来是什么问题"，不回答为什么。从 profiling 读出表象并判定瓶颈类型：

- 设备利用率低 + Free 大 → Host-Bound（host 喂不动设备）
- 利用率高 + mac 占主导 → Compute-Bound 嫌疑
- 利用率高 + mte 占主导 → Memory-Bound 嫌疑
- Communication 占比高 → Comm-Bound 嫌疑
- empty_tensor 高频 + Free 大 → Allocator-Bound 嫌疑

瓶颈类型决定归因层往哪个方向查（Host-Bound 查 host 侧浪费、Compute-Bound 查 compute 饱和…）。判定用利用率/硬件占比/通信占比，具体阈值是负载相关默认值（见工具映射层），非普适判据。

### 2. 归因层：属于哪类浪费

把"慢"归到一类可消除的浪费。每类给**识别信号（概念）+ 量化上限（概念）**——上限是该类浪费占总时间的比例，是收益的理论天花板。排序见「收益排序」。

1. **显式同步开销**——host 主动等 device（.item/.numpy/显式 sync/empty_cache）。信号：host 侧出现 D→H 同步类操作且占比高。上限：同步类 host 时间占比。典型：HuggingFace Trainer 每步 grad clip/NaN check 调 .item()。
2. **dispatch/调度开销**——host 在框架调度（Module.__call__、hook、分发）而非计算。信号：host dispatch 时间占比高、设备 idle 但 host 在框架层忙碌。上限：dispatch 类 host 时间 / dispatch 延迟占比。
3. **内存管理阻塞**——host 卡在内存管理 API（分配/释放/映射）。信号：内存管理类 host 时间占比高、高频分配释放。上限：内存管理类 host 时间占比。
4. **在线编译/重编译**——每步重新编译算子。信号：编译事件贯穿全程（非仅预热期）。上限：编译时间占比。
5. **内存带宽受限**——kernel 在搬数据而非算。信号：mte 占比远大于 mac、带宽利用率接近峰值。上限：带宽受限段的 device 时间。
6. **compute 饱和**——计算密集且硬件已充分利用。信号：mac 高 + 并行度满 + 利用率高。上限：该 kernel 群 device 时间（且总可优化空间 <10% 时此类别为主）。
7. **布局/格式转换**——运行时 transpose/cast/format 转换。信号：非 ND 格式占比高 / Transpose·Cast 类耗时多。上限：数据搬运类耗时占比。
8. **通信同步等待**——通信时间花在等而非传。信号：通信 Wait 占比高。上限：通信等待时间。
9. **小算子碎片**——大量短 kernel 串行。信号：短 kernel 占比高、kernel 数极多。上限：碎片段累计耗时。
10. **延迟未掩盖**——存在可并行的独立工作但未重叠。信号：device idle 但有可并行计算、多流并发占比低。上限：可掩盖的 idle/延迟。

> 归因用五种分析模式推理（见后文），不套固定 pattern。多类浪费并存时各自量化上限，按上限排序。

## 收益排序（横切操作）

多类浪费并存时，按归因层的**量化上限降序**排候选——上限是该类浪费占总时间的比例。`parse_step_trace` 的可优化空间是总上限，各归因类别的开销占比是单项上限。

- 主排序：按上限降序（数据驱动，非固定全局序；随瓶颈类型变——Host-Bound 时 host 侧类别靠前，Compute-Bound 时 compute 饱和到最前）。
- 次级筛选（决定本轮是否实施，不进主排序）：单点零风险→直接做；根因级需验证→微基准先行；高风险→子分支探索；终局（compute 饱和且可优化空间<10%）→判定无空间即止。

## 五种分析模式（推理方法）

归因层推理用的思维工具。脚本给数据和疑点，从疑点到决策需要推理——不是套固定 pattern，而是用以下五种模式自行构造。

### 模式 1：横向关联（广度结合）
同一问题从多个文件/维度交叉验证收敛。目的：消除单信号歧义，多来源指向同方向才可信。何时用：单脚本判断模糊（如利用率 50% 难判 host/device）、需排除 profiler 自身开销假象时。关键：真问题会在多维度同时留痕。

### 模式 2：纵向深入（沿一个点逐层钻入）
从高层信号层层钻到源码行级根因。典型路径：op_statistic（哪类算子最耗时）→ kernel_details --filter（为什么慢）→ operator_details --filter（谁触发）→ 源码（为什么这样写）。关键：不跳层，先确认"确实是这个算子的问题"再深入。

定位到源码后，沿调用链双向追溯：

- **向上追溯**（谁调用了它）：这个函数被哪个循环/模块调用？循环次数和 profiling 中的算子 count 是否一致？是框架代码在调用（如 `Module.__call__` → `forward`）还是项目代码直接调用？调用频次异常时，向上找是哪层循环/递归导致的。
- **向下深入**（它内部做了什么）：函数内部的控制流有没有推理时走了不必要的路径（如 `if training` / `if self.use_cache`）？每次调用是否都重新分配 tensor？是否有 `.contiguous()` / `.transpose()` / `.to()` 触发了物理拷贝？

### 模式 3：差异对比（两个状态的 delta）
看变化量而非绝对值。适用：优化前后（确认收益）、L0 vs L1（分离 profiler 注入开销）、GPU vs NPU（定位差距环节）。关键：只允许一个变量不同。

### 模式 4：异常定位（分布中找离群）
看分布找显著偏离的少数点。关键：离群=线索≠结论，需模式 2/1 确认。脚本的 Suspect Kernels、Top Stalls、dispatch top-N 本质都是异常定位。

### 模式 5：时序因果（时间轴前后关系）
用事件时间顺序推断因果——A 总紧挨 B 之前 → A 可能导致/阻塞 B。何时用：知道"哪里慢"不知"为什么"、需区分"host 喂不动"还是"device 等同步"时。

**协作顺序**（典型，非固定）：异常定位圈定嫌疑 → 纵向深入钻到候选根因 → 横向关联交叉验证 → 时序因果确认因果方向 → 差异对比确认收益。简单问题 1→2 即定位，复杂问题在 2↔3 反复。

## 工具映射（可替换层）

方法论里的每个"信号/上限"由哪个脚本/字段回答。本层可替换——脚本改 section 不影响方法论正文。

| 归因类别 | 识别信号来源 | 量化上限来源 |
|---|---|---|
| 显式同步 | operator_details Host Time by Category(sync) + api_statistic(sync) | sync 类 host 时间占比 |
| dispatch 调度 | operator_details Host Time by Category(dispatch) + trace_view dispatch latency | dispatch 类 host 时间 / dispatch-kernel ratio |
| 内存管理阻塞 | api_statistic(memory-mgmt) + operator_memory 重复分配 + trace_view idle 成因 | memory-mgmt 类 host 时间占比 |
| 在线编译 | trace_view Suspect 编译分类(A/B) + api_statistic(compile) | 编译时间占比 |
| 内存带宽受限 | kernel_details mte/mac + trace_view counter HBM 带宽 | 带宽受限段 device 时间 |
| compute 饱和 | kernel_details 真 compute-bound + step_trace 可优化空间 | kernel 群 device 时间 |
| 布局转换 | kernel_details 非 ND 格式 + op_statistic data movement | 搬运类耗时占比 |
| 通信同步 | communication Wait% + step_trace 通信占比 | 通信等待时间 |
| 小算子碎片 | kernel_details fusible + short kernel ratio | 碎片段累计耗时 |
| 延迟未掩盖 | trace_view stream concurrency + idle 成因 | 可掩盖 idle/延迟 |

## 脚本信息不够时的深入方法

当脚本输出不足时直接读原始文件：

| 想了解什么 | 去哪里 | 看什么 |
|---|---|---|
| 某算子实际 input shape | kernel_details.csv | Input Shapes 列 |
| 某次分配时系统内存多满 | operator_memory.csv | Allocation Total Allocated(MB) |
| 某算子在 forward 中的位置序列 | kernel_details.csv | 按 Start Time 排序搜目标算子 |
| 完整 Python 调用链 | operator_details.csv | Call Stack 列 |
| 两个 kernel 间真实 gap | kernel_details.csv | Start Time − 上一kernel 的(Start+Duration) |
| 某 step 独立数据 | kernel_details.csv | 按 Step Id 列过滤 |
| L0 vs L1 采集差异 | 两份 profiling | 分别跑脚本对比，L1 bubble 可能被 profiler barrier 夸大 |
