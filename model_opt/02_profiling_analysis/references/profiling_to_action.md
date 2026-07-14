# Profiling 分析推理指南

## 核心方法论: 五种分析模式

脚本给出数据和疑点,但从疑点到优化决策之间需要**推理**。推理不是套用固定 pattern,而是运用以下五种分析模式——它们覆盖了所有性能分析场景,agent 遇到预定义路径覆盖不到的新情况时,用这五种模式自行构造推理。

### 模式 1: 横向关联(广度结合)

同一个问题从多个 profiling 文件/维度同时观察,交叉验证收敛到可靠结论。

- **目的**: 消除单信号歧义——单个脚本的输出往往有多种解释,多个来源指向同一方向才可信
- **方法**: 针对同一怀疑点,分别从不同文件(step_trace / kernel_details / trace_view / operator_details / memory_record)提取相关信息,检查它们是否收敛
- **何时用**: 单一脚本给出模糊判断时(如 step_trace 利用率 50% 附近难判 host/device bound);需要排除"是不是 profiler 自身开销造成的假象"时
- **关键**: 各文件侧重不同维度——step_trace 给总体分布、kernel_details 给单 kernel 硬件细节、trace_view 给时序因果、operator_details 给源码映射。真问题会在多个维度同时留痕

### 模式 2: 纵向深入(沿一个点逐层钻入)

从高层信号出发,层层递进到更细粒度的数据,直到定位到源码行级的根因。

- **目的**: 从"模糊的慢"逐步收窄到"具体哪行代码、为什么慢、该怎么改"
- **方法**: 每一步用更细粒度的文件/工具回答上一步留下的"为什么"
- **何时用**: 已通过异常定位或横向关联确定一个可疑点,需要追到根因时
- **典型路径**: op_statistic(哪类算子最耗时) → kernel_details --filter(该算子为什么慢: shape/硬件占比/Block Dim) → operator_details --filter(谁触发的: Call Stack) → 源码阅读(为什么这样写)
- **关键**: 不要跳层——先确认"确实是这个算子的问题"再深入,避免在错误方向上浪费精力

### 模式 3: 差异对比(两个状态的 delta)

比较两份 profiling 数据,变化量本身就是信息。

- **目的**: 隔离变量——不看绝对值,看什么变了什么没变
- **方法**: 对比同一指标在两个状态下的差异
- **适用对比**:
  - 优化前 vs 优化后: 确认收益来自预期改动(用 diff_profiling)
  - L0 vs L1 采集: 分离 profiler 注入开销(L1 的 Free/Preparing 可能被 profiler barrier 夸大)
  - GPU vs NPU: 确定差距在哪个环节(如计算时间相近但 NPU 多出 dispatch gap)
- **何时用**: 判断优化是否生效;区分 profiler 开销与真实瓶颈;跨平台定位差异根源
- **关键**: 对比的两份数据必须只有一个变量不同(相同输入/相同 schedule/相同 step)

### 模式 4: 异常定位(分布中找离群)

看同类数据的分布,找打破 pattern 的异常点。

- **目的**: 从大量正常数据中精确定位少数真正有问题的点
- **方法**: 看分桶/排序/分位数,找显著偏离均值的条目
- **何时用**: 全局统计看起来"还行"但实际有隐藏瓶颈时;需要从上千 kernel 中圈定嫌疑对象时
- **关键**: 离群 = 线索,不等于结论。发现离群点后需用模式 2(纵向深入)或模式 1(横向关联)确认它是真问题还是正常变异
- **脚本已做的**: Suspect Kernels(高耗时低利用率)、Top Stalls(聚合后最大的空隙)、dispatch latency top-N——这些本质都是异常定位

### 模式 5: 时序因果(时间轴上的前后关系)

利用事件的时间顺序推断因果关系。

- **目的**: 建立"谁导致了谁"/"谁阻塞了谁"的因果链
- **方法**: 在 trace_view 的时间线上观察事件的前后关系——A 总是紧挨在 B 之前出现,说明 A 可能导致/阻塞了 B
- **何时用**: 知道"哪里慢"(如某处有大 gap)但不知道"为什么慢"时;需要区分"是 host 来不及喂"还是"device 在等某个同步"时
- **关键**: parse_trace_view 的 stall 聚合(按 kernel 对)是时序因果的结构化输出;compile 事件紧贴在 device gap 前 = 编译导致的因果

### 五种模式的协作

实际分析不是只用一种模式,典型组合顺序:

```
1. 异常定位 → 从全局数据中圈定嫌疑点(从上千 kernel 缩到几个)
2. 纵向深入 → 沿嫌疑点逐层钻到候选根因
3. 横向关联 → 用其他文件交叉验证候选根因(单来源不可信)
4. 时序因果 → 在 trace 中确认事件间的因果方向
5. 差异对比 → 优化后确认收益确实来自预期改动
```

不是每次分析都走完五步——简单问题可能 1→2 就定位了;复杂问题可能在 2↔3 之间反复。

---

## 参考路径(已验证的信号组合实例)

以下是经真实数据验证的典型信号组合,作为**快速匹配入口**——agent 遇到匹配的组合可直接走对应方向,遇到覆盖不到的新场景则用五种模式自行推理。这里只保留每类瓶颈最具代表性的一条,更多组合靠模式推导。

### Host-Bound + compile 贯穿全程

step_trace 利用率低 + trace_view compile B 类(贯穿全程) + kernel_details 硬件占比正常

→ **每步在线编译**,非算子问题,是执行模式问题
→ 分析模式: 横向关联(step_trace + trace_view + kernel_details 三方收敛)
→ 行动：关 jit_compile / 固定 shape / 图编译

### 利用率高 + mte_ratio >> mac_ratio

step_trace 利用率高 + kernel_details 硬件单元中 mte（搬运）远大于 mac（计算）

→ **Memory-Bound**：kernel 在忙但大部分时间在搬数据而非计算
→ 分析模式: 纵向深入(--filter 缩到具体算子和 shape)
→ 行动：shape 不友好考虑 pad/合并；全局带宽瓶颈考虑减少同时存活大 tensor

### 某算子 mac 高 + Block Dim 满 + cube_util 高

kernel_details 中某算子 mac_ratio 高 + Block Dim 已达硬件上限 + cube_utilization 高

→ **真 Compute-Bound**：计算密集且硬件利用已充分
→ 分析模式: 异常定位(从分布中找已是上限的点) → 判定终局
→ 行动：考虑量化/换算法降复杂度,或判定为终局(无优化空间)

### memory_record 高频抖动 + 同尺寸反复分配 + empty host 耗时高

memory_record 高频抖动 + operator_memory 同尺寸反复分配 + trace_view 中 empty 有高 host 耗时

→ **Allocator-Bound**: 反复申请释放同一 buffer
→ 分析模式: 横向关联(memory_record + operator_memory + trace_view 交叉验证)
→ 行动：预分配 + `out=` 写入(复用维度)

### 小算子 > 50% + Block Dim=1 多 + 利用率低

→ **Decode 场景碎片化**：per-token shape 太小导致并行度不足 + dispatch 占比高
→ 分析模式: 异常定位(avg duration <20us 离群)
→ 行动：fp16/bf16 启用融合算子减少 kernel 数,或图编译

### Communication Wait% > 80% + step_trace 通信占比高

communication.json 中 Wait Time 占总通信时间 >80% + step_trace Communication 列占比高

→ **同步瓶颈（非带宽问题）**：通信时间几乎全在等，不在传
→ 分析模式: 横向关联(communication + trace_view + step_trace 三方收敛)
→ 行动：查通信-计算重叠（hide_latency）、查 straggler rank（某 rank 慢导致其他等）、减少同步点

---

## 脚本信息不够时的深入方法

当脚本输出不足以做判断时，直接读原始文件:

| 想了解什么 | 去哪里 | 看什么 |
|-----------|--------|--------|
| 某算子的实际 input shape | `kernel_details.csv` | Input Shapes 列 |
| 某次分配时系统内存有多满 | `operator_memory.csv` | Allocation Total Allocated(MB) |
| 某算子在 forward 中出现的位置序列 | `kernel_details.csv` | 按 Start Time 排序搜目标算子 |
| 完整的 Python 调用链 | `operator_details.csv` | Call Stack 列 |
| 两个 kernel 之间的真实 gap | `kernel_details.csv` | Start Time - 上一个的 (Start Time + Duration) |
| 某个 step 的独立数据 | `kernel_details.csv` | 按 Step Id 列过滤 |
| L0 vs L1 采集差异 | 两份 profiling | 分别运行脚本对比，Level1 bubble 可能被 profiler barrier 夸大 |

---

## 多问题并存时的优先级

按收益确定性和实施风险排序:

```
1. 显式同步（.item / .numpy / H2D / empty_cache）→ 消除（单点修复，收益确定，零风险）
2. 在线编译→ 关 jit_compile / 固定 shape（根因级修复，收益大）
3. 图编译可行？→ 尝试（收益上限最高，但可能不兼容）
4. allocator 同步（empty_tensor 占比高）→ 预分配（收益确定，改动较大）
5. 框架 dispatch 开销 → flat forward（收益大，改动大）
6. 碎片算子融合 / 等价替换（融合算子、换 API）→ 逐个验证
7. 数据布局（Transpose 多）→ 预转置 + 改布局
8. kernel 本身慢（Compute-Bound）→ 降精度 / 换算法 / 判定终局
```
