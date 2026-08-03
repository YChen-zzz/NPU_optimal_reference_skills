# Line B 分析路线：从现象到源码根因

> 方法论脊柱：**现象 → 归因 → 源码定位**。
> 下界分析（[bound_analysis.md](../../references/bound_analysis.md)）是 Phase 2 前置步骤，为此处瓶颈定位提供方向参考和优化空间上界。

## 方法论脊柱

```
现象层  观察到什么（表象 + 瓶颈类型）
  ↓ 为什么慢 / 属于哪类开销
归因层  属于哪类浪费（识别信号 + 量化上限）
  ↓ 哪行源码导致的
源码层  源码中的具体位置 + 根因分析
```

定位到根因后，产出候选清单（问题 + 位置 + 影响范围 + 反事实收益上限），进入 ★A 用户确认。候选收益评估方法见本文 §候选评估。优化维度选择和手段设计是 Phase 3 的工作。

### 1. 现象层：观察到什么

只回答"看起来是什么问题"，不回答为什么。从 profiling 读出表象并判定瓶颈类型：

- 设备利用率低 + Free 大 → Host-Bound（host 喂不动设备）
- 利用率高 + mac 占主导 → Compute-Bound 嫌疑
- 利用率高 + mte 占主导 → Memory-Bound 嫌疑
- Communication 占比高 → Comm-Bound 嫌疑
- empty_tensor 高频 + Free 大 → Allocator-Bound 嫌疑

瓶颈类型决定归因层往哪个方向查（Host-Bound 查 host 侧浪费、Compute-Bound 查 compute 饱和…）。判定用利用率/硬件占比/通信占比，具体阈值是负载相关默认值（见工具映射层），非普适判据。

### 2. 归因层：属于哪类浪费

把"慢"归到一类可消除的浪费。每类给**识别信号（概念）+ 量化上限（概念）**——上限是该类浪费占总时间的比例，是收益的理论天花板。候选排序见 §候选评估。

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

## 分析模式（推理方法）

归因层推理用的思维工具。脚本给数据和疑点，从疑点到决策需要推理——不是套固定 pattern，而是用以下两种模式自行构造。

> 脚本已自动完成异常定位（Top-N 排序、DEFINITE/SIGNAL 信号、Suspect Kernels），agent 直接以脚本输出的显著信号为起点，不需要自行"找离群"。

### 模式 1：横向关联（广度结合）

同一问题从多个文件/维度交叉验证收敛。目的：消除单信号歧义，多来源指向同方向才可信。何时用：单脚本判断模糊（如利用率 50% 难判 host/device）、L0 vs L1 交叉验证（分离 profiler 注入开销）、跨平台对比（GPU vs NPU 定位差距环节）时。关键：真问题会在多维度同时留痕。

### 模式 2：纵向深入（沿一个点逐层钻入到源码）

从高层信号层层钻到源码行级根因，是定位瓶颈的主线模式。典型路径：op_statistic（哪类算子最耗时）→ kernel_details --filter（为什么慢）→ operator_details --filter（谁触发）→ 源码（为什么这样写）。关键：不跳层，先确认"确实是这个算子的问题"再深入。Call Stack 断桥（"(no stack)"）时，用 Input Shapes 反推计算语义，或切换到 Line A 穿透框架层。

模式 1 是模式 2 的补充——当模式 2 单线钻入的结论有歧义时，用模式 1 从其他维度交叉验证。简单问题模式 2 即可定位，复杂问题在 1↔2 间反复。

## 源码定位

> 归因层确定了"属于哪类浪费"后，需要从 profiling 数据跨接到源码具体位置——这是方法论脊柱的第三层（源码层）。

Profiling 只能告诉你"哪个算子慢"，要动手优化必须先把它跨接到"源码里哪一行/哪个模块"。这一跨接依赖 profiling 文件里几个特定字段作为"桥"。

> 前提：这些桥全部依赖采集参数正确。若采集缺失对应开关，字段不会生成，桥即断裂——此时不能假装能精确定位，只能按降级路径做有限推断。采集参数见 [profiling_collection.md](../../01_preparation/references/profiling_collection.md)。

### 桥接工具

| 桥 | 字段 / 文件 | 采集前提 | 作用 |
|----|------------|---------|------|
| **Call Stack** | `operator_details.csv` 的 `Call Stack` 列 | `activities` 含 CPU + `with_stack=True` | 唯一能把一次算子调用映射到 Python 源码函数/行号的字段 |
| **Input Shapes** | `kernel_details.csv` / `operator_details.csv` 的 `Input Shapes` 列 | `record_shapes=True` | 区分同一算子类型的不同调用点；Call Stack 断桥时，可从 shapes 反推计算语义（如 one-hot 向量 × 权重 = gather） |
| **下发时序** | `trace_view.json` 的 HostToDevice flow、`Node@launch` 的 `connection_id`、`async_npu(torch_to_npu)` flow、`AscendCL@opCompile` 事件（`parse_trace_view.py`） | NPU 采集即有；Python 调用栈/源码映射需 `with_stack=True` | host→device 下发链与在线编译停顿（A 预热 / B 每步）。`async_npu` flow 本身携带 `cpu_op` 的 Call Stack，因此下发时序是包含 Call Stack + host-device 时序关系的 richer 数据源——适合 host-device 交互类问题（如设备空等、下发延迟），而非"某个慢算子"类问题。**NPU 推理场景注意**：异步流水线（TASK_QUEUE_ENABLE=2）下 host-device 时序 gap 可能是 profiler 伪影，需先用 L0 交叉验证确认 gap 真实性再做因果推断 |

### 定位路径

定位的入口取决于问题类型：

**设备侧问题**（某算子慢/开销高）：脚本信号（op_statistic / kernel_details 发现异常算子）→ `--filter` 获取该算子的 Call Stack（来自 `operator_details.csv`）→ Call Stack 有源码信息？

- **有源码**（正常情况）：直接定位到函数/行号。如需区分同一算子类型的多个调用点，再用 Input Shapes 确认是哪种 shape。产出候选。
- **"(no stack)"**（框架动态生成的算子，如 JIT/e3nn codegen/opt_einsum_fx）：用 **Input Shapes** 反推计算语义（如 `(72,1)×(72,1)` = one-hot 向量 × 权重 = gather），推断出操作的实际含义后，切换到 **Line A 穿透框架层**：沿算子类型 → 框架入口 → 代码生成逻辑 → 生成出的算子序列，追溯该算子是哪段框架代码生成的。结论标注为"Line A 推断"。详见 [proactive_source_analysis.md](proactive_source_analysis.md)

**host-device 交互问题**（设备空闲 / 下发延迟 / 在线编译停顿）：
- 信号来源：step_trace（Free 大 / 利用率低）、trace_view（Host2Device Bound Regions、idle 成因分解）
- 入口：直接用**下发时序**（来自 `trace_view.json`，经 `parse_trace_view.py` 解析）。`async_npu` flow 本身携带 host 操作的 Call Stack——下发时序同时提供"哪个 host 操作在下发"和"设备在等什么"
- `connection_id` 配对 `Node@launch`↔设备算子，定位"设备空等 host"的时间区段和对应的 host 操作 → 从该 host 操作的 Call Stack 定位源码
- **Call Stack 断桥**（host 操作也无 "(no stack)"）：同设备侧路径，用 Input Shapes 反推或切换 Line A 穿透框架层
- **NPU 推理场景注意**：需先用 L0 交叉验证确认 host-device gap 非 profiler 伪影

例：`op_statistic` 发现 Transpose 占 15% → `operator_details --filter Transpose` 用 **Call Stack** 定位到源码行 → `kernel_details --filter Transpose` 用 **Input Shapes** 确认是哪种 shape → 产出候选（问题 + 位置 + 影响范围）。

## 候选评估：反事实收益上限

定位到根因后产出候选清单。候选排序不应基于 profiling 中的 self-time 占比（"时间花在哪"），而应基于**反事实收益上限**（"消除此浪费后端到端最多改善多少"）。self-time 占比 ≠ 优化收益——被异步流水线重叠的 host 开销虽然占比高，但消除它对端到端几乎无影响。

### 估算方法

对每个候选，在实施前估算反事实收益上限：

1. **确定该候选消除的浪费量**：
   - 设备侧浪费（如冗余 Transpose/Cast 占 L0_Computing 的比例）→ 消除量 = 该类算子的 L0_Computing 占比
   - host 侧浪费（如 dispatch 开销）→ 消除量 ≤ gap B（不能超过真实 host 开销）

2. **Amdahl 约束**：若消除量为 p（占总时间比例），端到端加速上限 = 1/(1−p)。但实际收益取决于该浪费是否在关键路径上。

3. **异步流水线修正**：
   - 若消除的是设备侧浪费（冗余算子）→ 不受异步流水线影响，实际收益 ≈ self-time 占比
   - 若消除的是 host 侧浪费 → 大部分可能已被重叠，实际收益 << self-time 占比，以 gap B 为上界

4. **标注反事实收益上限**：每条候选标注"反事实收益上限"（而非 self-time 占比），按此降序排列。

## 脚本信息不够时的深入方法

当脚本输出不足或桥梁断裂时，直接读原始文件：

| 想了解什么 | 去哪里 | 看什么 |
|---|---|---|
| 某算子实际 input shape | kernel_details.csv | Input Shapes 列 |
| 某次分配时系统内存多满 | operator_memory.csv | Allocation Total Allocated(MB) |
| 某算子在 forward 中的位置序列 | kernel_details.csv | 按 Start Time 排序搜目标算子 |
| 完整 Python 调用链 | operator_details.csv | Call Stack 列 |
| 两个 kernel 间真实 gap | kernel_details.csv | Start Time − 上一kernel 的(Start+Duration) |
| 某 step 独立数据 | kernel_details.csv | 按 Step Id 列过滤 |
| L0 vs L1 采集差异 | 两份 profiling | 分别跑脚本对比，L1 bubble 可能被 profiler barrier 夸大 |
