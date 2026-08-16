# Profiling 解析脚本使用指南

## 概述

本 skill 的 `scripts/` 目录（即 `02_bottleneck_analysis/scripts/`）下的脚本用于解析 CANN profiler 产出的 CSV 文件，将百万级原始数据压缩为结构化摘要，供 agent 消费后做优化决策。

这些脚本**只依赖 CANN profiler 的固定 CSV 格式**，不依赖任何特定项目的代码结构或路径。对于需要项目上下文才能判断的信息（如 Call Stack 中哪些是项目代码），脚本原样输出，由 agent 自行解读。

> **路径说明**：以下示例中的命令均以脚本名直接调用。实际执行时 agent 需使用脚本的完整路径（基于本 skill 目录的实际位置）。

## 脚本列表

| 脚本 | 对应文件 | 典型大小 | 用途 |
|------|---------|---------|------|
| `parse_op_statistic.py` | `op_statistic.csv` | ~100 行 | 算子级耗时分布，定位瓶颈类型 |
| `parse_api_statistic.py` | `api_statistic.csv` | ~百行 | CANN 运行时 API 耗时（tiling/launch/sync/memory-mgmt 分解），L1 产出 |
| `parse_step_trace.py` | `step_trace_time.csv` | ~几行 | 设备利用率，判断瓶颈在 host 侧还是 device 侧 |
| `parse_kernel_details.py` | `kernel_details.csv` | ~1K-100K 行 | Kernel 硬件单元利用、小算子、并行度、流水 stall |
| `parse_memory_record.py` | `memory_record.csv` | ~30K-1M 行 | 内存时间线，峰值定位，OOM 预判 |
| `parse_operator_details.py` | `operator_details.csv` | ~100K-20M 行 | 单算子耗时 + 完整调用栈（流式 Top-K） |
| `parse_operator_memory.py` | `operator_memory.csv` | ~10K 行 | 内存分配热点 |
| `parse_communication.py` | `communication.json` + `communication_matrix.json` | — | 多卡通信分析：时间分解、带宽、等待占比 |
| `parse_trace_view.py` | `trace_view.json` | 4MB-1GB+ | 时序：host→device 下发链、device 空隙、在线编译停顿、Call stack 源码栈 |
| `diff_profiling.py` | 两份 profiling 目录 | — | 对比两次采集的算子耗时和内存变化 |

## 通用调用方式

所有脚本遵循相同的接口约定：

```bash
python <script>.py <profiling_dir> [--rank N] [--top-k K] [--output file.txt]
```

- `<profiling_dir>`：profiling 输出的根目录，脚本会自动递归查找 `ASCEND_PROFILER_OUTPUT/`
- `--rank N`：多卡场景指定 rank，会在 `<profiling_dir>/rank_N/` 下查找
- `--top-k K`：输出的 Top-K 条目数量
- `--output`：输出到文件，默认输出到 stdout

## 各脚本详细说明

### parse_op_statistic.py

**输入**：`op_statistic.csv`（CANN 自动按算子类型聚合的统计）

**输出包含两部分**：

1. **算子排名表**（`--top-k` 控制显示数量，默认 30）：按总耗时降序列出各算子类型，每行含调用次数、总耗时、平均单次耗时、占比、累计占比。用于快速看到哪些算子消耗了最多的 device 时间。

2. **Suspect Signals**（自动生成，基于排名表的统计特征）：
   - **Top-3 集中度**：集中度高（>80%）意味着优化少数几个算子就能获得显著收益；集中度低则需要系统性优化。
   - **数据搬运开销**：Transpose/Cast/Copy 等非计算算子的占比。占比高说明数据布局不匹配或存在不必要的类型转换，应追溯源码消除。
   - **高频低耗时算子（碎片化信号）**：调用次数极多但单次很短，可能是冗余的细碎操作，存在合并或消除空间。
   - **低频高耗时算子（重型算子）**：调用少但单次很长，通常是大 shape 算子，需要看是否可以拆分或换更高效实现。

**何时使用**：每次拿到新 profiling 后首先运行，快速定位瓶颈类型。

```bash
python parse_op_statistic.py /path/to/profiling --top-k 30
```

---

### parse_api_statistic.py

**输入**：`api_statistic.csv`（CANN 运行时 API 调用耗时统计，L1 产出，L0 无）

**定位**：host dispatch 开销的 **CANN API 层**视角。与 operator_details（op 名层）、trace_view（per-instance 时间线）互补——回答"host 时间里多少是 tiling、多少是 launch、多少是 memory-mgmt、多少是 sync"。

**输出**：按 Level（acl/communication/node）汇总 + acl 层 Top API + **类别分解**（memory-mgmt / sync / tiling / launch）并标 dominant 类别。node launch count 与 trace_view Node@launch 对应。

**何时使用**：host-bound 时下钻 host 开销的 CANN API 成因（如 aclrtFreePhysical/aclrtSynchronizeStream 占比）。

```bash
python parse_api_statistic.py /path/to/profiling --top-k 20
```

---

### parse_step_trace.py

**输入**：`step_trace_time.csv`（每 step 的 Computing/Free/Communication/Preparing 时间）

**输出包含以下部分**：

1. **Overall 统计**：设备利用率（Computing / (Computing + Free)）、各时间分量总和与占比、瓶颈侧判定：
   - 利用率 <20%：严重 Host-Bound，设备空闲 >80%
   - 利用率 <50%：中度 Host-Bound
   - 利用率 ≥50%：瓶颈在 device 侧，需 kernel 级分析区分 compute-bound 和 memory-bound（高利用率 ≠ Compute-Bound）

2. **Per-Step Breakdown**：逐 step 列出各时间分量和利用率，用于观察是否有某个 step 异常偏离。

3. **Preparing Analysis**（仅当 CSV 中 Preparing 列有值时输出）：对比 Preparing 与 Computing 的平均值。Preparing > Computing 说明 profiler 本身的 trace-writing 开销占主导（Level1 常见），需对比 Level0 采集结果区分真实 host gap 和 profiler 开销。
4. **Optimization Ceilings**（Amdahl 式）：把本 step 时间拆成 compute floor / host-dispatch ceiling (Free) / communication ceiling，按上限排序候选优先级；并指向 operator_details（sync/alloc 子类）与 kernel_details（fusible 子类）做更细分解。喂饱确认节点 A 的"理论收益上限"。
5. **Suspect Signals**：单步推理也输出（[INFO] 标注单步、variance/spread 信号在多步时才激活）；
   - Step 间利用率波动大（>20% 差距）：部分 step 效率显著低于其他，可能是 warmup 或数据依赖行为
   - Step 间耗时差距大（>2x）：可能有首步编译、动态 shape 或缓存效应

**何时使用**：首先判断瓶颈在 host 侧还是 device 侧，决定后续分析方向。

```bash
python parse_step_trace.py /path/to/profiling
```

---

### parse_kernel_details.py

**输入**：`kernel_details.csv`（每个 NPU kernel 的完整执行信息，含硬件单元细分）

这是 profiling 中信息量最大的文件之一，包含每个 kernel 的算子名、类型、加速核类型、Block Dim、Input/Output Shapes、执行时间、等待时间、以及各硬件单元（mac/mte1/mte2/vec/scalar）的耗时和占比。当其他文件提供的信息不足以深入分析时，应回到此文件做进一步挖掘。

**输出包含以下维度**（AI CPU Fallback 仅在有非通信 AI_CPU kernel 时出现）：

1. **加速核分布**：AI_CORE vs AI_VECTOR_CORE 的 kernel 数和耗时占比，了解模型计算主要落在 cube 核还是 vector 核上。

2. **硬件单元利用率**：
   - AI_CORE：mac_ratio（计算）vs mte1/mte2_ratio（搬运）vs **fixpipe_ratio**（量化/定点）的平均值，判断 kernel 群整体是 compute-dominated 还是 memory-dominated
   - AI_VECTOR_CORE：vec_ratio vs mte2/mte3
   - **icache miss rate**（AI_CORE/AI_VECTOR_CORE 各列）：指令 cache 压力，miss 高 = kernel 调度效率低
   - Cube utilization 的平均和低利用率 kernel 数量
   - **duration 加权**（[duration-weighted]）：少数重 kernel 主导时，算术平均掩盖 bimodal；加权值反映重 kernel 的真实占比，与算术值对比即可看出是否两极分化
   - **Input Formats 分布**：ND vs NCL/NCHW/NCDHW 等，非 ND 格式占比高 = layout 转换成本（数据布局优化信号）

3. **AI CPU Fallback [DEFINITE]**（仅有非通信 AI_CPU kernel 时）：落在 AI_CPU 的非通信算子（通信算子排除——它们本就走 AI_CPU）。fallback 算子是"换实现/改 dtype"靶点。
4. **Kernel Duration 分布**：按耗时分桶（<5/5-20/20-50/50-200/>200us），短 kernel 占比高 = 碎片化/融合机会。
5. **小算子识别**（`--small-threshold` 控制，默认 5us）：duration 小于阈值的 kernel 数量、累计时间、Type 分布。大量小算子是碎片化信号，可能存在融合机会。

6. **Block Dim 分布**：反映 kernel 的并行度。大量 Block Dim=1 的 kernel 说明 shape 太小导致硬件并行利用不足。

7. **Suspect Kernels**：duration 高但硬件计算单元占比（mac/vec）极低的 kernel 列表，覆盖 AI_CORE 和 AI_VECTOR_CORE。列出嫌疑，不做判定——可能是 shape 导致的必然结果，也可能有优化空间，由 agent 决定是否用 `--filter` 深入。

8. **Wait Time 分布 + 高等待上下文**：wait time 的分桶统计，以及 wait 超过阈值的 kernel 及其**同流时间邻居**（按 Start Time 排序，非文件行序），用于识别流水断裂点的原因。
9. **Suspect Signals**：低利用率高耗时 kernel（融合/换实现靶点）、**真 compute-bound**（高耗时高 mac_ratio，替换/量化/拆分靶点）、可融合小算子序列（**按流分组**检测，跨流不会误报为可融合）。

**何时使用**：
- 需要判断 kernel 群整体是 compute-bound 还是 memory-bound 时
- 需要找出可融合的小算子时
- 需要定位硬件利用率低的具体 kernel 时
- 需要分析流水 stall 的具体位置和原因时
- 其他 profiling 文件信息不够深入时，回到此文件做进一步分析

**两种使用模式**：

```bash
# 全局分析模式（默认）
python parse_kernel_details.py /path/to/profiling --top-k 15 --small-threshold 5 --wait-threshold 500

# 过滤模式：针对特定算子深入分析
python parse_kernel_details.py /path/to/profiling --filter MatMul
python parse_kernel_details.py /path/to/profiling --filter Transpose Softmax --top-k 10
```

过滤模式（`--filter`）接受一个或多个算子名/类型关键词（子串匹配，大小写不敏感），输出匹配 kernel 的：聚合统计、硬件单元利用率、Input Shape 分布、Block Dim 分布、Duration 分位数、Top-K 实例详情。适合在全局分析发现某类算子值得深入后，做针对性检查。

---

### parse_memory_record.py

**输入**：`memory_record.csv`（按时间记录的内存占用，含 APP 定时采样和 PTA 事件触发两种记录）

**输出包含以下部分**：

0. **Peak Reserved by Component**：按 Component 分段（WORKSPACE = 算子 workspace，可经 tiling/env 控制；APP/PTA = tensor 内存），区分可控部分。
0a. **Active Memory（真实活集）**：Total Active 的 min/max。Active < Allocated = 可复用缓存空间；用 Active（非 Allocated）估 batch 上限。
1. **Reserved Memory（allocator 池大小）**：min/max/range + 分桶时间线。
2. **Allocated Memory（实际张量占用）**：仅 PTA 行有此数据，展示实际使用量的 min/max/range。
3. **Pool Fragmentation（Reserved - Allocated）**：空闲池大小的变化趋势，碎片化程度。
4. **Top-K Reserved Jumps**：最大的池增长和收缩跳变。
5. **Suspect Signals**：
   - 内存增长趋势：Reserved 是否持续上升
   - 高频抖动：大量 >50MB 的跳变（频繁大块分配释放）
   - 碎片化增长：Reserved - Allocated 的差距在扩大
   - OOM 风险：峰值接近 HBM 容量

**何时使用**：定位 OOM 风险、分析内存峰值出现在哪个阶段、判断碎片化程度和 buffer 复用不足。

```bash
python parse_memory_record.py /path/to/profiling --buckets 20 --top-k 10
```

---

### parse_operator_details.py

**输入**：`operator_details.csv`（每次算子调用的详细信息，含 Call Stack）

**注意**：此文件可达数千万行，脚本使用流式处理，不会全量加载。

**定位**：源码定位工具。此文件的独有价值是 Call Stack（唯一能关联操作到 Python 源码行的文件）和 per-op Host Duration。当其他脚本发现疑点后，来这里定位源码位置。

**两种模式**：

1. **默认模式**（轻量 host 开销概览）：
   - 按 host time 排序的算子列表（不重复 device 分析，那是 kernel_details 的事）
   - 纯 host 操作占比（框架 dispatch 开销量化）
   - **Host Time by Category**：把 host Self 时间按类别分解（sync D→H / H2D-copy / alloc-metadata / dispatch / framework / other）——sync 与 dispatch 优化方向相反，分解后定方向
   - **Host Time by Call-Chain Layer**：按调用链首个项目帧聚合 inclusive Host Total——喂饱 Line A「穿透层级量化」门禁（任何层 >10% host time 须有候选）
   - Suspect Signals：纯 host 操作占比过高、host/device ratio 极端的算子、**设备时间落在 AI_CPU 的算子**（Device Self Duration With AICore 占比高 = fallback，替换/换 dtype 靶点）

2. **Filter 模式**（`--filter`，核心用法）：
   - 给定算子名，输出该算子所有调用的 Call Stack，按调用位置分组
   - 每组显示：出现次数、host/device 时间、源码位置（文件:行号）、example shapes
   - 直接回答"这个操作是从源码哪里触发的、触发了多少次"

**何时使用**：
- 其他脚本发现某算子有疑点（如 op_statistic 发现 Transpose 占 15%）→ 用 `--filter Transpose` 定位是哪行代码触发的
- 想看 host 框架开销的整体占比时用默认模式

```bash
# 默认：host 开销概览
python parse_operator_details.py /path/to/profiling

# 定位源码：某算子的所有 Call Stack
python parse_operator_details.py /path/to/profiling --filter empty_tensor
python parse_operator_details.py /path/to/profiling --filter aclnnMatmul aten::view
```

---

### parse_operator_memory.py

**输入**：`operator_memory.csv`（每个 tensor 的分配生命周期记录）

**定位**：与 memory_record.csv 互补。memory_record 是全局内存曲线（按时间点采样），operator_memory 是**逐 tensor 的生命周期**（谁分配的、多大、活了多久、何时释放）。

**输出包含以下部分**：

1. **Top 分配 by Size**：最大的 tensor 分配，附带 lifetime 和分配时的全局内存状态。Lifetime 短的大 tensor 是 buffer 复用候选。
2. **Op 聚合**：每种 op 的累计分配量、次数、平均 size 和平均 lifetime。快速看出谁是内存分配大户。
3. **短命大 tensor**（size>100KB, lifetime<1ms）：分配后很快释放的大块——最强的 buffer 预分配信号。按 op 分组并列出典型 sizes。判定用 **Active Duration**（真实被引用时长）而非 pool Duration——caching allocator 保留的 tensor pool Duration 长但 Active 短，用 Active 才不漏判复用候选。
4. **Suspect Signals**：
   - 重复同尺寸分配：同一个 op 反复分配相同大小（预分配复用信号）
   - 短命大 tensor 累计量大：大量快速分配释放的内存抖动
   - 单一 op 垄断内存分配
5. **Parallelism Trigger Analysis**：大 tensor 按 short-lived（waste）/long-lived（essential）分类，估算消除 waste 后的投影峰值，判断是否需要切分并行
6. **Peak Attribution**：峰值时刻存活的 tensor 列表（按 size 降序）——直接回答"峰值是哪些 tensor 同时存活造成的"，优先减/复用这些

**何时使用**：
- 需要找"哪些 tensor 可以预分配复用"时
- 分析内存峰值由哪些 tensor 同时存活导致时
- memory_record 发现高频抖动后，来这里确认是哪些 op 造成的

```bash
python parse_operator_memory.py /path/to/profiling --top-k 20
```

---

### parse_trace_view.py

**输入**：`trace_view.json`（Chrome Trace 格式，唯一记录 host↔device 时序关系的文件）

**定位**：时序分析工具。CSV 文件只有 device 侧算子的顺序，`trace_view.json` 额外提供 host 下发与 device 执行的时间关联，是分析下发链、编译停顿、流水空隙的唯一来源。文件可达 GB 级，脚本流式解析（1GB 约 5s、内存 <50MB）。

**内容随采集开关变化**（脚本自动探测并在 §0 报告，不是按训练/推理区分）：
- 始终有：device kernel（`Task Type`）+ HostToDevice 下发 flow
- 仅 NPU / 未开 with_stack：host 侧为 CANN `AscendCL@...` 事件
- 开 CPU activity + `with_stack`：host 侧为 `cpu_op` + `python_function`，且 `cpu_op` 带 **Call stack**（源码栈）

**输出包含**：
0. **Detected Layers**：探测到的各层事件数；若缺 cpu_op/Call stack，明确提示"需重采开启 with_stack"，不静默出空
1. **Device Timeline**：只列含 compute 任务的 stream（span/active/busy%/kernel 数），其余通信/同步/DMA 流折叠成一行；compute 任务间的 gap 分布
2. **Device Stalls**：≥ 阈值的空隙**按(前→后 kernel 对)聚合**，给出出现次数、累计 gap、平均、最大，按累计降序——反复出现且累计大的才值得优化，避免被大量个例淹没
3. **Dispatch Latency**：HostToDevice flow 配对得到的下发延迟分布（avg/max/p50/p90）+ 最慢的 top-N（附最近 device kernel 名）
4. **Host2Device Bound Regions**：用 `Node@launch` 的 `connection_id` 与设备算子配对，算 `gap = device_start - launch_ts`；同 stream 上连续 ≥3 个 gap<50us 的算子视为一段 host2device-bound 区段（设备在等 host 下发、队列空转）。每段输出起止时间/算子链/设备空闲占比，并经 `async_npu(torch_to_npu)` flow 回连到 `cpu_op` 的 Call stack 定位源码。与 §3 互补：§3 是全局下发延迟统计，§4 是时间局部区段 + 源码定位
5. **Resource Utilization Timeline (counters)**：聚合 counter 事件——per-die HBM Read/Write 带宽、LLC Hit Rate/Throughput、L2/MAC Bw Level、内存占用、AI Core 频率。带宽/cache 命中率时间线是动态判 memory-bound vs compute-bound、定位带宽饱和时刻的依据
5b. **Stream Concurrency (掩盖维度)**：扫描所有 compute 流的 kernel 区间，统计"同时有 0/1/2+ 流 busy"的时间占比。四维度里"掩盖/重叠"的唯一量化产出——是否有多流并行可挖
5c. **Idle 成因分解**：联合扫描 device-busy 与 host `AscendCL@` 事件区间，当 device idle 时归因到 host 此刻在做的事（mem-mgmt / sync / compile / launch / residual）。回答"为什么 device idle"——host-bound 的根因定位
6. **Suspect Signals**：host2device-bound 摘要（[SIGNAL] 概述区段数/host-bound 算子数/最差区段链，引向 §4 看详情）、在线编译分类（**A 类**集中预热期 → `skip_first` 跳过；**B 类**贯穿全程 → 关 jit_compile / 定 shape / 图编译）、预取/预分配候选（`aten::to`/`copy_`/`empty` 等**不换算子**的优化点，附精简 Call stack）、AI Core 降频、Python GC、频繁 stream 同步

**Filter 模式**（`--filter NAME`）：给定算子名，输出匹配事件的 Call stack 和 Input Dims，直接定位源码位置。

**何时使用**：
- step_trace 判定 host-bound 后，用它定位 host 侧到底在忙什么（下发 / 编译 / 同步）
- 定位 host2device bound 区段：第 4 节直接给出"哪段时间、哪段代码 host 喂不动设备"，配合 §3 全局下发延迟判断是局部还是系统性问题
- 需要 host→device 下发链、区分首次编译（A 类，采集可解）与每步在线编译（B 类，执行模式问题）时
- 找预取 / 预分配 / buffer 复用等**不换算子**的优化点，并用 Call stack 定位到源码
- `operator_details.csv` 缺失但 trace_view 有 Call stack 时，作为源码定位的替代来源

```bash
python parse_trace_view.py /path/to/profiling --top-k 15 --gap-threshold 50
python parse_trace_view.py /path/to/profiling --filter aten::addmm
```

---

### parse_communication.py

**输入**：`communication.json` + `communication_matrix.json`（多卡场景 profiler_level >= Level1 时产出）

**定位**：通信分析工具。唯一能回答"通信为什么慢"的脚本——区分"真在传数据(Transit)"还是"在等其他 rank(Wait/Sync)"，以及带宽是否正常。

**输出包含**：
1. **Summary**：总通信时间分解（Transit/Wait/Sync/Idle）——一眼看出是带宽瓶颈还是同步瓶颈
2. **By Op Type**：按通信算子类型（allGather/alltoall/allReduce）聚合，含 Wait% 和 Transit%
3. **Top Ops by Elapse**：最耗时的通信算子排名
3b. **P2P Ops**：send/recv 详细时序（原仅计数）——PP 推理 P2P 是通信主体时定位瓶颈；wait 占比高指向流水 bubble
4. **Per-Link Bandwidth**：从 communication_matrix 提取的 per-link 带宽（min/avg/max + 最高/最低 link）
5. **Suspect Signals**：
   - [DEFINITE] Wait 占比 >80% → 同步瓶颈（非带宽问题），查通信-计算重叠 / straggler / 同步点
   - [SIGNAL] 某类算子 Wait% >90% → 交叉验证 trace_view 看 compute-comm 重叠
   - [SIGNAL] 低带宽 link（<30% 均值）→ 瓶颈 link
   - [SIGNAL] 小包占比 >30% → 考虑 batch 减少 small-packet overhead

**何时使用**：
- step_trace 显示 Communication 占比高时
- 多卡场景排查 straggler（某 rank 慢导致其他 rank 等待）
- 通信-计算重叠分析（配合 trace_view）

```bash
python parse_communication.py /path/to/profiling --rank 0 --top-k 15
```

---

### diff_profiling.py

**输入**：两个 profiling 目录（before / after）

**输出**：
- 算子耗时 diff（按变化量排序）
- 消失的算子和新增的算子
- 内存峰值变化（**优先用 Total Active**，Reserved 受 pool 保留干扰）
- **Comparability Guard**：L0/L1 口径一致性检查（operator_details.csv 是否存在）+ step 数差异检测；不一致时警告并按 per-step 归一化
- **Utilization / Free Diff**：before/after 利用率、Free、Computing、Comm 对比（per-step 归一化）——确认 host-bound 是否真改善、检测瓶颈转移
- **Bottleneck Type Shift**：before/after 瓶颈类型（Host-Bound / Comm-Bound / Device-side）对比——瓶颈转移是优化常见结果，下一轮应针对新瓶颈
- **Host Overhead Diff**（L1）：host self / pure-host 占比对比——确认 dispatch/sync 是否降
- **Kernel Hardware Diff**：AI_CPU fallback kernel 数 + AI_CORE mac_ratio 对比——确认 fallback 是否减少、compute/memory boundness 是否变

**何时使用**：实施优化后对比效果——确认算子被消除/融合、总耗时下降、利用率提升、host-bound 解除。注意同口径（L0 vs L0 或 L1 vs L1）同 step 数对比才有效。

```bash
python diff_profiling.py /path/to/before /path/to/after --top-k 20
```

---


## 注意事项

- 所有脚本都输出 **Suspect Signals** 部分——列出有疑点的数据，不做最终判定，由 agent 决定是否深入
- 脚本只负责**数据提取和压缩**，不做优化决策——决策由 agent 结合源码分析完成
- Call Stack 不做任何过滤，保证信息完整性
- 对于 `operator_details.csv` 超大文件（>10M 行），脚本耗时约 15-20s 是正常的
- 所有数值单位以 CANN 原始单位为准（时间: us，内存: KB/MB），脚本会转换为可读格式
