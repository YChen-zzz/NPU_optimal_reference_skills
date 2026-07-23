# Parse 脚本数据采集与分析说明

> 逐脚本记录：读取哪个文件、采集哪些字段、做什么分析、**为什么分析这个 / 想定位什么问题**。
> 所有脚本只依赖 CANN profiler 固定文件格式，与具体模型无关；只做数据提取与压缩，不做优化决策。每个分析点都对应一个推理优化需要回答的问题。

---

## parse_step_trace.py

**定位的核心问题**：瓶颈在 host 侧还是 device 侧？最多能优化多少？优化收益主要在哪一档？

- **数据源**：`step_trace_time.csv`
- **采集字段**：`Computing`、`Free`、`Communication`、`Preparing`（per step）
- **分析与定位目标**：
  - 设备利用率 = Computing/(Computing+Free)：**定位瓶颈侧**——利用率低 = host 喂不动设备（Host-Bound）；高 = 瓶颈在 device（需 kernel 级分析区分 compute/memory bound）。这是整个分析的入口，决定后续方向。
  - 可优化空间 = Free 占比：**回答"理论上限"**——Free 大 = 有压缩空间；Free 小 = 已接近极限，优化收益有限。
  - Optimization Ceilings（compute floor / host-dispatch ceiling=Free / comm ceiling）：**回答"收益主要在哪一档"**——按上限排序候选，避免在低收益方向投入。host-dispatch ceiling 最大 → 先治 host；comm 最大 → 先重叠通信。
  - Per-step 利用率/耗时方差：**定位异常 step**——某些 step 偏慢可能是 warmup/编译/动态 shape。
  - Preparing 分析：**区分 Free 是真实 host gap 还是 profiler 注入开销**——L1 采集的 Preparing 含 trace-writing 开销，Preparing>Computing 时 Free 可能是假象。
  - 单步 [INFO]：推理常单步，提示 variance 信号不适用，转向 utilization + 交叉验证。
- **产出**：Overall 利用率与瓶颈判定、Optimization Ceilings、Per-Step Breakdown、Preparing 分析、Suspect Signals

## parse_op_statistic.py

**定位的核心问题**：device 时间花在哪类算子？瓶颈是否集中？有多少是数据搬运浪费？

- **数据源**：`op_statistic.csv`
- **采集字段**：`OP Type`、`Count`、`Total Time(us)`（Avg 自算）
- **分析与定位目标**：
  - 按 Total Time 降序排名 + 累计占比：**定位优化靶点优先级**——哪类算子最该先优化。
  - Top-3 集中度：**回答"瓶颈是否集中"**——集中则聚焦少数算子收益大；分散则需系统性优化。
  - 数据搬运开销（Transpose/Cast/Copy 等关键词）：**定位非计算浪费**——有多少时间花在 layout/dtype 转换而非真实计算（对应"数据布局"优先级）。
  - 高频低耗时（碎片化）：**定位融合机会**——大量细碎算子可合并（去重维度）。
  - 低频高耗时（重型）：**定位大 shape 重型算子**——可拆分或换更高效实现。
- **产出**：算子排名表、Suspect Signals（集中度/数据搬运/碎片化/重型）

## parse_kernel_details.py

**定位的核心问题**：kernel 群是 compute-bound 还是 memory-bound？哪些算子 fallback 到 AI_CPU？哪些可融合/已算到顶？

- **数据源**：`kernel_details.csv`
- **采集字段**：`Name`、`Type`、`Accelerator Core`、`Duration(us)`、`Wait Time(us)`、`Block Dim`、`cube_utilization(%)`、`Start Time(us)`、`Stream ID`、`Input Formats`、aic 的 `mac/mte1/mte2/scalar/fixpipe _ratio` + `aic_icache_miss_rate`、aiv 的 `vec/mte2/mte3/scalar _ratio` + `aiv_icache_miss_rate`
- **分析与定位目标**：
  - 加速核分布（AI_CORE/AI_VECTOR_CORE/AI_CPU）：**定位计算落在哪类核**——理解计算性质，AI_CPU 多 = fallback 严重。
  - 硬件单元利用率（mac 计算 vs mte 搬运）：**定位 compute-bound vs memory-bound**——决定方向（compute→换算法/量化；memory→改访问/布局）。
  - duration 加权：**避免少数重 kernel 被大量小 kernel 稀释**——算术平均掩盖 bimodal，加权值反映真实占大头的 kernel 性质。
  - fixpipe：**量化/定点算子占比**——量化模型会漏的维度，fixpipe 重说明量化开销大。
  - icache miss：**指令 cache 压力**——miss 高 = kernel 调度效率低（可能是 kernel 过多/过大）。
  - Input Formats 非 ND 占比：**定位 layout 转换开销**——非 ND 格式意味着运行时 transpose/cast（数据布局优先级的数据支撑）。
  - AI CPU Fallback（非通信）：**定位没落 AI Core 的算子**——fallback 要换实现/改 dtype（替换维度）。
  - Duration 分桶/小算子：**定位碎片化程度**——短 kernel 占比高 = 融合机会。
  - Block Dim 分布：**定位并行度**——Block Dim=1 多 = shape 太小没充分利用硬件并行。
  - Wait Time + 高等待上下文（按 Stream 分组、按 Start Time 排序）：**定位流水断裂点**——wait 大的 kernel 前后是什么导致等待（依赖/同步）；按流分组避免跨流误判因果。
  - 真 compute-bound（高耗时高 mac）：**定位已算到顶的 kernel**——这类是换算法/量化/拆分靶点（kernel 本身慢），不是融合靶点。
  - 可融合序列（按流分组）：**定位同流连续小算子可合并**——融合机会（去重维度）。
- **产出**：§1-§8（加速核/硬件单元/AI_CPU/Duration/小算子/Block Dim/Wait/Suspect）
- **Filter 模式**：`--filter` 按算子名深入，给 shape→性能关联 + per-instance 硬件分解 + wait 上下文

## parse_operator_details.py

**定位的核心问题**：host 时间花在哪类操作、哪层调用链？哪些算子 device 时间落在 AI_CPU？

- **数据源**：`operator_details.csv`
- **采集字段**：`Name`、`Host Self Duration(us)`、`Device Self Duration(us)`、`Host Total Duration(us)`、`Device Total Duration(us)`、`Device Self Duration With AICore(us)`、`Call Stack`、`Input Shapes`
- **分析与定位目标**：
  - 按 host Self 排序：**定位 host 时间花在哪些算子**——host 开销的 op 名视角。
  - 纯 host op 占比：**定位无 device work 的浪费**——多少 host 时间没产生 device 计算（纯框架 dispatch 开销）。
  - Host Time by Category（sync/H2D-copy/alloc/dispatch/framework）：**定位 host 时间属于哪类**——sync 与 dispatch 优化方向相反（sync→消除 .item；dispatch→flat forward/图编译），不分类无法定方向。
  - Host Time by Call-Chain Layer（inclusive Host Total 按项目帧聚合）：**定位哪层调用链贡献 host 开销**——框架 wrapper 层 Self≈0 但 Total 大，靠 inclusive 才看得到；喂 Line A 穿透层级门禁（任何层 >10% host time 须有候选）。
  - AI_CPU 归因（Device With AICore 占比）：**定位哪些 op 的 device 时间落在 AI_CPU**——fallback 靶点，与 kernel_details 交叉验证。
- **产出**：Top ops by host、Pure Host Ops、Host Time by Category、Host Time by Call-Chain Layer、Suspect Signals
- **Filter 模式**：`--filter` 按 op 名分组 Call Stack 定位源码调用点

## parse_memory_record.py

**定位的核心问题**：内存峰值多大、何时发生？峰值里多少可控？是否接近 OOM？

- **数据源**：`memory_record.csv`
- **采集字段**：`Timestamp(us)`、`Total Reserved(MB)`、`Total Allocated(MB)`、`Total Active(MB)`、`Component`
- **分析与定位目标**：
  - Reserved/Allocated/碎片化(Reserved−Allocated) 时间线：**定位内存趋势**——池大小、实际占用、碎片化的变化；增长 = 可能泄漏，抖动 = 频繁分配释放。
  - Peak by Component（WORKSPACE vs APP/PTA）：**定位峰值里多少可控**——WORKSPACE 是算子 workspace（调 tiling/env），tensor 是分配/复用；不分开不知道动哪个旋钮。
  - Active Memory：**定位真实活集**——batch 上限由峰值 Active 决定；Reserved/Allocated 含 cache 会高估可压缩空间。
  - Top Reserved Jumps：**定位内存大幅扩张时刻**——什么操作导致池增长。
  - 增长趋势/高频抖动/OOM 风险：**定位泄漏/碎片化/容量风险**——峰值接近 HBM = OOM 风险。
- **产出**：§0 Peak by Component、§0a Active、§1-§4（Reserved/Allocated/Fragmentation/Jumps）、Suspect Signals

## parse_operator_memory.py

**定位的核心问题**：哪些 tensor 可预分配复用？峰值是哪些 tensor 同时存活造成的？

- **数据源**：`operator_memory.csv`
- **采集字段**：`Name`、`Size(KB)`、`Duration(us)`、`Active Duration(us)`、`Allocation Time(us)`、`Release Time(us)`、`Allocation Total Allocated(MB)`
- **分析与定位目标**：
  - Top by size + op 聚合：**定位分配大户**——谁分配了大 tensor、谁是高频分配 op。
  - 短命大 tensor（用 Active Duration 判定）：**定位 buffer 复用候选**——分配后很快不用的 tensor 该预分配复用（复用维度）；用 Active 而非 pool Duration，避免 caching allocator 保留的 tensor 被误判为长命而漏掉。
  - 重复同尺寸分配：**定位预分配信号**——同一 op 反复分配相同大小（复用维度）。
  - 短命 churn：**定位内存抖动**——大量快速分配释放，预分配可消除。
  - Parallelism Trigger（大 tensor short/long-lived + 投影峰值）：**定位是否需要切分并行**——消除 waste 后峰值仍超 HBM 则需并行。
  - Peak Attribution（峰值时刻存活 tensor）：**定位峰值成因**——峰值是哪些 tensor 同时存活造成的，优先减/复用这些。
- **产出**：§1-§3 + Suspect + Parallelism Trigger + Peak Attribution

## parse_api_statistic.py

**定位的核心问题**：host 开销在 CANN 运行时 API 层属于哪类（内存管理/同步/tiling/launch）？这是 host-bound 成因的最深下钻。

- **数据源**：`api_statistic.csv`（L1 产出，L0 无）
- **采集字段**：`Level`、`API Name`、`Count`、`Time(us)`、`Avg(us)`、`Max(us)`
- **分析与定位目标**：
  - 按 Level（acl/communication/node）汇总：**定位 host API 开销在哪层**——acl 是算子运行时、communication 是 HCCL、node 是 launch。
  - acl 层类别分解（memory-mgmt / sync / tiling / launch / other）并标 dominant：**定位 host API 时间属于哪类操作**——
    - memory-mgmt 大（aclrtFree/Unmap/Malloc）= 频繁分配释放阻塞 host → buffer 复用/预分配；
    - sync 大（SynchronizeStream/Device）= 显式同步 → 消除 .item/无效同步；
    - tiling 大 = 每次重算 tiling → cache/graph compile（静态 shape 时）；
    - launch 大 = 下发开销 → 减少 op 数（融合/图编译）。
  - node launch count：**与 trace_view Node@launch 对应**，验证一致性。
  - communication API（Notify_*）：**定位通信同步开销**——Notify_Wait 多 = 通信等待（流水 bubble）。
- **产出**：By Level、ACL API Top、类别分解、Node Launch、Communication API、Suspect Signals

## parse_trace_view.py

**定位的核心问题**：host 与 device 的时序关系——下发是否健康？设备为何 idle？多流并行用了没？带宽是否饱和？这是唯一能回答时序因果的文件。

- **数据源**：`trace_view.json`（Chrome Trace，流式解析）
- **采集字段/事件**：device kernel（`Task Type`/`ts`/`dur`/`connection_id`/pid,tid）、`Node@launch`、`HostToDevice` flow、`async_npu`(torch_to_npu) flow、`cpu_op`（`Call stack`/`Input Dims`/`Input type`）、`AscendCL@*` 事件（opCompile/aclrtFree/Unmap/Malloc/Map/SynchronizeStream）、counter 事件（`MHz`/`Read,Write(MB/s)`/`Hit Rate(%)`/`Throughput(MB/s)`/`L2,MAC Bw Level`/`KB`/`value`）、GC、enqueue/dequeue
- **分析与定位目标**：
  - §0 Detected Layers：**确认采集开关到位**——缺 cpu_op/Call stack 则源码定位不可用，提示重采。
  - §1 Device Timeline：**定位各 compute 流的繁忙度与空隙**——stream busy% 低 = 流没喂满；gap 分布看空隙大小。
  - §2 Device Stalls（按 kernel 对聚合）：**定位反复出现的流水断裂点**——哪些 kernel 对之间总有空隙（聚合后避免被个例淹没）。
  - §3 Dispatch Latency（HostToDevice flow 配对）：**定位 host→device 下发延迟**——p50/p90/top；dispatch/kernel-active 比例高 = dispatch 开销大，可能未充分重叠。
  - §4 Host2Device Bound Regions（connection_id 配对 launch↔device）：**定位设备空等 host 下发的时间区段**——连续算子 gap 极小 = 设备一 launch 就执行、队列空转（host 喂不动）；经 async_npu 回连 cpu_op Call stack 定位源码。
  - §5 Resource Utilization Timeline（counter 聚合）：**定位动态资源利用率**——HBM 带宽是否饱和、LLC 命中率、利用率时间线；区分"某段时间 memory-bound vs compute-bound"（静态 ratio 看不出时间相变），定位带宽饱和时刻。
  - §5b Stream Concurrency（0/1/2+ 流同时 busy 时间占比）：**定位多流并行是否被利用**——掩盖维度；≥2 流并发占比低 = 有 overlap 空间可挖（double-buffer/多流）。
  - §5c Idle Cause Breakdown（device idle 时归因 host 在做什么）：**定位 device idle 成因**——idle 时 host 在 mem-mgmt/sync/compile/launch 还是 residual；host-bound 根因定位（回答"为什么 device 空闲"）。
  - §6 Suspect Signals：h2d 摘要、在线编译 A 类（预热可 skip）/B 类（每步编译要治）、预取候选、降频、GC、stream 同步。
- **Filter 模式**：`--filter` 按算子名输出 Call stack + Input Dims

## parse_communication.py

**定位的核心问题**：通信慢是带宽问题还是同步问题？哪个 link/哪类算子是瓶颈？

- **数据源**：`communication.json` + `communication_matrix.json`
- **采集字段**：collective/p2p op 的 `Communication Time Info`（Elapse/Transit/Wait/Sync/Idle）、`Communication Bandwidth Info`（Bandwidth/Transit Size/Transport Type/Size Distribution）
- **分析与定位目标**：
  - Summary 时间分解（Transit/Wait/Sync/Idle）：**定位通信慢的性质**——Wait 大 = 同步瓶颈（等别人，非带宽）；Transit 大 = 带宽瓶颈。
  - By Op Type（Wait%/Transit%）：**定位哪类通信算子等待多**——如 allReduce Wait% 高 = 集合通信同步重。
  - Top Ops by Elapse：**定位最耗时通信算子**。
  - P2P Ops（send/recv 详细时序）：**定位 P2P 瓶颈**（PP 场景）——wait 占比高 = 流水 bubble。
  - Per-Link Bandwidth：**定位瓶颈 link**——低带宽 link + 小包占比高 = 可 batch 减开销。
  - Suspect：Wait>80% 同步瓶颈、低带宽 link、小包占比高。
- **产出**：Summary、By Op Type、Top Ops、P2P Ops、Per-Link Bandwidth、Suspect Signals

## diff_profiling.py

**定位的核心问题**：优化是否真生效？瓶颈是否转移？有没有伪收益（算子快了但利用率没涨）？

- **数据源**：两个 profiling 目录（before/after），读 `op_statistic.csv`、`memory_record.csv`、`step_trace_time.csv`、`operator_details.csv`、`kernel_details.csv`
- **分析与定位目标**（before→after 对比）：
  - Op Time Diff（per-op total、消失/新增）：**确认优化作用于预期算子**——哪些变快、是否消除/新增算子（替换型优化的验证）。
  - Memory Peak（优先 Active）：**确认峰值是否真降**——Reserved 受 pool 保留干扰可能不降，Active 才反映真实。
  - Comparability Guard（L0/L1 口径 + step 数）：**确保对比有效**——口径不同或 step 数不同则对比失真（L1 含 profiler 注入开销）。
  - Utilization/Free Diff（per-step 归一化）：**确认 host-bound 是否真改善**——算子快了不等于利用率涨（可能瓶颈转移）。
  - Bottleneck Type Shift：**定位优化后瓶颈是否转移**——下一轮应针对新瓶颈（如 Host-Bound→Compute-Bound）。
  - Host Overhead Diff（host self/step + pure-host 占比，L1）：**确认 host 开销是否降**。
  - Kernel Hardware Diff（AI_CPU fallback 数 + mac_ratio 均值）：**确认 fallback 是否减少、compute/memory boundness 是否变**。
  - Suspect Signals：utilization 未改善（伪收益/瓶颈转移）、口径不一致。
- **未对比的缺口**：trace_view（h2d/dispatch/compile/counter）、operator_details 类别分解、operator_memory 复用候选、api_statistic——这些维度 before/after 变化尚未纳入 diff（D3 类待补）。
