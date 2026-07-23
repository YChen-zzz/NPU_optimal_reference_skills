# Parse 脚本原始信息捕获完整性审计（Phase 1）

> 范围：只审计"profiling 文件中存在的原始信息，parse 脚本是否都读到了"，**不评估后处理逻辑是否合适**（那是 Phase 2 的任务）。
> 数据样本：`/Users/kaili/Downloads/20260325/rank_0/.../ASCEND_PROFILER_OUTPUT/`（CANN profiler 真实产出）。
> 方法：逐文件列出全部列/字段 → 比对脚本实际读取的字段 → 标注未捕获项及影响等级。

## 结论速览

主干文件的"大块时间/主要耗时"信息捕获到位（step_trace 的 Computing/Free、op_statistic 的 Total Time、kernel_details 的 Duration/Wait/硬件占比、operator_details 的 Call Stack/Host Self、memory_record 的 Reserved/Allocated、trace_view 的下发链）。

> **重要修订**：原始信息在 profiling 文件间有大量重复。下文逐文件标注的"未捕获项"，有一部分在别的文件里**已经被捕获**（如 kernel_details 没读 Start Time，但 trace_view 的设备算子 `ts` 已用于时间线分析）。因此"该脚本没读"≠"信息全局缺失"。判定盲区前先看「跨文件冗余核对」一节。

剔除跨文件冗余后，真正全局缺失的系统性盲区是：

1. **格式维度**——`Formats`（ND/NZ 等内存格式）仅 kernel_details 有且未读；Data Types 在 trace_view cpu_op 有 `Input type` 也未读。数据布局优化（优先级 7）无数据根基。
2. **动态资源利用率时间线**——`trace_view` 的 176 万 counter 事件（HBM 读写带宽、Cache Hit Rate、Throughput、L2/MAC 带宽级别、利用率），脚本只用了其中 2 个 MHz 事件。kernel_details 的硬件 ratio 是静态 per-kernel 快照，不替代时间线。
3. **inclusive / active 维度**——operator_details 的 Host/Device Total 与 With AICore、operator_memory 的 Active Duration、memory_record/operator_memory 的 Total Active，均未读。

而原先看似缺失的 **Start Time / Stream ID**：kernel_details 自身没读，但 trace_view 已用设备算子 `ts`/`(pid,tid)` 做了时间线与流级分析——属"该脚本内不可用但全局已捕获"，归为 Phase 2 处理问题而非原始信息丢失。

---

## 跨文件冗余核对（盲区判定修订）

对高/中影响遗漏逐项核对：该信息是否在别处已捕获。**真盲区**=所有文件都没读；**部分盲区**=别处有但粒度/口径不同或也未读；**非盲区**=别处已捕获（原脚本的缺失归 Phase 2 处理问题）。

| 遗漏项（原文件） | 别处同源信息 | 是否被捕获 | 修订判定 |
|---|---|---|---|
| kernel_details `Start Time(us)` | trace_view 设备算子事件 `ts` | 是（trace_view 用于 timeline/gap/stall） | **非盲区**——kernel_details 内 wait-context/可融合序列靠行序，属 Phase 2 |
| kernel_details `Stream ID` | trace_view 设备算子 `(pid,tid)` 即流身份 | 是（trace_view 按 stream 聚合 busy%/gap） | **非盲区**——流级分析在 trace_view 已做 |
| kernel_details `Input Data Types` | trace_view cpu_op args 有 `Input type` | 否（trace_view 也未提取该字段） | **部分盲区**——dtype 数据存在但两处都没读 |
| kernel_details `Input/Output Formats` | 无别处来源 | 否 | **真盲区**——格式转换优化无数据根基 |
| memory_record `Total Active(MB)` | operator_memory 有 `Allocation/Release Total Active(MB)` | 否（两脚本都没读） | **真盲区**（数据存在但未捕获） |
| op_statistic `Core Type` | kernel_details `Accelerator Core` 同源 | 是（parse_kernel_details 按 core 聚合） | **非盲区**——聚合视图缺，kernel 粒度 core 归属已捕获 |
| op_statistic `Min/Max Time` | kernel_details filter 模式按 shape 给 min/max | 部分（仅 filter 模式） | **部分盲区**——全局聚合无方差，需 filter 才看得到 |
| trace_view C(counter) 带宽/cache/利用率 | 无（kernel_details 是静态 ratio，非时间线） | 否 | **真盲区** |
| trace_view enqueue/dequeue | 无别处来源 | 否 | **真盲区** |
| api_statistic.csv | 无别处来源 | 否（无 parse 脚本） | **真盲区** |
| operator_details Host/Device Total、With AICore | 无别处来源 | 否 | **真盲区** |
| operator_memory Active Duration、释放点 Active | 无别处来源 | 否 | **真盲区** |
| kernel_details fixpipe / icache_miss_rate / 绝对 hw 时间 | 无别处来源 | 否 | **真盲区** |
| step_trace Overlapped/Stage/Bubble | 无别处来源 | 否 | **真盲区**（PP 场景） |
| communication P2P 详细时序 | 无别处来源 | 否（仅计数） | **真盲区**（P2P 密集场景） |

---

## 1. step_trace_time.csv

**全部列**：`Device_id, Step, Computing, Communication(Not Overlapped), Overlapped, Communication, Free, Stage, Bubble, Communication(Not Overlapped and Exclude Receive), Preparing`

**脚本读取**（`parse_step_trace.py`）：`Computing, Free, Communication, Preparing`

**未捕获**：

| 字段 | 含义 | 影响 |
|------|------|------|
| `Overlapped` | 计算与通信重叠的时间 | 中——直接衡量计算-通信重叠度（掩盖维度的关键指标），目前只能从 Communication 总量推断 |
| `Stage` | 流水并行阶段编号 | 中——流水并行（PP）场景下区分各 stage，单卡推理无值故未暴露，但 PP 训练场景缺失 |
| `Bubble` | 流水 bubble 时间 | 中——PP bubble 是流水并行的核心开销，PP 场景必备 |
| `Communication(Not Overlapped)` / `... Exclude Receive` | 不同口径的纯通信时间 | 低——已有 Communication 总量，细分口径对深挖有用但非必需 |
| `Device_id` / `Step` | 设备号、步号 | 低——单 rank 单步时无意义；多卡多步时可用于对齐 |

**判定**：单卡推理场景基本完整；流水并行训练场景缺 Stage/Bubble/Overlapped，是真实盲区。

---

## 2. op_statistic.csv

**全部列**：`Device_id, OP Type, Core Type, Count, Total Time(us), Min Time(us), Avg Time(us), Max Time(us), Ratio(%)`

**脚本读取**（`parse_op_statistic.py`）：`OP Type, Count, Total Time(us)`（Avg 自行计算）

**未捕获**：

| 字段 | 含义 | 影响 |
|------|------|------|
| `Core Type` | 该算子落在 AI_CORE / AI_CPU / AI_VECTOR_CORE | 中高——聚合视图下哪个算子类型落 AI_CPU（fallback）一眼可见，目前要回到 kernel_details 才能看；且按 Core 拆分耗时分布缺失 |
| `Min Time(us)` / `Max Time(us)` | 同类算子耗时的极值 | 中——同 op 类型不同调用的耗时方差（Min/Max 跨度大 = shape 敏感或动态行为），目前完全看不到方差 |
| `Ratio(%)` | profiler 预计算的占比 | 低——脚本自行重算，等价 |
| `Device_id` | 设备号 | 低 |

**判定**：核心耗时分布捕获到位；缺 Core Type（fallback 识别的聚合入口）和 Min/Max（方差信号）。

> **跨文件冗余注**：`Core Type` 与 kernel_details `Accelerator Core` 同源——parse_kernel_details 已按 core 聚合，故全局非盲区，仅聚合视图缺；`Min/Max` 可用 kernel_details filter 模式按 shape 查看方差。

---

## 3. kernel_details.csv（信息最密集的文件）

**全部列**（46 列）：`Device_id, Model ID, Task ID, Stream ID, Name, Type, OP State, Accelerator Core, Start Time(us), Duration(us), Wait Time(us), Block Dim, Mix Block Dim, HF32 Eligible, Input Shapes, Input Data Types, Input Formats, Output Shapes, Output Data Types, Output Formats, Context ID, aicore_time(us), aic_total_cycles, aic_mac_time(us), aic_mac_ratio, aic_scalar_time(us), aic_scalar_ratio, aic_mte1_time(us), aic_mte1_ratio, aic_mte2_time(us), aic_mte2_ratio, aic_fixpipe_time(us), aic_fixpipe_ratio, aic_icache_miss_rate, aiv_time(us), aiv_total_cycles, aiv_vec_time(us), aiv_vec_ratio, aiv_scalar_time(us), aiv_scalar_ratio, aiv_mte2_time(us), aiv_mte2_ratio, aiv_mte3_time(us), aiv_mte3_ratio, aiv_icache_miss_rate, cube_utilization(%)`

**脚本读取**：`Name, Type, Accelerator Core, Duration(us), Wait Time(us), Block Dim, cube_utilization(%), Input Shapes` + aic 的 `mac/mte1/mte2/scalar _ratio` + aiv 的 `vec/mte2/mte3/scalar _ratio`

**未捕获**（按影响分组）：

**高影响**：

| 字段 | 含义 | 影响 |
|------|------|------|
| `Start Time(us)` | kernel 精确启动时间戳 | 高——脚本的"高等待上下文"和"可融合序列"全靠**文件行顺序**当时间序，无真正时间戳；且无法与 trace_view / memory_record 做跨文件时序对齐 |
| `Input Data Types` / `Input Formats` / `Output Shapes` / `Output Data Types` / `Output Formats` | 数据类型与内存格式 | 高——只读 Input Shapes。"数据布局/format 优化"（profiling_to_action 优先级 7、NPU 的 ND/NZ 格式转换）完全没有数据支撑；format_cast/Transpose 的根因分析缺输入 |
| `Stream ID` | kernel 所在流 | 高——kernel 粒度的流归属缺失，无法做流级并行度/流间依赖分析（trace_view 有 pid/tid 但 kernel_details 这边断了） |

**中影响**：

| 字段 | 含义 | 影响 |
|------|------|------|
| `aic_fixpipe_time/ratio` | fixpipe 单元（量化/定点）耗时与占比 | 中——硬件单元覆盖不全，fixpipe 重的量化模型会漏 |
| `aic_icache_miss_rate` / `aiv_icache_miss_rate` | 指令 cache miss 率 | 中——icache 压力是 kernel 调度效率的信号，完全丢弃 |
| `aic_mac_time / mte1_time / mte2_time ...`（绝对时间） | 各硬件单元绝对耗时 | 中——只用了 ratio（占比），无法聚合"全部 kernel 的 mte 总耗时 = X ms"，绝对量级信息丢失 |
| `aicore_time(us)` / `aiv_time(us)` | 核上实际执行时间 | 中——与 Duration（含开销）不同，区分 kernel overhead 与纯计算 |
| `aic_total_cycles` / `aiv_total_cycles` | 总周期数 | 中——绝对吞吐上下文 |
| `OP State` | dynamic / static 状态 | 中——动态 shape kernel 识别（与在线编译/重编译相关） |
| `Mix Block Dim` | 混合 block 维度 | 中——与 Block Dim 配合反映并行度细节 |

**低影响**：`Model ID, Task ID, Context ID`（标识符）、`HF32 Eligible`（HF32 优化提示，量少）、`Device_id`。

> **跨文件冗余注**：`Start Time(us)` 与 `Stream ID` 虽在本文件未读，但 trace_view 已用设备算子 `ts`/`(pid,tid)` 做了时间线与流级分析——属"本脚本内不可用但全局已捕获"，归 Phase 2 处理问题（见「跨文件冗余核对」）。`Input Data Types` 在 trace_view cpu_op `Input type` 有但也未读；`Formats` 则全文件仅此一处，真盲区。

**判定**：硬件占比/小算子/Block Dim/等待分布捕获到位；本文件真盲区为 **Formats / Data Types / fixpipe / icache / 绝对时间**；Start Time 与 Stream ID 因 trace_view 已覆盖不计全局盲区。

---

## 4. operator_details.csv

**全部列**：`Name, Input Shapes, Call Stack, Host Self Duration(us), Host Total Duration(us), Device Self Duration(us), Device Total Duration(us), Device Self Duration With AICore(us), Device Total Duration With AICore(us)`

**脚本读取**（`parse_operator_details.py`）：`Name, Input Shapes, Call Stack, Host Self Duration(us), Device Self Duration(us)`

**未捕获**：

| 字段 | 含义 | 影响 |
|------|------|------|
| `Host Total Duration(us)` | 含子 op 的 inclusive host 耗时 | 中——Self 是框架 dispatch 净开销，Total 是含调用的总开销；穿透层级量化（Line A 门禁）需要 Total，目前只能估 |
| `Device Total Duration(us)` | inclusive device 耗时 | 中——Self 排除了被调子算子，看一个 op 的端到端 device 成本需要 Total |
| `Device Self/Total Duration With AICore(us)` | 含/不含 AICore 的 device 耗时 | 中——区分算子的 device 时间里有多少落在 AI_CPU（fallback），目前要回 kernel_details 看 Accelerator Core |

**判定**：源码定位（Call Stack）和 host self 捕获到位；inclusive 耗时与 AICore 归属缺失，影响 host 开销的穿透层级量化。

---

## 5. memory_record.csv

**全部列**：`Component, Timestamp(us), Total Allocated(MB), Total Reserved(MB), Total Active(MB), Stream Ptr, Device Type`

**脚本读取**（`parse_memory_record.py`）：`Timestamp(us), Total Reserved(MB), Total Allocated(MB), Component`

**未捕获**：

| 字段 | 含义 | 影响 |
|------|------|------|
| `Total Active(MB)` | 真正被活跃张量占用的内存（未进 cache 的活集） | 中高——脚本用 `Reserved - Allocated` 当碎片化指标，但 Active 才是真实活集；Allocted 含 allocator cache，二者差额 = 可复用缓存。丢弃 Active 使碎片化判断偏粗 |
| `Stream Ptr` | 关联流指针 | 低——可把内存事件链到流，但价值有限 |
| `Device Type` | `NPU:0` 等 | 低——多卡区分 |

**判定**：Reserved/Allocated 时间线捕获到位；Active（活集）缺失使碎片化/可复用判断不够精确。

> **跨文件冗余注**：`Total Active(MB)` 在 operator_memory 也有（`Allocation/Release Total Active(MB)`），但 parse_operator_memory 同样未读——故 Active 是真盲区（数据存在于两文件但都未捕获），而非单文件遗漏。

---

## 6. operator_memory.csv

**全部列**：`Name, Size(KB), Allocation Time(us), Release Time(us), Active Release Time(us), Duration(us), Active Duration(us), Allocation Total Allocated(MB), Allocation Total Reserved(MB), Allocation Total Active(MB), Release Total Allocated(MB), Release Total Reserved(MB), Release Total Active(MB), Stream Ptr, Device Type`

**脚本读取**（`parse_operator_memory.py`）：`Name, Size(KB), Duration(us), Allocation Time(us), Release Time(us), Allocation Total Allocated(MB)`

**未捕获**：

| 字段 | 含义 | 影响 |
|------|------|------|
| `Active Duration(us)` / `Active Release Time(us)` | 张量真正活跃（被使用）的时长 | 中高——脚本用 `Duration`（在 pool 中的时长，含缓存期）判"短命大 tensor=复用候选"，但一个被分配后立即释放进 cache 的 tensor Duration 长、Active Duration 短。用错维度会把可复用的判成不可复用 |
| `Allocation Total Reserved(MB)` / `Allocation Total Active(MB)` | 分配时刻的全局 Reserved / Active | 中——只用了 Allocated，Reserved 和 Active 在分配点的快照丢失，峰值时刻的池/活集状态不全 |
| `Release Total Allocated/Reserved/Active(MB)` | 释放时刻的全局内存状态 | 中——释放点的内存下降量是判断"释放是否有效缓解峰值"的依据，丢弃 |
| `Stream Ptr` / `Device Type` | 流/设备 | 低 |

**判定**：tensor 的 size/生命周期基本捕获；但用了 pool Duration 而非 Active Duration，且释放点全局状态丢失，影响复用候选判定的准确性（属 Phase 2 处理问题，但根因是 Active 字段未读取）。

---

## 7. trace_view.json（时序因果的唯一来源）

基于流式扫描的事件类别清单（3.59M 事件）与脚本实际处理分支比对。

**脚本捕获**：device kernel（Task Type）→ stream/gap/stall/h2d；HostToDevice flow → dispatch latency；cpu_op → prefetch + callstack + h2d 回连；AscendCL@opCompile → 在线编译分类；async_npu(torch_to_npu) → h2d 源码定位；Node@launch → h2d 配对；C 事件中 `MHz` → 降频；GC 事件；SynchronizeStream；python_function（仅计数）。

**未捕获**：

**高影响**：

| 事件类 | 含义 | 影响 |
|--------|------|------|
| C (counter) 事件 176 万个，仅用了 2 个 `MHz` | 见下表 | 高——动态资源利用率时间线几乎全丢 |

C 事件 arg 键实际分布（脚本只取 MHz）：

| arg 键 | 事件数 | 含义 |
|--------|--------|------|
| `value` + `acc_id` | 167 万 / 76 万 | AI Core 利用率等通用计数器时间线（MindStudio 的利用率色带） |
| `KB` | 2.7 万 | 内存占用计数器 |
| `Read(MB/s)` / `Write(MB/s)` | 1.8 万 | **HBM 读写带宽时间线** |
| `Hit Rate(%)` | 9074 | **Cache 命中率时间线** |
| `Throughput(MB/s)` | 9074 | **内存吞吐时间线** |
| `L2 Buffer Bw Level` / `Mata Bw Level` | 4438 | L2 cache / MAC 带宽级别 |
| `MHz` | 2 | 频率（唯一被用的） |

HBM 带宽 / Cache 命中率 / 利用率时间线是判断 memory-bound vs compute-bound 的**动态**依据，远比 kernel_details 的静态 ratio 丰富。当前完全盲区。

**中影响**：

| 事件类 | 含义 | 影响 |
|--------|------|------|
| `enqueue` / `dequeue` 事件 + `async_task_queue` flow | host 侧任务队列的入队/出队时序与耗时 | 中——dequeue duration 实测首 op 达 51ms（launch 线程阻塞），是 host dispatch 瓶颈的直接证据，未捕获 |
| M (metadata) 事件：`process_name`/`thread_name` | pid/tid → 可读名（如 `Python`/`CPU`） | 低中——脚本输出原始 `(297976224, 2)` 元组，人不识其含义；MindStudio 显示命名流。本样本 thread 名多为 `Thread N` 价值有限，但 process_name/labels 至少能区分 host/device pid |

**判定**：下发链/编译/h2d 捕获到位且较深；但 counter 时间线（带宽/cache/利用率）是最大盲区，enqueue/dequeue 队列阻塞次之。

---

## 8. communication.json + communication_matrix.json

**脚本读取**（`parse_communication.py`）：collective op 的 `Communication Time Info`（Elapse/Transit/Wait/Sync/Idle）、`Communication Bandwidth Info`（Bandwidth/Transit Size/Transit Time/Transport Type/Size Distribution）；matrix 的 per-link 带宽。

**未捕获**：

| 项 | 含义 | 影响 |
|----|------|------|
| P2P op 详细时序 | `step_data["p2p"]` 仅计数，未取每个 P2P op 的耗时分解 | 中——send/recv 密集的流水并行场景，P2P 是通信主体，只数个数不看耗时无法判断 P2P 瓶颈 |
| `Total Op Info` 中的其他字段 | 每步聚合信息只取了 Size Distribution | 低——可能还有 op 数等聚合字段未用 |

**判定**：collective 分析完整；P2P 仅计数是真实缺口（依赖场景）。

---

## 9. 无专用 parse 脚本的文件

| 文件 | 列/规模 | 内容 | 是否应纳入 |
|------|---------|------|-----------|
| `api_statistic.csv` | `Device_id, Level, API Name, Time, Count, Avg, Min, Max, Variance`（220 行） | Level=acl(216)/communication(2)/node(1)，CANN runtime API 级统计（如 `Add_Tiling`、aclnn 启动 API 耗时） | **应纳入**——host-bound 分析时，acl API 耗时是 dispatch 开销的直接来源，目前整套脚本无此视角 |
| `npu_module_mem.csv` | `Device_id, Component, Timestamp, Total Reserved, Device`（35 万行） | CANN 模块级内存（SLOG 等），多为内部组件 | 可不纳入——偏 CANN 内部，对模型优化 actionable 度低 |
| `FRAMEWORK/torch.*`（6 个文件） | 二进制格式 | torch 的 op_range/op_mark/python_tracer/memory_usage 等 | 不纳入——二进制非文本可解析，且其信息已被 trace_view.json/operator_details.csv 聚合体现 |

**判定**：`api_statistic.csv` 是真实遗漏（host-bound 的 API 级视角缺失）；npu_module_mem 与 FRAMEWORK 二进制文件不纳入合理。

---

## 汇总：未捕获原始信息按影响排序

> 已剔除跨文件冗余：标 **[真盲区]**=全局未捕获；**[部分盲区]**=别处有但口径不同/也未读；**[非盲区→P2]**=别处已捕获，原脚本缺失属 Phase 2 处理问题。

### 高影响（真盲区，建议优先补）
1. **[真盲区] trace_view C(counter) 事件**：HBM Read/Write 带宽、Cache Hit Rate、Throughput、L2/MAC Bw Level、利用率时间线（176 万事件，仅用 2 个 MHz）。无任何文件替代（kernel_details 的 ratio 是静态快照）。
2. **[真盲区] kernel_details `Input/Output Formats`**：ND/NZ 等内存格式，仅此文件有且未读。数据布局优化（优先级 7）无数据根基。
3. **[真盲区] api_statistic.csv 整文件**：CANN runtime API 耗时（acl 层 216 条），无 parse 脚本，host-bound 的 API 级视角缺失。
4. **[部分盲区] kernel_details `Input Data Types`**：dtype 在 trace_view cpu_op `Input type` 有但也未读；Formats 则是真盲区。

### 中影响（真盲区）
5. **[真盲区] kernel_details**：`aic_fixpipe`、`icache_miss_rate`、硬件单元**绝对时间**（只用 ratio）、`OP State`、`aicore_time/aiv_time`。
6. **[真盲区] operator_details**：`Host Total` / `Device Total` / `With AICore` 变体（inclusive 耗时与 AI_CPU 归属）。
7. **[真盲区] memory_record `Total Active(MB)` + operator_memory Active 列**：真实活集，两文件都有 Active 字段但都未读。
8. **[真盲区] operator_memory `Active Duration`** / 释放点全局状态（用错 Duration 维度的根因）。
9. **[真盲区] trace_view `enqueue/dequeue`**：队列阻塞（dequeue 实测 51ms）。
10. **[真盲区] step_trace `Overlapped` / `Stage` / `Bubble`**：PP 场景。
11. **[真盲区] communication P2P** op 详细时序（仅计数）。

### 非盲区 → 归 Phase 2（别处已捕获）
12. **[非盲区] kernel_details `Start Time(us)`**：trace_view 设备算子 `ts` 已用于时间线/stall。kernel_details 内靠行序做 wait-context/可融合序列属处理问题。
13. **[非盲区] kernel_details `Stream ID`**：trace_view `(pid,tid)` 已做流级分析。
14. **[非盲区] op_statistic `Core Type`**：kernel_details `Accelerator Core` 已按 core 聚合。
15. **[部分盲区] op_statistic `Min/Max`**：kernel_details filter 模式可看，全局聚合无。

### 低影响
16. op_statistic `Ratio%`（自算等价）、`Device_id`。
17. trace_view M 事件（thread 名多为 `Thread N`，价值有限）。
18. 各文件 `Stream Ptr` / `Device Type` / 标识符列。
19. npu_module_mem.csv（CANN 内部）。

---

## 方法论说明

- 本审计**只判定"原始字段是否被读取"**，不评判读取后的处理是否合理。例如 operator_memory 用 `Duration` 而非 `Active Duration` 判复用候选——字段未读属 Phase 1 发现，但"该用哪个字段"属 Phase 2 议题，此处仅标注根因。
- **跨文件冗余是本审计的关键校正**：某字段在某脚本未读，若别处已捕获则不计为全局盲区，而归为"该脚本未利用已有信息"的 Phase 2 处理问题（典型：Start Time / Stream ID / Core Type）。
- 影响等级按"该信息缺失会让某个分析方向完全盲区/部分盲区/仅细节缺失"评定。
- 样本为单 rank 推理采集，部分列（Stage/Bubble/多卡字段）在该样本中无值，但列定义存在，PP/多卡场景下会激活，仍计为遗漏。

**下一步（Phase 2）**：在确认原始信息捕获完整后，审计基于已捕获信息的处理是否充分——是否把已读字段的值榨干、聚合维度是否合理、suspect signals 是否覆盖了已有信息能支撑的所有信号；以及"非盲区→P2"项（别处已捕获但本脚本未利用）是否应在脚本内打通。
