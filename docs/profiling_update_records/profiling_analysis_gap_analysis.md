# Profiling 分析能力补充评估

> 参考来源: msprof-analyze 库的 `PROFILING_ANALYSIS_REPORT.md`(22 个 Checker 的分析逻辑) + `/Users/kaili/Downloads/20260424` 真实多卡训练 profiling 数据。
> 目的: 对比 model_opt 当前的 profiling 分析能力与 msprof-analyze 的差距,找出可补充的分析维度。

---

## 1. 现状对比总览

model_opt 当前的 profiling 分析脚本(7 个)覆盖的维度:
- step_trace: Host/Device bound 判定
- op_statistic: 算子类型聚合
- kernel_details: 硬件单元利用、小算子、Block Dim、wait stall
- operator_details: Call Stack 源码定位、host/device ratio
- memory_record / operator_memory: 内存时间线、tensor 生命周期
- trace_view: device timeline、dispatch latency、compile A/B、prefetch 候选

msprof-analyze 的 22 个 Checker 覆盖的维度(按 6 类):
- computation: BlockDim 对齐、算子流水线瓶颈、AI Core 性能(cube/FA/vector 分治)、AI CPU fallback、动态 shape、可融合算子序列、图融合规则、AI Core 降频
- schedule: 流同步、GC、推测性 GC、SyncBN、Dataloader
- communication: 通信包大小、字节对齐、带宽竞争、通信重传
- overall: 环境变量
- cluster: 集群通信
- memory: 内存操作

---

## 2. msprof-analyze 有而 model_opt 缺失的分析维度

以下逐项说明: msprof-analyze 做了什么、model_opt 当前是否覆盖、是否值得补、如何补。

### 2.1 通信分析(完全缺失) 🔴 P0

**msprof-analyze 做了什么**:
- `communication.json` + `communication_matrix.json` 解析,提取每个 HCCL 算子(allgather/allreduce/alltoall)的:
  - 通信时间分解: Transit Time(实际传输)、Wait Time(等待)、Synchronization Time(同步)、Idle Time
  - 带宽信息: per-link 的 Transit Size / Transit Time → Bandwidth(GB/s)
  - 通信包大小检查(PacketChecker): 异常小包占比
  - 字节对齐检查(ByteAlignmentChecker): size % 512 != 0
  - 带宽竞争检查(BandwidthContentionChecker): SDMA 与 matmul 时间重叠 + 带宽低于阈值
  - 通信重传检查(CommunicationRetransmissionChecker): RDMA transit time 异常

**真实数据验证**(`/Users/kaili/Downloads/20260424`):
- `communication.json` 存在,含 step0 的 collective 通信(allgather/alltoall)
- `communication_matrix.json` 存在,含 per-link 带宽(如 `0-0: LOCAL, 517MB, 3.67ms, 141GB/s`)
- op_statistic 显示 `allgatherAicpuKernel` 占 42.95%(36.4s)、`alltoallAicpuKernel` 占 7.92%——**通信是这份 profiling 的头号瓶颈**,但 model_opt 的脚本完全没分析它

**model_opt 现状**: 完全缺失。step_trace 能看到 Communication 列,但无法深入到"哪个通信算子慢、为什么慢(等待 vs 传输 vs 同步)、带宽是否正常"。

**补充方案**:
- 新增 `parse_communication.py` 解析 `communication.json` + `communication_matrix.json`
- 输出:
  1. 通信算子排名(按 Elapse Time / Transit Time / Wait Time)
  2. 时间分解(Transit vs Wait vs Sync vs Idle)——区分"真在传"还是"在等"
  3. per-link 带宽统计(min/avg/max + 离群低带宽 link)
  4. 通信包大小分布(异常小包检测)
  5. 字节对齐检测(size % 512)
- Suspect Signals: Wait Time Ratio 高(在等不在传)、带宽离群、小包占比高、不对齐

### 2.2 可融合算子序列检测(缺失) 🟡 P1

**msprof-analyze 做了什么**(FusibleOperatorChecker):
- 滑动窗口扫描 op_summary 的连续算子序列
- 匹配规则文件 `fusible_operator.yaml` 中的可融合 pattern
- 计算 wall time vs NPU time 比例 → host 瓶颈标志
- 计算 mte_time / aicore_time → 内存传输瓶颈标志
- 按出现频次 + 总时间占比排序

**model_opt 现状**: op_statistic 能看单算子占比,但**不能检测"连续 N 个小算子可融合"**这种序列级 pattern。kernel_details 的 small kernel 统计是单算子维度的。

**补充方案**:
- 在 `parse_kernel_details.py` 或 `parse_op_statistic.py` 中增加"连续小算子序列"检测
- 不做规则匹配(msprof-analyze 的 yaml 规则维护成本高),而是检测"连续 N 个 duration < 阈值的算子、累计耗时占比高"作为 SIGNAL
- agent 结合源码判断是否可融合

### 2.3 AI Core 降频检测(缺失) 🟡 P1

**msprof-analyze 做了什么**(AICoreFreqChecker):
- 从 trace_view.json 解析 AI Core 频率事件
- 计算降频比例 = sum(max_freq - freq) / (max_freq * count)
- 降频 ≥ 5% → 标记(高温/功耗限制导致性能下降)

**model_opt 现状**: parse_trace_view 没解析频率事件。trace_view.json 里确实有 `AI Core Freq` 泳道(前面验证过)。

**补充方案**: 在 `parse_trace_view.py` 中增加频率解析——检测 `AI Core Freq` 泳道的事件,计算降频比例,作为 DEFINITE 信号。

### 2.4 流同步检测(缺失) 🟡 P1

**msprof-analyze 做了什么**(SynchronizeStreamChecker):
- 检测 NODE_LAUNCH 紧接 SYNC_STREAM 的共现模式
- 共现比例高 → 可能设置了 `ASCEND_LAUNCH_BLOCKING=1`(同步执行)

**model_opt 现状**: 没有检测。trace_view 里有 CANN 层事件但没解析这个 pattern。

**补充方案**: 在 `parse_trace_view.py` 的 CANN 事件中检测 sync stream pattern,作为 DEFINITE 信号(环境变量问题)。

### 2.5 GC 检测(缺失) 🟢 P2

**msprof-analyze 做了什么**(GcChecker + ConjecturedGcChecker):
- 直接检测 GC 事件(cat=GC)
- 推测性 GC: 当无显式 GC 事件时,通过 free 期间 ACL 活动低推测 GC

**model_opt 现状**: 没有检测。trace_view 里有 GC 事件(cat=GC,前面验证训练数据有)。

**补充方案**: 在 `parse_trace_view.py` 中检测 GC 事件,统计总耗时和 Top-K 长GC,作为 SIGNAL。

### 2.6 环境变量检查(缺失) 🟢 P2

**msprof-analyze 做了什么**(EnvironmentVariableChecker):
- 检查 profiling 数据中的环境变量(ASCEND_LAUNCH_BLOCKING、ACLNN_CACHE_LIMIT、PYTORCH_NPU_ALLOC_CONF 等)

**model_opt 现状**: `01_preparation` 有环境变量设置指引(TASK_QUEUE_ENABLE、CPU_AFFINITY_CONF),但**没有从 profiling 数据中回检环境变量是否设对**。

**补充方案**: 新增 `parse_env_check.py` 或在现有脚本中增加环境变量检查,从 `profiler_info_0.json` / `profiler_metadata.json` 中提取环境变量,对照检查清单。

### 2.7 Block Dim 对齐检查(部分覆盖) 🟢 P2

**msprof-analyze 做了什么**(BlockDimChecker):
- 检查 block_dim % core_num == 0(AI Core 核数或 AI Vector 核数)
- 不对齐 → 硬件利用不充分

**model_opt 现状**: kernel_details 有 Block Dim 分布统计,但**没有检查与核数的对齐**——只统计了 Block Dim=1 的数量,没做 `% core_num` 检查。

**补充方案**: 在 `parse_kernel_details.py` 的 Block Dim 部分增加对齐检查(需知道硬件核数,可从环境或参数传入)。

### 2.8 Cube 内轴对齐检查(缺失) 🟢 P2

**msprof-analyze 做了什么**(AICorePerformanceChecker._check_cube_inner_axis):
- 对 cube 算子(matmul)检查内轴是否对齐 256(NZ 格式)或 128(ND 格式)
- 不对齐 → tiling 效率低

**model_opt 现状**: 没有检测。这对 matmul 密集的 LLM 推理场景很有价值。

**补充方案**: 在 `parse_kernel_details.py` 的 filter 模式中,对 matmul 类算子增加内轴对齐检查(从 Input Shapes 提取维度,检查 K 轴 % 256/128)。

---

## 3. model_opt 有而 msprof-analyze 没有的(保持优势)

- **Call Stack 源码定位**: model_opt 的 operator_details + profiling_to_source 的"五座桥"方法论,msprof-analyze 没有(它只做数据层检查,不做源码归因)
- **compile A/B 分类**: model_opt 的 trace_view 脚本区分预热编译 vs 每步在线编译,msprof-analyze 的 DynamicShapeChecker 只检测是否动态 shape
- **五种分析模式方法论**: model_opt 教 agent 怎么推理,msprof-analyze 是固定 Checker
- **prefetch/prealloc 候选**: model_opt 筛 H2D/alloc 操作 + Call stack,msprof-analyze 没有

---

## 4. 优先级与落地清单

| 优先级 | 补充项 | 落地方式 | 依据 |
|--------|--------|---------|------|
| 🔴 P0 | 通信分析(communication.json/matrix) | 新增 `parse_communication.py` | 真实数据显示通信占 50%+,但当前完全没分析 |
| 🟡 P1 | 可融合算子序列检测 | 扩展 `parse_kernel_details.py` 或 `parse_op_statistic.py` | 当前只有单算子维度,缺序列级 |
| 🟡 P1 | AI Core 降频检测 | 扩展 `parse_trace_view.py` | trace_view 有频率数据但没解析 |
| 🟡 P1 | 流同步检测 | 扩展 `parse_trace_view.py` | ASCEND_LAUNCH_BLOCKING 误设是常见问题 |
| 🟢 P2 | GC 检测 | 扩展 `parse_trace_view.py` | 训练场景 GC 可能占比高 |
| 🟢 P2 | 环境变量回检 | 新增 `parse_env_check.py` | 防止环境变量设错 |
| 🟢 P2 | Block Dim 对齐检查 | 扩展 `parse_kernel_details.py` | 当前只统计分布没检查对齐 |
| 🟢 P2 | Cube 内轴对齐检查 | 扩展 `parse_kernel_details.py` filter 模式 | matmul 密集场景有价值 |

### 不做的事

- **图融合规则匹配**(GraphFusionChecker): msprof-analyze 依赖计算图 + yaml 规则,model_opt 没有计算图解析基础设施,维护成本高。agent 通过可融合序列检测 + 源码分析可达到类似效果。
- **集群通信分析**(ClusterCommunicationChecker): 需要多 rank 的 communication 数据交叉对比,当前多卡分析能力尚不完善,等通信分析基础打好后再考虑。
- **SyncBN / Dataloader 检测**: 场景较窄(训练特定),当前聚焦推理,优先级低。

---

## 5. 真实数据验证记录

对 `/Users/kaili/Downloads/20260424` 这份多卡训练 profiling 的验证:

| 文件 | 存在 | model_opt 当前是否解析 | msprof-analyze 是否解析 |
|------|------|----------------------|----------------------|
| kernel_details.csv | ✅(18K行) | ✅ | ✅(OpSummary) |
| operator_details.csv | ✅(1.9M行) | ✅ | ✅ |
| trace_view.json | ✅(1GB) | ✅ | ✅(Msprof) |
| op_statistic.csv | ✅ | ✅ | ✅ |
| step_trace_time.csv | ✅ | ✅ | ❌ |
| memory_record.csv | ✅ | ✅ | ❌ |
| operator_memory.csv | ✅ | ✅ | ❌ |
| **communication.json** | ✅ | ❌ | ✅ |
| **communication_matrix.json** | ✅ | ❌ | ✅ |
| **api_statistic.csv** | ✅ | ❌ | ❌(用 DB 路径) |
| **npu_module_mem.csv** | ✅ | ❌ | ❌ |
| profiler_info_0.json | ✅ | ❌ | ✅(环境变量) |

**关键发现**: 这份数据的 op_statistic 显示 `allgatherAicpuKernel` 占 42.95%、`alltoallAicpuKernel` 占 7.92%——**通信是头号瓶颈**,但 model_opt 的 7 个脚本没有任何一个能分析通信。这是最大的缺口。
