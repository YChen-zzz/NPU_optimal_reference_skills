# OPT-Skills Profiling 改进方案

> 综合两阶段审计（Phase 1 原始信息捕获、Phase 2 处理充分性与信息分析全面性）+ diff_profiling 审查，形成完整改进路线。
> 标尺：架构无关的推理优化可用性。样本（AlphaFold 单次推理）的特例特性（叶 op 主导 host、无 caching）不外推为通用需求。
> 来源标注：[P1]=Phase 1 字段缺失，[P2]=Phase 2 处理不足，[C]=信息分析维度缺口，[D]=diff 验证闭环。

## 改进分五类

| 类 | 含义 | 典型 |
|----|------|------|
| **A 捕获补全** | 字段在文件内但脚本没读（含真盲区） | counter 时间线、Formats、Host Total |
| **B 处理修复** | 字段已读或在文件内，处理退化/用错维度 | 行序代时间、ratio 平均化、Duration 代 Active |
| **C 分析维度补全** | 数据支持但无脚本产出的分析 | host 类别分解、掩盖量化、收益上限 |
| **D 验证闭环** | diff_profiling 缺口 | 多维 diff、瓶颈转移、归一化 |
| **E 配套** | 阈值/filter/文档同步 | 阈值适配、文档自描述 |

---

## A. 捕获补全

| # | 项 | 来源 | 类型 | 推理收益 | 工作量 |
|---|----|------|------|---------|--------|
| A1 | trace_view counter 时间线：per-die HBM Read/Write 带宽、LLC Hit Rate/Throughput、L2/MAC Bw Level、APP/HBM 占用、利用率累加器（176 万事件仅用 2 个 MHz） | P1 真盲区 | 捕获+处理 | 动态 bound 判定、阶段分析、带宽饱和——最大单项收益 | 大 |
| A2 | kernel_details `Input/Output Formats`（ND/NZ） | P1 真盲区 | 捕获 | 数据布局优化（优先级 7）有数据根基 | 小 |
| A3 | kernel_details `Input Data Types` + trace_view cpu_op `Input type` | P1 部分盲区 | 捕获 | dtype 转换开销可见 | 小 |
| A4 | api_statistic.csv 新 parse 脚本（CANN runtime API 耗时，acl 层 216 条） | P1 真盲区 | 新脚本 | host-bound 的 API 级归因（tiling/launch 开销） | 中 |
| A5 | operator_details `Host/Device Total` + `With AICore` 变体 | P1 in-file | 捕获 | inclusive 耗时与 AI_CPU 归属 | 小 |
| A6 | operator_memory `Active Duration` + `Allocation/Release Total Active` | P1 in-file | 捕获 | 真实活集、复用候选正确分类 | 小 |
| A7 | memory_record `Total Active(MB)` | P1 in-file | 捕获 | batch 上限估算准 | 小 |
| A8 | kernel_details `fixpipe`/`icache_miss_rate`/绝对 hw 时间/`OP State` | P1 in-file | 捕获 | 硬件单元全覆盖、icache 压力、动态 shape kernel | 小 |
| A9 | step_trace `Overlapped`/`Stage`/`Bubble` | P1 in-file | 捕获 | 多卡/PP 推理重叠与 bubble | 小 |
| A10 | trace_view `enqueue`/`dequeue` + dequeue duration | P1 in-file | 捕获 | host 队列阻塞（实测首 op 51ms） | 小 |
| A11 | communication P2P op 详细时序 | P1 in-file | 捕获 | PP 推理 P2P 瓶颈 | 中 |

> 注：kernel_details `Start Time`/`Stream ID` 不在 A 类——trace_view 已捕获（非盲区），归 B 类（kernel_details 自身处理要用）。

---

## B. 处理修复

| # | 项 | 来源 | 类型 | 推理收益 | 工作量 |
|---|----|------|------|---------|--------|
| B1 | kernel_details 读 `Start Time`+`Stream ID`，按流分组、按时间排序做 fusible/wait-context（替代行序） | P2 | 纯处理 | 修正 fusible 跨流误报、wait-context 错位 | 中 |
| B2 | kernel_details 硬件 ratio 改 duration 加权 + top-N 重 kernel 单列（替代算术平均） | P2 | 纯处理 | compute/memory bound 不被 bimodal 稀释 | 小 |
| B3 | kernel_details suspect 增"高耗时高利用率"列表（真 compute-bound 靶点） | P2 | 纯处理 | 替换/量化靶点浮现（优先级 8） | 小 |
| B4 | operator_details 用 `Host Total` + 按 `Call Stack` 层级聚合（穿透层级归因） | P2/C6 | 纯处理 | 通用分层热路径的 dispatch 层瓶颈可见；喂饱 Line A 门禁 | 中 |
| B5 | operator_memory 短命判定改用 `Active Duration` | P2 | 纯处理 | 服务场景下修正复用候选漏判 | 小 |
| B6 | step_trace 去掉 `len>1` 门控，单步推理也产 suspect signal + 推理专用信号 | P2 | 纯处理 | 单步推理不再零信号 | 小 |
| B7 | memory_record 按 `Component` 分段（WORKSPACE vs APP/PTA） | P2 | 纯处理 | 区分可控 workspace 与 tensor 内存 | 小 |

---

## C. 分析维度补全（信息分析全面性缺口）

| # | 项 | 来源 | 数据支持 | 推理收益 | 工作量 |
|---|----|------|---------|---------|--------|
| C1 | host 时间类别分解（sync/dispatch/alloc/compile/python） | C | operator_details op 名+host time + 归类规则 | host-bound 定方向（sync 与 dispatch 优化相反） | 中 |
| C2 | Free 成因分解（sync-wait/dispatch-starved/comm-wait/no-work） | C | trace_view h2d/stall/sync 统一聚合 | 定位"为什么 idle" | 中 |
| C3 | 掩盖维度量化（stream overlap / comm-compute overlap / 可并行算子识别） | C | trace_view 多流 timeline + step_trace Overlapped | 四维度唯一无产出的维度，double-buffer/重叠收益可量化 | 中 |
| C4 | 各优化类型量化收益上限（Amdahl 式排序） | C | 已有信号（Free/类别分解/小算子总量） | 喂饱确认节点 A"理论收益上限"，候选按数据排序 | 中 |
| C5 | 重复计算检测（区别于重复分配） | C | operator_details Call Stack + 相同输入 shape | 去重维度靶点（同结果多次算） | 中 |
| C6 | 调用链层级归因 | C | = B4 | per-layer inclusive host/device 占比 | 中 |
| C7 | 动态 shape 跨 request 重编译归因 | C | trace_view compile 事件 ts × op shape 变化 | 非 LLM 推理 host 大户 | 中 |
| C8 | 数据预处理/H2D pipeline 独立开销量化 | C | trace_view prefetch + cpu_op 时间线 | 非 LLM 推理常是主瓶颈 | 中 |
| C9 | request-invariant 预计算识别 | C | operator_details Call Stack + 跨 request 不变性 | 复用维度：跨 request 可缓存计算 | 大 |
| C10 | 内存峰值归因（峰值时刻在跑什么） | C | memory_record 峰值 ts × operator_memory 分配窗 | 峰值是哪段代码造成 | 中 |
| C11 | 内存带宽饱和分析 | C | 依赖 A1 counter | 区分"带宽饱和"vs"访问低效" | 大 |

---

## D. 验证闭环补全（diff_profiling）

| # | 项 | 来源 | 推理收益 | 工作量 |
|---|----|------|---------|--------|
| D1 | diff 增 step_trace 对比（utilization/Free/Computing） | D | 确认 host-bound 是否真改善 | 小 |
| D2 | diff 增 host 开销对比（operator_details pure host %/host self） | D | 确认 dispatch/sync 是否降 | 小 |
| D3 | diff 增 h2d/dispatch 对比（trace_view h2d 区段/dispatch latency） | D | 确认 h2d 是否解除 | 中 |
| D4 | diff 增 kernel hw 占比对比（compute/memory boundness） | D | 确认瓶颈性质是否变 | 小 |
| D5 | 瓶颈类型转移检测（before/after Host/Compute/Memory/Allocator） | D | 瓶颈转移是优化常见结果 | 小 |
| D6 | memory peak 用 Active 非 Reserved | D | 真实峰值变化（Reserved 受 pool 保留干扰） | 小 |
| D7 | 归一化（step/样本数）+ L0/L1 口径守卫 + 多次运行方差 | D | 避免伪收益/口径错比 | 中 |

---

## E. 配套

| # | 项 | 推理收益 | 工作量 |
|---|----|---------|--------|
| E1 | thresholds.py 阈值适配推理（`fusible_small_us` 等，LLM decode 小 kernel 场景） | 信号阈值合理 | 小 |
| E2 | filter 模式分组帧数可配置（operator_details 前 3 帧→可调，适配 HF 深链） | 调用点定位唯一 | 小 |
| E3 | 文档同步机制：脚本输出 section 索引自描述，references 引用脚本元数据而非手抄 | 避免能力变更漏同步（本次已发生） | 中 |
| E4 | thresholds.py 补 A/B/C/D 新项的阈值 | 新分析可调 | 小 |

---

## 优先级路线图

### P0 — 先做（纯处理/低成本/修正性强，数据已在文件内）
- **B1** kernel_details Start Time + Stream ID 按流分组
- **B4/C6** operator_details Host Total + Call Stack 层级聚合
- **B5** operator_memory 改 Active Duration
- **B6** step_trace 单步 suspect signal
- **B7** memory_record WORKSPACE 分段
- **B2/B3** kernel_details ratio 加权 + 真 compute-bound suspect
- **C1** host 时间类别分解（归类规则）
- **C4** 各优化类型量化收益上限（Amdahl）
- **D6/D7** diff 用 Active + 归一化/口径守卫

> P0 全部是"已有数据没处理/用错维度"的修复，无新捕获依赖，应优先闭合。

### P1 — 中成本（新捕获或跨脚本聚合）
- **A2/A3/A5/A8** kernel_details/operator_details 字段补读（Formats/Data Types/Total/fixpipe/icache）
- **A9/A10** step_trace Overlapped + trace_view enqueue/dequeue
- **C2** Free 成因分解
- **C3** 掩盖维度量化（stream overlap）
- **C10** 内存峰值归因
- **D1–D5** diff 多维对比 + 瓶颈转移
- **A4** api_statistic 新脚本

### P2 — 高成本（依赖 counter 捕获或外部输入）
- **A1** trace_view counter 时间线（捕获+处理，最大单项收益）
- **C11** 内存带宽饱和（依赖 A1）
- **C7** 动态 shape 重编译归因
- **C8** 预处理/H2D 独立量化
- **C9** request-invariant 预计算识别
- **C5** 重复计算检测
- **A11** communication P2P detail

---

## 与现有框架/门禁的衔接

| 现有门禁/机制 | 当前问题 | 改进项 |
|--------------|---------|--------|
| 确认节点 A "理论收益上限"排序 | priority list 仅定性，无量化 | C4 各优化类型量化收益上限 |
| Line A "穿透层级量化"门禁（>10% host time 须候选） | 无 per-layer inclusive 产出，喂不饱 | B4/C6 Host Total + 层级聚合 |
| 确认节点 B "profiling 确认收益" | diff_profiling 只 diff 算子+peak，靠人工跨脚本 | D1–D7 验证闭环补全 |
| 四维度（去重/复用/掩盖/替换） | 掩盖维度无分析支持 | C3 掩盖量化 |
| profiling_to_action 优先级 7（数据布局） | 无 Formats 数据根基 | A2 Formats 捕获 |
| 优先级 8（kernel 本身慢） | suspect 只筛低利用率，漏真 compute-bound | B3 |
| Suspect Signals 机制 | 单步推理零信号；成因不分解 | B6、C1、C2 |

---

## 风险与验证

- **样本偏差风险**：本方案判断已剥离 AlphaFold 单次推理特例（叶 op 主导、无 caching）。实施后须在**至少 2 类负载**验证——单次推理 + 多 request 服务场景，确认 Active Duration/Host Total 在服务场景确有差异（本样本测不出）。
- **counter 语义风险**：A1 的 `value`/`acc_id` 累加器含义需先确认（已确认 HBM bw/LLC hit rate 等，但 value/acc_id 待定）再处理。
- **回归风险**：B 类纯处理改动（行序→时间序、Self→Total）可能改变既有输出，需保留旧输出作对比、用 evidence_db 已有案例回归。
- **门槛**：C 类新分析若产出不稳定信号，按现有 Suspect Signals 惯例标 [SIGNAL] 不 [DEFINITE]，避免误判。

---

## 方法论说明

- 本方案合并 Phase 1（字段捕获）+ Phase 2（处理与分析维度）+ diff 审查，是 profiling 子系统的完整改进视图。
- 优先级逻辑：**补分析维度（C）与补字段（A）同样重要，但 C 中"数据已支持的"（C1/C2/C4/C6）比 A 中"需新捕获的"（A1）成本低，故 C1/C4/C6 入 P0、A1 入 P2**。
- 掩盖维度（C3）是四维度结构性失衡的唯一无产出维度，无论成本应纳入 P1。
- 验证闭环（D）是推理优化"确认收益"的最后一环，当前残缺使确认节点 B 实质靠人工，P0 先补 D6/D7（低成本），P1 补多维 diff。
