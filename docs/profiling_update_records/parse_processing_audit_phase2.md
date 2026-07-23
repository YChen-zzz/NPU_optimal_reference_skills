# Parse 脚本处理充分性审计（Phase 2）——推理优化视角

> 范围：审计脚本**对已捕获字段的处理是否充分**，是否把信息榨干、是否产出推理优化可用的信号。不再讨论"字段有没有读到"（Phase 1 已完成，见 `parse_coverage_audit.md`）。
> 视角：**推理优化**（非训练），以架构无关的通用轴为标尺（适用于 LLM / 视觉 / 扩散 / 科学计算 / 语音等）。推理与训练的核心区别：延迟而非吞吐、单步无反向、host 开销常是主瓶颈、内存峰值决定 batch 上限、首 request 有编译/冷缓存开销。LLM 自回归的 prefill/decode 等家族特异项单列，不混入通用判断。

## 评估框架

**推理优化的核心诉求**（架构无关标尺——适用于 LLM / 视觉 / 扩散 / 科学计算 / 语音 / 推荐等所有推理负载）：

| 维度 | 通用推理关切 | 训练区别 |
|------|---------|---------|
| 端到端延迟 | 每 request 延迟、首 request/warmup 延迟 | 训练关心吞吐/step 时间 |
| host 开销 | eager 推理 host 常是主瓶颈（dispatch / sync / Python 逻辑） | 训练有反向掩盖 |
| memory bandwidth | 带宽密集算子是否饱和 | 训练梯度也吃带宽 |
| format/layout | NPU ND/NZ 转换是推理常见暗坑 | 同 |
| 小算子碎片 | eager 推理 kernel 数极多，融合收益大 | 图模式训练已融合 |
| 内存峰值 | 决定最大 batch（吞吐上限） | 训练峰值含激活+梯度 |
| warmup/compile | 首 request 编译/冷缓存延迟 | 训练可 skip_first |
| 数据预处理/H2D | 输入解码/增强/特征提取 + 输入搬运（非 LLM 常是主瓶颈） | 训练 DataLoader 可多 worker |
| 动态 shape | 跨 request shape 变化触发重编译 | 训练 shape 通常固定 |
| 控制流同步 | shape/数值驱动的分支强制 `.item()` 同步 | 同 |

> **模型家族差异**：上述是通用轴，不同家族各有侧重——LLM 自回归有 prefill/decode 两相 + KV cache；视觉 CNN 关心 conv/NCHW 格式 + 图像预处理；扩散模型关心迭代去噪循环（同图重复 N 步）；科学计算（如本样本 AlphaFold）关心长序列 attention + 重 Python 预处理 + 动态 shape 控制流；语音关心流式实时约束。本审计以**通用轴**为标尺，家族特异项单列（见末尾「模型家族特异」），不混入通用判断。

**Phase 2 判定准则**：
1. 已读字段是否被处理到极限（有无退化近似）。
2. 聚合维度是否合理（平均/求和是否会抹平推理关键特征）。
3. suspect signals 是否覆盖了"已捕获信息能支撑的推理模式"。
4. 是否有推理特有分析，数据本支持但无人做。

**与 Phase 1 的衔接**：部分字段脚本自己文件里就有但没处理（如 `kernel_details` 的 Start Time / Stream ID、`operator_memory` 的 Active Duration、`operator_details` 的 Host Total）——Phase 1 从"全局是否盲区"角度判定（有些在别处有），Phase 2 从"该脚本处理是否充分"角度判定（自己有却没用 = 处理不足）。本阶段聚焦后者。

---

## 总体结论

主干推理信号**已覆盖**：host/device bound（step_trace）、耗时算子排名（op_statistic）、compute/memory bound（kernel_details）、源码定位（operator_details）、内存峰值（memory_record / operator_memory）、下发链 / h2d / 编译（trace_view）。推理的主要优化方向都有对应脚本的输出支撑。

但处理层面有**三类削弱推理价值**的不足：

1. **有字段不处理，退化为近似**——`kernel_details` 有 `Start Time` 却用文件行序做时序分析、有 `Stream ID` 却不分组；`operator_memory` 有 `Active Duration` 却用 pool `Duration`；`operator_details` 有 `Host Total` 却只用 `Host Self`。
2. **聚合抹平推理关键特征**——`kernel_details` 硬件 ratio 简单算术平均，掩盖少数重 kernel 的 bimodal 分布（推理常是几个大 kernel 主导）；suspect 筛选只看"高耗时低利用率"，漏掉"高耗时高利用率但绝对值大"的真 compute-bound。
3. **推理特有维度缺分析**——无 host 时间类别分解（sync/dispatch/alloc）、无 stream overlap 量化、无内存峰值归因（峰值时刻在跑什么）、无 host 开销的穿透层级量化（Host Total 未用，Line A 门禁喂不饱）、无动态 shape 跨 request 重编译归因。

---

## 实证验证修订（运行脚本后的修正）

> 上述结论最初基于**读代码**推断。随后在真实数据上跑了全部脚本，发现部分判断被实证**修正**——有的高估、有的低估、有的漏看。以下以实证为准。

**实际瓶颈画像（实证）**：本样本是 host-bound（utilization 22%），根因是：
- `operator_details`：`aten::_local_scalar_dense`（D→H 同步，即 `.item()` 类）287 次占 host 时间 63%；纯 host op 占 host 时间 98.2%。
- `op_statistic`：`broadcastAicpuKernel`（AI_CPU 通信）占 device 时间 31.7%。
- 脚本**已正确浮现**这两个根因——但要注意：本样本的 host 瓶颈是**叶 op 同步**（Self-by-name 直接可抓），这正是 Self-by-name 分析擅长的情况。**不能据此外推"脚本对 host 瓶颈捕获强"**：框架包裹重的推理（dispatch 层瓶颈，Self≈0）和带宽瓶颈，Self-by-name 抓不到，恰恰需要 Host Total / 层级聚合 / counter 时间线等下文维持高影响的维度。本样本"叶 op 主导"是特例，与 Host Total、Active Duration 两处降级错误同根。

**逐条修正**：

| 原判断 | 实证结果 | 修订 |
|--------|---------|------|
| operator_memory 用 Duration 而非 Active Duration → 复用候选误判（高影响 #2） | 实测**本样本** Duration 与 Active Duration 差异小：15/11290 行差 >1ms，max 123ms；短命分类仅 18 个翻转 | **维持中影响**（修正此前降级）。本样本是单次推理（Steps=1，caching allocator 几乎不保留 tensor）→ Active≈Duration，是**特例**。通用推理服务场景（多 request、allocator 保留 tensor 供复用）下 Active<<Duration，被缓存的可复用 tensor 的 Duration 长（在 pool 中）、Active 短（仅被引用时）——用 Duration 会判为"长命"而漏掉，而这正是复用候选最该抓的对象。Active Duration 在"复用 mattered"的场景才关键，与单次推理的特例相反 |
| operator_details 缺 Host Total → Line A 门禁喂不饱（高影响 #4） | 实测本样本 top host op 全是叶节点（`_local_scalar_dense`/`copy_`/`empty_tensor`），Self≈Total | **维持高影响**（修正此前降级）。本样本叶 op 主导是**特例**；通用推理热路径通常分层（框架 wrapper → forward → 子模块 → 叶 op），wrapper/dispatch 层 Self≈0 但 Total 大，Self-by-name 看不到该层。Line A"穿透层级量化"门禁（任何层 >10% host time 须有候选）是通用要求，无 per-layer Total 喂不饱。真正需要的是"Host Total + 按 Call Stack 层级聚合"（两块数据都在文件内，脚本都没用于此） |
| step_trace 处理基本充分 | 实测单步推理（Steps=1）时，**Suspect Signals 整节被 `len(step_data)>1` 门控跳过**，输出零信号 | **新增缺口**：单步推理（推理常见）从 step_trace 拿不到任何 suspect signal；Only Overall（utilization+optimizable space）可用。这是训练导向的多步假设所致 |
| memory_record 处理充分 | 实测有 `WORKSPACE` 组件（4257 条）与 APP/PTA 混在 timeline，未按 Component 分段 | **新增缺口**：WORKSPACE（算子 workspace，可经 tiling/env 控制）与 tensor 内存混算，推理内存优化时无法区分可控部分 |
| **counter 时间线是带宽/cache/利用率**（推断） | 实证确认且更丰富：`HBM 0/Read`、`HBM 1/Write`（per-die 带宽）、`LLC 0 Read/Hit Rate`、`LLC 0 Read/Throughput`、`L2 Buffer Bw Level`、`Mata Bw Level`、`APP/HBM` 占用、`read_bandwidth`/`read_ost` 累加器 | **确认高影响真盲区**，#6 建议**强化**：这是动态资源利用率时间线，阶段分析（warmup/稳态/带宽饱和时刻）的最强依据 |
| op_statistic Core Type 缺失（Phase 1 中影响） | 实测 #1 op `broadcastAicpuKernel` 占 31.7% 是 AI_CPU，但无 Core Type 列仅能从名字猜 | **确认**：Core Type 缺失使 #1 耗时 op 是否 fallback 不可直接判定，本数据正中此场景 |

**仍未审计（诚实标注）**：
- ~~`diff_profiling.py` 的处理~~ → 已审计：仅 diff `op_statistic`（op 耗时）+ memory peak（Reserved），**不 diff 利用率/host 开销/h2d/kernel 硬件占比**；无运行长度归一化。推理收益验证闭环不完整——"op 变快了"看得到，"utilization 提升 / host 开销下降 / h2d 解除"看不到。
- ~~filter 模式对推理深调用链的适配~~ → 已审计：`operator_details --filter _local_scalar` 实测正确把同步定位到 `attention.py:49`（3 帧分组在本项目浅链下足够）。HF generate 深调用链下 3 帧可能不唯一，属场景性风险，非系统缺口。
- 阈值是否适配推理（`fusible_small_us=10` 等，LLM decode 小 kernel 场景）——未实证，留待 Phase 3 调参。
- LLM prefill/decode 分相可行性——属**家族特异**项（见末尾），非通用缺口；CANN 数据不含语义相位标签，需用户标记，有条件可行。

---

## 信息分析全面性：分析维度覆盖评估（本阶段核心）

> 焦点转移：不再问"脚本有没有读到字段"，而问"**推理优化需要回答的全部分析问题，当前分析框架（7 脚本 + 五种分析模式 + profiling_to_action 模式 + 四维度）是否都覆盖**"。是否有整条分析维度缺失，比单字段缺失更致命。

### 核心判断

当前框架偏**描述性**（时间花在哪、谁慢、谁占得多），缺**决策性**分析（为什么慢的成因分解、能并行多少、各优化能省多少）。两个最大空白：

1. **掩盖（overlap / 并行）维度几乎无分析支持**——四维度（去重/复用/掩盖/替换）中唯一没有量化产出的维度。推理优化的"双 buffer / 通信-计算重叠 / 多流并行"全靠 agent 凭经验，无数据支撑。
2. **成因分解缺失**——host-bound 到底是 sync 还是 dispatch 还是 alloc？device Free 到底是等同步、等下发、还是等通信？都没有分解输出，信号散落在多个脚本里靠 agent 拼。

### 覆盖矩阵（按分析问题）

**诊断 / 归因层**（"时间花在哪、为什么"）

| 分析问题 | 现状 | 覆盖 |
|---|---|---|
| device 时间按算子分布 | op_statistic | ✓ |
| device 时间按硬件单元（compute/memory） | kernel_details ratio | ✓（平均化） |
| host 时间按算子分布 | operator_details | ✓ |
| **host 时间按类别分解**（sync/dispatch/alloc/compile/python） | 无——列 op 名但不归类 | ✗ 关键 |
| 时间按**调用链层级**归因（穿透层级） | 无脚本产出，Line A 门禁却要求 | ✗ |
| 时间按**流**归因 | trace_view busy% 但非时间占比 | △ |
| 时间按**阶段**（warmup/稳态；LLM 另有 prefill/decode） | 无 | ✗ |
| **Free/空闲成因分解**（sync-wait/dispatch-starved/comm-wait/no-work） | 散落（h2d/stall/sync）无统一分解 | ✗ 关键 |

**去重维度**（"有没有多余工作"）

| 分析问题 | 现状 | 覆盖 |
|---|---|---|
| 可融合小算子序列 | kernel_details fusible | ✓（行序近似） |
| 重复同尺寸分配 | operator_memory | ✓ |
| **重复计算**（同结果多次计算） | 无——只检测重复分配，不检测重复计算 | ✗ |
| 冗余框架 dispatch（纯 host op） | operator_details pure host | ✓ |

**复用维度**（"结果/资源能否复用"）

| 分析问题 | 现状 | 覆盖 |
|---|---|---|
| buffer 复用候选（短命大 tensor） | operator_memory | ✓ |
| 跨步不变常量可预计算 | trace_view prefetch 候选（部分） | △ |
| **重算 vs 缓存权衡**（中间结果该缓存还是重算） | 无 | ✗ |

**掩盖维度**（"延迟能否并行"）← **整条维度支持最弱**

| 分析问题 | 现状 | 覆盖 |
|---|---|---|
| 计算流内空泡（device stall） | trace_view stalls | ✓ |
| **流间重叠率 / 可掩盖空泡** | 无——只有 busy%，无 overlap 量化 | ✗ 关键 |
| **通信-计算重叠** | step_trace `Overlapped` 未读，无量化 | ✗ |
| **算子间数据依赖 / 可并行算子识别** | 无 | ✗ |
| host-device 重叠（h2d） | trace_view h2d | ✓ |

**替换维度**（"有没有更便宜的等价"）

| 分析问题 | 现状 | 覆盖 |
|---|---|---|
| AI_CPU fallback 算子 | kernel_details aicpu | ✓ |
| 低利用率高耗时算子 | kernel_details suspect | ✓ |
| **高耗时高利用率真 compute-bound** | 无（只筛低利用率） | ✗ |
| 格式/layout 转换（ND/NZ） | 无 Formats 数据 | ✗（Phase 1） |

**验证层**（"优化是否生效"）

| 分析问题 | 现状 | 覆盖 |
|---|---|---|
| 优化前后算子耗时 diff | diff_profiling | ✓ |
| **优化前后利用率/Free diff** | 无 | ✗ |
| **优化前后 host 开销 diff** | 无 | ✗ |
| **优化前后 h2d/dispatch diff** | 无 | ✗ |
| 端到端延迟对比 | 无（需外部计时） | ✗ |

**上限 / 决策层**

| 分析问题 | 现状 | 覆盖 |
|---|---|---|
| 理论加速上限（Free 占比） | step_trace optimizable | ✓ |
| **各优化类型量化收益上限**（消除 sync 省 X / 融合省 Y / 重叠省 Z，Amdahl 式排序） | 无——priority list 仅定性 | ✗ 关键 |
| 内存峰值→batch 上限 | operator_memory parallelism trigger | ✓ |
| 内存带宽是否饱和 | 无（counter） | ✗ |

### 关键缺口详述（按推理影响）

1. **host 时间类别分解**——本数据实证：`_local_scalar_dense`（sync）占 host 63%，但无脚本把它归为"sync 类"并给出"sync=X% / dispatch=Y% / alloc=Z%"的分解。sync 与 dispatch 的优化方向相反（sync→消除 .item；dispatch→flat forward/图编译），不分解就无法定方向。数据支持（operator_details 有 op 名+host time），缺的是**归类规则**。
2. **Free 成因分解**——step_trace 给 Free 总量（73%），但不分解成因。trace_view 有 h2d/sync/stall 等碎片信号，但无统一"Free = sync-wait + dispatch-starved + comm-wait + no-work"的分解。推理 host-bound 时这是定位"为什么 idle"的核心。
3. **掩盖维度量化**——四维度里唯一无产出的维度。流间重叠率、通信-计算重叠、可并行算子识别全缺。推理的 double-buffer / comm-compute overlap 是大类优化，无分析支撑等于该维度"靠猜"。
4. **各优化类型量化收益上限**——priority list 是定性排序，但"消除 sync 省多少 ms / 融合小算子省多少 / 重叠通信省多少"无数据量化。候选应按量化上限排序（SKILL.md 确认节点 A 已要求"理论收益上限"），但脚本不提供这个数。
5. **重复计算检测**——operator_memory 检测重复**分配**，但重复**计算**（同一中间结果每步重算）无检测。推理常量的重复计算是去重维度的典型靶点。
6. **调用链层级归因**——Line A 门禁要求"任何层 >10% host time 须有候选"，但无脚本按调用链层级聚合 host/device 时间。数据支持（Call Stack + Host Total），缺层级聚合。
7. **验证闭环缺口**——diff_profiling 只 diff 算子耗时与内存峰值，不 diff 利用率/host 开销/h2d。优化后"算子快了但利用率没涨"这种伪收益检测不到。

### 与已有框架的关系

- **五种分析模式 + profiling_to_action 模式**是推理方法，能填补部分缺口（agent 手动拼），但**未系统化为脚本产出**，依赖 agent 经验、不可复现。缺口 1/2/4 本质是"把 agent 的拼装工作固化成脚本"。
- **四维度覆盖不均**：去重/复用/替换 有对应脚本信号，**掩盖几乎无**。这是框架结构性失衡，不是个别脚本问题。
- **本评估的价值**：指明"补哪些分析"比"补哪些字段"更重要——字段是原料，分析维度是产出。即便 Phase 1 字段全补齐，若不补这些分析维度，推理优化决策仍靠 agent 经验。

---

## 逐脚本评估（推理视角）

### step_trace —— 推理价值高，处理基本充分
- **推理价值**：Computing/Free 划分是推理 host-bound 判定的入口；Free = 可优化空间 = 推理延迟可压缩上限。
- **处理不足**：`Overlapped` 未读（Phase 1），导致计算-通信重叠度无法量化——多卡推理（TP/PP）场景下重叠是"掩盖"优化的核心指标，单卡推理无通信则不影响。
- **推理影响**：单卡推理充分；多卡推理缺重叠量化。

### op_statistic —— 推理价值高，处理充分
- **推理价值**：算子耗时排名直接给出推理 device 侧优化靶点；数据搬运占比（Cast/Transpose）是推理 format 暗坑信号；碎片化信号对应 eager 推理的小算子融合机会。
- **处理不足**：无（Core Type/Min-Max 属 Phase 1 字段缺失，处理逻辑本身合理）。
- **推理影响**：充分。

### kernel_details —— 推理价值最高，但处理退化最明显
- **推理价值**：compute/memory bound 判定、小算子融合、Block Dim 并行度、流水 stall——全是推理优化核心。
- **处理不足（推理影响大）**：
  1. **时序用行序近似**：`all_kernels` 按文件行顺序追加，wait-context（高等待前后 kernel）和可融合序列（连续小 kernel）都基于行序。但 CSV 行序不保证等于 Start Time 序（多流交错时尤其）。`Start Time(us)` 列就在本文件里却没读 → 时序分析是近似的，可能把不同流/错位的 kernel 当成"连续"。推理 eager 模式多流交错常见，影响 fusible 判定的正确性（跨流的 kernel 不能融合）。
  2. **无流分组**：`Stream ID` 未读，可融合序列和高等待上下文都不区分流。可融合的前提是同流串行，跨流"连续"是假象。
  3. **硬件 ratio 算术平均**：`aic_mac_sum/aic_kernels` 简单平均。推理常由少数大 kernel 主导（如一个大 MatMul compute-bound + 一堆小 Cast memory-bound），平均后 mac/mte 都"中等"，看不出 bimodal。应做**duration 加权**或按 kernel 分桶。
  4. **suspect 只筛"低利用率"**：`mac_ratio < 0.2 且 dur > 10us` 抓的是"高耗时但没在算"的 kernel（好）；但漏了"高耗时且高利用率"的真 compute-bound（这些是替换/量化/拆分的靶点，优先级 8）。推理的大 MatMul 常落这里。
- **推理影响**：fusible 序列可能误报（跨流）、compute/memory bound 判定被平均稀释、真 compute-bound 靶点不浮现。

### operator_details —— 推理价值高，缺 inclusive 维度
- **推理价值**：host dispatch 开销是 eager 推理主瓶颈；纯 host op 占比、H/D ratio、Call Stack 源码定位——推理优化直接用。
- **处理不足**：
  1. **只用 Host Self，不用 Host Total**：Self 是该 op 自身 host 开销（框架 dispatch 净值），Total 是含子调用的 inclusive。Line A 的"穿透层级量化"门禁要求"调用链任何层贡献 >10% total host time 须有候选"——这需要 Total（inclusive per-layer cost），Self 撑不起。推理 host-bound 时穿透层级定位是关键，缺 Total 使门禁喂不饱。
  2. **无时间维度**：host 开销是否集中在某段（如某 forward 阶段）看不到，只有按 op 名聚合。
- **推理影响**：host 开销的层级归因不精确，影响"改哪一层 dispatch"的决策。

### memory_record —— 推理价值高，峰值归因缺失
- **推理价值**：内存峰值决定推理最大 batch（吞吐上限）；OOM 风险；碎片化影响峰值。
- **处理不足**：
  1. **碎片化用 Reserved−Allocated，忽略 Active**（Phase 1 字段缺失的下游）：Active 是真实活集，Allocated 含 allocator cache。推理 batch 上限由峰值 Active 决定，用 Allocated 会高估可压缩空间。
  2. **无峰值归因**：给出了峰值时刻（max_res_time），但不把峰值与当时在跑的算子/阶段关联。推理想知道"峰值是哪段计算的激活/中间结果造成的"（如 attention 中间态 vs FFN 激活 vs 预处理 buffer），需要跨 operator_memory/trace_view 的时间对齐，本脚本不做。
- **推理影响**：batch 上限估算偏乐观；峰值归因需人工跨文件。

### operator_memory —— 推理价值高，用错 Duration 维度
- **推理价值**：buffer 复用候选（推理 eager 反复分配）、短命大 tensor、重复同尺寸分配——推理内存优化核心。
- **处理不足（推理影响大）**：
  1. **用 pool Duration 而非 Active Duration 判"短命"**：`short_lived_large` 用 `Duration(us) < 1ms`。但 Duration 是 tensor 在 allocator pool 中的存活时间（含被释放进 cache 后仍"存活"的时段），Active Duration 才是真正被使用的时间。一个分配后立即释放进 cache 的 tensor：Duration 长（直到 cache 被驱逐）、Active Duration 短（只用一次）。用 Duration 会把"可复用"判成"不可复用"（Duration 长看似不能复用），漏掉推理最该复用的对象。`Active Duration(us)` 列就在本文件里。
  2. **复用候选无 peak 归因**：parallelism trigger 分析了 peak 时刻的 waste，但"哪些 op 的分配落在峰值时刻"没列。
- **推理影响**：复用候选判定用错维度，可能漏掉推理最高频的复用机会。

### trace_view —— 推理价值最高，处理深但缺 counter 时间线
- **推理价值**：h2d bound 区段（推理 host 喂不动的直接证据）、dispatch latency、在线编译分类（首请求延迟）、device stall、prefetch 候选——推理优化主力。
- **处理不足**：
  1. **counter 时间线未处理**（Phase 1 字段缺失的下游）：HBM 带宽/Cache 命中率/利用率是时间序列，能区分"某段时间 memory-bound"vs"compute-bound"、定位带宽饱和时刻、区分 warmup 与稳态。当前只有静态 kernel ratio，看不出时间相变。
  2. **enqueue/dequeue 未处理**：dequeue duration 实测首 op 51ms（launch 线程阻塞），是 host dispatch 瓶颈的直接证据，未捕获。
- **推理影响**：无法做阶段分析（warmup/稳态/带宽饱和）与动态 bound 判定；host 队列阻塞不可见。h2d/compile 等已处理部分很深。

### communication —— 推理价值场景相关
- 单卡推理不触发；多卡推理（TP/PP）时 wait/transit 分解、带宽、小包信号都充分。P2P 详细时序缺失（Phase 1）影响 PP 推理。

### diff_profiling —— 验证闭环不完整（推理收益判定的关键缺口）
- **推理价值**：优化前后对比，确认收益真实来自预期改动——推理优化闭环的最后一环。
- **处理不足**：
  1. **只 diff op_statistic + memory peak（Reserved）**：不 diff step_trace（utilization/Free/Computing）、kernel_details（hw 占比/小算子/Block Dim）、operator_details（host self/纯 host 占比）、trace_view（h2d 区段/dispatch latency/compile）、communication（wait/transit）。后果：优化可能"算子变快了"但 utilization 没涨（瓶颈转移到别处），diff_profiling 看不到这种**伪收益**。
  2. **不检测瓶颈类型转移**：从 Host-Bound 优化到 Compute-Bound 是常见结果（瓶颈转移），无 before/after 瓶颈类型对比。
  3. **memory peak 用 Reserved 非 Active**：Reserved 受 pool 保留影响，优化减少实际占用后 Reserved 可能不降；用 Active 才反映真实峰值变化（与 Phase 1/2 的 Active 盲区同根）。
  4. **无归一化 / L0-L1 口径守卫**：before/after 若 step 数/样本数不同则总量不可比；若一个 L0 一个 L1 则对比无效（L1 含 profiler 注入开销）。无检测无警告。
  5. **无方差/显著性**：单次 before/after，推理延迟有噪声，delta 可能是噪声。无多次运行处理。
  6. **op 按 Type 名匹配**：替换型优化（Transpose→融合算子）能识别（eliminated/new），但同语义改名不匹配；无 shape 维度分解（total 变化是"变快"还是"shape 分布变了"不可区分）。
- **推理影响**：验证闭环残缺——"op 快了 + peak 降了"看得到，"utilization 提升 / host 开销下降 / h2d 解除 / 瓶颈转移 / 收益是否显著"看不到。这使确认节点 B 的"profiling 确认收益"门禁实际依赖人工跨脚本拼，不可复现。

---

## 跨脚本处理缺口（按推理影响排序）

### 高影响（直接削弱推理优化决策）
1. **kernel_details 时序用行序 + 无流分组**：fusible 序列可能跨流误报、wait-context 错位。修复：读 `Start Time` + `Stream ID`，按流分组、按时间排序。`Start Time`/`Stream ID` 列已在文件内，纯处理改造。
2. **operator_memory 用 Duration 而非 Active Duration**：复用候选判定用错维度。修复：短命判定改用 `Active Duration`。字段已在文件内。
3. **kernel_details 硬件 ratio 算术平均**：掩盖 bimodal。修复：duration 加权 + 按耗时分桶（如 top-10 重 kernel 单独列其 mac/mte）。
4. **operator_details 缺 Host Total**：Line A 穿透层级门禁喂不饱。修复：overview 增加 Host Total 聚合，支持"哪层 host 开销 >10%"。

### 中影响
5. **memory_record/operator_memory 忽略 Active**：batch 上限估算偏乐观、碎片化不准。修复：读 Active 列，峰值用 Active。
6. **无内存峰值归因**：峰值时刻在跑什么需人工跨文件。修复：memory_record 峰值时刻 × operator_memory 分配时间窗，定位峰值贡献者。
7. **kernel_details suspect 漏"高耗时高利用率"**：真 compute-bound 靶点不浮现。修复：增加"高耗时且 mac_ratio 高"的 kernel 列表（替换/量化靶点）。

### 推理特有缺失（数据支持但无人做）
8. **动态 shape 跨 request 重编译归因**：非 LLM 推理（视觉/科学计算/变长输入）shape 随 request 变化触发重编译，是 host 开销大户。trace_view compile 分类（A 预热 / B 每步）能看出"有重编译"，但不归因到"shape 变化驱动"。数据支持（compile 事件 ts × op 输入 shape 变化），缺归因分析。
9. **数据预处理 / H2D pipeline 开销**：非 LLM 推理（图像解码/增强、蛋白质特征提取、音频前端）的预处理常是主瓶颈。trace_view prefetch 候选只覆盖 `aten::to`/`copy_`/`empty`，不把"预处理阶段"作为独立开销类别量化。本样本 AlphaFold 的重 Python 预处理即属此类。
10. **request-invariant 预计算机会**：推理服务多 request 时，与 request 无关的计算（位置编码、attention bias、权重预处理、常量 tensor）可跨 request 缓存。无脚本识别"哪些计算输出不随 request 变化"（复用维度），只检测重复分配不检测重复计算。
11. **batch padding 浪费**：变长输入 pad 到同形状 batch 会浪费 compute。无 padding ratio 分析。
12. **控制流同步归因**：shape/数值驱动的分支强制 `.item()` 同步（本样本 `_local_scalar_dense` 占 host 63% 即此）。operator_details 能列出该 op，但不归类为"控制流同步"并归因到"哪个分支判断触发的"。属成因分解缺口（见 #1）的具体形态。
9. **无 stream overlap 量化**：多流推理的"掩盖"收益无量化。数据支持：trace_view 的多流 timeline 已有，但只输出 busy%，不算流间重叠率/可掩盖空泡。
10. **无 warmup vs steady 区分**：首请求编译/冷启动延迟 vs 稳态延迟未分。数据支持：step_trace 多 step + trace_view compile 事件时间分布。当前 compile 分类只分 A（预热）/B（每步），不量化首请求延迟占比。

---

## 建议（按推理收益排序）

| # | 改造 | 类型 | 推理收益 | 改造量 |
|---|------|------|---------|--------|
| 1 | kernel_details 读 Start Time + Stream ID，按流分组、按时间排序做 fusible/wait-context | 纯处理 | 修正 fusible 误报、流级并行可见 | 中 |
| 2 | trace_view 处理 counter 时间线（Phase 1 先补捕获） | 捕获+处理 | 阶段分析（warmup/稳态）、动态 bound 判定、带宽饱和定位（实证确认 per-die HBM 带宽/LLC hit rate 时间线存在） | 大 |
| 3 | kernel_details 硬件 ratio 改 duration 加权 + top-N 重 kernel 单列 | 纯处理 | compute/memory bound 不被稀释 | 小 |
| 4 | step_trace 单步推理也产 suspect signal（去掉 len>1 门控，加推理专用信号） | 纯处理 | 单步推理不再零信号（实证发现的真实缺口） | 小 |
| 5 | memory_record 按 Component 分段（WORKSPACE vs tensor） | 纯处理 | 区分可控 workspace 与 tensor 内存 | 小 |
| 6 | operator_details 增加 Host Total + 按 Call Stack 层级聚合（穿透层级归因） | 纯处理 | 通用推理热路径分层，喂饱 Line A 门禁；本样本叶 op 主导是特例，一般情况 wrapper/dispatch 层瓶颈靠此浮现 | 中 |
| 7 | kernel_details suspect 增"高耗时高利用率"列表 | 纯处理 | 真 compute-bound 靶点浮现 | 小 |
| 8 | 内存峰值归因（memory_record × operator_memory） | 跨脚本 | 峰值是哪段代码造成 | 中 |
| 9 | trace_view stream overlap 量化 | 纯处理 | "掩盖"收益可量化 | 中 |
| ~~原#2~~ | ~~operator_memory 改用 Active Duration~~ | 纯处理 | **经通用视角复核后恢复为 #11**：单次推理样本下差异微小是特例，通用服务场景（caching allocator）下漏判复用候选 | 小 |
| 10 | 动态 shape 重编译归因 + request-invariant 预计算识别 | 新分析 | 非 LLM 推理（视觉/科学计算）的 host 大户：重编译归因、跨 request 可缓存计算识别 | 中 |
| 11 | operator_memory 短命判定改用 Active Duration | 纯处理 | 通用服务场景（caching allocator）下修正复用候选漏判；单次推理特例下无差异 | 小 |

> 修订逻辑：经通用视角复核，原 #2（operator_memory Active Duration）与原 #4（operator_details Host Total）**均维持影响**——本样本"单次推理 + 叶 op 主导 host"是特例，使 Self-by-name 与 pool Duration 恰好够用；通用推理（框架包裹分层 + caching allocator 服务场景）下两者都是关键缺失。新增 step_trace 单步零信号（#4）与 memory_record WORKSPACE 分段（#5）两个实证缺口；counter 时间线（#2）因实证确认而强化。仍以"纯处理、低成本"项先行；2/10 依赖捕获或外部输入，成本高。
>
> **方法论警示**：实证可验证/修正具体判断，但本样本有两个特例特性——(a) 单次推理（无 caching，Active≈Duration）、(b) 叶 op 主导 host（Self≈Total，Self-by-name 够用）——曾导致 Active Duration 与 Host Total 两处被错误降级。判断通用需求时须剥离样本特例，从"一般推理是否分层 / 是否服务多 request"出发。

---

## 模型家族特异项（不混入通用判断）

通用分析轴（上文）适用于所有推理负载。不同家族有特异关切，单列如下——这些不是通用缺口，而是"通用轴在某家族的具体化"，是否补取决于目标家族：

| 家族 | 特异关切 | 通用轴映射 | 当前覆盖 |
|------|---------|-----------|---------|
| LLM 自回归 | prefill/decode 两相、KV cache 内存、首 token 延迟、per-token decode host-bound | 阶段分析、内存峰值、host 开销 | 两相分相无（需用户标记）；KV cache 无专项；其余靠通用轴 |
| 视觉 CNN | conv 优化、NCHW/ND 格式、图像预处理 CPU 开销 | format/layout、预处理/H2D | 格式数据缺（Phase 1）；预处理未独立量化 |
| 扩散模型 | 迭代去噪循环（同图 N 步）、UNet、timestep 调度 | 小算子碎片、重复结构、warmup | 重复步计算无跨步复用识别 |
| 科学计算（本样本 AlphaFold） | 长序列 attention、重 Python 预处理、动态 shape 控制流同步 | host 开销、动态 shape、控制流同步 | 同步已浮现（63% host）；预处理/动态 shape 归因缺 |
| 语音 | 流式实时、conv+RNN/transformer 混合 | 端到端延迟、流式分段 | 流式分段无 |
| 推荐/嵌入 | embedding 查表、稀疏算子、小 batch 延迟 | host 开销、内存带宽 | 稀疏算子无专项 |

> 本审计样本为 AlphaFold（科学计算/encoder 式单次推理），故"prefill/decode""首 token""KV cache"等 LLM 项对样本**不适用**——样本瓶颈（控制流同步 63% host + AICPU 广播 31.7% device）恰好落在通用轴上，证明通用轴标尺的正确性。家族特异项的补全优先级取决于实际目标负载。

---

## 方法论说明

- 本阶段**只评处理充分性**，不重复 Phase 1 的字段捕获判定。"字段在文件内但没处理"归本阶段（如 kernel_details 的 Start Time/Stream ID、operator_memory 的 Active Duration、operator_details 的 Host Total）。
- 评估标尺是**架构无关的推理优化可用性**：能否支撑"减少延迟 / 提高 batch / 消除 host 开销 / 融合小算子 / 重叠并行"的决策。LLM 等家族特异项不混入通用标尺。
- 训练特有的关切（梯度对齐、收敛、optimizer state）不在本阶段标尺内，相关脚本的训练能力不评。
- "推理特有缺失"指数据本支持但无脚本产出的分析，是本阶段相对 Phase 1 的增量发现。
