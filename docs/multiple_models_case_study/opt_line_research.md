# 面向任意模型推理的低层性能瓶颈定位与优化上界探索

## 摘要

本报告研究的问题是：给定任意机器学习模型的推理程序、目标硬件和运行时环境，如何不依赖 attention、卷积、MLP 等预定义模块，而在张量程序、数据移动和硬件映射层面定位性能瓶颈，并估计各类优化的端到端收益上界。

现有工作已经分别覆盖了这一问题的多个组成部分：张量编译器能够将模型下沉为循环、索引映射、归约、布局及 schedule；加速器设计空间探索工具能够根据循环嵌套、存储层级和数据流估计时延、能耗与带宽需求；profiler 能够采集真实执行中的算子、kernel、CPU/GPU、通信和内存事件；因果 profiling 与关键路径分析能够回答“优化某段是否真正改善端到端性能”。

但公开工作尚未形成一个统一闭环：**以任意 tensor-IR 为输入，将程序语义、真实 trace、硬件资源模型和反事实干预结合起来，自动输出根因、可行动的优化变量，以及语义 Oracle / 硬件可实现 / 软件栈可达三档性能上界。**

---

## 1. 问题重述：从“模块变慢”到“程序—硬件映射失配”

传统推理优化常以模型模块作为分析单位，例如“卷积慢”“attention 带宽受限”“embedding 是热点”。这种做法有三个根本局限：

1. 新模型、融合算子和自定义算子无法被稳定归类；
2. 同一模块在不同 shape、布局、batch、硬件和运行时下可能有完全不同瓶颈；
3. 模块耗时占比不等于优化该模块能带来的端到端收益。

更通用的问题定义为：

> 给定模型程序 $P$、输入分布 $X$、硬件 $H$、编译器/运行时 $R$ 和优化目标 $O$，识别使端到端推理性能受限的低层程序结构与资源约束，并估计对可选优化动作 $a$ 的端到端反事实收益 $\Delta T(a)$。

其中“低层程序结构”至少包括：

- 迭代空间与循环边界；
- 张量索引与地址映射；
- reduction、scan、scatter/gather 等依赖；
- 精度、数据布局与稀疏格式；
- tile、loop order、并行映射、向量化和线程绑定；
- 张量在寄存器、片上 SRAM/shared memory、cache、HBM/DRAM 和网络中的驻留与搬运；
- kernel launch、同步、内存分配、通信和 host-device 协作。

因此，最终定位结论应该类似：

> “该 region 的 reduction 轴过短，无法填满执行资源；而 fusion 又抬高寄存器压力，导致 occupancy 降低。其对端到端关键路径的可优化空间上限为 12%。”

而不是：“这是 attention 瓶颈。”

---

## 2. 统一抽象：四层 IR 与执行图

一个可行的研究框架需要至少四层表示。

| 层次 | 表示内容 | 作用 |
|---|---|---|
| 语义图层 | 张量依赖、控制流、shape 约束 | 保证变换语义正确，构造全图依赖 |
| 张量程序层 | 循环、索引映射、归约、layout、dtype | 消除对预定义模块的依赖 |
| 映射层 | tile、reorder、parallel、vectorize、fusion、memory placement | 表示如何在硬件上执行 |
| 事件层 | kernel、copy、同步、CPU 调度、通信、队列事件 | 用真实 trace 校准并构造端到端关键路径 |

### 2.1 张量程序原子

对任意可编译 region，可使用如下抽象：

$$
\mathcal{R}=(\mathcal{I},\ \mathcal{A},\ \mathcal{D},\ \mathcal{L},\ \mathcal{S})
$$

- $\mathcal{I}$：迭代域，例如每个循环轴的范围及动态约束；
- $\mathcal{A}$：张量索引映射，可为 affine 或非 affine；
- $\mathcal{D}$：数据依赖，如逐元素、归约、scan、随机访问和控制依赖；
- $\mathcal{L}$：数据类型、布局、稀疏表达；
- $\mathcal{S}$：当前 schedule 与硬件映射。

这一表示可将 matmul、卷积、attention、归一化、卷积后处理、推荐中的 embedding lookup、GNN message passing 等统一到“迭代—索引—依赖—数据移动”的框架中。

[TensorIR](https://arxiv.org/abs/2207.04296) 将张量计算作为一等对象，扩展传统 loop-nest 表示以承载 tensorization 和自动映射，是这一方向的重要代表。

### 2.2 数据移动图

仅有 FLOPs 不足以解释推理性能。应为每个 region 建立多级存储访问量：

$$
Q_r=(Q_{reg},Q_{L1},Q_{L2},Q_{HBM},Q_{PCIe},Q_{NIC})
$$

并记录：

- 每个张量在各存储层的 working set；
- 重用距离与是否能被 tile 捕获；
- 连续/合并访问、bank conflict、cache miss；
- 布局变换和量化/反量化带来的额外读写；
- 跨 region 的中间张量是否可保留在片上存储。

[Timeloop/Accelergy](https://timeloop.csail.mit.edu/) 的核心即是对张量工作集随时间在存储层级中的流动进行解析建模，并据此估算性能、能耗和面积。

---

## 3. 性能上界：不应只有一个 Roofline 数字

### 3.1 Region 级资源下界

对 region $r$，给出其计算、数据移动和同步等工作量后，可构造理想执行时间下界：

$$
t_r^{lb}=\max \left(
\frac{F_r}{P^{ub}_{compute}},
\max_l\frac{Q_{r,l}}{B^{ub}_l},
\frac{S_r}{R^{ub}_{sync}}
\right)
$$

其中 $F_r$ 是必要计算量，$Q_{r,l}$ 是对存储层级 $l$ 的必要数据移动量，$P^{ub}_{compute}$ 与 $B^{ub}_l$ 是相应约束下的可达计算与带宽上限，$S_r$ 是不可消除的同步或依赖次数。

经典 Roofline 是其中计算与主存带宽的二维投影；它有效，但不足以表达 cache 容量、访存延迟、同步、通信及并行不足造成的 ceiling。NVIDIA 的 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 已将 Roofline 和峰值性能边界纳入 kernel 分析。

### 3.2 图级与端到端上界

令 $G_E$ 为由 kernel、copy、通信与同步事件构成的执行依赖图。端到端理想时延受以下两类约束：

$$
T^{lb}_{e2e}\geq
\max\left(
CP(G_E,\{t_r^{lb}\}),
\max_h \frac{\sum_r W_{r,h}}{C_h}
\right)
$$

其中 $CP$ 是关键路径长度，后一项是硬件资源总容量约束。

建议输出三档边界：

| 上界层次 | 假设 | 解释的差距 |
|---|---|---|
| 语义 Oracle 上界 | 最优复用、映射、并行与 overlap；不违反程序依赖 | 算法与硬件物理极限 |
| 硬件可实现上界 | 加入寄存器、SRAM、线程、指令、cache、互联等真实约束 | 编译器/代码生成仍可能弥合的差距 |
| 软件栈可达上界 | 加入现有 runtime、kernel 库、编译时间和服务约束 | 短期工程优化空间 |

以时延表示时，这些是下界；以吞吐表示时，才对应上界。报告中应避免混用“性能上界”和“时延上界”。

### 3.3 Amdahl 与关键路径约束

若某局部区域占总时延的比例为 $p$，即使将其消除，端到端加速上限也受 Amdahl 定律约束：

$$
Speedup_{max}=\frac{1}{1-p}
$$

但对于并发图，这还不够。非关键路径 region 即使耗时很长，也可能对延迟几乎无影响；相反，一个耗时短但位于串行链上的同步点可能具有极高价值。因此优化优先级应基于反事实关键路径增益，而非 self-time 排序。

---

## 4. 核心方法：反事实瓶颈定位

传统 profiler 回答的是：“时间花在哪里？”

研究问题真正需要回答的是：“若消除某项约束，端到端会改善多少？”

可将优化动作定义为 $a$，包括：

- 修改 tile、loop order、并行轴或 vector width；
- 调整 layout；
- 融合/拆分 region；
- 改变某张量的 memory placement；
- 允许异步 copy 或通信计算 overlap；
- 改变 batch、实例数、图编译与执行策略；
- 替换 precision 或稀疏格式。

其价值定义为：

$$
\Delta T(a)=T_{e2e}^{current}-T_{e2e}^{counterfactual}(a)
$$

[Coz](https://arxiv.org/abs/1608.03676) 提出的 causal profiling 提供了重要思想基础：通过“虚拟加速”测量一段代码变快时对整体性能的影响，而不是仅报告采样时间占比。

在模型推理场景中，反事实不应限于“加速某行代码”，而应在低层映射变量上操作：

$$
\mathrm{do}(Q_{HBM}^{r}\downarrow,\ occupancy_r\uparrow,\ latency_{copy}\downarrow,\ \text{fusion}(r_i,r_j)=1)
$$

然后以依赖图重放或离散事件模拟重新计算关键路径。由此可自然得到：最值得优化的循环、张量边或调度决策；单一优化的收益上限；多优化之间的重叠、替代与互补关系；以及当前实现与三档上界之间的 gap 分解。

---

## 5. 相关工作梳理

### 5.1 张量编译与 schedule 搜索

**TVM / Ansor / MetaSchedule**

[Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) 面向任意 tensor program，通过层次化搜索空间、进化搜索和 learned cost model 生成高性能实现，并可同时优化模型中的多个子图。

TVM [MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) 将任务提取、候选 schedule 生成、性能预测、真实测量与调度预算分配构造成标准调优闭环。

**贡献：**

- 不依赖固定模型模块，而以 tensor computation / schedule 作为优化对象；
- 提供了可操作的优化空间；
- 真实硬件测量能校正纯解析模型误差。

**局限：**

- cost model 的目标是预测候选程序快慢，通常不解释资源根因；
- 多以单 kernel 或子图 tuning 为主；
- 不直接估计端到端关键路径上的反事实收益和整体上界。

[TLP](https://arxiv.org/abs/2211.03578) 直接从 schedule primitives 学习性能模型，而不是高度依赖人工设计的算子特征，增强了跨硬件调优效率。这说明“以调度序列作为分析对象”已经可行，但其输出仍主要是候选排序，不是可解释的性能归因。

### 5.2 图级变换与跨算子优化

**TASO**

[TASO](https://www.cs.cmu.edu/~zhihaoj2/papers/sosp19.pdf) 自动生成并验证计算图等价替换，再以 cost-based search 找到更优图，而非依赖人工维护的大量模块级 rewrite rule。

**贡献：**

- 处理图级结构而非固定模块；
- 将 graph rewrite 变为可搜索、可验证的优化空间；
- 对 fusion、算子替换、冗余消除等全图优化有价值。

**局限：**

- cost 仍主要基于候选图的性能估计；
- 没有将“为什么该 rewrite 会改善关键路径”归因到数据移动或硬件资源；
- 对动态 shape、真实 runtime 干扰和复杂执行重叠处理有限。

[BOLT](https://proceedings.mlsys.org/paper_files/paper/2022/hash/1f8053a67ec8e0b57455713cefdd8218-Abstract.html) 尝试结合硬件原生模板库与 auto-tuning，并指出传统 auto-tuner 对硬件细节较为不透明，同时支持 graph、operator 和 model 多层优化。

### 5.3 数据流、存储层级与加速器映射

**MAESTRO**

[MAESTRO](https://research.nvidia.com/publication/2019-10_understanding-reuse-performance-and-hardware-cost-dnn-dataflows-data-centric) 使用数据中心化表示描述时空复用和资源占用，以解析 cost model 估算数据流的时延和能耗。

**Timeloop / Accelergy / Sparseloop**

该体系将 workload、硬件架构、mapping 和能耗模型显式分离，可探索 loop tiling、loop permutation、spatial mapping、buffer bypass、稀疏表示与数据流。其输出包含不同存储层访问次数、带宽、周期、吞吐与能耗。参见 [Timeloop/Accelergy](https://timeloop.csail.mit.edu/) 与 [Sparseloop](https://sparseloop.mit.edu/documents/2022-micro-sparseloop.pdf)。

**ZigZag**

[ZigZag](https://arxiv.org/abs/2007.11360) 用 memory-centric nested-for-loop 作为算法、存储层级和映射之间的统一表示，特别强调不均匀 mapping 带来的数据移动优化。

**贡献：**

- 最接近低层“循环—数据空间—硬件层级”分析；
- 可直接揭示复用不足、带宽不足、buffer 容量不足、空间并行不足等根因；
- 可作为 Oracle 与硬件可实现上界建模的基础。

**局限：**

- 主要针对空间/专用加速器及较规则 tensor algebra；
- 对 GPU kernel 库内部实现、复杂 cache 行为、CPU runtime、动态控制流和跨 kernel 并发建模不足；
- 通常优化单层或逐层 mapping，而非从真实端到端 trace 出发。

### 5.4 动态 shape 与动态执行结构

[DISC](https://arxiv.org/abs/2103.05288) 为动态 shape 设计 IR，并将一部分 shape-dependent runtime flow 前置到编译期，扩大 host-device 协同优化空间。

[DietCode](https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html) 构造 shape-generic search space 和 cost model，使多个动态 shape 共享调优过程，并报告端到端动态模型的自动调度结果。

[Event Tensor](https://arxiv.org/abs/2604.13327) 将 tile 任务间依赖表示为事件，支持 shape 和数据相关动态性，并基于静态/动态调度变换生成高性能 persistent kernel。它主要在 LLM 负载验证，但其抽象具有更一般的编译意义。

这些工作证明非静态模型图可以进入编译器抽象；但尚未把动态性转化为可解释的性能上界和根因输出。

### 5.5 性能观测与关键路径

[ONNX Runtime](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) 可以产生包含线程、算子延迟和 GPU kernel 信息的 trace。

TensorRT 的推荐流程是“测量—分析—优化”的闭环，并支持 layer、kernel、CPU/GPU timeline 和硬件指标分析。参见 [TensorRT Benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/11.1.0/performance/benchmarking.html)。

**贡献：**

- 提供真实行为，而非只依赖静态模型；
- 能发现 launch gap、同步、copy、CPU 发射、通信与 kernel 的关系；
- 可为分析模型提供校准数据。

**局限：**

- 工具输出以 trace 与指标为主，根因解释和上界估计仍主要依赖专家；
- 通常无法自动回答“应优化什么、端到端最多改善多少”。

[CRISP](https://www.usenix.org/conference/atc22/presentation/zhang-zhizhou) 用关键路径分析处理大规模异步服务 trace，强调真正影响端到端延迟的路径与局部耗时并不等价。

[dPRO](https://proceedings.mlsys.org/paper_files/paper/2022/file/b422680f3db0986ddd7f8f126baaf0fa-Paper.pdf) 面向分布式 DNN 训练，基于 profiling 和关键路径识别优化机会并触发图变换。

二者提供了端到端关键路径的思路，但并未深入到 tensor iteration、数据复用和低层映射变量。

---

## 6. 业界现状

业界工具已覆盖“测量”和“局部搜索”，但尚未公开提供统一因果上界系统。

| 工具/体系 | 强项 | 不足 |
|---|---|---|
| TensorRT + Nsight | GPU kernel、图编译、精度、timeline 分析 | 对专家依赖强；无自动反事实上界 |
| ONNX Runtime profiler | 跨 execution provider 的算子/线程 trace | 更偏观测，缺少映射级解释 |
| TVM MetaSchedule | 低层 schedule 搜索和测量闭环 | 主要输出更优实现，不输出根因图 |
| Triton Model Analyzer | batch、并发、实例数、显存和 SLO 配置搜索 | 服务配置级，不深入张量程序本体 |
| Timeloop/Accelergy | architecture/dataflow/memory 的 Oracle 型分析 | 与通用 GPU runtime 和全图 trace 脱节 |

工业界的常见流程仍是：profiler 找热点 → 人工判断资源约束 → 修改图/代码/配置 → benchmark 验证。研究机会在于将这条人工链条自动化，并显式报告假设与上界。

---

## 7. 建议的系统设计

建议构建一个“跨层性能归因与上界估计器”，包括五个组件。

### 7.1 前端：IR 与 trace 采集

输入：

- ONNX、Torch FX、StableHLO/MLIR、TVM IR 等模型表示；
- 编译后的 TensorIR/PTX/LLVM IR 或可获得的 schedule；
- runtime trace：kernel、copy、同步、CPU 线程、通信；
- 硬件 profile：吞吐、带宽、cache、occupancy、指令利用率；
- 输入 shape 与数据分布。

对于无法下沉的外部库 kernel，先将其作为 opaque region，用 profile 建立替代 cost model；若后续获得源码或可重写实现，再下沉至张量程序层。

### 7.2 中间层：资源增强的执行依赖图

构建资源增强图：

$$
G=(V,E_{data},E_{control},E_{resource})
$$

- $V$：region、kernel、copy、通信和同步事件；
- $E_{data}$：张量生产—消费依赖；
- $E_{control}$：shape guard、条件分支与 runtime 调度依赖；
- $E_{resource}$：同一 stream、SM、memory channel、NIC 等资源竞争关系。

每个节点绑定：

$$
\phi(v)=(F,Q_{reg},Q_{cache},Q_{HBM},Q_{net},\text{occupancy},\text{launch},\text{shape},\text{mapping})
$$

### 7.3 成本模型：解析模型与学习模型结合

建议采用“解析约束 + 学习残差”的混合模型：

$$
\hat{t}_r=t_{analytic}(\phi_r,H)+\epsilon_\theta(\phi_r,H,\text{trace})
$$

解析部分保证可解释性和跨硬件外推能力；学习残差吸收 cache、warp 调度、库实现等难以精确建模的微架构效应。

输出不只是一项时延预测，而应分解为：

$$
\hat{t}_r=t_{compute}+t_{memory}+t_{sync}+t_{launch}+t_{contention}-t_{overlap}
$$

这些项可非严格可加，但应保持同一归因口径，并通过实验验证其反事实有效性。

### 7.4 反事实引擎

对每个候选优化动作 $a$，执行：

1. 在 mapping/资源图中施加干预；
2. 更新该动作影响的计算量、数据移动量、并行度、依赖或资源竞争；
3. 用事件重放/离散事件模拟重算关键路径；
4. 输出收益、置信区间和代价。

候选动作的优先级可定义为：

$$
Priority(a)=\frac{\mathbb{E}[\Delta T(a)]}{Cost(a)}\times Confidence(a)
$$

其中 $Cost(a)$ 可是工程改动成本、重新编译成本或精度风险。

### 7.5 上界搜索

分别搜索：

- **Oracle mapping space**：允许所有语义正确的 mapping 与最优复用；
- **implementable mapping space**：受硬件和 codegen 约束；
- **available mapping space**：受当前 runtime 与工程约束。

由于全局搜索通常组合爆炸，不应将“最优”表述为无条件数学证明。应报告已搜索空间、未覆盖空间、解析 bound、搜索得到的 best-known point、预测误差与置信区间。

---

## 8. 评估设计

### 8.1 工作负载

为证明不依赖固有模块，应覆盖至少四类：

| 类别 | 代表特性 |
|---|---|
| 规则稠密张量图 | CNN、ViT、BERT；验证 loop/dataflow 基础能力 |
| 动态 shape | 检测、分割、语音、变长 Transformer；验证 shape 适应性 |
| 非规则访问 | 推荐、GNN、稀疏模型；验证 gather/scatter 与 latency-bound 判断 |
| 生成/状态模型 | 扩散、LLM、流式 ASR；验证控制流、缓存与长关键路径 |

### 8.2 评估指标

- 时延：p50/p95/p99、关键路径时延；
- 吞吐与资源利用；
- 能耗、峰值显存、数据移动量；
- 上界误差：预测下界与最优已知实现之间的距离；
- 归因准确性：预测最优优化方向与真实 ablation 的一致性；
- 排序质量：Top-k 推荐中真正有效优化的命中率；
- 搜索效率：达到相同性能所需编译/测量次数；
- 可解释性：是否能定位到具体 loop、tensor edge 或 mapping 决策。

### 8.3 最重要的验证：反事实校验

对于诊断器认为最有价值的动作 $a$，实际实现或尽量接近地实施，并比较：

$$
Error_{cf}(a)=\left|\Delta T_{pred}(a)-\Delta T_{measured}(a)\right|
$$

这比单纯报告 latency predictor 的 MAPE 更重要；研究目标是“指导优化”，而不是仅预测当前性能。

---

## 9. 研究创新点与风险

### 潜在创新

1. 从 module-centric profiling 转向 tensor-IR / data-movement-centric diagnosis；
2. 统一静态程序结构、硬件性能模型与真实 execution trace；
3. 将因果 profiling 的反事实思想下沉到 loop、layout、memory placement 和 schedule 层；
4. 明确区分 Oracle、硬件可实现和软件栈可达三档上界；
5. 从单 kernel 最优转向全图关键路径与组合优化收益。

### 主要风险

- 低层模型足够泛化时，解释性与预测精度会存在张力；
- 通用 GPU 的 cache 与 warp 调度难以准确解析建模；
- 动态控制流使静态依赖图不完整，需要 profile-conditioned graph；
- 优化动作之间存在强交互，不能将局部收益简单相加；
- 若缺乏编译器或 kernel 修改权限，只能对 opaque region 做黑箱归因。

### 风险缓解

- 采用“解析模型 + 残差学习 + trace 校准”的混合方案；
- 先覆盖可下沉的 tensor region，再逐步处理 opaque library kernel；
- 对动态模型采用 profile cluster / shape cluster 建立多版本模型；
- 输出置信区间和可验证假设，避免把启发式搜索结果误称为严格全局上界。

---

## 10. 最终判断

该方向已有充分技术基础，但尚未被完整解决。

- **Timeloop / MAESTRO / ZigZag** 已解决低层数据流、存储层级和映射空间如何建模；
- **TVM / TensorIR / Ansor / MetaSchedule** 已解决任意 tensor program 如何表达、变换和搜索；
- **TASO / BOLT** 已解决如何扩展至图级优化；
- **Coz / CRISP / dPRO** 已解决为什么热点不等于优化机会，以及如何用关键路径或反事实分析端到端影响；
- **Nsight / TensorRT / ONNX Runtime** 已提供真实运行时证据。

真正的研究空白位于这些体系的交叉处：

> 面向任意模型推理图，构建一个由低层 IR、数据移动语义、硬件资源模型和真实 trace 共同驱动的反事实性能诊断器；它能够将性能瓶颈定位到具体的循环轴、张量数据边和映射决策，并给出端到端优化收益与多档上界。

这比“推理引擎优化”更具一般性，也更容易形成同时覆盖编译器、体系结构和系统优化的研究贡献。