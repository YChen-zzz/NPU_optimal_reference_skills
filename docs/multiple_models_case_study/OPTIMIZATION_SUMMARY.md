# NPU 模型推理优化总结报告

## 1. 优化总览

对 7 个模型在 Ascend 910 NPU 上进行了基于 profiling 驱动的推理优化，共执行 22 轮优化迭代，产出 31 个 evidence_db 案例文件。

### 1.1 最终结果

| 模型 | 架构 | 参数量 | 轮次 | Wall-clock 提升 | L0 利用率 | 终局原因 |
|------|------|--------|------|-----------------|-----------|----------|
| timm resnet50 | CNN (ResNet50) | 23.5M | 2 | -28.3% (5.10→3.66ms) | 89% | TransData 是 CANN 优化 |
| dinov2-base | ViT-B/16 | 86.6M | 4 | -43.1% (5.07→2.89ms) | 97% | NPU 非结合律浮点运算 |
| wav2vec2-base | Conformer | 94.4M | 3 | -43.4% (7.55→3.15ms) | 92% | TransData 是 CANN 优化 |
| CLIP | ViT+Text | 149.6M | 3 | -73.2% (32.6→8.7ms) | 58% | Python 框架开销下限 |
| ESM2 t33 650M | Transformer (Encoder) | 652M | 5 | -56.6% (19.7→8.5ms) | 57% | 图编译被 CANN 阻塞，接近 L0 下界 |
| Qwen3-1.7B | Transformer (Decoder, GQA) | 1.72B | 3 | -51.6% (24.8→12.0ms) | 80% | Python 框架开销下限 |
| Stable Diffusion v1-4 | UNet+VAE+CLIP | ~1.0B | 5 | -22.9% (293→226ms) | 94% | 图编译不可用，量化未授权 |

### 1.2 优化手段分类

#### 1.2.1 去重（Eliminate Redundancy）

| 手段 | 适用模型 | 效果 | 风险 |
|------|----------|------|------|
| Flat forward（绕过 Module.__call__）| 全部 7 个 | -15%~73% wall-clock | 零（数学等价） |
| QKV 合并（3 个同源 Linear → 1 GEMM）| dinov2, wav2vec2, CLIP, ESM2, Qwen3 | -48~99 MatMul kernel | 零（数学等价） |
| K+V 合并（GQA 模型 K/V 同源）| Qwen3 | -28 MatMul kernel | 零 |
| gate+up 合并（SwiGLU 同源 Linear）| Qwen3 | -28 MatMul kernel | 零 |
| 权重折叠（scale/sqrt2 吸收进权重）| ESM2, Qwen3 | -33~66 kernel | 零（数学等价） |
| FRACTAL_Z 权重预转换（消除冗余权重格式转换）| SD | -625 TransData kernel (-27ms) | 零（内存布局重排，数值不变） |
| CACHE_MODE=force（跳过逐次缓存验证）| SD | -36ms host overhead | 零（仅影响编译策略） |
| 预计算 causal mask | Qwen3 | -56 kernel (Triu+OnesLike) | 零（预计算不变量） |
| 消除 Identity/Dropout no-op | dinov2, wav2vec2, CLIP | -36 Module 调用 | 零 |

#### 1.2.2 替换（Equivalent Substitution）

| 手段 | 适用模型 | 效果 | 风险/限制 |
|------|----------|------|-----------|
| npu_rms_norm（7→1 kernel/实例）| Qwen3 | -679 kernel | 零（精度验证通过） |
| npu_fusion_attention BSND | dinov2, wav2vec2, CLIP, ESM2, SD | -48~768 Transpose | 中（GQA 需 repeat，BSND 内核慢22%但转置消除更优） |
| Safety checker NPU 预处理（PIL 往返→NPU tensor 操作）| SD | -35ms host overhead | 零（safety_checker 不修改图像） |
| allow_internal_format toggle（转换时启用，推理时关闭）| SD | -100ms Online-Compile | 零（FRACTAL_Z 权重持久） |
| npu_rotary_mul（融合 rotary 位置编码）| ESM2 | -462 kernel (Cast+Mul+Slice+Neg+Concat) | 零（bit-identical，NPU 内部处理 float32 精度） |
| npu_fast_gelu | CLIP | -48 kernel (Sigmoid+Mul→1) | 零（quick_gelu 等价） |
| F.gelu 替代手动 gelu | ESM2 | -99 kernel (Erf+Mul+Add→1) | 零（NPU F.gelu 精度更好：diff=0.008 vs 手动 0.012） |
| npu_add_layer_norm | wav2vec2, CLIP | -48 kernel (Add+LN→1) | 中（pre-norm 不适用，post-norm 通过） |

#### 1.2.3 复用（Reuse and Precompute）

| 手段 | 适用模型 | 效果 | 风险 |
|------|----------|------|------|
| 预移动 input 到 NPU | 全部 | -0.6~5ms H2D | 零 |
| 预计算 rotary embedding | ESM2, Qwen3 | 消除每层 rotary 重算 | 零（序列长度固定时不变） |
| 预计算位置编码插值 | dinov2 | -155us UpsampleBicubic2d | 零（输入尺寸固定时不变） |
| Flat forward feature_extractor | wav2vec2 | -0.4ms Free | 零 |

---

## 2. 精度验证方法论评估

### 2.1 当前做法

- **指标**：cosine_similarity + max_abs_diff
- **阈值**：cosine ≥ 0.9999, max_abs_diff < 1e-4（float32）/ < 1e-3（float16）/ < 0.1（33 层 float16）
- **样本数**：3~10 个 profiling subset 样本
- **对比对象**：原始模型（Module.__call__ 路径）输出 vs 优化后 flat forward 输出
- **确定性条件**：model.eval() + torch.no_grad() + 固定输入 + 固定随机种子

### 2.2 问题与不足

1. **阈值不统一**：不同模型用了不同阈值（1e-4 / 1e-3 / 0.1 / 0.15），缺乏系统性论证
   - resnet50/dinov2/wav2vec2/CLIP：1e-4（合理，float32 或少量层 float16）
   - ESM2/Qwen3：0.1~0.15（33/28 层 float16 累积，但阈值选择缺乏理论支撑）
   - SD：图像级 cosine（完全不同的精度空间）

2. **样本数不足**：大部分模型只测了 3 个样本
   - ESM2 Round 2 发现 sample 1 精度超限而 sample 0/2 通过——说明 3 个样本不够
   - 不同序列长度/氨基酸组成/文本长度可能触发不同精度退化模式

3. **缺乏逐层精度追踪**：只在最终输出对比精度
   - ESM2 Round 3 发现精度退化是逐层累积的——如果逐层追踪可以更早定位问题
   - Qwen3 的 rotary int64 bug 如果逐层对比可以更快发现

4. **缺乏 baseline 自一致性验证**：没有先验证原始模型的 D2H 确定性
   - ESM2 原始模型自一致性 diff=0.0（完全确定），但优化后 diff=0.043~0.117
   - 如果原始模型本身有非确定性（如随机 dropout），精度对比会更复杂

5. **float16 精度验证缺乏统计意义**：只做了 max_abs_diff 和 cosine
   - 没有做相对误差（relative error）
   - 没有做 KL divergence（对生成模型更重要）
   - 没有做 top-k 匹配率（对分类/LM 模型更重要）

### 2.3 改进建议

1. **建立分层精度验证体系**：
   - Level 0：最终输出 cosine + max_abs_diff（快速检查）
   - Level 1：逐层输出 cosine（定位退化层）
   - Level 2：top-k 匹配率 / KL divergence（语义级别）
   - Level 3：大规模样本统计（100+ 样本的精度分布）

2. **阈值应基于模型特性推导**：
   - float32 单层：max_abs_diff < 1e-6（机器精度）
   - float16 单层：max_abs_diff < 1e-3（half 精度）
   - N 层累积：max_abs_diff < N × single_layer_diff × growth_factor
   - growth_factor 需要通过实验确定（通常 1.5~3x）

3. **必须验证 baseline 自一致性**：优化前先运行原始模型两次，确认 diff=0

---

## 3. 性能验证方法论评估

### 3.1 当前做法

- **指标**：wall-clock median（20 次取中位数）
- **L0 profiling**：NPU Only，3 warmup + 1 profiled iteration
- **L1 profiling**：CPU + NPU，with_stack + record_shapes + profile_memory
- **L0/L1 交叉验证**：L1 utilization 低于 L0 超过 20pp → profiler 伪影
- **A/B benchmark**：优化前后交错运行（interleaved），取中位数

### 3.2 问题与不足

1. **L0 采集方式不一致**：
   - 有些模型 L0 用 `export_chrome_trace`（无 step_trace_time.csv）
   - 有些用 `tensorboard_trace_handler`（有 step_trace_time.csv）
   - resnet50 Round 1 因此无法做 L0/L1 交叉验证，导致误判

2. **wall-clock 计时范围与 profiling 不对齐**：
   - 有些 benchmark 包含 processor/tokenizer（CLIP Round 1）
   - 有些不包含（CLIP Round 2 修正后）
   - **ESM2 R4 发现关键问题**：wall-clock 只框了 `flat_forward`，而 L0 step_trace 的范围包含 H2D + rotary 预计算 + flat_forward。两者口径不一致导致 wall-clock / L0_Computing 比值失真
   - ESM2 实测：Scope A（仅 flat_forward）= 8.04ms，Scope B（H2D + rotary + flat_forward，对齐 L0）= 8.54ms，差异 0.5ms
   - **正确做法**：wall-clock 的计时块必须与 L0 step_trace 覆盖的代码完全一致，否则两者的对比（如 wall-clock / L0_Computing）无意义

3. **warmup 次数不足**：
   - SD 只做了 2 次 warmup（5 步 × 2 = 10 UNet calls）
   - 第一次 UNet call 可能有 JIT 编译开销
   - 建议 warmup ≥ 5 次或直到连续 3 次延迟差异 < 5%

4. **缺乏统计显著性检验**：
   - 只取中位数，没有 p-value 或置信区间
   - 20 次采样可能不够区分 < 5% 的改进
   - ESM2 Round 2 的 +2% 改进在统计上可能不显著

5. **profiler 开销已量化（ESM2 R4 实测）**：
   - L0 profiler 开销：L0 total (Computing + Free) = 12.5ms vs 对齐 wall-clock = 8.54ms，profiler 引入 3.96ms 额外 Free
   - L1 profiler 开销：L1 total = 48.5ms vs 对齐 wall-clock = 8.54ms，profiler 引入 ~40ms 额外开销（barrier 注入破坏异步流水线）
   - **L0 Computing 受 profiler 影响小**（仅被动记录 NPU 活动，不注入 barrier），可作为设备执行时间的可靠估计
   - **L0 Free 和 L1 host time 严重高估真实 host 开销**：真实 host 开销 = 对齐 wall-clock - L0 Computing = 8.54 - 7.1 = 1.44ms，而 L0 Free = 5.4ms（高估 3.7x），L1 host time = 42ms（高估 29x）

### 3.3 改进建议

1. **统一 L0 采集格式**：全部使用 `tensorboard_trace_handler`
2. **wall-clock 计时范围必须与 L0 step_trace 对齐**：
   - 在 L0 profiling 脚本中明确记录 step_trace 覆盖的代码范围
   - wall-clock benchmark 使用完全相同的代码范围
   - 示例（ESM2）：两者都包含 `ids.to(device) + compute_rotary + flat_forward`
3. **增加采样数**：50 次取中位数 + p10/p90 + Wilcoxon 检验
4. **profiler overhead 量化方法**：
   - 对齐 wall-clock = 无 profiler 的真实性能
   - L0 Computing ≈ 真实设备执行时间（profiler 对 device 影响小）
   - 真实 host 开销 = 对齐 wall-clock - L0 Computing
   - profiler 开销 = L0 total - 对齐 wall-clock
   - L1 host time 不可直接作为 host 开销的估计——barrier 注入使其膨胀 10-30x

---

## 4. 优化停止条件（上界分析）

### 4.1 当前问题

当前完全依赖人工判断"何时停止优化"，缺乏系统性上界。具体表现：
- 有时过早停止（SD Round 1 没深入分析源码就下结论）
- 有时过晚停止（dinov2 Round 4 测试了一个已知会失败的方案）
- 不同模型的"终局"标准不一致

### 4.2 建议的上界构建方法

#### 4.2.1 理论下界（Theoretical Lower Bound）

模型推理的理论最快时间 = 纯计算时间（无任何 host 开销）。

**计算方法**：
```
theoretical_min = sum(all_kernel_duration)  # L0 Computing
```

这个值可以从 L0 profiling 直接获取。对于 7 个模型：

| 模型 | L0 Computing (理论下界) | 实际 wall-clock | 差距 | 差距占比 |
|------|------------------------|-----------------|------|----------|
| resnet50 | 3.66ms | 3.41ms | -0.25ms | -7%* |
| dinov2 | 2.81ms | 2.89ms | +0.08ms | +3% |
| wav2vec2 | 2.49ms | 3.15ms | +0.66ms | +21% |
| CLIP | 4.98ms | 8.73ms | +3.75ms | +43% |
| ESM2 | 7.1ms | 8.54ms | +1.44ms | +17% |
| Qwen3 | 10.96ms | 11.96ms | +1.00ms | +8% |
| SD v1-4 | 203.3ms | 226ms | +22.7ms | +11% |

*resnet50 wall-clock < L0 Computing 因为 L0 Computing 包含 profiler 开销

**当 wall-clock / L0_Computing < 1.1 时，可以认为已接近理论下界**。

ESM2 经过 5 轮优化后 wall-clock / L0_Computing = 8.54 / 7.1 = 1.20x（对齐口径后），接近但未达到 1.1 阈值。剩余 1.44ms 差距来自 Python dispatch 开销，只能通过图编译消除——而图编译被 CANN 8.5.0 阻塞（详见 4.4 节）。

> **注意**：此处的 wall-clock 8.54ms 是与 L0 step_trace 同口径的测量（包含 H2D + rotary + flat_forward），不是仅 flat_forward 的 8.04ms。口径对齐方法详见 §3.2-3.3。

#### 4.2.1b Roofline 下界（硬件极限）

L0 Computing 不是真正的硬件极限。Roofline 模型给出了更底层的下界：

```
对每个操作: time >= max(FLOPs / peak_compute, bytes / HBM_bandwidth)
全模型下界 = sum(max(compute_time, memory_time) for each op)
```

**batch=1 推理时大多数操作是 memory-bound**，因为权重远大于激活：

| 操作 (seq_len=100, fp16) | FLOPs | 内存访问 | compute time | memory time | Roofline | 实际 kernel |
|--------------------------|-------|----------|-------------|-------------|----------|-------------|
| QKV MatMul [3840,1280] | 983M | 10.9MB | 3.1us | 9.1us | 9.1us | ~15us |
| FC1 MatMul [5120,1280] | 1.31G | 13.1MB | 4.1us | 10.9us | 10.9us | ~20us |
| FlashAttention | 512M | 1.0MB | 1.6us | 0.9us | 1.6us | ~60us |
| LayerNorm | ~0.5M | 0.77MB | ~0 | 0.6us | 0.6us | ~9us |

（Ascend 910 FP16 ~320 TFLOPS, HBM ~1.2 TB/s）

ESM2 全模型 Roofline 下界估算：
```
33 层 × 4 MatMul 权重 ≈ 1.4 GB 总内存访问
Roofline ≈ 1.4 GB / 1.2 TB/s ≈ 1.2 ms
```

**三级下界的层次关系**：

| 下界层次 | ESM2 数值 | 含义 | 优化手段 |
|----------|----------|------|---------|
| 绝对计算下界 (FLOPs/peak) | ~0.5ms | 物理极限，忽略一切开销 | 架构变更/量化 |
| Roofline 下界 | ~1.2ms | 硬件极限，考虑带宽 | CANN/OPP 算子级优化 |
| L0 Computing | 7.1ms | 实际 kernel 执行时间 | Python 层（图编译等） |
| 对齐 Wall-clock | 8.54ms | 加上未重叠的 host 开销 | 当前优化终点 |

L0 Computing 是 Roofline 的 ~5 倍，差距来自 kernel 内部开销（tiling、synchronization、小 kernel 无法饱和带宽）。这不在 Python 应用层可优化范围内。

**关键发现**：batch=1 推理的根本瓶颈是**权重加载的内存带宽**，不是计算。这解释了：
- 序列长度 40→300aa，wall-clock 仅从 8.8→8.9ms（权重不变，memory-bound 不随 seq_len 变化）
- cube utilization 仅 18.4%（计算单元闲置）
- batch 推理可显著提升 per-sequence 性能（分摊权重加载，提高算术强度）

#### 4.2.2 Host 开销下界（Host Overhead Floor）

当 L0 Free > 0 时，剩余的 Free 来自 Python 框架开销：

```
host_overhead = L0_Free = N_python_frames × avg_frame_cost
```

其中 `avg_frame_cost` 可以通过 `L0_Free / N_python_frames` 估算：

| 模型 | L0 Free | Python frames (L1) | avg_frame_cost | 占比 |
|------|---------|-------------------|----------------|------|
| resnet50 | 0.39ms | ~200 | ~2.0us | 11% |
| dinov2 | 0.20ms | ~650 | ~0.3us | 6% |
| wav2vec2 | 0.23ms | ~1000 | ~0.2us | 7% |
| CLIP | 3.62ms | ~746 | ~4.9us | 42% |
| ESM2 (R2) | 9.87ms | ~1400 | ~7.1us | 63% |
| ESM2 (R4) | 5.4ms | ~1379 | ~3.9us | 43% |
| Qwen3 | 2.81ms | ~1246 | ~2.3us | 23% |
| SD v1-4 | 13.0ms | ~86000 | ~0.2us | 6% |

**当 L0_Free / L0_Computing < 5% 时，可以认为 host 开销已不是瓶颈**。

#### 4.2.3 算子不可消除下界（Irreducible Kernel Floor）

某些算子是设计必需的，无法通过 Python 层面优化消除：

| 不可消除算子 | 原因 | 影响模型 |
|-------------|------|----------|
| ~~TransData（权重部分）~~ | ~~CANN 运行时 Conv 格式转换~~ | ~~resnet50, wav2vec2, SD~~ — **SD 已消除权重部分**：FRACTAL_Z 预转换消除 625 次权重 TransData (-27ms)；输入/输出部分仍不可消除 |
| ~~Cast (rotary fp16↔fp32)~~ | ~~数值稳定性设计~~ | ~~ESM2, Qwen3~~ — **已消除**：npu_rotary_mul 在 NPU 内部处理精度 |
| Triu+OnesLike (SDPA is_causal) | NPU SDPA 内部创建（可预计算消除）| Qwen3（已消除）, SD |
| Conv2D/FlashAttention | 核心计算 | 全部 |
| MatMulV2/V3 | 核心计算 | 全部 |

**当不可消除算子占 L0 Computing > 80% 时，可以认为已达到算子下界**。

#### 4.2.4 综合停止条件（建议加入 skill）

```
停止优化当满足以下任一条件：
1. wall_clock / L0_Computing < 1.1（接近理论下界）
2. L0_Free / L0_Computing < 5%（host 开销可忽略）
3. 不可消除算子 / L0_Computing > 80%（算子已是最小集）
4. 连续 2 轮优化均 < 2% 改进（边际收益低于工程成本）
5. 所有候选方案被拒绝且无新候选产生
```

### 4.3 上界构建的挑战

1. **L0 Computing 本身可能不是真正的下界**：
   - L0 Computing 包含 profiler 开销（虽然 L0 最小，但仍非零）
   - 某些 kernel 可能不是最优的（如 Conv2D 有更优的 tiling）
   - 但从 Python 层面无法优化 kernel 内部实现

2. **异步流水线使得 wall-clock ≠ L0_Computing + L0_Free**：
   - TASK_QUEUE_ENABLE=2 时 host 和 device 操作重叠
   - wall-clock 可能小于 L0_Computing + L0_Free
   - 需要建立异步重叠模型

3. **NPU 非结合律浮点运算限制了可优化空间**：
   - 任何改变计算顺序的优化（BN 折叠、layer scale 折叠、Add+LN 融合）都可能在 float16 多层模型上失败
   - 这不是"不想做"而是"做不了"——是硬件约束，不是软件约束
   - 需要区分"可优化但收益不足"和"不可优化因硬件限制"

### 4.4 图编译的 CANN 平台限制（ESM2 Round 4 实测）

ESM2 Round 4 穷尽了 6 条图编译路径，全部被 CANN 8.5.0 阻塞：

| 路径 | 失败原因 |
|------|----------|
| torchair.inference.cache_compile (dynamic=True) | npu_fusion_attention BSND layout 不被 GE converter 支持（仅 BSH/BNSD） |
| torchair.inference.cache_compile (dynamic=False, BNSD) | MatMulV2 tiling error: "tiling is illegal, actual is [None]" |
| torch.compile (npu backend) | 同 BSND 不支持问题 |
| torch.compile (inductor backend) | triton 未安装 |
| torch.jit.trace | npu_fusion_attention 返回 int 类型，jit.trace 不支持 |
| jit_compile=True | MatMulV2 tiling error（同上） |

**两个 CANN 限制**：
1. **npu_fusion_attention GE converter 仅支持 BSH/BNSD layout**——BSND 不支持，但 BSND 是避免 Transpose 的最优 layout
2. **MatMulV2 在图编译模式下 tiling 失败**——"tiling is illegal, actual is [None]"，影响所有包含 MatMul 的图编译

**结论**：CANN 8.5.0 的图编译能力对 Transformer 模型不可用。需 CANN 更新修复 MatMulV2 tiling 和扩展 npu_fusion_attention layout 支持。

### 4.5 "终局"判断的修订

ESM2 的案例表明，"Python 框架开销下限"作为终局原因可能过早：

| 阶段 | 终局原因 | 实际后续 |
|------|---------|---------|
| R3 | "Python 框架开销下限，需图编译" | R4 发现 npu_rotary_mul + F.gelu 可在 Python 层再减 561 kernel (+37.8%) |
| R4 | "图编译被 CANN 阻塞，接近 L0 下界" | R5 验证：npu_add_layer_norm/jit.trace/allow_internal_format 均无效，确认终局 |

**教训**：终局判断前必须穷尽 NPU 融合算子库（npu_rotary_mul、npu_gelu 等），不能仅因"手动实现中有 Cast"就判定为不可消除。R2 的 rotary analysis 错误地将 Cast 标记为"design-critical"——实际上 npu_rotary_mul 在 NPU 内部处理了 float32 精度，产出 bit-identical 结果。

---

## 5. 关键平台发现汇总

### 5.1 NPU 硬件行为

1. **NPU 浮点运算不满足结合律**：
   - `(W*x)*scale ≠ (W*scale)*x` 在 float16 上（max_diff ~6e-4）
   - `Add(x, y) + LayerNorm ≠ npu_add_layer_norm(x, y)` 在 pre-norm 模型上
   - 33 层 float16 模型累积差异可达 0.1+
   - **影响**：限制了所有"权重折叠"和"算子融合"类优化

2. **TransData 权重部分可消除，输入/输出部分不可消除**：
   - CANN 每次 Conv2D 调用插入 3 个 TransData：输入 NCHW→NC1HWC0、权重 NCHW→FRACTAL_Z、输出 NC1HWC0→NCHW
   - **权重部分**可通过 FRACTAL_Z 预转换消除（SD v1-4 验证：-27ms, -625 kernel, 零精度影响）
   - **输入/输出部分**是 CANN 运行时格式转换，无法通过 Python 层消除（需图编译）
   - 此前结论"TransData 是 CANN 优化，FRACTAL_Z 预转换更慢"基于 resnet50/wav2vec2 经验，在 SD v1-4 上不成立
   - **影响**：Conv2D-heavy 模型应测试 FRACTAL_Z 预转换，不能假设不可消除

3. **npu_fusion_attention 的 causal mask 参数不生效**：
   - pre_tockens/next_tockens 参数无法实现 causal masking
   - sparse_mode=1 也不正确
   - 必须用显式 atten_mask（bool True=mask 上三角）
   - **影响**：npu_fusion_attention 用于 causal 模型时需要额外 mask

4. **NPU SDPA bool mask 约定与 PyTorch 相反**：
   - PyTorch：True = masked（不 attend）
   - NPU SDPA：True = attend（下三角）
   - **影响**：预计算 causal mask 时必须用 tril（不是 triu）

5. **GQA 模型使用 npu_fusion_attention 需要 repeat_interleave**：
   - repeat_interleave 的开销抵消了 attention kernel 的收益
   - **影响**：npu_fusion_attention 不适用于 GQA 模型（Qwen3）

6. **rotary_emb 对 int 输入返回 int64 cos/sin**：
   - `model.rotary_emb(input_ids, position_ids)` 返回 int64
   - 必须用 float embeddings 调用：`rotary_emb(embed_tokens(input_ids), position_ids)`
   - **影响**：Qwen3 初始实现完全错误（cosine=0.23）

7. **npu_rotary_mul 是 NPU 融合 rotary 算子，bit-identical 替代手动实现**：
   - `torch_npu.npu_rotary_mul(input, cos, sin, rotary_mode='half')` 等价于 `cos * input + sin * rotate_half(input)`
   - NPU 内部处理 float32 精度，产出与手动 `q.float()*cos + rotate_half(q.float())*sin → .to(dtype)` **bit-identical** 的结果
   - 消除 264 Cast + 66 Mul + 66 Slice + 33 Neg + 33 Concat = 462 kernel（33 层模型）
   - 约束：D < 896 且为 2 的倍数；BSND layout 无 32 字节对齐限制（BNSD 有）
   - **影响**：ESM2 R4 最大优化项，之前 R2 错误标记 Cast 为"design-critical"

8. **F.gelu 在 NPU 上精度优于手动 gelu**：
   - NPU F.gelu 使用更高内部精度的 erf 实现
   - 33 层 float16 累积：F.gelu diff=0.008 vs 手动 gelu diff=0.012
   - **影响**：F.gelu 不仅减少 99 kernel，还提升了精度

9. **CANN 8.5.0 图编译对 Transformer 模型不可用**：
   - npu_fusion_attention GE converter 仅支持 BSH/BNSD，不支持 BSND
   - MatMulV2 在图编译模式下 tiling 失败（"tiling is illegal"）
   - 影响 cache_compile / torch.compile / jit_compile / jit.trace 全部路径
   - **影响**：图编译作为"高级优化"在当前 CANN 版本不可行

10. **npu_add_layer_norm 对 PRE-norm 模型不适用**：
    - PRE-norm 架构中 Add 结果（未归一化）需保留给下一层残差
    - npu_add_layer_norm 只能融合 final LN（省 1 kernel，可忽略）
    - POST-norm 模型可完全融合（wav2vec2, CLIP 验证通过）
    - **影响**：ESM2（PRE-norm）无法使用此融合算子

### 5.2 优化模式

1. **Flat forward 对 host-bound 模型最有效，对 compute-bound 模型无效**：
   - 对所有 host-bound 模型有效（CLIP -73%, ESM2 -57%, Qwen3 -52%）
   - 效果与 Python frames 数量成正比
   - 零精度风险（数学等价）
   - 对 compute-bound 模型（SD 84.5% utilization）无效——Free 太少，无 host 开销可消除
   - **但 compute-bound 不等于不可优化**（见下条）

2. **NPU 融合算子的适用性因模型而异**：
   - npu_rms_norm：通用（Qwen3 -679 kernel）
   - npu_fusion_attention：对非 GQA + 非 causal 模型最有效（dinov2 -84 Transpose）
   - npu_rotary_mul：对 rotary 模型通用（ESM2 -462 kernel，bit-identical）
   - F.gelu：对 gelu 模型通用（ESM2 -99 kernel，精度更好）
   - npu_add_layer_norm：仅 POST-norm 适用（wav2vec2），PRE-norm 不适用（ESM2）
   - npu_fast_gelu：仅适用于 quick_gelu 模型（CLIP）

3. **权重折叠的精度风险与层数成正比**：
   - 少层模型（CLIP 12+12 层）：通过
   - 多层模型（ESM2 33 层, Qwen3 28 层）：可能失败
   - 阈值：~20 层 float16 是权重折叠的安全边界

### 5.3 Profiling 方法论

1. **L0/L1 交叉验证是必须的**：
   - L1 的 host-bound 信号在所有 7 个模型上都是 profiler 伪影
   - L1 utilization 比 L0 低 40~70 个百分点
   - 但 L1 的算子级数据（op_statistic, kernel_details）仍然有效

2. **L0 Free 不是唯一的 host 开销指标**：
   - SD 的 L0 Free=19ms（1.3%）但 L1 显示 126ms _local_scalar_dense
   - 这些开销被异步流水线重叠，对 wall-clock 无影响
   - **结论**：L0 Free 是 host 开销的可靠指标，L1 host time 不是

3. **profiler 开销量化（ESM2 R4 实测）**：
   - L0 Free 高估真实 host 开销 3.7 倍（5.4ms vs 真实 1.44ms）
   - L1 host time 高估 29 倍（42ms vs 真实 1.44ms）
   - **正确估算方法**：对齐 wall-clock - L0 Computing = 真实 host 开销
   - **结论**：wall-clock / L0_Computing 的比值才有意义，但前提是两者口径对齐（计时范围一致）

4. **wall-clock 计时范围必须与 L0 step_trace 对齐**：
   - ESM2 R4 发现：wall-clock 只框 flat_forward (8.04ms)，L0 step_trace 框 H2D+rotary+flat_forward (8.54ms)
   - 口径不一致导致 wall-clock / L0_Computing 比值从 1.13x（错误）变为 1.20x（正确）
   - **结论**：wall-clock 的 `synchronize → t0 → 代码 → synchronize → t1` 中的"代码"必须与 L0 step_trace 覆盖的代码完全相同

5. **compute-bound 模型仍有优化空间**：
   - SD v1-4 从 84.5% utilization 优化到 94%，通过 4 种手段：
     a. Safety checker NPU 预处理（消除 PIL 往返的 host 开销，Free 45→11ms）
     b. BSND attention（消除 Q/K/V 转置，-19ms Transpose）
     c. FRACTAL_Z 权重预转换（消除冗余权重格式转换，-27ms TransData）
     d. allow_internal_format toggle + CACHE_MODE=force（消除 Online-Compile）
   - 前轮分析错误地认为 98.7% utilization 是终局——实际是排除了 safety_checker 且数据计算错误
   - **结论**：compute-bound 模型应检查 host 开销来源（D2H 转换、格式转换、编译开销），不能仅看 utilization 数字

---

## 6. Skill 改进建议

### 6.1 优化停止条件（新增）

在 SKILL.md 的"迭代退出条件"中增加量化标准：

```
停止优化当满足以下任一条件：
1. wall_clock / L0_Computing < 1.1
2. L0_Free / L0_Computing < 5%
3. 连续 2 轮优化均 < 2% 改进
4. 所有候选被拒绝且无新候选
```

### 6.2 精度验证改进（增强）

在 04_accuracy_assurance/SKILL.md 中增加：

1. **分层验证体系**：Level 0（最终输出）→ Level 1（逐层）→ Level 2（语义）
2. **阈值推导方法**：基于层数 × 单层精度 × 增长因子
3. **baseline 自一致性验证**：优化前必须验证原始模型两次运行 diff=0
4. **float16 多层模型的特殊处理**：> 20 层时阈值应放宽，但需论证

### 6.3 新增"不可消除算子"概念

在 02_bottleneck_analysis 中增加：

```
不可消除算子定义：经过源码分析和实测验证，确认无法通过 Python 层面优化消除的算子。

常见不可消除算子：
- TransData（CANN 运行时 Conv 格式转换优化）
- Cast from rotary fp16↔fp32（数值稳定性设计）
- Conv2D/FlashAttention/MatMul（核心计算）

验证方法：
1. 测试 FRACTAL_Z 权重预转换 → 如果更慢则输入/输出 TransData 不可消除（但权重部分总可消除）
   - 注意：前人经验"FRACTAL_Z 更慢"可能基于不同模型，必须在本模型实测
   - SD v1-4 实测：权重预转换 -27ms，但需要 toggle allow_internal_format 避免 Online-Compile
2. 测试 float16 rotary → 如果精度超限则 Cast 不可消除
3. 记录到 evidence_db 的 platform_findings 中
```

### 6.4 新增"NPU 融合算子适用性矩阵"

在 03_optimization/references 中增加：

```
| 融合算子 | 非 GQA + 非 causal | 非 GQA + causal | GQA + causal |
|---------|-------------------|----------------|-------------|
| npu_fusion_attention BSND | ✅ 直接使用 | ⚠️ 需 atten_mask | ❌ repeat 开销 |
| npu_rotary_mul | ✅ rotary 模型通用 (bit-identical) | ✅ | ✅ |
| F.gelu / npu_gelu | ✅ 精度更好 | ✅ | ✅ |
| npu_rms_norm | ✅ 通用 | ✅ 通用 | ✅ 通用 |
| npu_add_layer_norm | ✅ post-norm | ⚠️ pre-norm 不适用 | ⚠️ pre-norm 不适用 |
| npu_fast_gelu | ✅ quick_gelu 等价 | ✅ | ✅ |
```

### 6.5 新增"权重折叠安全边界"

```
权重折叠（将 scale/norm 参数吸收进 Linear 权重）的安全边界：
- float32：任意层数安全
- float16, ≤ 12 层：安全（CLIP 12+12 层验证通过）
- float16, 13~20 层：需逐层验证
- float16, > 20 层：高风险（ESM2 33 层验证失败, Qwen3 28 层部分失败）

注意：折叠 1/sqrt(2) 进 fc1 权重 + 手动 gelu 的精度（ESM2 diff=0.012）
不如直接用 F.gelu（ESM2 diff=0.008）。当 NPU 有 F.gelu 融合算子时，
应优先使用 F.gelu 而非权重折叠 + 手动 gelu。
```

### 6.6 L0 采集格式统一

在 01_preparation 中增加：

```
L0 profiling 必须使用 tensorboard_trace_handler（不是 export_chrome_trace）。
原因：tensorboard_trace_handler 产出 step_trace_time.csv，用于 L0/L1 交叉验证。
export_chrome_trace 只产出 trace.json，无法做交叉验证。
```

### 6.7 新增"异步流水线重叠模型"

```
当 TASK_QUEUE_ENABLE=2 时：
- wall_clock ≈ max(L0_Computing, L0_Computing - overlap + L0_Free)
- overlap = min(host_dispatch_time, device_compute_time_gap)
- 当 L0_Free < 5% L0_Computing 时，overlap ≈ L0_Free（几乎完全重叠）
- 当 L0_Free > 30% L0_Computing 时，overlap < L0_Free（部分重叠）

结论：L0_Free 是 host 开销的可靠上界，但不是精确值。
wall_clock 的下界是 L0_Computing（理论最快）。
```

---

## 7. 思考与展望

### 7.1 SD v1-4 的优化历程与教训

前4轮分析（2026-07-31）结论为"SD 完全 compute-bound（98.7%），无法优化"，产出 0% 提升。该结论基于两个严重缺陷：

1. **safety_checker 被禁用**（`safety_checker=None`）：所有 profiling 和 wall clock 排除了 safety_checker（CLIP Vision 24层模型，占 43ms/14.3%）
2. **evidence_db 数据错误**：声称 1489ms wall-clock，实际脚本日志为 270ms；L0 Computing=1433ms 是 3 次迭代累加值

重新分析后（2026-08-02/03），5 轮优化实现 -22.9% 提升（293→226ms）：

| 轮次 | 优化 | 手段 | Wall clock | 累计 |
|------|------|------|-----------|------|
| 基线 | 修正：启用 safety_checker | — | 293ms | — |
| R1 | Safety checker NPU 预处理 | 替换 | 266ms | -9.2% |
| R2 | BSND attention 消除转置 | 替换 | 256ms | -12.6% |
| R3 | FRACTAL_Z 权重预转换 | 去重 | 243ms | -17.1% |
| R4 | Toggle allow_internal_format | 执行模式 | 237ms | -19.1% |
| R5 | CACHE_MODE=force | 去重 | 226ms | -22.9% |

**教训**：
1. profiling 和 wall clock 必须包含全部功能代码（包括 safety_checker）
2. evidence_db 数据必须与脚本日志交叉验证
3. "compute-bound 终局"判断前必须检查 host 开销来源（D2H 转换、格式转换、编译开销）
4. 前人经验（"FRACTAL_Z 预转换更慢"）可能基于不同模型，必须在本模型上实测

### 7.2 图编译是下一步的方向吗？

当前所有优化都在 Python 层面（flat forward, F.* 调用）。对于 CLIP（58% utilization）和 ESM2（R4 后 57% utilization），剩余的 host 开销来自 ~1400 个 F.* 调用的 Python 框架开销。这些只能通过图编译（torch.compile, torchair）消除。

**ESM2 Round 4 图编译实测结果**（6 条路径全部失败）：
- torchair cache_compile：npu_fusion_attention BSND 不被 GE converter 支持 + MatMulV2 tiling 失败
- torch.compile (npu/inductor)：同上 + triton 未安装
- torch.jit.trace：npu_fusion_attention 返回 int 类型不支持
- jit_compile=True：MatMulV2 tiling 失败

**结论**：CANN 8.5.0 的图编译能力对使用 npu_fusion_attention 的 Transformer 模型不可用。需要：
1. CANN 修复 MatMulV2 图编译 tiling 问题
2. CANN 扩展 npu_fusion_attention GE converter 支持 BSND layout
3. 或在图编译版本中使用 SDPA 替代 npu_fusion_attention（但 SDPA 更慢且同样遇到 tiling 问题）

**建议**：在 skill 中增加图编译作为"高级优化"阶段，仅在 Python 层面优化穷尽后使用。同时记录 CANN 版本兼容性矩阵，标注当前版本的限制。

### 7.3 量化是另一个方向吗？

量化（float16 → int8）可以将 Conv2D 和 MatMul 的计算量减少 2~4 倍。对于 SD（compute-bound），这是唯一可行的加速路径。

但量化有以下风险：
1. 精度退化（需要校准数据集）
2. NPU int8 支持的算子覆盖范围有限
3. 量化感知训练可能需要重新训练

**建议**：在 skill 中明确"量化不在当前优化流程范围内"，但提供指向量化 skill 的链接。

### 7.4 优化过程的可重复性

当前优化过程高度依赖 agent 的判断：
- 选择哪个优化方向先做
- 什么时候停止
- 如何调试精度问题

建议增加更多结构化决策点：
1. **优化优先级矩阵**：按"收益上限 × 成功概率"排序所有候选
2. **调试决策树**：精度失败时按"逐层对比 → 检查 dtype → 检查 device → 检查 shape"的顺序排查
3. **停止条件检查表**：每轮结束时自动检查 5 个停止条件

### 7.5 模型架构与优化策略的映射

从 7 个模型的优化经验中，可以总结出架构到优化策略的映射：

| 架构特征 | 推荐优化策略 | 预期收益 |
|---------|-------------|----------|
| CNN (Conv2D-heavy) | flat forward only | 10~30% |
| ViT (pre-norm, non-GQA) | flat forward + QKV merge + npu_fusion_attn | 30~50% |
| ViT (post-norm) | flat forward + QKV merge + npu_add_layer_norm | 40~60% |
| Transformer (GQA, RMSNorm) | flat forward + npu_rms_norm + K+V merge + gate+up merge | 40~55% |
| Transformer (rotary, float16) | flat forward + QKV merge + npu_rotary_mul + F.gelu + rotary pre-compute | 50~60% |
| Diffusion (UNet+VAE) | SC NPU预处理 + BSND attention + FRACTAL_Z权重 + toggle + cache_force | 10~25% |
| 多组件 pipeline (CLIP, SD) | + preprocess 移出循环 | +10~60% |

这个映射可以作为 skill 中的"优化策略选择指南"。

---

## 8. 附录

### 8.1 Evidence DB 统计

共 31 个案例文件，分布如下：

| 模型 | 采纳案例 | 拒绝案例 | 总计 |
|------|---------|---------|------|
| resnet50 | 1 (flat forward) | 2 (BN fold, TransData) | 3 |
| dinov2 | 2 (flat forward+QKV, npu_fusion_attn) | 2 (layer scale fold, add+LN) | 4 |
| wav2vec2 | 2 (flat forward+fusion, full flat) | 1 (TransData) | 3 |
| CLIP | 3 (flat forward, preprocess fix, npu_fast_gelu) | 0 | 3 |
| ESM2 | 3 (flat forward+QKV+rotary, weight fold+npu_attn, npu_rotary_mul+F.gelu) | 3 (rotary analysis, deep analysis, graph compilation) | 6 |
| Qwen3 | 2 (flat forward+rms_norm, precomp mask) | 1 (npu_fusion_attn GQA) | 3 |
| SD v1-4 | 5 (SC预处理, BSND attention, FRACTAL_Z权重, toggle, cache_force) | 5 (4轮旧基线 + 1轮R2终局误判) | 10 |

### 8.2 所有平台发现汇总

共记录 46+ 条 platform_findings，关键条目：

1. NPU 浮点非结合律限制权重折叠和算子融合
2. TransData 是 CANN 优化不是浪费
3. npu_fusion_attention causal mask 参数不生效
4. NPU SDPA bool mask True=attend（反直觉）
5. GQA 模型 npu_fusion_attention repeat 开销抵消收益
6. rotary_emb 对 int 输入返回 int64
7. npu_rms_norm 7→1 kernel（最有效的融合算子）
8. Flat forward 对 host-bound 模型最有效
9. Compute-bound 模型（>95% util）Python 优化无效
10. 异步流水线完全重叠 host 开销（SD 105K frames 但 Free=1.3%）
11. npu_rotary_mul bit-identical 替代手动 rotary（NPU 内部处理 float32 精度）
12. F.gelu 在 NPU 上精度优于手动 gelu（更高内部精度 erf 实现）
13. CANN 8.5.0 图编译对 Transformer 不可用（MatMulV2 tiling + npu_fusion_attention BSND）
14. npu_add_layer_norm 仅适用于 POST-norm（PRE-norm 需保留未归一化残差）
15. batch=1 推理的根本瓶颈是权重加载的内存带宽（非计算），Roofline 下界远低于 L0 Computing
16. "终局"判断前必须穷尽 NPU 融合算子库，不能因手动实现有 Cast 就判定不可消除
17. wall-clock 计时范围必须与 L0 step_trace 对齐，否则 wall-clock / L0_Computing 比值失真（ESM2 差 0.5ms）
18. L0 Free 高估真实 host 开销 ~3.7x，L1 host time 高估 ~29x；真实 host 开销 = 对齐 wall-clock - L0 Computing
19. **FRACTAL_Z 权重预转换**：CANN 每次 Conv2D 都冗余转换权重（625次/推理, 27ms）。预转换消除全部冗余，零精度影响。前人结论"更慢"基于 resnet50/wav2vec2，在 SD v1-4 上不成立
20. **allow_internal_format toggle**：全局启用导致 CANN Online-Compile（100ms/推理，5076次内核重编译）。仅转换时启用、推理时关闭可兼顾 FRACTAL_Z 收益和零编译开销
21. **CACHE_MODE=force vs enable**：FRACTAL_Z 格式触发 enable 模式的逐次缓存验证（36ms host）。force 模式跳过验证，零精度影响
22. **npu.config.allow_internal_format**：`_npuConfig` 实现了 `__setattr__`（转发到 C++）但未实现 `__getattr__`，赋值生效但读取报 AttributeError
23. **Safety checker PIL 往返开销**：NPU tensor → PIL → CPU feature_extractor → NPU 产生 35ms 主机开销。直接 NPU tensor 操作（F.interpolate + normalize）完全消除，零精度影响
24. **profiling 和 wall clock 必须包含全部功能代码**：SD 前4轮排除 safety_checker 导致基线数据完全错误（1489ms vs 实际 270ms）
25. **BSND FlashAttention 内核比 BNSD 慢 22%**：但转置消除节省 19ms > 内核减速 9ms，净收益 -10ms。layout 选择需权衡转置开销与内核效率
26. **L0 profiler + CACHE_MODE=force 干扰**：force 模式下 L0 Free 不可靠（99ms vs 实际 13ms），应以 wall clock 为准
