# Skills 改进方案：从"时间花在哪"到"优化空间在哪"

> 结合 `opt_line_research.md` 的性能归因理论（三档上界、反事实瓶颈定位、self-time ≠ 优化价值）与 `OPTIMIZATION_SUMMARY.md` 中 7 模型 22 轮优化的实践经验，提出一个框架级的改进方案。
>
> 核心思想：当前 skills 的分析逻辑是 **"profiling → 算子 Top-N → 根因 → 候选"**（回答"时间花在哪"），应升级为 **"三档下界 → gap 分解 → 反事实收益上限 → 候选"**（回答"优化空间在哪、最多能改善多少"）。

---
# Skills 改进方案：从"时间花在哪"到"优化空间在哪"

> 结合 `opt_line_research.md` 的性能归因理论（三档上界、反事实瓶颈定位、self-time ≠ 优化价值）与 `OPTIMIZATION_SUMMARY.md` 中 7 模型 22 轮优化的实践经验，提出一个框架级的改进方案。
>
> 核心思想：当前 skills 的分析逻辑是 **"profiling → 算子 Top-N → 根因 → 候选"**（回答"时间花在哪"），应升级为 **"三档下界 → gap 分解 → 反事实收益上限 → 候选"**（回答"优化空间在哪、最多能改善多少"）。

---

## 1. 问题诊断：当前框架的根本缺陷

### 1.1 self-time 排序误导优化方向

当前 skills 的候选排序基于"该类浪费占总时间的比例"（profiling_to_action.md 归因层量化上限）。但 7 模型实测反复证明：**profiling 中的耗时占比不等于优化该部分能带来的端到端收益**。

最极端的案例是 SD：86000 个 Python frames 产生大量 host time（L1 显示 126ms `_local_scalar_dense`），但 L0_Free 仅 1.3%——异步流水线将这些 host 开销完全重叠。按 self-time 排序，host 开销是最大瓶颈；按反事实收益，消除 host 开销的收益 ≈ 0。

这正是 `opt_line_research.md` §3.3 指出的问题："非关键路径 region 即使耗时很长，也可能对延迟几乎无影响"。

### 1.2 缺少"能优化到什么程度"的量化框架

当前 skills 的"迭代退出条件"只有三条定性描述。Agent 不知道当前性能离物理极限有多远，也不知道剩余优化空间在哪一层。导致：
- ESM2 R3 判定终局，R4 又获得 +37.8% 提升
- SD 前 4 轮基于错误基线判定终局
- dinov2 R4 测试已知会失败的方案

`opt_line_research.md` §3 提出的三档上界框架可以直接解决这个问题——但要落地，需要用实测数据中已有的指标来实例化三档。

### 1.3 host 开销估算被 profiling 伪影扭曲

ESM2 R4 实测：L0 Free = 5.4ms，显著高于 1.44ms 的端到端关键路径残差；L1 host time 更高。当前归因层用 profiling 中的 host 时间占比作为“量化上限”，对 NPU 推理场景系统性高估。

---

## 2. 核心设计：三档下界 + gap 分解

### 2.1 三档下界的实例化

`opt_line_research.md` 提出三档上界（以时延表示则为下界）。用 7 模型实测中已有的指标实例化：

```
Tier 1: Roofline 下界（硬件物理极限）
  对每个关键操作: t >= max(FLOPs / peak_compute, bytes / HBM_bandwidth)
  全模型: sum of per-op Roofline
  数据来源: 模型结构（权重大小、FLOPs 估算）+ 硬件规格
  ESM2 实例: 1.4GB / 1.2TB/s ≈ 1.2ms

Tier 2: L0 NPU Computing（设备执行参考值）
  = L0 轻量 NPU-only 采集下的设备计算时间
  数据来源: L0 profiling 的 step_trace_time.csv
  ESM2 实例: 7.1ms

Tier 3: 对齐 wall-clock（实际端到端时间）
  = 无 profiler 的真实性能（计时范围与 L0 step_trace 对齐）
  数据来源: wall-clock benchmark
  ESM2 实例: 8.54ms
```

### 2.2 gap 分解：优化方向由 gap 决定

两个 gap 各自指向不同的优化层：

```
Tier 1 ──── gap A ──── Tier 2 ──── gap B ──── Tier 3
  Roofline           L0 Computing           wall-clock

gap A = L0_Computing - Roofline  （设备执行效率 gap）
  含义: kernel 内部的 tiling、occupancy、小 kernel 无法饱和带宽等
  可优化性: 纯 Python 调度通常无法改善；但融合算子、图编译、布局调整或算法替换可能改善
  ESM2: 7.1 - 1.2 = 5.9ms，Tier 2 / Tier 1 ≈ 5.9x，说明设备执行离理想参考下界仍较远

gap B = max(0, wall-clock - L0_Computing)  （端到端关键路径残差）
  含义: 在 L0 为轻量 NPU-only 采集、且 benchmark 与 L0 step 边界严格对齐时，
       近似表示未被 NPU Computing 覆盖的非设备计算时间；可作为仅优化 host 调度/dispatch 的收益上界
  可优化性: 纯 host 优化受该上界约束；融合算子等可能同时影响 gap A，不受此上界限制
  ESM2: 8.54 - 7.1 = 1.44ms (16.9%)，纯 host 优化仍有有限空间
  SD: 226 - 203 = 23ms (11%)，空间较小
  resnet50: 3.41 - 3.66 = -0.25ms，应标记为采样扰动或计时口径异常；gap B 按 0 处理，不作负 host 开销解释
```

**关键原则**：gap B 是仅优化 Python 调度/dispatch 的收益上限。当 gap B 在测量波动内时，这一类优化空间不可分辨；融合、图编译、算子替换等仍可能通过降低 Tier 2 获益。

### 2.3 gap B 的修正：异步流水线的影响

`opt_line_research.md` §3.3 指出 self-time ≠ 关键路径贡献。在 NPU 上，异步流水线（`TASK_QUEUE_ENABLE=2`）使大部分 host 开销被 device compute 重叠——这些 host 开销虽然出现在 profiling 中，但不在关键路径上。

```
端到端关键路径残差 = max(0, wall-clock - L0_Computing) = gap B
gap B 是轻量 L0 条件下 host-side 的近似上界，不是严格的 host 开销测量值
L0_Free 包含被重叠的 host 活动，通常高估 host 关键路径贡献
L1 host time = profiler barrier 注入后的 host 时间，高估约 29x
```

**结论**：在计时口径对齐且重复测量稳定时，gap B 是 host-side 候选的端到端收益上界；L0_Free 和 L1 host time 不能直接代表关键路径贡献。归因层中 host 侧类别的“量化上限”应基于 gap B，而非 profiling 中的 host 时间占比。

### 2.4 gap A 的诊断价值：理解 batch=1 推理的本质

`opt_line_research.md` §2.2 强调"仅有 FLOPs 不足以解释推理性能"。7 模型实测验证了这一点：

ESM2 Roofline 参考下界 1.2ms vs L0 Computing 7.1ms（Tier 2 / Tier 1 ≈ 5.9x）。Roofline 分析显示 batch=1 推理是 memory-bound（权重加载主导），cube utilization 仅 18.4%。这解释了：
- 序列长度 40→300aa，wall-clock 仅从 8.8→8.9ms（权重不变，memory-bound 不随 seq_len 变化）
- gap A 的大部分来自小 kernel 无法饱和 HBM 带宽（权重碎片化为多个小 MatMul）
- 减少 kernel 数量（融合算子）可以部分缩小 gap A——不是通过减少 FLOPs，而是通过减少 kernel launch 间隙和提升单次 kernel 的带宽利用率

---

## 3. 流程重构：Phase 2 新增"下界分析"前置步骤

### 3.1 新增 Phase 2 Step 0：下界分析

在 `run_analysis.py` 报告分析之前，先计算三档下界，确定优化方向：

```markdown
### Step 0: 下界分析（新增，Line B 的前置步骤）

计算三档下界，确定 gap A 和 gap B 的相对大小，决定本轮优化方向。

1. **Tier 1 (Roofline)**：估算模型关键操作的 Roofline 下界
   - 对模型中的主要 MatMul/Conv 操作，计算 max(FLOPs / peak_compute, weight_bytes / HBM_bandwidth)
   - 不需要精确——目的是给出物理极限的量级，不是精确预测
   - 对 batch=1 推理，大多数操作可能是 memory-bound，Roofline 可先以总权重大小 / HBM 带宽估算；须注明是否计入激活、格式转换、workspace 与权重复用假设

2. **Tier 2 (L0 Computing)**：从当前轮 L0 profiling 读取
   - = step_trace 的 Computing 值

3. **Tier 3 (对齐 wall-clock)**：从 benchmark 读取
   - 计时范围必须与 L0 step_trace 对齐（见标准化操作规范「计时口径对齐」）

4. **gap 分析**：
   - gap A = Tier2 - Tier1 → 设备执行相对理想参考下界的差距（普通调度优化通常难以影响）
   - gap B = max(0, Tier3 - Tier2) → 端到端关键路径残差（仅 host 调度/dispatch 候选的收益上界）
   - gap B / Tier3 = 仅优化 host 调度/dispatch 的收益上限占比；融合、图编译等可能同时降低 Tier 2，不受此项单独约束

5. **方向判定**：
   - 先重复测量 wall-clock 与 L0 Computing（建议各 ≥10 次），取中位数；以 MAD 或 P95-P50 估计测量波动 ε
   - |Tier3 - Tier2| ≤ ε → gap B = 0，标记“关键路径残差不可分辨”，不优先纯 host 优化
   - Tier3 < Tier2 - ε → 标记“采样扰动或计时口径异常”，不以该数据做归因
   - gap B / Tier3 > 15% 且跨轮稳定 → host-residual-dominated，优先验证 host 侧候选
   - gap B / Tier3 < 5% → 纯 host 优化空间很小；仍保留可能同时降低 Tier 2 的融合、图编译或算子替换方案
   - 5%~15% → 两条路都有空间，按具体候选的收益上限与验证成本排序
```

### 3.2 候选评估改为反事实收益上限

当前候选排序基于"该类浪费占总时间的比例"（self-time）。改为基于**反事实收益上限**：

```markdown
### 候选反事实收益上限估算

对每个候选，在实施前估算"若完全消除该浪费，端到端最多改善多少"：

1. **确定该候选消除的浪费量**：
   - 设备侧浪费（如 Transpose 占 L0_Computing 的 25.2%）→ 理想消除量 = 该类算子的 L0_Computing 占比；实际收益需验证
   - host 侧浪费（如 dispatch 开销）→ 消除量 ≤ gap B（不能超过 host-side 的端到端收益上界）

2. **Amdahl 约束**：
   - 若消除量为 p（占总时间比例），端到端加速上限 = 1/(1-p)
   - 但实际收益取决于该浪费是否在关键路径上

3. **异步流水线修正**：
   - 若消除的是 host 侧浪费且 gap B 很小 → 大部分被重叠，实际收益 << self-time 占比
   - 若消除的是设备侧浪费（Transpose/Cast 等冗余算子）→ 标注为消除该设备工作量的理想上界；融合后可能改变 kernel、layout 或重叠关系，需最小 A/B 验证

4. **标注反事实上限**：每条候选标注"反事实收益上限"（而非 self-time 占比），按此降序排列

例（ESM2 R2）：
- 候选A: 消除 rotary 的 462 个 Cast+Mul kernel → 设备侧，L0_Computing 占比 ~15% → 理想反事实上限 = 15% × L0_Computing / wall-clock ≈ 12%，需 A/B 验证
- 候选B: 消除 Module.__call__ dispatch → host 侧，L1 显示占比 63% → 但 gap B = 1.44ms (16.9%) → 反事实上限不超过 16.9%，且通常更低
```

### 3.3 停止条件自然涌现

三档下界框架使停止条件从定性判断变为定量计算：

```markdown
### 量化停止条件

每轮 ★C 时，基于三档下界判断是否继续：

1. **gap B 在测量波动内**（|wall-clock - L0_Computing| ≤ ε）：纯 host 调度优化空间不可分辨
   → 停止纯 host 优化支线；仍评估可同时缩小 Tier 2 的融合、图编译、布局调整、量化或算法替换
   
2. **gap A 显著且 gap B < 5%**：普通调度优化收益有限，优先评估融合、图编译、升级 CANN、量化或算法替换
   → 仅当这些路径均无可行候选时停止

3. **连续 2 轮 < max(2%, ε / Tier3) 改进**，且所有剩余候选的理想上限均不超过该门槛：边际收益不足
   → 停止

4. **候选穷尽**：所有候选被拒绝且无新候选
   → 停止

注意：终局判断前必须穷尽 NPU 融合算子库。融合算子可同时缩小 gap A（减少 kernel 数量、改善带宽利用率）和 gap B（减少 dispatch 次数）；图编译、布局调整与算子替换也可能影响 gap A。
```

---

## 4. 不可消除算子判定：gap A 的细分

`opt_line_research.md` §1 指出"模块耗时占比不等于优化该模块能带来的端到端收益"。在实践中，这体现为"有些算子耗时高但不可消除"。当前 skills 没有系统性的判定方法。

### 4.1 gap A 的可优化与不可优化部分

gap A = L0_Computing - Roofline 反映设备执行与理想参考下界的差距。但并非所有 kernel 都不可优化：

```
gap A 分解:
  ├── 可优化部分: 冗余算子（Transpose/Cast/格式转换）→ 消除后 L0_Computing 下降
  │   判定: 该算子是否由 CANN 运行时插入（TransData）/框架 codegen 生成（opt_einsum_fx 分解）
  │   方法: 融合算子替换 / 权重预转换 / 自定义实现
  │
  └── 难以在应用层消除部分: 核心计算算子（MatMul/Conv/FlashAttention）
      判定: 该算子是否是模型设计的必要计算
      方法: 通常通过 kernel 效率（CANN/OPP）、图编译、量化或换算法改善
```

### 4.2 不可消除算子判定流程

在归因层识别出"布局/格式转换"或"compute 饱和"类浪费后，执行：

1. **该算子是否有 NPU 融合算子覆盖？**（查 npu_operator_catalog.yaml + 已验证适用性矩阵）
   → 有：候选（替换维度），注意适用性约束
2. **该算子是否由 CANN 运行时自动插入？**（如 TransData for Conv2D）
   → 权重部分可预转换（FRACTAL_Z），输入/输出部分需图编译
3. **该算子是否由框架 codegen 生成？**（如 opt_einsum_fx 分解）
   → 可通过自定义实现替换
4. **该算子是否是核心计算？**（Conv2D, MatMul, FlashAttention）
   → 不可在 Python 层消除，属于 gap A 的不可优化部分

**关键原则**（7 模型实测教训）：
- 不能因"手动实现中有 Cast"就判定不可消除——npu_rotary_mul 在 NPU 内部处理 float32 精度，bit-identical 替代手动实现（ESM2 R2 误判，R4 纠正）
- 前人经验（"FRACTAL_Z 更慢"）可能基于不同模型——resnet50/wav2vec2 上慢，SD 上 -27ms，必须在本模型实测
- 判定为"不可消除"须记录到 evidence_db 的 platform_findings 中，附实测依据

---

## 5. 吸收已有补丁

以下改进被三档下界框架自然吸收，不再是独立补丁：

| 原编号 | 内容 | 如何被吸收 |
|--------|------|-----------|
| P0-1 | wall-clock 与 L0 口径对齐 | Tier 2→Tier 3 gap 计算的前提条件，框架内强制要求 |
| P0-4 | 量化停止条件 | gap B 在测量波动内且无可行 Tier 2 候选时停止相应支线，§3.3 自然涌现 |
| P0-5 | 异步流水线高估 host 开销 | gap B = max(0, wall-clock - L0_Computing) 替代 L0_Free，作为 host-side 收益上界，§2.3 |
| P2-5 | Roofline 下界 | 成为 Tier 1，框架的核心组成部分 |

以下改进与三档下界框架正交，仍作为独立补丁保留：

| 编号 | 内容 | 理由 |
|------|------|------|
| P0-2 | L0 采集格式禁止 export_chrome_trace | 基础设施问题，与下界分析无直接关系 |
| P0-3 | NPU 浮点非结合律约束 | 精度问题，影响优化可行性判断，不影响收益上限估算 |
| P0-6 | profiling 全覆盖要求 | 数据采集正确性问题，是所有分析的前提 |
| P1-1 | NPU 融合算子适用性矩阵 | 优化知识库，支撑 §4 的判定流程 |
| P1-2 | 逐层精度追踪 | 精度调试工具 |
| P1-3 | baseline 自一致性验证 | 精度验证前提 |
| P1-4 | flat forward 方法论 | gap B 的主要优化手段，补充 §3.2 的候选评估 |

---

## 6. 实施方案

### 6.1 改动清单

| 改动 | 文件 | 内容 | 优先级 |
|------|------|------|--------|
| 新增 Step 0 下界分析 | `02_bottleneck_analysis/SKILL.md` | Line B 流程 step 0 新增三档下界计算 + gap 分解 + 方向判定 | P0 |
| 候选评估改为反事实收益 | `02_bottleneck_analysis/references/profiling_to_action.md` | 候选排序节改为反事实收益上限估算（含 Amdahl + 异步流水线修正） | P0 |
| 停止条件量化 | `model_opt/SKILL.md` | 迭代退出条件改为基于 gap 的量化条件 | P0 |
| 归因层 host 残差修正 | `02_bottleneck_analysis/references/profiling_to_action.md` | host 侧类别的量化上限改为基于 gap B（host-side upper bound）而非 L1 host time，并处理测量波动 ε | P0 |
| 不可消除算子判定 | `02_bottleneck_analysis/references/profiling_to_action.md` | 归因层后新增不可消除算子判定流程 | P1 |
| 计时口径对齐 | `references/standardized_operations.md` | 新增计时口径对齐规范（Tier 2→3 gap 的前提） | P0 |
| L0 格式强制 | `01_preparation/references/profiling_collection.md` | 禁止 export_chrome_trace | P0 |
| 全覆盖要求 | `references/standardized_operations.md` | profiling 必须覆盖全部功能代码 | P0 |
| 浮点非结合律约束 | `03_optimization/references/equivalent_substitution.md` | 新增权重折叠/算子融合的精度安全边界 | P0 |
| 融合算子适用性矩阵 | `03_optimization/references/equivalent_substitution.md` | 新增已验证的 7 模型融合算子适用性表 | P1 |
| flat forward 方法论 | `03_optimization/references/eliminate_redundancy.md` | 框架调度层消除扩展为完整方法论 | P1 |
| 逐层精度追踪 | `04_accuracy_assurance/SKILL.md` | Level 3 增加逐层对比方法 | P1 |
| baseline 自一致性 | `04_accuracy_assurance/SKILL.md` | 基线管理增加自一致性验证 | P1 |

### 6.2 Phase 2 重构后的完整流程

```
Phase 2 Line B:
  Step 0: 下界分析（新增）
    ├─ Tier 1 (Roofline): 估算物理极限
    ├─ Tier 2 (L0 Computing): 从 profiling 读取
    ├─ Tier 3 (对齐 wall-clock): 从 benchmark 读取
    ├─ gap A = T2-T1（设备执行效率参考差距）, gap B = max(0, T3-T2)（关键路径残差）
    ├─ 重复测量并估计波动 ε；异常负 gap 不作归因
    └─ 方向判定: gap B/T3 > 15% 且稳定 → 优先 host 候选；< 5% → 优先能影响 Tier 2 的候选
    
  Step 1: run_analysis.py 报告 + L0/L1 交叉验证（已有）
  
  Step 2: 归因层推理（已有，修正 host-side 收益上界估算）
    └─ host 侧类别上限基于 gap B（host-side upper bound），不基于 L1 host time
    
  Step 3: 根因追踪（已有）
  
  Step 4: 候选评估（改为反事实收益上限）
    ├─ 每个候选标注反事实收益上限（Amdahl + 异步流水线修正）
    ├─ 不可消除算子判定（新增：区分 gap A 可优化 vs 不可优化）
    └─ 按反事实收益上限降序排列
    
  Step 5: ★A 用户确认（已有，候选已带反事实上限）
```

### 6.3 与 `opt_line_research.md` 的对应关系

| 论文概念 | 本方案实例化 | 落地程度 |
|---------|------------|---------|
| 三档上界 | Tier 1 Roofline（理想参考下界）/ Tier 2 L0 Computing / Tier 3 wall-clock | 完全落地，用已有指标；需声明 Roofline 的流量与复用假设 |
| 反事实收益 ΔT(a) | 候选反事实收益上限（Amdahl + 异步流水线修正） | 简化落地，不做离散事件模拟 |
| self-time ≠ 优化价值 | gap B 替代 L0_Free/L1 host time 作为 host-side 收益上界 | 完全落地，需以重复测量与 ε 处理异常 |
| 关键路径分析 | gap B/T3 比值判断 host-side 残差是否显著 | 简化落地，不做图级 CP 分析 |
| 数据移动建模 | Tier 1 Roofline 用 bytes/bandwidth（memory-bound 场景） | 部分落地，不做多级存储建模 |
| 不可消除算子 | gap A 的可优化/不可优化细分 | 完全落地 |
| opaque region 黑箱归因 | CANN kernel 作为 opaque，用 L0 Computing 做替代 cost model | 已有（L0 Computing 本身就是黑箱 cost model） |

**未落地的部分**（需 tensor IR 下沉、离散事件模拟等基础设施）：
- 张量程序层分析（循环/索引/归约级别的根因定位）
- 映射层 schedule 搜索
- 资源增强执行依赖图的精确构建
- 多优化间交互的精确估算（当前用 Amdahl 独立估算，不建模交互）

这些是 `opt_line_research.md` 描述的完整研究系统，超出当前 skills 的能力边界。本方案取其思想中**可用现有 profiling 数据实例化**的部分，将 skills 从"self-time 排序"升级为"gap 分解 + 反事实上限"。

## 1. 问题诊断：当前框架的根本缺陷

### 1.1 self-time 排序误导优化方向

当前 skills 的候选排序基于"该类浪费占总时间的比例"（profiling_to_action.md 归因层量化上限）。但 7 模型实测反复证明：**profiling 中的耗时占比不等于优化该部分能带来的端到端收益**。

最极端的案例是 SD：86000 个 Python frames 产生大量 host time（L1 显示 126ms `_local_scalar_dense`），但 L0_Free 仅 1.3%——异步流水线将这些 host 开销完全重叠。按 self-time 排序，host 开销是最大瓶颈；按反事实收益，消除 host 开销的收益 ≈ 0。

这正是 `opt_line_research.md` §3.3 指出的问题："非关键路径 region 即使耗时很长，也可能对延迟几乎无影响"。

### 1.2 缺少"能优化到什么程度"的量化框架

当前 skills 的"迭代退出条件"只有三条定性描述。Agent 不知道当前性能离物理极限有多远，也不知道剩余优化空间在哪一层。导致：
- ESM2 R3 判定终局，R4 又获得 +37.8% 提升
- SD 前 4 轮基于错误基线判定终局
- dinov2 R4 测试已知会失败的方案

`opt_line_research.md` §3 提出的三档上界框架可以直接解决这个问题——但要落地，需要用实测数据中已有的指标来实例化三档。

### 1.3 host 开销估算被 profiling 伪影扭曲

ESM2 R4 实测：L0 Free = 5.4ms 高估真实 host 开销 3.7x，L1 host time 高估 29x。当前归因层用 profiling 中的 host 时间占比作为"量化上限"，对 NPU 推理场景系统性高估。

---

## 2. 核心设计：三档下界 + gap 分解

### 2.1 三档下界的实例化

`opt_line_research.md` 提出三档上界（以时延表示则为下界）。用 7 模型实测中已有的指标实例化：

```
Tier 1: Roofline 下界（硬件物理极限）
  对每个关键操作: t >= max(FLOPs / peak_compute, bytes / HBM_bandwidth)
  全模型: sum of per-op Roofline
  数据来源: 模型结构（权重大小、FLOPs 估算）+ 硬件规格
  ESM2 实例: 1.4GB / 1.2TB/s ≈ 1.2ms

Tier 2: L0 Computing（设备执行下界）
  = 实际所有 kernel 执行时间之和
  数据来源: L0 profiling 的 step_trace_time.csv
  ESM2 实例: 7.1ms

Tier 3: 对齐 wall-clock（实际端到端时间）
  = 无 profiler 的真实性能（计时范围与 L0 step_trace 对齐）
  数据来源: wall-clock benchmark
  ESM2 实例: 8.54ms
```

### 2.2 gap 分解：优化方向由 gap 决定

两个 gap 各自指向不同的优化层：

```
Tier 1 ──── gap A ──── Tier 2 ──── gap B ──── Tier 3
  Roofline           L0 Computing           wall-clock

gap A = L0_Computing - Roofline  （kernel 实现效率 gap）
  含义: kernel 内部的 tiling、occupancy、小 kernel 无法饱和带宽等
  可优化性: 不在 Python 应用层可优化范围内（需 CANN/OPP 算子级优化或图编译）
  ESM2: 7.1 - 1.2 = 5.9ms (5.9x)，说明 kernel 效率极低但 Python 层无法改善

gap B = wall-clock - L0_Computing  （host 开销 gap）
  含义: Python dispatch、同步、内存分配等未被异步流水线重叠的开销
  可优化性: Python 层可优化（flat forward、QKV merge、融合算子等）
  ESM2: 8.54 - 7.1 = 1.44ms (20%)，Python 层还有优化空间
  SD: 226 - 203 = 23ms (11%)，空间较小
  resnet50: 3.41 - 3.66 = -0.25ms，wall-clock < L0_Computing（profiler 自身开销）
```

**关键原则**：gap B 是 Python 层优化的收益上限。当 wall-clock ≈ L0_Computing（gap B → 0）时，Python 层优化空间耗尽，只能通过缩小 gap A（换算子/图编译/量化）继续优化。

### 2.3 gap B 的修正：异步流水线的影响

`opt_line_research.md` §3.3 指出 self-time ≠ 关键路径贡献。在 NPU 上，异步流水线（`TASK_QUEUE_ENABLE=2`）使大部分 host 开销被 device compute 重叠——这些 host 开销虽然出现在 profiling 中，但不在关键路径上。

```
真实 host 开销 = wall-clock - L0_Computing = gap B
L0_Free = host 开销的上界（含被重叠的部分），高估约 3.7x
L1 host time = profiler barrier 注入后的 host 时间，高估约 29x
```

**结论**：gap B（= wall-clock - L0_Computing）是 host 开销的可靠估计，L0_Free 和 L1 host time 不是。归因层中 host 侧类别的"量化上限"应基于 gap B，而非 profiling 中的 host 时间占比。

### 2.4 gap A 的诊断价值：理解 batch=1 推理的本质

`opt_line_research.md` §2.2 强调"仅有 FLOPs 不足以解释推理性能"。7 模型实测验证了这一点：

ESM2 Roofline 下界 1.2ms vs L0 Computing 7.1ms（5.9x gap A）。Roofline 分析显示 batch=1 推理是 memory-bound（权重加载主导），cube utilization 仅 18.4%。这解释了：
- 序列长度 40→300aa，wall-clock 仅从 8.8→8.9ms（权重不变，memory-bound 不随 seq_len 变化）
- gap A 的大部分来自小 kernel 无法饱和 HBM 带宽（权重碎片化为多个小 MatMul）
- 减少 kernel 数量（融合算子）可以部分缩小 gap A——不是通过减少 FLOPs，而是通过减少 kernel launch 间隙和提升单次 kernel 的带宽利用率

---

## 3. 流程重构：Phase 2 新增"下界分析"前置步骤

### 3.1 新增 Phase 2 Step 0：下界分析

在 `run_analysis.py` 报告分析之前，先计算三档下界，确定优化方向：

```markdown
### Step 0: 下界分析（新增，Line B 的前置步骤）

计算三档下界，确定 gap A 和 gap B 的相对大小，决定本轮优化方向。

1. **Tier 1 (Roofline)**：估算模型关键操作的 Roofline 下界
   - 对模型中的主要 MatMul/Conv 操作，计算 max(FLOPs / peak_compute, weight_bytes / HBM_bandwidth)
   - 不需要精确——目的是给出物理极限的量级，不是精确预测
   - 对 batch=1 推理，大多数操作是 memory-bound，Roofline ≈ 总权重大小 / HBM 带宽

2. **Tier 2 (L0 Computing)**：从当前轮 L0 profiling 读取
   - = step_trace 的 Computing 值

3. **Tier 3 (对齐 wall-clock)**：从 benchmark 读取
   - 计时范围必须与 L0 step_trace 对齐（见标准化操作规范「计时口径对齐」）

4. **gap 分析**：
   - gap A = Tier2 - Tier1 → kernel 实现效率 gap（Python 层不可优化）
   - gap B = Tier3 - Tier2 → host 开销 gap（Python 层可优化）
   - gap B / Tier3 = Python 层优化的收益上限占比

5. **方向判定**：
   - gap B / Tier3 > 15% → host-bound，优先 host 侧优化（flat forward、合并、融合算子）
   - gap B / Tier3 < 5% → host 开销已极小，只能缩小 gap A（换算子/图编译/量化）
   - 5%~15% → 两条路都有空间，按具体候选的收益上限排序
```

### 3.2 候选评估改为反事实收益上限

当前候选排序基于"该类浪费占总时间的比例"（self-time）。改为基于**反事实收益上限**：

```markdown
### 候选反事实收益上限估算

对每个候选，在实施前估算"若完全消除该浪费，端到端最多改善多少"：

1. **确定该候选消除的浪费量**：
   - 设备侧浪费（如 Transpose 占 L0_Computing 的 25.2%）→ 消除量 = 该类算子的 L0_Computing 占比
   - host 侧浪费（如 dispatch 开销）→ 消除量 ≤ gap B（不能超过真实 host 开销）

2. **Amdahl 约束**：
   - 若消除量为 p（占总时间比例），端到端加速上限 = 1/(1-p)
   - 但实际收益取决于该浪费是否在关键路径上

3. **异步流水线修正**：
   - 若消除的是 host 侧浪费且 gap B 很小 → 大部分被重叠，实际收益 << self-time 占比
   - 若消除的是设备侧浪费（Transpose/Cast 等冗余算子）→ 不受异步流水线影响，实际收益 ≈ self-time 占比

4. **标注反事实上限**：每条候选标注"反事实收益上限"（而非 self-time 占比），按此降序排列

例（ESM2 R2）：
- 候选A: 消除 rotary 的 462 个 Cast+Mul kernel → 设备侧，L0_Computing 占比 ~15% → 反事实上限 = 15% × L0_Computing / wall-clock ≈ 12%
- 候选B: 消除 Module.__call__ dispatch → host 侧，L1 显示占比 63% → 但 gap B = 1.44ms (17%) → 反事实上限 ≤ 17% × overlap_factor
```

### 3.3 停止条件自然涌现

三档下界框架使停止条件从定性判断变为定量计算：

```markdown
### 量化停止条件

每轮 ★C 时，基于三档下界判断是否继续：

1. **gap B ≈ 0**（wall-clock / L0_Computing < 1.1）：host 开销已极小，Python 层优化空间耗尽
   → 除非有缩小 gap A 的手段（图编译/量化/换算法），否则停止
   
2. **gap A / Tier1 > 5x 且 gap B < 5%**：性能受限于 kernel 实现效率，Python 层无法改善
   → 停止，建议升级 CANN 或使用图编译

3. **连续 2 轮 < 2% 改进**：边际收益不足
   → 停止

4. **候选穷尽**：所有候选被拒绝且无新候选
   → 停止

注意：终局判断前必须穷尽 NPU 融合算子库。融合算子可同时缩小 gap A（减少 kernel 数量提升带宽利用率）和 gap B（减少 dispatch 次数），是 Python 层唯一能影响 gap A 的手段。
```

---

## 4. 不可消除算子判定：gap A 的细分

`opt_line_research.md` §1 指出"模块耗时占比不等于优化该模块能带来的端到端收益"。在实践中，这体现为"有些算子耗时高但不可消除"。当前 skills 没有系统性的判定方法。

### 4.1 gap A 的可优化与不可优化部分

gap A = L0_Computing - Roofline 来自 kernel 实现效率。但并非所有 kernel 都不可优化：

```
gap A 分解:
  ├── 可优化部分: 冗余算子（Transpose/Cast/格式转换）→ 消除后 L0_Computing 下降
  │   判定: 该算子是否由 CANN 运行时插入（TransData）/框架 codegen 生成（opt_einsum_fx 分解）
  │   方法: 融合算子替换 / 权重预转换 / 自定义实现
  │
  └── 不可优化部分: 核心计算算子（MatMul/Conv/FlashAttention）
      判定: 该算子是否是模型设计的必要计算
      方法: 只能通过缩小 gap A 的 kernel 效率（CANN/OPP 级优化）或缩小 Tier1（量化/换算法）
```

### 4.2 不可消除算子判定流程

在归因层识别出"布局/格式转换"或"compute 饱和"类浪费后，执行：

1. **该算子是否有 NPU 融合算子覆盖？**（查 npu_operator_catalog.yaml + 已验证适用性矩阵）
   → 有：候选（替换维度），注意适用性约束
2. **该算子是否由 CANN 运行时自动插入？**（如 TransData for Conv2D）
   → 权重部分可预转换（FRACTAL_Z），输入/输出部分需图编译
3. **该算子是否由框架 codegen 生成？**（如 opt_einsum_fx 分解）
   → 可通过自定义实现替换
4. **该算子是否是核心计算？**（Conv2D, MatMul, FlashAttention）
   → 不可在 Python 层消除，属于 gap A 的不可优化部分

**关键原则**（7 模型实测教训）：
- 不能因"手动实现中有 Cast"就判定不可消除——npu_rotary_mul 在 NPU 内部处理 float32 精度，bit-identical 替代手动实现（ESM2 R2 误判，R4 纠正）
- 前人经验（"FRACTAL_Z 更慢"）可能基于不同模型——resnet50/wav2vec2 上慢，SD 上 -27ms，必须在本模型实测
- 判定为"不可消除"须记录到 evidence_db 的 platform_findings 中，附实测依据

---

## 5. 吸收已有补丁

以下改进被三档下界框架自然吸收，不再是独立补丁：

| 原编号 | 内容 | 如何被吸收 |
|--------|------|-----------|
| P0-1 | wall-clock 与 L0 口径对齐 | Tier 2→Tier 3 gap 计算的前提条件，框架内强制要求 |
| P0-4 | 量化停止条件 | gap B → 0 即停止条件，§3.3 自然涌现 |
| P0-5 | 异步流水线高估 host 开销 | gap B = wall-clock - L0_Computing 替代 L0_Free 作为 host 开销估计，§2.3 |
| P2-5 | Roofline 下界 | 成为 Tier 1，框架的核心组成部分 |

以下改进与三档下界框架正交，仍作为独立补丁保留：

| 编号 | 内容 | 理由 |
|------|------|------|
| P0-2 | L0 采集格式禁止 export_chrome_trace | 基础设施问题，与下界分析无直接关系 |
| P0-3 | NPU 浮点非结合律约束 | 精度问题，影响优化可行性判断，不影响收益上限估算 |
| P0-6 | profiling 全覆盖要求 | 数据采集正确性问题，是所有分析的前提 |
| P1-1 | NPU 融合算子适用性矩阵 | 优化知识库，支撑 §4 的判定流程 |
| P1-2 | 逐层精度追踪 | 精度调试工具 |
| P1-3 | baseline 自一致性验证 | 精度验证前提 |
| P1-4 | flat forward 方法论 | gap B 的主要优化手段，补充 §3.2 的候选评估 |

---

## 6. 实施方案

### 6.1 改动清单

| 改动 | 文件 | 内容 | 优先级 |
|------|------|------|--------|
| 新增 Step 0 下界分析 | `02_bottleneck_analysis/SKILL.md` | Line B 流程 step 0 新增三档下界计算 + gap 分解 + 方向判定 | P0 |
| 候选评估改为反事实收益 | `02_bottleneck_analysis/references/profiling_to_action.md` | 候选排序节改为反事实收益上限估算（含 Amdahl + 异步流水线修正） | P0 |
| 停止条件量化 | `model_opt/SKILL.md` | 迭代退出条件改为基于 gap 的量化条件 | P0 |
| 归因层 host 开销修正 | `02_bottleneck_analysis/references/profiling_to_action.md` | host 侧类别的量化上限改为基于 gap B 而非 L1 host time | P0 |
| 不可消除算子判定 | `02_bottleneck_analysis/references/profiling_to_action.md` | 归因层后新增不可消除算子判定流程 | P1 |
| 计时口径对齐 | `references/standardized_operations.md` | 新增计时口径对齐规范（Tier 2→3 gap 的前提） | P0 |
| L0 格式强制 | `01_preparation/references/profiling_collection.md` | 禁止 export_chrome_trace | P0 |
| 全覆盖要求 | `references/standardized_operations.md` | profiling 必须覆盖全部功能代码 | P0 |
| 浮点非结合律约束 | `03_optimization/references/equivalent_substitution.md` | 新增权重折叠/算子融合的精度安全边界 | P0 |
| 融合算子适用性矩阵 | `03_optimization/references/equivalent_substitution.md` | 新增已验证的 7 模型融合算子适用性表 | P1 |
| flat forward 方法论 | `03_optimization/references/eliminate_redundancy.md` | 框架调度层消除扩展为完整方法论 | P1 |
| 逐层精度追踪 | `04_accuracy_assurance/SKILL.md` | Level 3 增加逐层对比方法 | P1 |
| baseline 自一致性 | `04_accuracy_assurance/SKILL.md` | 基线管理增加自一致性验证 | P1 |

### 6.2 Phase 2 重构后的完整流程

```
Phase 2 Line B:
  Step 0: 下界分析（新增）
    ├─ Tier 1 (Roofline): 估算物理极限
    ├─ Tier 2 (L0 Computing): 从 profiling 读取
    ├─ Tier 3 (对齐 wall-clock): 从 benchmark 读取
    ├─ gap A = T2-T1 (kernel 效率), gap B = T3-T2 (host 开销)
    └─ 方向判定: gap B/T3 > 15% → host-bound; < 5% → compute-bound
    
  Step 1: run_analysis.py 报告 + L0/L1 交叉验证（已有）
  
  Step 2: 归因层推理（已有，修正 host 开销估算）
    └─ host 侧类别上限基于 gap B，不基于 L1 host time
    
  Step 3: 根因追踪（已有）
  
  Step 4: 候选评估（改为反事实收益上限）
    ├─ 每个候选标注反事实收益上限（Amdahl + 异步流水线修正）
    ├─ 不可消除算子判定（新增：区分 gap A 可优化 vs 不可优化）
    └─ 按反事实收益上限降序排列
    
  Step 5: ★A 用户确认（已有，候选已带反事实上限）
```

### 6.3 与 `opt_line_research.md` 的对应关系

| 论文概念 | 本方案实例化 | 落地程度 |
|---------|------------|---------|
| 三档上界 | Tier 1 Roofline / Tier 2 L0 Computing / Tier 3 wall-clock | 完全落地，用已有指标 |
| 反事实收益 ΔT(a) | 候选反事实收益上限（Amdahl + 异步流水线修正） | 简化落地，不做离散事件模拟 |
| self-time ≠ 优化价值 | gap B 替代 L0_Free/L1 host time 作为 host 开销估计 | 完全落地 |
| 关键路径分析 | gap B/T3 比值判断 host 开销是否在关键路径上 | 简化落地，不做图级 CP 分析 |
| 数据移动建模 | Tier 1 Roofline 用 bytes/bandwidth（memory-bound 场景） | 部分落地，不做多级存储建模 |
| 不可消除算子 | gap A 的可优化/不可优化细分 | 完全落地 |
| opaque region 黑箱归因 | CANN kernel 作为 opaque，用 L0 Computing 做替代 cost model | 已有（L0 Computing 本身就是黑箱 cost model） |

**未落地的部分**（需 tensor IR 下沉、离散事件模拟等基础设施）：
- 张量程序层分析（循环/索引/归约级别的根因定位）
- 映射层 schedule 搜索
- 资源增强执行依赖图的精确构建
- 多优化间交互的精确估算（当前用 Amdahl 独立估算，不建模交互）

这些是 `opt_line_research.md` 描述的完整研究系统，超出当前 skills 的能力边界。本方案取其思想中**可用现有 profiling 数据实例化**的部分，将 skills 从"self-time 排序"升级为"gap 分解 + 反事实上限"。
