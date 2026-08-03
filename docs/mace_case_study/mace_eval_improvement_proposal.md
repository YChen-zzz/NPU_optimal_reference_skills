# 基于 MACE 实测的 Skills 改进方案

> 以 MACE-MP-0 Large (L2) 在 Ascend910 上的推理优化（12 采纳 / 7 拒绝 / 9 evidence_db / 3 轮交互）为实测依据，评估 model_opt 各设计点的实效，提出 5 项改进。
> 范围：仅针对 model_opt 当前实现 + MACE 案例，不考虑历史 docs 记录。

## 1. 统一 Profiling 分析入口

### 1.1 问题

当前 02_bottleneck_analysis/SKILL.md 要求 Agent 依次手动运行 8 个 parse 脚本。实测中：
- Session 3 只运行了 5/8 个脚本（跳过了 parse_trace_view、parse_operator_memory、parse_communication）
- 脚本输出分散在 8 份独立文本中，Agent 需要自行在上下文中维护全部输出
- bound reference 的加载也依赖 Agent 记忆——跑完脚本后"必须加载"对应 reference 的规则常被跳过

### 1.2 方案：统一入口脚本，不组合只重组

新增 `02_bottleneck_analysis/scripts/run_analysis.py`，作为 Phase 2 的唯一分析入口：

```bash
python run_analysis.py <l1_profiling_dir> [--l0-dir <l0_dir>] [--rank N] [--output report.txt]
```

**设计原则**：
- 内部依次调用全部 8 个 parse 脚本（多卡时追加 communication），捕获各自的 stdout
- **不做任何跨脚本信息组合、不做 pattern 匹配、不设人为阈值**——每个脚本的输出原样保留
- 只对输出做**归类和排序**：用 section header 将 8 份输出组织成一个有逻辑层次的单文件报告
- Agent 拿到一份完整报告后，自行做跨脚本推理

**报告结构**（按 Agent 推理的自然顺序组织）：

```
=== Phase 2 Profiling 分析报告 ===
L1 目录: <path>
L0 参考目录: <path or "未提供">

--- A. 全局视角 ---
[parse_step_trace.py L1 输出]
[parse_step_trace.py L0 输出]（如提供 --l0-dir，用于 L0/L1 交叉验证，见 §2）

--- B. 设备侧：算子分布 ---
[parse_op_statistic.py 输出]

--- C. 设备侧：Kernel 级详情 ---
[parse_kernel_details.py 输出]

--- D. Host-Device 交互 ---
[parse_trace_view.py 输出]

--- E. 源码定位 ---
[parse_operator_details.py 输出]

--- F. 内存 ---
[parse_memory_record.py 输出]
[parse_operator_memory.py 输出]

--- G. CANN 运行时 ---
[parse_api_statistic.py 输出]

--- H. 通信（仅多卡）---
[parse_communication.py 输出]
```

**为什么不组合**：MACE 实测表明，高价值的优化发现都来自 Agent 对多个脚本输出的交叉推理（如 "Transpose 25.2% from op_statistic" + "73/77 无 Call Stack from operator_details" + "读 e3nn 源码发现 opt_einsum_fx 分解"）。这种推理是 Agent 的核心能力，人为设定阈值和 pattern 反而会限制发现范围或产生误判。统一入口解决的是"Agent 漏跑脚本"和"输出太分散"问题，不应越界解决"Agent 不会推理"问题。

### 1.3 对 SKILL.md 的改动

`02_bottleneck_analysis/SKILL.md` 的「强制脚本检查清单」改为：

```
Phase 2 Line B 分析的入口为 `scripts/run_analysis.py`，一次性运行全部 parse 脚本并输出归类报告。
运行后，Agent 须阅读完整报告，对报告中每个脚本的 DEFINITE 信号和 WARNING 警告执行根因追踪。

如需对单个脚本做 --filter 深入查询（如 parse_operator_details --filter Transpose 获取 Call Stack），
可单独调用对应脚本。单脚本深入查询是 run_analysis.py 的补充，不替代它。
```

「必读参考」的绑定关系不变——Agent 在阅读报告各 section 后，仍须按原绑定表加载对应 reference。但触发方式从"运行脚本后加载"变为"阅读报告对应 section 后加载"。

---

## 2. L0/L1 交叉验证作为 Phase 2 强制前置步骤

### 2.1 问题

MACE 实测中，L1 profiler 的 barrier 注入造成假的 "SEVERE Host-Bound" 信号（L1 利用率 10.1%），实际 L0 显示 89.8% 利用率。Agent 花了大量时间排除这个假信号。`profiling_to_action.md` 模式 3（差异对比）提到了 "L0 vs L1 分离 profiler 注入开销"，但只是五种分析模式之一，Agent 不一定选择使用。

### 2.2 方案

在 `02_bottleneck_analysis/SKILL.md` 的 Line B 流程中，**在跑完 run_analysis.py 后、开始根因追踪前**，插入一个强制步骤：

```
L0/L1 交叉验证（强制，不可跳过）

对比 run_analysis.py 报告 A 节中的 L0 和 L1 两份 step_trace 输出：

1. 记录 L0 Computing% / Free% / Utilization
2. 记录 L1 Computing% / Free% / Utilization
3. 若 L1 Utilization 显著低于 L0（如 L1 < L0 - 20pp），标注"profiler 伪影警告"：
   L1 的低利用率可能由 profiler barrier 注入导致，不代表真实瓶颈类型。
4. 瓶颈类型判定以 L0 的利用率为准；L1 的算子级数据（op_statistic、kernel_details 等）
   仍然有效，但 step_trace 的 host time / Free time 不可直接作为瓶颈判据。

交叉验证结论记入分析报告开头，后续所有根因追踪和候选排序均基于此结论。
```

### 2.3 L0 数据来源

| 轮次 | L0 来源 | L1 来源 |
|------|---------|---------|
| 第 0 轮 | Phase 1 采集的 L0 基线 | 第 0 轮 Phase 2 采集的 L1 |
| 第 i 轮（i≥1） | 第 i-1 轮 Phase 4 采集的 L0（阶段末收益验证用） | 第 i 轮 Phase 2 采集的 L1 |

run_analysis.py 的 `--l0-dir` 参数指向上述 L0 目录。如 L0 目录不可用（首次优化前未采集 L0），标注"L0 不可用，L1 结论未经交叉验证，须谨慎"。

---

## 3. 四维度"正交性"声明修正

### 3.1 问题

`03_optimization/SKILL.md` 通用原则中声明："优化方向正交性：四维度是正交的。一个方向的失败不影响其他方向。"

MACE 实测证伪了此声明。7 个被拒绝方案中至少 3 个的失败原因是"去重"或"替换"改变了操作序列，破坏了 NPU 异步流水线（`TASK_QUEUE_ENABLE=2`）的自动延迟掩盖——即"去重/替换"方向与"掩盖"方向在 NPU 上强耦合。

### 3.2 方案

将 `03_optimization/SKILL.md` 的"优化方向正交性"段落替换为：

```markdown
- **四维度逻辑正交 + NPU 硬件耦合**：四个维度（去重/复用/掩盖/替换）在逻辑层面正交
  ——它们各自回答不同的优化问题（见上表）。但在 NPU 上，维度间通过**内存分配模式**和
  **异步流水线**产生硬件耦合：任何改变操作数量或操作顺序的优化（去重/替换），都可能改变
  NPU 异步流水线的重叠模式，导致预期外的性能回退。

  **实践规则**：
  - "一个方向的逻辑失败不影响其他方向"——逻辑层面仍然成立，不要因一个替换方案失败
    就放弃独立的去重方案。
  - "但任何改变操作序列的优化必须用 L0 端到端 benchmark 验证"——不能只看 profiling 中
    的算子级数据，因为异步流水线的重叠效果只在端到端时间中体现。
  - 若优化导致端到端回退但 profiling 显示算子级改善，根因是异步流水线耦合——记录此发现，
    可考虑通过自适应阈值（如不同输入规模用不同实现）规避。
```

---

## 4. 方向放弃标准分级

### 4.1 问题

`03_optimization/SKILL.md` 的"方向放弃标准"要求所有方向放弃前都满足：≥2 种实现、每种记录 timing/baseline 对比、微基准→小样本→全量三步验证。MACE 实测中 7 个拒绝方案无一满足此标准——Agent 对所有方向都绕过了标准，包括那些本应深入探索的方向（如 FastLinear 仅调试后说"调试成本太高"就放弃）。标准过于刚性，导致失去约束力。

### 4.2 方案：按改动复杂度分三级

将"方向放弃标准"替换为：

```markdown
### 方向放弃标准（分级）

放弃一个优化方向前，须满足该方向所属级别的全部条件：

**Level 1 — 微调级**（改参数 / flag / 跳过单步操作 / 1-3 行代码改动）
- A/B benchmark 对比（优化前 vs 优化后，同一输入，≥3 次取中位数）
- 记录：改动描述、耗时对比、慢/无效的原因
- 如回退，一句话说明原因即可（如"异步流水线耦合导致回退 +0.8ms"）

**Level 2 — 替换级**（换等价实现 / 融合算子 / 改数据流路径）
- 至少 1 种实现（框架提供或自定义均可）
- Level 1 精度验证（代表性样本，快速检查）
- A/B benchmark 对比
- 记录：实现描述、精度结果、耗时对比、失败原因归类（概念错误 / 框架 overhead / 硬件不友好 / 异步流水线耦合）
- 如仅有框架实现且失败，须尝试 1 种自定义实现后才可放弃

**Level 3 — 架构级**（重写算子 / 改变计算图结构 / 需要 checkpoint 重映射）
- 至少 2 种实现（1 框架 + 1 自定义/bare）
- 每种实现经过微基准 → 小样本 → 全量三步验证
- 每步记录具体数值（微基准 diff、小样本精度、全量耗时）
- 失败原因归类 + 是否尝试过调整参数重试
- 适用原"方向放弃标准"的全部要求

**通用规则**（所有级别）：
- "我觉得不会通过"（未实测）始终为不合理放弃原因
- 优化方向正交性（逻辑层面）仍然成立——一个方向的失败不连带放弃独立方向
```

---

## 5. 断桥降级路径与 Line A 的显式连接

### 5.1 问题分析

MACE 实测中，根因追踪的核心瓶颈是 Call Stack 桥断裂——73/77 个 Transpose 算子的 Call Stack 为 "(no stack)"，因为它们是 e3nn codegen + opt_einsum_fx 动态生成的，不在 Python 调用栈中。

最初的分析建议"新增第六座桥（框架源码分析）"，但经审视，这与 Line A 的"穿透框架层"方法论本质相同——都是"读源码定位根因"。五座桥的定义是"profiling 元数据 → 源码位置"的映射工具（Call Stack、Input Shapes、AI Core 指标、Accelerator Core、下发时序），它们利用 profiling 文件中的特定字段做精确定位。"读框架源码"不依赖任何 profiling 字段，不具备桥的属性，不应作为第六座桥。

### 5.2 方案：将断桥降级路径显式连接到 Line A

修改 `profiling_to_source.md` 的「断桥时的降级路径」表，将 Call Stack 断桥的降级做法从模糊的"人工推断"改为明确指向 Line A：

```markdown
| 缺失字段 | 现象 | 降级做法 |
|---------|------|---------|
| Call Stack 为 "(no stack)" | 算子由框架 codegen 动态生成（如 e3nn、opt_einsum_fx、torch.compile），不在 Python 调用栈中 | **切换到 Line A 方法论**：用 Line A 的"穿透框架层"方法，沿算子类型 → 框架 codegen 入口 → 代码生成逻辑 → 生成出的算子序列，追溯该算子是哪段框架代码生成的、生成逻辑是否可替换。结论标注为"Line A 推断"。 |
```

同时，在 `02_bottleneck_analysis/SKILL.md` 的根因追踪执行规则中增加一条：

```markdown
- 当 Call Stack 桥断裂（"(no stack)"）时，该信号的根因追踪自动从 Line B 切换到 Line A。
  此时该信号的追踪产出格式不变，但"使用的桥梁"列标注为"断桥 → Line A 穿透框架层"。
  这不是降级——Line A 的源码分析能力天然覆盖 codegen 生成的算子，只是定位路径从
  "profiling 字段映射"变为"源码结构追踪"。
```

### 5.3 本质关系

五座桥是 Line B（profiling 驱动）的工具。当桥断裂时，自然的过渡是切换到 Line A（源码驱动）。这个过渡点本身就是双线分析模型的设计意图——Line B 覆盖可见瓶颈（有 profiling 信号的），Line A 覆盖 profiling 盲区（包括 codegen 算子这种"桥不通"的情况）。不需要新增桥梁，只需要让这个过渡显式化，让 Agent 知道"桥断了不是死路，是切换到 Line A 的信号"。

---

## 6. 落地改动清单

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `02_bottleneck_analysis/scripts/run_analysis.py` | 新增 | 统一入口脚本，调用全部 parse 脚本，输出归类报告 |
| `02_bottleneck_analysis/SKILL.md` | 修改 | 脚本检查清单改为 run_analysis.py 入口；增加 L0/L1 交叉验证步骤；增加断桥→Line A 切换规则 |
| `02_bottleneck_analysis/references/profiling_to_source.md` | 修改 | 断桥降级路径表：Call Stack 断桥的降级做法指向 Line A 穿透框架层 |
| `03_optimization/SKILL.md` | 修改 | "优化方向正交性"改为"逻辑正交 + NPU 硬件耦合"；"方向放弃标准"改为三级分级 |
