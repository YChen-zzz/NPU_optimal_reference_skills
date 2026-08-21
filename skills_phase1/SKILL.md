---
name: npu-supernode-tuning
description: NPU 训练性能优化。以 GPU compiled evidence 为参照，自动拆分计算图为 Supernode，逐节点穷尽优化方法并用多卡 ablation 验证。适用于 GPU→NPU 迁移调优、NPU 原生训练加速。
---

# NPU Supernode 调优

## 核心原则

1. **逐 Supernode 做到极致。** 一个 SN 的所有优化层级测完再做下一个。
2. **GPU 是参照不是答案。** 对齐优化意图，用 NPU 方法实现。
3. **Forward + backward + 真实 shape 才算有效 benchmark。**
4. **单机 benchmark ≠ 多卡增益。** 必须 ablation 验证。
5. **改动成本越低越优先尝试。**
6. **不因单次失败放弃。** 追根因，换方式再试。

---

## Step 1: Baseline + GPU 参照

### 1.1 NPU Baseline

运行原始脚本，记录:
- `train_time`, `step_avg`, `val_loss`, `peak_memory`
- 多 regime 时分别记录各 regime 的 step 耗时

### 1.2 GPU Teacher 信息提取

从 GPU evidence pack 中提取（详见 [references/gpu_teacher_reading.md](references/gpu_teacher_reading.md)）:

若没有现成 evidence pack，需要在 GPU 机器上采集（详见 [references/gpu_teacher_collection.md](references/gpu_teacher_collection.md)）。

- **Compile 前 FX graph**: 每个 op 的 dtype、shape、调用顺序
- **Compile 后 IR (ir_post_fusion.txt)**: 哪些 ops 被融合为一个 kernel
- **Profiling summary**: 各类 kernel 的时间占比
- **GPU source vs NPU source 差异**: 精度路径、API、layout

**产出**: 1 个 markdown 对比表:
```
| 功能 | GPU (compile 后) | NPU 当前 | 差异类型 |
```

---

## Step 2: Supernode 划分 + Lab 骨架创建

### 划分方法

详见 [references/supernode_analysis.md](references/supernode_analysis.md)

**GPU IR 快速提取** (不需要读完 13000+ 行):
```bash
# 提取所有 fusion group 概览
grep -E "^op[0-9]|FusedSchedulerNode|ExternKernel" ir_post_fusion.txt | head -80

# 提取不可融合的大算子
grep "ExternKernel" ir_post_fusion.txt

# 提取特定范围的 fusion detail
sed -n '<start>,<end>p' ir_post_fusion.txt
```

**核心流程**:

1. 用上述命令提取 GPU fusion groups 概览
2. 按**语义功能**分组：服务同一计算目的的相邻 fusion groups 合为一个 Supernode
3. 在 NPU source 中标注每个 Supernode 对应的代码范围
4. 估算每个 SN 占 step 时间的比例 → 确定优化优先级

### ⚠️ 强制产出: Lab 骨架文件

Step 2 的产出**不是 markdown 表格**，而是为每个 SN 创建 benchmark 骨架：

```bash
mkdir -p benchmarks/supernodes
```

为每个 SN 创建 `benchmarks/supernodes/sn_<name>.py`，结构如下：

```python
"""
SN-<NAME> Supernode Lab
========================
GPU fusion groups: op<X>_op<Y>_... (N ops → 1 kernel)
GPU pre-compile ops:
  - <op1>: <dtype> <shape>
  - <op2>: ...
GPU post-compile result:
  - <what got fused/eliminated>
NPU current implementation:
  - <file>:<line> <op1>
  - <file>:<line> <op2>
  - ...
Identified gaps:
  - precision: <GPU dtype vs NPU dtype>
  - layout: <transpose differences>
  - API params: <GPU passes X, NPU doesn't>
  - redundancy: <repeated/unnecessary ops>
"""

# ===== Real shapes from training (all regimes) =====
SHAPES = {
    "regime_0": ...,
    "regime_1": ...,
    "regime_2": ...,
}

# ===== B0: Control (current NPU implementation) =====
def B0_control():
    pass  # TODO: extract from training code

# ===== Optimization candidates =====
# TODO L0: <from API param gaps above>
# TODO L1: <from redundancy gaps above>
# TODO L2: <from dir(torch_npu) search>
# TODO L3: <from layout/expression gaps above>
# TODO L4: <compile scope candidates>

# ===== Benchmark harness =====
# TODO: warmup + timing + grad check
```

**这就是 Step 2 的全部产出。** 分析直接嵌入代码注释里——不存在"先写表格再写代码"两步。

**门禁规则**: 每个 SN 的 lab 骨架文件中，docstring 的 "GPU fusion groups" 和 "Identified gaps" 必须已填写，才能开始实现 TODO 中的优化方案。如果这些注释为空，说明 GPU 分析被跳过了——回去补。

---

## Step 3: 逐 Supernode 优化

在 Step 2 创建的 Lab 骨架基础上，逐个 SN 填充并运行。

### 3a. 填充 Lab 骨架的 GPU 对齐信息

打开当前 SN 的 `sn_<name>.py`，填写 docstring 中的所有字段：

**精度对齐**:
- GPU 在该 SN 用什么 dtype（input/compute/accumulator/output/saved_for_backward）？
- NPU 是否多了 cast（.float()/.type_as()/.to()）？
- 哪些 cast 是精度必需的，哪些是移植遗留？
- ⚡ **常见陷阱**: GPU→NPU 移植时经常在 loss/norm 计算前插入 `.float()` 但 GPU 训练实际用 bf16。必须读 GPU source 确认 training path 的真实 dtype — 不要假设 f32 是必需的。移除移植遗留的 `.float()` 同时还能省去一次全 tensor 的 cast 开销。

**Layout 对齐**:
- 有无不必要的 transpose / .contiguous() / .view() vs .reshape()？
- F.linear(x, w.T) 是否可改为 torch.matmul(x, w) 避免 double-transpose？

**冗余操作**:
- 有无重复计算（同一输入多次调用相同函数）？
- 有无不必要的 tensor 分配（intermediate 可否复用/in-place）？
- 有无 host-device 同步点（.item(), .tolist(), CPU scalar → device）？

**API 参数**:
- 对该 SN 涉及的每个 NPU API，打印其完整签名/docstring，**逐个参数**检查是否有未使用的。
- 对该 SN 的 GPU 等价 API，列出其全部参数。做**逐参数对照映射**：GPU 有但 NPU 没传的，查 NPU API 是否有同语义的参数（名称可能完全不同）。
- 特别注意：窗口/稀疏/mask 相关参数、精度控制参数、计算范围限制参数 — 这类参数不传时通常走最慢的默认路径。

### 3a-bis. 从 Gap 推导候选方案

docstring 填完后，将每个 gap 映射为 TODO 候选：

| Gap 类型 | 映射到 |
|---------|--------|
| GPU API 有参数 NPU 没传 | → TODO L0 |
| NPU 多了 cast/sync/重复计算 | → TODO L1 |
| GPU 用了 fused API | → TODO L2 (搜 NPU 等价) |
| GPU compile 消除了 transpose/简化表达式 | → TODO L3 |
| GPU compile 将多 elementwise 融合为 1 kernel | → TODO L4 |
| GPU backward 保存更少 tensor | → TODO L5 |

将具体候选方案写入 lab 脚本的 TODO 注释中。

### 3a-ter. GPU Compile 对齐候选 (强制)

**除了上面按 gap 类型推导的通用候选外，每个 SN 必须至少两组专门尝试复现 GPU compile 结果的候选。**

GPU compile 后的状态是 ground truth — 它证明了这些 ops **可以**被融合/消除/简化。NPU 的目标是用任何可行方法达到等效状态（不限于 compile，任何能对齐 kernel 数量 + 精度 + 效率的方式都算）。

**强制步骤**:

1. **记录 GPU compile 后的目标状态**:
   - 该 SN 在 GPU compile 后是几个 kernel？（从 ir_post_fusion.txt 数 FusedSchedulerNode + ExternKernel）
   - 哪些 ops 被消除了？哪些被融合了？
   - 输入/输出 dtype 是什么？中间有无 materialization？

2. **记录 NPU 当前状态**:
   - 该 SN 在 NPU eager 下是几个 kernel launch？
   - 有无多余的 cast/copy/materialization？

3. **生成 "对齐 GPU" 候选** — 尝试多种方式逼近 GPU 的 kernel 数和效率:
   - `L-GPU-a`: `torch.compile(backend='npu')` 包裹 GPU fusion group 对应的 scope
   - `L-GPU-b`: 搜索 NPU 官方 fused API 覆盖同 scope（一个 API call = GPU 的一个 fused kernel）
   - `L-GPU-c`: 手动改写消除 GPU compile 消除的 ops（去 cast、去 materialization、去 transpose）
   - `L-GPU-d`: 组合方案（API + compile + 手动的混合）
   - 如果以上都不能对齐 kernel 数 → 记录差距和原因，标注为 NPU 平台限制

4. **验证对齐程度** — Lab 报告中必须包含:
   ```
   GPU compile 后: N kernels, dtype=bf16, 无中间 materialization
   NPU eager:      M kernels, dtype=..., 有/无额外 cast
   NPU 优化后:     K kernels, dtype=..., (对齐程度: K/N)
   ```

**这不是 "试一次 compile 不行就算了"** — 如果 `backend='npu'` 的 compile 不能一步到位，要拆开看：GPU 融合的 ops 中，哪些子集可以被 NPU compile？哪些需要用 API 替代？哪些需要手动消除？分而治之，逼近目标。

### 3b. 实现并运行 Lab 中的候选方案

按成本从低到高逐级尝试。**每级都必须测试或记录跳过原因。**

| Level | 类型 | 说明 |
|-------|------|------|
| **L0** | API 参数/环境变量 | 0-1 行。改 API 参数、加环境变量 |
| **L1** | 消除冗余 | 去多余 cast、去重复计算、去不必要 sync |
| **L2** | NPU 官方融合 API | 搜索 `dir(torch_npu)` 找等价融合算子（详见 [references/npu_optimization_patterns.md](references/npu_optimization_patterns.md)） |
| **L3** | 等价手动改写 | 改表达式不改语义。消除 double-transpose、buffer 复用、表达式简化 |
| **L4** | torch.compile | `@torch.compile(backend='npu', dynamic=False)` 包裹函数 |
| **L5** | Custom autograd | 当 API forward 快但 backward 有问题时 |
| **L6** | AscendC kernel | 最后手段 |

**每级内穷举多个方案**:

在每个层级内，不要只尝试一种方法就进入下一级。主动发散：

- **L0**: 查阅该 SN 所用 API 的完整签名，列出所有可调参数逐个测试；搜索相关环境变量。
- **L1**: 该 SN 内有几处独立的冗余（cast/重复计算/sync）？每处作为独立方案分别测试。同时检查参数存储 dtype——如果 weight 是 f32 但 activation 是 bf16，考虑将 weight 永久转为 bf16（消除每次 forward 的隐式 cast）。
- **L2**: 按功能关键词搜索 `dir(torch_npu)`，可能有多个相关 API 变体。每个验证语义等价性+性能。
- **L3**: 同一个改写目标可能有多种等价实现（不同数学表达、不同 layout、不同内存策略）。全部写出对比。
- **L4**: compile 的 scope 可以不同（只包核心计算？包整个函数？包含上下游 norm？）。每种 scope 独立测试。
- **L5**: 同一 forward API 的 backward 可能有多种手动等价推导。对比精度和速度。

**方案命名**: Lab 中用 `B0`(control) + `L<级别><序号>` 命名（如 `L2a`, `L2b`, `L3a`...）。

**累计搜索**: 每级的 winner 成为下一级的 parent baseline。最终报告 cumulative gain vs B0。

**L4 (compile) 关键知识**:

⚡ **必须用 `backend='npu'`**:
```python
@torch.compile(backend='npu', dynamic=False)  # 不是默认 inductor!
def fn(x):
    ...
```
默认 `backend='inductor'` 在 NPU 上通常失败或无增益。`backend='npu'` 使用昇腾专用编译路径，对 elementwise chain 有显著 fusion 效果。

⚡ **PATH 排错**: 如果报 "npuc" / "bishengir" / "Invalid bishengir path" 错误:
```bash
# 找到编译器实际位置
find /usr/local/Ascend -name "bishengir-compile"
# 加入 PATH (通常在 bisheng_toolkit/bishengir/bin/ 下)
export PATH="/usr/local/Ascend/<version>/bisheng_toolkit/bishengir/bin:$PATH"
```

⚡ **Compile scope 很重要**: 不要只 compile 最小子表达式。尝试逐步扩大 scope（只包 activation → 包 activation+linear → 包 norm+linear+activation+linear），更大 scope 通常有更好的 fusion 效果。每种 scope 作为独立候选方案在 Lab 中对比。

**适用判断**:
- ✅ 适合: 多个 elementwise/pointwise 链（sigmoid+mul, relu+mul, div+sigmoid+mul）
- ❌ 不适合: 已是单个大算子（mm, attention, norm API）、tensor 很小（dispatch overhead > fusion gain）
- 注意: `dynamic=False` 在多 regime 训练中每个 shape 编译一次后缓存

⚡ **多卡注意**: 首次多卡 compile 可能崩溃（"unable to open output file kernel_meta/..."）→ 每 rank 独立 cache 路径:
```bash
export TORCH_NPU_COMPILE_CACHE_DIR="/tmp/npu_compile_cache_${RANK:-0}"
```

### 3c. Supernode Lab (强制)

**在修改任何训练代码之前**，必须为当前 SN 创建独立的 benchmark 脚本：`benchmarks/supernodes/sn_<name>.py`

这个脚本是该 SN 的**隔离实验室** — 所有优化方案在此验证后才能进入训练代码。

**脚本结构要求**:

```python
"""
Supernode Lab: SN-<NAME>
GPU Teacher: <GPU compile 前后做了什么，几个 kernel>
NPU Current: <当前实现，几个 kernel launch>
"""

# 1. 真实 shape/dtype/layout (从训练代码中提取)
# 2. Control: 当前实现 (forward + backward)
# 3. L0-L6 各级候选方案 (每个都测)
# 4. Benchmark: 每个方案跑 warmup + reps，报告 median/p95
# 5. Correctness: 对比 control 的梯度 cosine similarity
# 6. 所有 regime shape 都测 (不只测一个)
```

**报告格式** (脚本输出):
```
=== SN-<NAME> Supernode Lab ===
GPU: <对齐信息>
NPU: <当前信息>

Method              | T=8192  | T=16384 | T=24576 | Grad cos | vs Ctrl
----------------------------------------------------------------------
B0: control         | X.XXXms | X.XXXms | X.XXXms | 1.0000   | 1.00x
L0: <api param>     | ...     | ...     | ...     | ...      | ...
L1: <eliminate>     | ...     | ...     | ...     | ...      | ...
L2: <npu api>       | ...     | ...     | ...     | ...      | ...
L3: <rewrite>       | ...     | ...     | ...     | ...      | ...
L4: <compile>       | ...     | ...     | ...     | ...      | ...
```

**强制规则**:
- 每个 SN 必须有对应的 lab 脚本，否则不允许修改训练代码
- Lab 中必须测试所有适用的 L0-L6 方案（不适用的标注跳过原因）
- 所有方案必须含 backward（优化前向但破坏梯度的无效）
- Lab 中发现的 winner 进入 3d 多卡 ablation；ablation 通过后才改训练代码
- Lab 脚本**永久保留**，作为优化决策的证据（为什么选了这个方案、为什么跳了那个）

### 3d. 多卡 Ablation (~60s 快速验证)

**构造短跑脚本** (首次需创建，后续复用):

从完整训练脚本派生一个 ~60 秒的短跑版本:
- 等比压缩 total steps（如 2090 → 210 步，约 1/10）
- 按相同比例映射所有 step-dependent schedule（lr、batch size、window size 的切换点）
- 保持相同 seed、数据、初始状态
- 输出格式不变（step_avg 可直接对比）

```python
# 短跑关键修改 (相对完整训练)
num_scheduled_iterations = FULL_STEPS // 10  # 等比压缩
num_extension_iterations = FULL_EXT // 10
val_loss_every = num_scheduled_iterations + num_extension_iterations  # 最后才 validate
```

**⚠️ 短跑与 Full Training 的关系（单向派生，禁止反推）**:

短跑脚本是从原始 `train_gpt.py` 的训练参数**单向派生**的产物。Full training 的参数**永远以 baseline git commit 中的原始 `train_gpt.py` 为唯一真相来源**，禁止从短跑脚本反向推算 full training 参数。

- 短跑创建时，记录派生来源：在短跑脚本头部注释中标注 `# Derived from: train_gpt.py @ commit <hash>` 和原始完整参数值
- Full training 恢复时：直接 `git show <baseline_commit>:train_gpt.py` 获取原始训练参数，在当前优化版本上恢复这些参数值
- **禁止**从短跑参数 × 压缩比来还原 full training 参数（多次修改短跑后比例可能漂移）

**Ablation 执行规则**:

1. 先跑一次 baseline 短跑 → 记录 `step_avg` 作为对照 → 存入 `logs/baseline_short.log`
2. 每个有增益的方案放入 `ablations/` 子目录（见下方「目录结构」），不要放在项目根目录
3. 提交多卡短跑 → 结果写入 `logs/sn_<name>_L<N>.log`
4. 对比 baseline 的 `step_avg`
5. **只有 step_avg 下降才接受**

**注意**: 对于 Lab 阶段"严格通过"的方案，短跑的 `val_loss` 不用于判断正确性（步数太少未收敛），短跑只验证多卡环境下的真实速度增益。

**条件通过方案的短跑验证**:

Lab 精度检查采用分级验收（详见 [references/npu_optimization_patterns.md](references/npu_optimization_patterns.md)「验证门禁」）。对于"条件通过"的方案（高增益 + 误差在 dtype 精度可解释范围内，如 bf16 下 relative_error ~1e-3 量级），短跑同时承担**速度验证 + 精度观察**双重角色：

1. 正常跑短跑 ablation，记录 `step_avg` 和 `val_loss`
2. 如果 `val_loss` 在自然波动内（与 baseline 短跑的 val_loss 差距合理）→ 暂时 accept
3. 如果 `val_loss` 明显异常 → 该方案降级为拒绝，不进入组合版本
4. 条件通过的方案在 progress.md 中标记为 `✅cond`（区别于严格通过的 `✅`），便于 full training 阶段追溯

---

## Step 4: 组合 + Full Run

合并所有 ablation 通过的优化 → full run → 验证 train_time + val_loss。

**Full Run 参数来源**：full training 的训练参数（total steps、schedule 等）从 baseline git commit 的原始 `train_gpt.py` 中获取，不从短跑脚本反推。

如果 full run 结果 < target：完成。
如果未达标：回到 Step 2 检查是否有遗漏 SN，或在现有 SN 上继续深入下一层级。

### 回退策略（val_loss 超标时）

当 full training 的 val_loss 超出要求时，优先排查"条件通过"的方案（progress.md 中标记为 `✅cond` 的条目）：

1. **按嫌疑度排序**：Lab 阶段误差越大的方案嫌疑越高（cosine 越低 / relative_error 越大 → 排越前）
2. **逐个回退验证**：从最高嫌疑开始，去掉该方案后重跑 full training（或先用短跑快速筛查）
3. **定位元凶后决策**：
   - 确认是某个条件通过方案导致 → 移除该方案，保留其余优化
   - 多个条件通过方案的组合效应 → 逐步缩小组合范围
   - 所有条件通过方案去掉后仍超标 → 问题在严格通过方案的组合，按常规 debug 流程处理

**注意**：回退不意味着"条件通过"机制有问题——大多数情况下这些方案会通过 full training 验证。回退机制只是保底手段，让高增益方案有机会证明自己。

---

## 持续试错机制

Agent 容易在连续失败后放弃或在 "看起来够好" 时过早停止。以下机制强制持续探索：

### 进度文件 (强制维护)

每次 Supernode Lab 运行后，更新 `benchmarks/supernodes/progress.md`:

```markdown
# 优化进度

## 目标: train_time < XXXs, val_loss < X.XX
## 当前最优: XXXs (step_avg=XXXms)
## 剩余 gap: XXs

| SN | L0 | L1 | L2 | L3 | L4 | L5 | L6 | Winner | Ablation | 状态 |
|----|----|----|----|----|----|----|----|----|---------|------|
| SN-Loss | ✅skip | ✅-15s | ✅L2a好,L2b差 | ✅fail | ✅1.5x | - | - | L4c | ✅pass | done |
| SN-Attn-Core | ✅-30s! | - | - | - | - | - | - | L0a | ✅pass | done |
| SN-MLP | - | - | 🔄testing | - | - | - | - | - | - | in_progress |
| SN-Rotary | - | - | - | - | - | - | - | - | - | pending |
```

- 每个格子: ✅ (测完+结果), 🔄 (进行中), ❌ (失败但已记录原因), `-` (未开始), `skip:原因`
- **Agent 每次启动时先读这个文件**，从上次停下的地方继续
- **只有所有高优先级 SN 的所有格子都填满后才能声明完成**

### 停止条件 (唯一允许停止的情况)

Agent **只有**满足以下**全部**条件时才能停止当前 SN：

1. L0-L4 每级至少测试过 4 个方案（或有明确 skip 原因写入 progress）
2. 每级内的方案都跑过 forward + backward + 多 shape
3. 如果某级报错，已追过根因并尝试至少 2 种不同的修复方式
4. Winner 已通过多卡 ablation（或确认增益 < 测量噪声）

不允许因为以下原因停止：
- "这个 API 报错了" → 追根因（PATH？格式？参数？）
- "benchmark 显示更慢" → 检查精度是否对齐、shape 是否真实
- "感觉没什么能做的了" → 检查 progress 表是否有未填的格子
- "增益太小不值得" → 记录数值，但仍然完成所有级别的测试

### 失败重试规则

同一方向失败时：
1. 第 1 次失败: 记录错误，分析根因
2. 第 2 次（换方式）: 修改参数/格式/组合再试
3. 第 3 次（换条件）: 改变 shape/dtype/scope 再试
4. 3 次全失败: 记录 "L<N>: ❌ 原因=..., 尝试=3次, 结论=当前环境不支持"，进入下一级

### 上下文恢复

如果对话过长需要重新开始：
1. 读 `benchmarks/supernodes/progress.md` 恢复状态
2. 读已存在的 `benchmarks/supernodes/sn_*.py` lab 脚本恢复历史结果
3. 读 `logs/sn_*.log` 恢复多卡验证结果
4. 从 progress 表中第一个未完成的格子继续

---

## 何时重新分析

- 所有高优先级 SN 的 L0-L6 都测完，总增益仍不够
- 代码结构大改后旧 benchmark 失效
- 发现新的 NPU API 或环境变量
- Regime/shape 变化使旧结论失效

---

## 项目目录结构

Phase 1 会产生大量实验脚本（Lab、短跑变体、probe、profiling 变体等）。**所有派生文件必须放入对应子目录，禁止在项目根目录平铺。**

```
项目根目录/
├── train_gpt.py                    # 唯一的主训练脚本（持续就地修改）
├── run.sh                          # 唯一的主启动脚本
├── benchmarks/
│   └── supernodes/
│       ├── progress.md             # 优化进度追踪
│       └── sn_<name>.py            # 各 SN 的 Lab 脚本
├── ablations/                      # 多卡 ablation 用的短跑变体
│   ├── train_gpt_short_baseline.py # 短跑 baseline（从主脚本派生，仅一份）
│   ├── train_gpt_short_<opt>.py    # 各方案的短跑脚本
│   ├── run_short_baseline.sh
│   └── run_short_<opt>.sh
├── probes/                         # 单算子探测脚本
│   ├── probe_<name>.py
│   └── run_probe_<name>.sh
├── logs/                           # 所有运行日志
│   ├── baseline_short.log
│   ├── sn_<name>_L<N>.log
│   └── final_full_run.log
├── profiling/                      # profiling 输出（.gitignore）
└── custom_op/                      # 自定义算子（如有）
```

**核心规则**：
- `train_gpt.py` 是唯一的主脚本——ablation 通过的优化合入这里，不创建 `train_gpt_full_*.py` 变体
- 短跑变体和 run 脚本放 `ablations/`，probe 脚本放 `probes/`
- Ablation 通过后，对应的 `ablations/` 文件可以保留作为记录，但不是必须的；优化决策的证据在 Lab 脚本（`benchmarks/supernodes/sn_*.py`）中
- profiling 相关的 run 脚本也放 `ablations/`（如 `run_profile_<opt>.sh`），或直接在主 `run.sh` 上加 profiler 参数

## Git 与日志管理

### Git 分支策略

借鉴 Phase 2 的分支管理，Stage 1 也使用工作分支：

```
main（稳定版本，不直接修改）
  └── optimize/stage1（主工作分支）
        ├── 逐 SN 实施优化，每个 SN ablation 通过后 commit
        └── 所有 SN 完成后合入 main
```

1. **优化开始前**: 从 main 创建 `optimize/stage1` 分支，`git commit -am "baseline before optimization"` 作为安全回滚点
2. **每个 SN ablation 通过并合入 train_gpt.py 后**: `git commit -am "SN-<name>: <winner> (step_avg Xms→Yms)"`
3. **需要回滚**: `git checkout -- train_gpt.py` 回到上一个 checkpoint
4. **Stage 1 完成后**: 用户确认后合入 main

不要在 git 操作上花超过 1 分钟。

### 日志 (集中命名)

所有训练 run 的日志保存到 `logs/` 目录，文件名编码实验内容：

```
logs/
├── baseline.log                    # 初始 baseline
├── baseline_short.log              # 短跑 baseline
├── sn_loss_L4c_compile_sig.log     # SN-Loss L4c ablation
├── sn_attn_L0a_pre_tockens.log     # SN-Attn L0a ablation
├── v5_all_combined.log             # 组合版本
└── final_full_run.log              # 最终 full run
```

每个 run script 统一格式：
```bash
torchrun ... ablations/train_gpt_short_X.py 2>&1 | tee logs/<descriptive_name>.log
```

**快速对比所有实验结果**：
```bash
grep "step:.*val_loss" logs/*.log
```

不要依赖 job 平台的远程日志查看（ID 难记、job 结束后可能不可访问）。

---

## 禁止事项

- ❌ 优化前写大量文档/schema
- ❌ 因一次报错放弃方向
- ❌ 用单机 benchmark 直接宣称多卡增益
- ❌ 一次加多个未 ablation 的优化
- ❌ 宣布"没有优化空间"（换方式再试）
- ❌ 在项目根目录创建 `train_gpt_*.py` 或 `run_*.sh` 变体文件（放 `ablations/` 或 `probes/`）
