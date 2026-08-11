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

## Step 2: Supernode 划分

### 划分方法

详见 [references/supernode_analysis.md](references/supernode_analysis.md)

**核心流程**:

1. 读 GPU `ir_post_fusion.txt`，识别所有 `FusedSchedulerNode` 和 `ExternKernel`
2. 按**语义功能**分组：服务同一计算目的的相邻 fusion groups 合为一个 Supernode
3. 在 NPU source 中标注每个 Supernode 对应的代码范围
4. 估算每个 SN 占 step 时间的比例 → 确定优化优先级

**划分原则**:
- 语义边界优先（不以 kernel 名划分）
- GPU 把哪些 ops 融合在一起 → 说明它们之间有优化机会
- 按时间权重排优先级：先做最大的 SN
- 大 SN 可继续细拆（如 Attention 拆为 prologue/core/epilogue）

**产出**:
```
| SN | 语义 | GPU kernels 数 | NPU 当前 kernel 数 | 占 step % | 优先级 |
```

---

## Step 3: 逐 Supernode 优化

对每个 SN 依次执行 3a → 3b → 3c → 3d。

### 3a. GPU 对齐分析

对当前 SN 回答（详见 [references/supernode_analysis.md](references/supernode_analysis.md)）：

**精度对齐**:
- GPU 在该 SN 用什么 dtype（input/compute/accumulator/output/saved_for_backward）？
- NPU 是否多了 cast（.float()/.type_as()/.to()）？
- 哪些 cast 是精度必需的，哪些是移植遗留？

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

### 3b. 优化层级 (L0-L6)

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
- **L1**: 该 SN 内有几处独立的冗余（cast/重复计算/sync）？每处作为独立方案分别测试。
- **L2**: 按功能关键词搜索 `dir(torch_npu)`，可能有多个相关 API 变体。每个验证语义等价性+性能。
- **L3**: 同一个改写目标可能有多种等价实现（不同数学表达、不同 layout、不同内存策略）。全部写出对比。
- **L4**: compile 的 scope 可以不同（只包核心计算？包整个函数？包含上下游 norm？）。每种 scope 独立测试。
- **L5**: 同一 forward API 的 backward 可能有多种手动等价推导。对比精度和速度。

**方案命名**: Lab 中用 `B0`(control) + `L<级别><序号>` 命名（如 `L2a`, `L2b`, `L3a`...）。

**累计搜索**: 每级的 winner 成为下一级的 parent baseline。最终报告 cumulative gain vs B0。

**L4 (compile) 使用判断**:
- ✅ 适合: 多个 elementwise/pointwise 链（sigmoid+mul, relu+mul, div+sigmoid+mul）
- ❌ 不适合: 已是单个大算子（mm, attention, norm API）、tensor 很小（dispatch overhead > fusion gain）
- 注意: `dynamic=False` 在多 regime 训练中每个 shape 编译一次后缓存

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

### 3d. 多卡 Ablation

**每个有增益的方案**:
1. 创建独立文件（`train_gpt_short_X.py` + `run_short_X.sh`）
2. 提交多卡短跑 → 结果写入独立 `ablation_X.log`
3. 对比 baseline `step_avg`
4. **只有 step_avg 下降才接受**

---

## Step 4: 组合 + Full Run

合并所有 ablation 通过的优化 → full run → 验证 train_time + val_loss。

如果 full run 结果 < target：完成。
如果未达标：回到 Step 2 检查是否有遗漏 SN，或在现有 SN 上继续深入下一层级。

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
2. 读已存在的 `sn_*.py` lab 脚本恢复历史结果
3. 读 `ablation_*.log` 恢复多卡验证结果
4. 从 progress 表中第一个未完成的格子继续

---

## 何时重新分析

- 所有高优先级 SN 的 L0-L6 都测完，总增益仍不够
- 代码结构大改后旧 benchmark 失效
- 发现新的 NPU API 或环境变量
- Regime/shape 变化使旧结论失效

---

## Git 与日志管理

### Git (最小化)

1. **优化开始前**: `git commit -am "baseline before optimization"` — 安全回滚点
2. **每个 SN ablation 通过并合入 train_gpt.py 后**: `git commit -am "SN-<name>: <winner>"` — checkpoint
3. **需要回滚**: `git checkout -- train_gpt.py` 回到上一个 checkpoint

不要为每个实验创建 branch（用独立文件隔离代替），不要在 git 操作上花超过 1 分钟。

### 日志 (集中命名)

所有训练 run 的日志保存到 `logs/` 目录，文件名编码实验内容：

```
logs/
├── baseline.log                    # 初始 baseline
├── sn_loss_L4c_compile_sig.log     # SN-Loss L4c ablation
├── sn_attn_L0a_pre_tockens.log     # SN-Attn L0a ablation
├── v5_all_combined.log             # 组合版本
└── final_full_run.log              # 最终 full run
```

每个 run script 统一格式：
```bash
torchrun ... train_gpt_short_X.py 2>&1 | tee logs/<descriptive_name>.log
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
