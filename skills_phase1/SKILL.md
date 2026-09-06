---
name: npu-supernode-tuning
description: NPU 训练性能优化。以 GPU compiled evidence 为参照，自动拆分计算图为 Supernode，逐节点穷尽优化方法并用多卡 ablation 验证。适用于 GPU→NPU 迁移调优、NPU 原生训练加速。
---

# NPU Supernode 调优

## 核心原则

1. **按 step 占比从大到小做。** 大头优先，低占比 SN 可精简。
2. **GPU 是参照不是答案。** 对齐优化意图，用 NPU 方法实现。
3. **移植遗留 ≠ 数学语义。** GPU 训练中没有的 `.float()` 是移植遗留，移除它是恢复原始语义，即使用户说"不改数学语义"也合规。
4. **Forward + backward + 真实 shape 才算有效 benchmark。**
5. **单机 benchmark ≠ 多卡增益。** 必须 ablation 验证。
6. **不因单次失败放弃。** 追根因，换方式再试。

---

## Step 1: Baseline + GPU 参照

运行原始脚本，记录 `train_time`、`step_avg`、`val_loss`、`peak_memory`。

从 GPU evidence pack 提取 compile 前 FX graph、compile 后 IR、profiling summary、source 差异。
详见 [references/gpu_teacher_reading.md](references/gpu_teacher_reading.md)。

**产出**: GPU vs NPU 对比表。

---

## Step 2: Supernode 划分

详见 [references/supernode_analysis.md](references/supernode_analysis.md)

1. 从 GPU IR 提取 fusion groups
2. 按语义功能分组为 Supernode
3. 在 NPU source 中标注每个 SN 对应的代码范围
4. 估算每个 SN 占 step 时间的比例
5. **progress.md 中的 SN 行按占比降序排列**，Step 3 严格按此顺序执行

为每个 SN 创建 Lab 骨架 `benchmarks/supernodes/sn_<name>.py`，使用 [references/lab_template.py](references/lab_template.py) 中的模板。

**Step 2 必须创建 `benchmarks/supernodes/progress.md`**，格式见下方「进度文件」。这是 Step 3 的前置条件——没有 progress.md 不允许开始优化。

**门禁**: Lab 骨架中 `DTYPE_PATH`、`CAST_AUDIT` 和 `SHAPES` 未填写时，禁止实现候选方案。

---

## Step 3: 逐 Supernode 优化

**按 step 占比从大到小**逐个 SN 执行。禁止跳过高占比 SN 去做低占比 SN。

**每次只能有一个 SN 处于 `in_progress`**。必须把当前 SN 做到 `done`（或有明确的占比跳过理由）后，才能开始下一个 SN。禁止同时推进多个 SN。

**低占比 SN（如低于 1%）**：完成 L0-L2，如果没有明显优化点，可跳过 L3-L4，在 progress.md 标注 `skip: 占比 X%，L0-L2 无明显优化点`。

### 每个 SN 的强制流程

#### Gate 1: 精度对齐（填 Lab 中的 DTYPE_PATH + CAST_AUDIT）

**Part A**: 记录该 SN 在 GPU 和 NPU 上的完整 dtype 路径（input/compute/output/weight），找出不一致的地方。

**Part B**: 对该 SN 中的**每个** `.float()` / `.type_as()` / `.to(torch.float32)`：

1. 读 GPU source 同一位置，确认 GPU 是否有这个 cast
2. GPU 没有 → 标记为「移植遗留」，归入 L1 候选
3. GPU 也有 → 标记为「原始设计」，不动

**不允许因用户语义约束而跳过此分析。** 必须填入 Lab 的 `DTYPE_PATH` 和 `CAST_AUDIT`，assert 会检查。

#### Gate 2: 候选生成

从 Gap 推导候选（详见 [references/supernode_analysis.md](references/supernode_analysis.md)）：

| Gap 类型 | 映射到 |
|---------|--------|
| GPU API 有参数 NPU 没传 | L0 |
| NPU 多了 cast/sync/重复计算 | L1 |
| GPU 用了 fused API | L2 (搜 NPU 等价) |
| GPU compile 消除了 transpose/简化表达式 | L3 |
| GPU compile 将多 elementwise 融合为 1 kernel | L4 |

#### Gate 3: L4 compile 候选覆盖（填 Lab 中的 COMPILE_AUDIT）

L4 必须覆盖以下候选（缺一不可，不适用的标注原因）：

1. **compile 当前累计 winner**（不管来自 L0/L1/L2/L3 哪级）
2. **compile GPU fusion 对应的代数表达族**（可能和 eager winner 不同）
3. **不同 scope**（可选，仅在认为有必要时测试）
4. **覆盖全部真实 shape**（不只测一个 regime）

**必须用 `backend='npu'`**，不是默认 inductor：
```python
@torch.compile(backend='npu', dynamic=False)
def fn(x): ...
```

首次编译耗时与 steady-state 分开记录，不因 cold-start 否决。
更多 compile 知识（PATH 排错、多卡注意事项等）见 [references/compile_guide.md](references/compile_guide.md)。

### L0-L6 层级定义

| Level | 类型 | 说明 |
|-------|------|------|
| L0 | API 参数/环境变量 | 0-1 行改动 |
| L1 | 消除冗余 | 去多余 cast、去重复计算、去不必要 sync |
| L2 | NPU 融合 API | `dir(torch_npu)` 搜索等价融合算子 |
| L3 | 等价手动改写 | 改表达式不改语义 |
| L4 | torch.compile | `backend='npu', dynamic=False` |
| L5 | Custom autograd | forward 快但 backward 有问题时 |
| L6 | AscendC kernel | 最后手段 |

**累计搜索**: 每级的 winner 成为下一级的 parent baseline。

详见 [references/npu_optimization_patterns.md](references/npu_optimization_patterns.md)。

### Lab → Ablation → 合入

1. 在 Lab 中测试所有候选（forward + backward + 全部 regime shape）
2. Lab winner 进入多卡短跑 ablation（详见 [references/ablation_protocol.md](references/ablation_protocol.md)）
3. step_avg 下降 → accept，合入 `train_gpt.py`，git commit
4. Lab 脚本永久保留作为证据

---

## Step 4: 组合 + Full Run

合并所有 ablation 通过的优化 → full run → 验证 train_time + val_loss。

Full training 参数从 baseline git commit 的原始 `train_gpt.py` 获取，不从短跑反推。

未达标 → 回到 Step 2 检查遗漏 SN 或在现有 SN 上继续深入。

---

## 进度文件

每次 Lab 运行后更新 `benchmarks/supernodes/progress.md`：

```markdown
## 目标: train_time < XXXs, val_loss < X.XX
## 当前最优: XXXs (step_avg=XXXms)

| SN | 占比 | L0 | L1 | L2 | L3 | L4 | Winner | Ablation | 状态 |
|----|------|----|----|----|----|----|--------|---------|------|
| SN-Attn | 30% | ✅ | ✅ | ✅ | ✅ | ✅ | L0a | ✅pass | done |
| SN-MLP | 25% | ✅ | ✅ | 🔄 | - | - | - | - | in_progress |
```

每个格子: ✅(测完) / 🔄(进行中) / ❌+原因(失败) / skip:原因

Agent 每次启动时先读此文件，从上次停下的地方继续。

---

## Stage 1 完成门禁

切换到 Stage 2 前，必须逐步执行并输出：

1. 读取 progress.md 全表
2. 逐 SN 检查：每个格子必须是 ✅ / ❌+原因 / skip:原因，不允许有 `-` 或 🔄
3. 输出完成度统计（总 SN 数、完成数、未完成数及原因）
4. 完成率 100% 才允许切换

低占比 SN 的 L3-L4 标注 `skip: 占比 X%，L0-L2 无明显优化点` 视为已完成。

---

## 项目结构与工程规范

详见 [references/project_structure.md](references/project_structure.md)

核心规则：
- `train_gpt.py` 是唯一的主脚本，不创建根目录变体
- 短跑放 `ablations/`，探测放 `probes/`，日志放 `logs/`

---

## 禁止事项

- ❌ 因用户语义约束跳过 `.float()` 分析（必须先查 GPU source 再决定）
- ❌ L4 只编译 eager winner 不编译其他代数表达族
- ❌ 同时推进多个 SN（必须逐个完成）
- ❌ 跳过高占比 SN 去做低占比 SN
- ❌ 因一次报错放弃方向
- ❌ 用单机 benchmark 直接宣称多卡增益
- ❌ 一次加多个未 ablation 的优化
- ❌ 在项目根目录创建变体文件
