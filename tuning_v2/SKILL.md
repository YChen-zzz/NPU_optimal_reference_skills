---
name: npu-supernode-tuning
description: NPU 训练/推理性能优化。以 GPU compiled evidence 为参照，自动拆分计算图为 Supernode，逐节点穷尽优化方法并用多卡 ablation 验证。适用于 GPU→NPU 迁移调优、NPU 原生训练加速。
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
- NPU API 是否有未使用的加速参数？
- GPU 传了什么参数 NPU 没传？

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

**L4 (compile) 使用判断**:
- ✅ 适合: 多个 elementwise/pointwise 链（sigmoid+mul, relu+mul, div+sigmoid+mul）
- ❌ 不适合: 已是单个大算子（mm, attention, norm API）、tensor 很小（dispatch overhead > fusion gain）
- 注意: `dynamic=False` 在多 regime 训练中每个 shape 编译一次后缓存

### 3c. Benchmark

写入 `benchmarks/supernodes/sn_<name>.py`

**必须包含**:
- Forward + backward（`.sum().backward()`）
- 所有 regime 的 shape
- 梯度正确性: `cosine_similarity > 0.9999`
- 对照 control 的 speedup

**格式**:
```
Method | Shape1 fwd+bwd | Shape2 fwd+bwd | Shape3 fwd+bwd | Grad cos | vs Control
```

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

## 何时重新分析

- 所有高优先级 SN 的 L0-L6 都测完，总增益仍不够
- 代码结构大改后旧 benchmark 失效
- 发现新的 NPU API 或环境变量
- Regime/shape 变化使旧结论失效

---

## 禁止事项

- ❌ 优化前写大量文档/schema
- ❌ 因一次报错放弃方向
- ❌ 用单机 benchmark 直接宣称多卡增益
- ❌ 一次加多个未 ablation 的优化
- ❌ 宣布"没有优化空间"（换方式再试）
- ❌ 把 compile 留到最后
