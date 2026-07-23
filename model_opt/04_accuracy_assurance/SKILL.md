---
name: npu-accuracy-assurance
description: Phase 4 精度保证。确保 NPU 优化后的模型精度正确，包括基线管理、分层验证和精度调试。当用户关注精度验证、输出对比或精度问题定位时触发。执行前参见根 SKILL.md 全流程。
---

# NPU 精度保证

## 核心思路

优化可能在任何环节引入精度问题。精度验证的目标是：确认优化后的输出与可信 baseline 在数值上足够接近。

推理和训练的验证复杂度差异很大：

- **推理**：比较最终推理产物即可，验证链路短
- **训练**：需要三层递进验证——单步数值对齐 → 训练过程对齐 → 最终结果对齐

根据任务类型选择对应的验证路径。

## 基线管理

精度对比的第一步是确认 baseline 来源。来源不对，整个结论无效。

**优先级**：官方基线（README/benchmark）→ 用户 GPU 基线 → CPU 仅作调试辅助。

**强制规则**：
- 先查官方基线，再写对比脚本
- 官方基线不足时必须询问用户 GPU 基线来源
- 禁止默认使用 CPU 结果作为最终对齐结论
- 阈值必须在比较前声明
- 样例级对齐 ≠ 模型级对齐

详见 [baseline_policy.md](references/baseline_policy.md)

## 验证策略：按场景分流

### 推理场景

比较最终推理产物，分层递进：

**Level 1 快速验证**（每次修改后）：少量代表性样本，检查输出与基线的数值差异。三要素：功能、精度、性能。

**Level 2 全量验证**（每批提交前，提交门禁之一）：完整测试集，覆盖所有推理路径，关注 worst case 而非平均值。**必须与 Profiling 确认收益一起作为提交前的两项门禁**——两项均通过后才可进入用户确认提交流程。

**Level 3 精度调试**（验证未通过时）：详见 [debugging_guide.md](references/debugging_guide.md)

### 训练场景：三层递进验证

训练精度对齐比推理复杂得多，需要从微观到宏观逐层验证。

#### Layer 1：单步数值对齐（微观）

固定相同输入、相同初始权重、关闭所有随机性，对比 GPU 与 NPU 单步训练的数值。

| 对齐点 | 方法 | 指标 | 参考阈值 |
|--------|------|------|----------|
| 前向输出 | 逐层 hook 抓取激活值，对比 mean/max/min | cosine sim + max abs diff + 相对误差 | 单步相对误差 < 5% |
| Loss 值 | 同一输入对比 loss 标量 | 绝对误差 + 相对误差 | 相对误差 < 1% |
| 反向梯度 | `loss.backward()` 后取 `param.grad`，逐参数对比 | cosine sim + 梯度范数比 + max abs diff | cosine > 0.999 |
| 参数更新 | `optimizer.step()` 后对比权重变化量 `(w_new - w_old)` | 相对误差 + max abs diff | 相对误差 < 1% |

**问题定位方法**：逐模块比对 → 模块内二分 → 定位到具体算子。

#### Layer 2：训练过程对齐（中观）

跑多步或完整 epoch，对比训练曲线和动态行为。

| 对齐点 | 方法 | 指标 | 参考阈值 |
|--------|------|------|----------|
| Loss 曲线 | 逐 step 对比 GPU/NPU loss 值 | 平均相对误差 + 最大绝对偏差 + 曲线趋势可视化 | 平均相对误差 < 5% |
| 收敛速度 | 对比达到目标 loss 的 step 数 | step 数比值 | 偏差 < 10% |
| 训练稳定性 | 对比 loss 方差、梯度范数变化 | loss 滑动方差比 + grad norm 曲线 | 无异常震荡或发散 |
| 优化器状态 | 确认 optimizer state 一致 | 逐 step state_dict 对比 | 完全一致 |

#### Layer 3：最终结果对齐（宏观）

完整训练后，对比最终模型质量。

| 对齐点 | 方法 | 指标 | 参考阈值 |
|--------|------|------|----------|
| 任务指标 | 在相同验证/测试集上评估 | 任务相关（accuracy / F1 / BLEU / perplexity 等） | 在 GPU 多次运行自然波动范围内 |
| 最终权重 | 对比训练结束后的 state_dict | cosine sim + max abs diff | cosine > 0.99 |
| 泛化一致性 | 对比多个评估集上的表现分布 | 指标分布差异 | 无系统性偏差 |

#### 训练对齐验证流程

```
Layer 1 单步对齐 ── 通过 ──→ Layer 2 过程对齐 ── 通过 ──→ Layer 3 最终结果
      │                            │                            │
    不通过                        不通过                        不通过
      ↓                            ↓                            ↓
  逐层二分定位问题算子        检查超参/数据一致性         检查累积误差/随机性
```

## 指标选择原则

### 推理指标

| 输出类型 | 思路 | 常用指标 |
|---------|------|----------|
| 连续向量 | 方向一致性 + 幅值偏差 | cosine similarity + max abs diff |
| 离散输出 | 匹配率（自回归场景对微小偏差敏感） | 匹配率（仅参考） |
| 聚合标量 | 相对误差 | \|a-b\|/\|a\| |

### 训练指标

| 对齐层级 | 核心指标 | 辅助指标 |
|---------|---------|---------|
| 单步前向 | cosine sim + 相对误差 | 逐层 max abs diff |
| 单步梯度 | cosine sim + 梯度范数比 | 逐参数 max abs diff |
| 训练过程 | loss 平均相对误差 + 曲线可视化 | grad norm 曲线、收敛 step 数 |
| 最终结果 | 任务指标（accuracy/F1/BLEU 等） | 最终权重 cosine sim |

单一指标不充分，必须联合判断。所有阈值为参考值，以同模型 GPU 多次运行的自然波动为最终基准。

## 确定性保证

NPU 结果必须可复现。推理和训练各有侧重：

**推理**：若结果不可复现，排查：随机性未关闭、JIT 编译引入不确定性、NPU 内部格式转换引入偏差。

**训练**（确定性要求更严格，是单步对齐的前提）：
- 固定所有随机种子：`random.seed()` / `np.random.seed()` / `torch.manual_seed()` / `torch.npu.manual_seed_all()`
- `torch.backends.cudnn.deterministic = True`，关闭 `benchmark`
- `torch.use_deterministic_algorithms(True)`（如适用）
- DataLoader 设置 `worker_init_fn` 固定各 worker 种子
- NPU 侧关闭私有格式等可能引入差异的配置（如 `FLAGS_npu_storage_format=0`）
- 确认 NPU 确定性计算模式已开启

## 常见误区

1. 只验证第一步就判定精度正确——某些 bug 只在后续步骤暴露
2. 将浮点累积偏差误判为 bug——关键看 diff 增长是否平滑
3. 与优化后的模型自己对比——必须与原始未优化的 baseline 对比
4. 只看平均指标——必须关注 worst case
5. 对离散输出要求 bit-exact——自回归/随机采样中不可能
6. 未确认基线就写对比脚本——基线来源不对，整个结论无效
7. 看到 mismatch 后临时放宽阈值——阈值必须在比较前声明
8. 训练场景只看 loss 曲线——必须同时验证单步梯度和最终模型质量
9. 跳过单步对齐直接看收敛——单步就有问题时，收敛结果不可信
10. 未固定随机性就做训练对齐——确定性是单步对齐的前提

## 参考资料索引

| 文件 | 加载时机 |
|------|----------|
| [baseline_policy.md](references/baseline_policy.md) | 需要确认基线来源或向用户询问时 |
| [debugging_guide.md](references/debugging_guide.md) | 验证未通过，需要定位精度问题时 |
| [checklists.md](references/checklists.md) | 需要快速核对配置项或证据项时 |
| [model_family_hints.md](references/model_family_hints.md) | 需要按模型类型选择比较对象和判断重点时 |

## 配套脚本（参考实现）

以下脚本位于本 skill 的 `scripts/` 目录下，作为参考实现。agent 应根据具体项目的输出格式和对比需求编写适配的对比脚本，但设计原则必须一致：一键可运行、指标和阈值在运行前声明、结果保存为文件。执行时需使用脚本的完整路径。

| 脚本 | 用途 | 适用场景 |
|------|------|----------|
| `scripts/compare_inference.py` | 比较两份推理输出（JSON/CSV/npy/目录） | 输出为标准格式文件时可直接使用 |
| `scripts/compare_loss.py` | 比较两份训练日志中的标量信号（loss 等） | 日志格式规整时可直接使用 |
| `scripts/scan_baseline_bundle.py` | 扫描用户提供的基线目录，识别可用文件 | 首次接触基线目录时快速了解结构 |

如果项目输出格式特殊（如自定义二进制、非标准日志），agent 应参考这些脚本的设计模式编写新的对比工具。
