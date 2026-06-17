---
name: NPU 精度保证
description: 确保 NPU 优化后的模型精度正确，包括基线管理、分层验证和精度调试。当用户关注精度验证、输出对比或精度问题定位时触发。
---

# NPU 精度保证

## 核心思路

优化可能在任何环节引入精度问题。精度验证的目标是：确认优化后的输出与可信 baseline 在数值上足够接近。

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

## 验证策略：分层递进

**Level 1 快速验证**（每次修改后）：少量代表性样本，检查输出与基线的数值差异。三要素：功能、精度、性能。

**Level 2 全量验证**（每批提交前）：完整测试集，覆盖所有推理路径，关注 worst case 而非平均值。通过后才可 commit。

**Level 3 精度调试**（验证未通过时）：详见 [debugging_guide.md](references/debugging_guide.md)

## 指标选择原则

| 输出类型 | 思路 | 常用指标 |
|---------|------|----------|
| 连续向量 | 方向一致性 + 幅值偏差 | cosine similarity + max abs diff |
| 离散输出 | 匹配率（自回归场景对微小偏差敏感） | 匹配率（仅参考） |
| 聚合标量 | 相对误差 | \|a-b\|/\|a\| |

单一指标不充分，必须联合判断。阈值因任务而异，以同模型 GPU 多次运行的自然波动为参考。

## 确定性保证

NPU 推理结果必须可复现。若不可复现，排查：随机性未关闭、JIT 编译引入不确定性、NPU 内部格式转换引入偏差。

## 常见误区

1. 只验证第一步就判定精度正确——某些 bug 只在后续步骤暴露
2. 将浮点累积偏差误判为 bug——关键看 diff 增长是否平滑
3. 与优化后的模型自己对比——必须与原始未优化的 baseline 对比
4. 只看平均指标——必须关注 worst case
5. 对离散输出要求 bit-exact——自回归/随机采样中不可能
6. 未确认基线就写对比脚本——基线来源不对，整个结论无效
7. 看到 mismatch 后临时放宽阈值——阈值必须在比较前声明

## 参考资料索引

| 文件 | 加载时机 |
|------|----------|
| [baseline_policy.md](references/baseline_policy.md) | 需要确认基线来源或向用户询问时 |
| [debugging_guide.md](references/debugging_guide.md) | 验证未通过，需要定位精度问题时 |
| [checklists.md](references/checklists.md) | 需要快速核对配置项或证据项时 |
| [model_family_hints.md](references/model_family_hints.md) | 需要按模型类型选择比较对象和判断重点时 |

## 配套脚本

| 脚本 | 用途 |
|------|------|
| `scripts/compare_inference.py` | 比较两份推理输出（JSON/CSV/npy/目录） |
| `scripts/compare_loss.py` | 比较两份训练日志中的标量信号（loss 等） |
| `scripts/scan_baseline_bundle.py` | 扫描用户提供的基线目录，识别可用文件 |
