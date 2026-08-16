# 推理场景等价性验证协议

## 定位

每次对模型推理路径做**等价替换**（换 API、换融合算子、换算法）后，必须验证替换前后计算等价。本文定义单步即时验证流程——比较替换前后的 tensor 输出，确认数值一致。

> 等价性验证是所有优化的前提约束——不通过验证的改动不允许进入后续流程。
> 本文关注单步等价（tensor 级数值对比）。下游功能指标（benchmark 分数、LPIPS、RMSD 等）的验证见 SKILL.md Level 2 全量验证。

## 验证步骤

### 1. 构造代表性输入

覆盖以下维度:
- **长度**: 最短(1 token) / 中位数 / 最长(接近 max_length)
- **Batch**: batch=1 和 batch>1
- **边界**: padding 位置、mask 边界、空序列(如果模型支持)
- **数值**: 正常范围 + 含极值(接近 fp16 上溢的大值)

输入一旦确定，**固定不变**(存文件复用)，确保每次验证可比。

### 2. 关闭随机性

```python
model.eval()
torch.manual_seed(42)
# NPU 确定性条件:
torch.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format = False
# 确认无 dropout / random 路径
```

### 3. Forward 对比

分别跑原始实现和替换后的实现，保存输出 tensor：

```python
out_orig = model_orig(input)
out_new = model_new(input)
```

按输出类型选择距离函数（此处都是 tensor 级数值对比）：

**连续向量**（logits、forces、embedding、feature map）：
- cosine similarity（方向一致性）+ max abs diff（worst case）
- 两者同时满足才通过

**聚合标量**（energy、loss）：
- 相对误差 |a-b|/|a|（当 baseline 接近 0 时用绝对误差）

**分布**（attention weights、概率分布）：
- KL 散度

**离散序列**（token ids）：
- 完全匹配率（离散输出无渐变）

### 4. 距离判定

参考起点：fp32 cosine >= 0.9999 / max_abs < 1e-4；fp16/bf16 cosine >= 0.999 / max_abs < 1e-2。最终以项目自然波动基准为准。


### 5. 确定性验证

同一输入跑两次替换后的实现，确认 bit-exact:

```python
out_run1 = model_new(input)
out_run2 = model_new(input)
assert torch.equal(out_run1, out_run2)
```

不确定意味着当前 NPU 配置有问题，验证结果不可信。

### 6. 多组输入全部通过

对步骤 1 构造的**所有代表性输入**都重复步骤 3-5。单一输入通过不够——边界 case 可能暴露问题。

## 不通过时的处理

| 情况 | 可能原因 | 处理 |
|------|---------|------|
| cosine 高但 max_abs_diff 超阈值 | 某几个位置有大偏差(通常是 mask/padding 位置) | 检查是否是"无效位置"的垃圾值，若是则排除该位置后重新判定 |
| cosine 低 | 计算逻辑不等价(bug) | 不是精度问题，是代码 bug，回去查实现 |
| 不确定(两次运行不一致) | jit_compile 未关 / allow_internal_format 未关 | 修复配置后重跑 |
| fp16 下偏差大但 fp32 下通过 | 精度累积放大 | 可接受(fp16 本身有此特性)，但需确认是否影响下游任务指标 |

## 与 04_accuracy_assurance 现有流程的关系

- 本协议用于**单次等价替换的即时验证**(Phase 3 中每步改动后)——tensor 级数值对比
- `04_accuracy_assurance/SKILL.md` 的 Level 2 全量验证(Phase 4)是**累积多步改动后的最终确认**——可能包含下游功能指标
- 两者不冲突: 等价性验证保证每步 tensor 等价 → Level 2 保证累积无退化且下游功能达标
