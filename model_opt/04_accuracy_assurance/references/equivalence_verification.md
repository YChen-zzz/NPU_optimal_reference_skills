# 推理场景等价性验证协议

## 定位

每次对模型推理路径做**等价替换**（换 API、换融合算子、换算法）后,必须验证语义不变。本文定义标准验证流程,适用于所有推理场景的等价性确认。

> 等价性验证是所有优化的前提约束——不通过验证的改动不允许进入后续流程。

## 验证步骤

### 1. 构造代表性输入

覆盖以下维度:
- **长度**: 最短(1 token) / 中位数 / 最长(接近 max_length)
- **Batch**: batch=1 和 batch>1
- **边界**: padding 位置、mask 边界、空序列(如果模型支持)
- **数值**: 正常范围 + 含极值(接近 fp16 上溢的大值)

输入一旦确定,**固定不变**(存文件复用),确保每次验证可比。

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

分别跑原始实现和替换后的实现,保存输出:

```python
# 原始
out_orig = model_orig(input)
np.save("orig_logits.npy", out_orig.cpu().float().numpy())

# 替换后
out_new = model_new(input)
np.save("new_logits.npy", out_new.cpu().float().numpy())
```

### 4. 指标判定

| 指标 | fp32 阈值 | fp16/bf16 阈值 | 含义 |
|------|----------|---------------|------|
| cosine similarity | >= 0.9999 | >= 0.999 | 整体方向一致性 |
| max_abs_diff | < 1e-4 | < 1e-2 | 最大逐元素偏差 |
| mean_abs_diff | < 1e-5 | < 1e-3 | 平均偏差 |

**判定规则**:
- cosine 和 max_abs_diff **同时满足**才通过
- 任一不满足 → 不通过,需排查原因(见下方"不通过时的处理")

### 5. 确定性验证

同一输入跑两次替换后的实现,确认 bit-exact:

```python
out_run1 = model_new(input)
out_run2 = model_new(input)
assert torch.equal(out_run1, out_run2), "NPU 结果不确定,检查 jit_compile / allow_internal_format"
```

不确定意味着当前 NPU 配置有问题,验证结果不可信。

### 6. 多组输入全部通过

对步骤 1 构造的**所有代表性输入**都重复步骤 3-5。单一输入通过不够——边界 case 可能暴露问题。

## 不通过时的处理

| 情况 | 可能原因 | 处理 |
|------|---------|------|
| cosine 高但 max_abs_diff 超阈值 | 某几个位置有大偏差(通常是 mask/padding 位置) | 检查是否是"无效位置"的垃圾值,若是则排除该位置后重新判定 |
| cosine 低 | 计算逻辑不等价(bug) | 不是精度问题,是代码 bug,回去查实现 |
| 不确定(两次运行不一致) | jit_compile 未关 / allow_internal_format 未关 | 修复配置后重跑 |
| fp16 下偏差大但 fp32 下通过 | 精度累积放大 | 可接受(fp16 本身有此特性),但需确认是否影响下游任务指标 |

## 与 04_accuracy_assurance 现有流程的关系

- 本协议用于**单次等价替换的即时验证**(Phase 3 中每步改动后)
- `04_accuracy_assurance/SKILL.md` 的全量精度验证(Phase 4 Level 2)是**累积多步改动后的最终确认**
- 两者不冲突: 等价性验证保证每步正确 → 全量验证保证累积无退化
