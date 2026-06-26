# 源码分析方法（根因定位）

本文描述如何从 Profiling 疑点出发，通过源码分析定位根因。

## 从 Profiling 信号到源码位置

### 步骤 1：确定入口

Profiling 告诉你"什么操作耗时"，但你需要找到它在源码中的位置：

- `operator_details.csv` 的 Call Stack 列直接给出 Python 帧路径和行号
- `kernel_details.csv` 的 Name 列给出 CANN 算子名（如 `aclnnMatmul_MatMulCommon_MatMulV2`），前缀 `aclnn` 后面的部分对应 PyTorch API
- 没有 Call Stack 时，在源码中 grep 相关 PyTorch API（如 `torch.matmul`、`F.linear`）

### 步骤 2：沿调用链追溯

找到直接调用位置后，不要停下——继续看上下文：

**向上追溯**（谁调用了它）：
- 这个函数被哪个循环/模块调用？循环次数和 profiling 中的算子 count 是否一致？
- 是框架代码在调用（如 `Module.__call__` → `forward`）还是项目代码直接调用？
- 调用频次异常时，向上找是哪层循环/递归导致的

**向下深入**（它内部做了什么）：
- 函数内部的控制流：有没有 `if training` / `if self.use_cache` 等分支在推理时走了不必要的路径？
- 内存行为：每次调用是否都重新分配 tensor，还是可以复用？
- 数据变换：是否有 `.contiguous()` / `.transpose()` / `.to()` 触发了物理拷贝？

### 步骤 3：判断根因类型

追溯完成后，根因通常属于以下几类：

| 根因类型 | 源码特征 | 举例 |
|---------|---------|------|
| 调用次数过多 | 循环/递归中的重复调用 | 逐 token 处理 vs batch 处理 |
| 实现方式低效 | 用了多步操作实现可一步完成的功能 | 手动 Pow+Mean+Rsqrt 实现 RmsNorm |
| 数据布局不匹配 | 每次调用前做 transpose/contiguous | weight 和算子期望的维度顺序不一致 |
| 框架开销 | Module.__call__ 的深层调度链 | 48 层模型每层都走完整 hook 链 |
| 不必要的同步 | .item()/.numpy() 等 D→H 操作 | 每步都把 loss 取回 host |
| 训练遗留 | 推理时仍执行 dropout(p=0) 等 | 未清理的训练专用分支 |

## 源码结构理解

### 理解模型整体架构

在做任何优化前，先建立对源码的全局认知：

1. **找入口**：从推理/训练脚本出发（通常是 `main()` → `model(input)`），找到 forward 的第一层调用
2. **画层级**：识别模型的层次结构——哪些子模块被循环调用（如 `for layer in self.layers`）、循环几次
3. **标计算路径**：forward 中的主计算路径是什么（attention → FFN → norm），哪些是分支逻辑
4. **找数据形变点**：tensor 在哪里改变 shape/dtype/device（reshape、transpose、cast、to），这些是潜在的性能拐点

### 关注框架交互

模型不是孤立运行的，它和框架（HuggingFace、PyTorch Lightning 等）的交互可能是瓶颈源：

- **generate 循环**：HF 的 `generate()` 函数几千行，每步都做 stopping criteria 检查、logits 处理、cache 更新——这些可能比模型计算本身还慢
- **Trainer 循环**：每步的 gradient clipping、NaN detection 可能调用 `.item()` 触发同步
- **Module.__call__**：PyTorch 的 Module 调度链（hook 检查、autograd 设置）在模型层数多时累积成本可观

### 关注 NPU 特异性

同一段 PyTorch 代码在 NPU 上的行为可能和 GPU 不同：

- 某些 PyTorch API 在 CANN 中拆成更多子 kernel（如 `F.one_hot` 可能插入额外的 Cast）
- inplace 操作（如 `sigmoid_`）可能触发 CANN 隐式同步
- 4D tensor 的 matmul 可能触发物理 Transpose，而 3D bmm 不会
- 某些 dtype 不支持特定融合算子（如 fp32 不支持 `npu_fusion_attention`）

## 静态代码扫描

不依赖 profiling，直接在源码中搜索已知的性能问题模式：

```bash
# H2D/D2H 同步点
grep -rn "\.item()\|\.numpy()\|\.cpu()" model/ --include="*.py"

# 推理时不需要的训练逻辑
grep -rn "dropout\|F\.dropout" model/ --include="*.py"

# 可能触发物理 Transpose 的操作
grep -rn "\.transpose\|\.permute\|\.contiguous" model/ --include="*.py"

# 每次 forward 都重新分配的 tensor
grep -rn "torch\.zeros\|torch\.empty\|torch\.ones" model/ --include="*.py"

# 框架级开销
grep -rn "torch\.cat\|F\.one_hot" model/ --include="*.py"
```

详见 [03_optimization/references/npu_checklist.md](../03_optimization/references/npu_checklist.md) 获取完整的 NPU 已知问题扫描清单。
