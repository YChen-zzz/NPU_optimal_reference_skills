# 去重——消除不必要的工作

## 原理

对关键路径上的每一个操作，问两个问题：

1. **这个操作的工作是必要的吗？** 如果它的输出从未被使用、或条件永远不满足、或等价于空操作——删除它。
2. **这个操作能和相邻操作合并吗？** 如果 N 个操作共享输入或彼此独立，合并为 1 个操作可消除 N-1 次 dispatch 开销。

## 判断方法

以下现象提示可以用"去重"手段优化：
- 同类算子调用次数异常高但单次耗时极低 → 碎片化调用，考虑合并为单次大算子
- 出现不预期的算子类型（非项目代码触发）→ 框架或 CANN 自动插入的冗余操作
- 多个独立 Linear/MatMul 共享同一输入 → 可合并为单次 GEMM
- 源码中结果被丢弃的计算分支 → 死代码，直接删除
- 推理时仍执行训练专用逻辑（dropout p=0、grad 相关）→ 清理

## 合并：N 次 → 1 次

### 多个同源 Linear → 单次 GEMM

N 个 Linear 共享同一输入时，运行时拼接权重做 1 次大 GEMM：

```python
# 2 次 GEMM → 1 次
pg = F.linear(x, torch.cat([W_proj, W_gate], dim=0))
proj, gate = pg[..., :c], pg[..., c:]
```

原理适用于所有"同一输入 × 不同权重"的场景。

### 通用库调用 → 原生操作

`einsum` 字符串解析 + 多步拆解、`einops.rearrange` 格式解析——这些在 Python 层有固定开销，且可能被后端拆成多个 kernel。直接用 `bmm`/`matmul`/`view`+`permute` 等原生调用减少 kernel 数。

### struct-of-arrays → packed tensor

分量独立存储（如 `x, y, z` 三个张量）导致每个运算拆成 3 路 dispatch。改为 `[N, 3]` packed tensor 后一次向量化操作完成。

### 融合算子替换

NPU 提供将已知算子模式合并为单 kernel 的融合算子（如 `npu_rms_norm`、`npu_fusion_attention`、`npu_swiglu`）。效果 = 消除 N-1 次 dispatch gap。

**注意**：融合不一定更快——当原始算子组计算量极小时，融合 kernel 的固定初始化开销可能超过节省的 dispatch gap。必须 profiling 对比。

## 消除：去掉不产生有用结果的操作

### 死代码

函数返回 tuple 但调用方只取第一个值——删除未使用分支的全部计算。保留 `__init__` 中的参数定义确保 checkpoint 兼容，只删 forward 中的执行路径。

### 冗余 layout 变换

反复 `permute` + `.contiguous()` 说明数据 layout 选择有问题。统一 layout 约定后整条 permute 链消失。

`matmul` 可以直接在 non-contiguous 转置视图上执行 transposed GEMM——大量 `.contiguous()` 是不必要的。

### 推理时的训练遗留

`dropout(p=0)`、`fp16 clamp`、dtype 检查、`tuple` 打包——推理时不产生效果但消耗 dispatch。通过 monkey-patch 或本地化代码去除。

### 全局 Hook 包装

某些导入（如 `transfer_to_npu`）注册全局 hook，给每个 tensor 操作套一层 Python 包装。确认代码已正确使用 `.to(device)` 后删除。

### 不必要的 H2D / D2H

形状元数据存为 device Tensor → `.numpy()` 触发 D2H → 改用 Python tuple。条件永远为空的 `torch.where` → Python 层跳过。先在 NPU 做 dtype cast 再搬 CPU → 先搬 CPU 再 cast。

## 框架调度层消除

每次 `nn.Module.__call__` 有固定 Python 开销（hook 检查、参数验证、dispatcher）。N 层 × M 子模块 = N×M 次调用。

极端情况下可将权重提取到普通数据结构，用扁平函数实现 forward，绕过 Module 调度链。判断依据：`Module.__call__` 相关 Host Self Duration 占比。
