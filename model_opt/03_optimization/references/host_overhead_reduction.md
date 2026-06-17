# Host 开销消减思路

## 问题本质

NPU 设备空闲等待 Host 下发下一个 kernel。profiling 表现为设备利用率低、kernel 间 wait time 大。根因不在算子本身，而在 Host 侧“准备下一个算子”的开销：

- Python 框架调度链（Module.__call__ → hooks → dispatcher → forward）
- 每次算子调用的 tensor 分配（empty_tensor）
- 权重的运行时转置（aten::t）
- 数据格式转换（format_cast）
- 全局 hook 的额外包装（transfer_to_npu）

## 解决思路：分层递进

核心思想是 **减少 Host 在两个 kernel 之间做的事情**，让 device 流水线保持忙碌。

### 思路 1：减少调用次数

**问题**：每次 `nn.Module.__call__` 有固定的 Python 开销（hook 检查、参数验证、dispatcher）。N 层模型每层 M 个子模块 → N×M 次调用。

**解法**：将权重提取到普通 Python 数据结构，用一个扁平函数实现整个 forward，绕过所有 Module 调度层。

**如何找到要改的位置**：
- `model.named_modules()` 列出完整的 Module 嵌套层级，确认调用深度
- `model.named_parameters()` 确认权重命名约定，用于提取到扁平结构

**判断依据**：operator_details.csv 中 `Module.__call__` 相关的 Host Self Duration 占比。

### 思路 2：消除运行时分配

**问题**：每次算子调用时，PyTorch 为输出分配新 tensor（触发 HBM malloc）。

**解法**：
- 预分配输出 buffer，算子通过 `out=` 参数写入已有 buffer
- 相同 shape 的计算复用同一组 buffer（按 shape 做 key 缓存）
- 用原地操作（`add_`、`mul_`）替代非原地版本减少新 tensor 创建

**判断依据**：operator_details.csv 中 `empty_tensor` 的次数和总耗时。

### 思路 3：把“运行时准备”提前到“初始化时”

**问题**：每次 forward 都重复做的事情（权重转置、position bias 计算等），其结果只依赖模型参数或序列长度，不依赖输入内容。

**解法**：
- 权重预转置：初始化时 `.t().contiguous()`，避免每次 forward 的 `aten::t`
- 中间结果缓存：按 key（如 seq_len）缓存结果，相同输入复用
- 跨层 prefetch：当前层 device 计算时，host 提前准备下一层的数据

**判断依据**：trace 中某个 kernel 前的 host gap 里在做什么（view/transpose/compute_bias）。

### 思路 4：消除数据格式转换

**问题**：NPU 算子对输入布局有要求，不符时自动插入 format_cast 或 transpose。

**解法**：
- 统一整个推理过程的维度约定（如全程 2D 避免 squeeze/unsqueeze）
- 选择对 NPU 友好的布局（如 3D bmm 替代 4D matmul）
- 确保张量在传入算子前是连续的（`.contiguous()`）

**判断依据**：profiling 中 format_cast / Transpose 的次数和总耗时。

### 思路 5：清理多余的框架逻辑

**问题**：推理框架中有很多训练时才需要的逻辑（dropout(p=0)、fp16 clamp、dtype 检查、tuple 打包），推理时这些是纯开销。

**解法**：通过 monkey-patch 或本地化代码去除，而不修改框架源码。

**如何定位可去除的位置**：
- 用 `inspect.getsource(Model.forward)` 或 `grep` 在框架源码中定位 forward 实现
- 搜索关键词：`dropout`、`clamp`、`isinstance`、`if hidden_states.dtype`、`(hidden_states,) +`
- 验证方式：注释掉候选行后跑快速验证，确认功能和精度无损再做正式 patch

**判断依据**：operator_details.csv 中 `aten::dropout` 等无效操作的 Host Self Duration。

### 思路 6：清理全局 Hook

**问题**：某些导入（如 `transfer_to_npu`）会注册全局 hook，给每个 tensor 操作套一层 Python 包装。

**解法**：确认代码已正确使用 `.to(device)` 后，删除该导入。

**判断依据**：operator_details 的 Call Stack 中含 `transfer_to_npu` 的 op 数量和总耗时。

## 显存管理原则

如果同时持有原始模型和优化后的模型，HBM 中会存在两份权重。这不仅浪费显存，还会导致 HBM bandwidth 竞争，表现为 **所有 kernel 均匀变慢**（不是某个算子变慢）。

**思路**：构建优化模型时引用原始权重（不是 clone），构建完成后 del 原始模型 + empty_cache。通过 `torch.npu.memory_allocated()` 确认显存未翻倍。

**典型案例**：同时保留原始权重 `_w` 和预转置权重 `_wt` → 显存翻倍 → HBM bandwidth 竞争 → 所有 kernel 均匀变慢（不是某个算子变慢，而是全局退化）。诊断方法：检查 `memory_allocated()` 是否超出模型权重的理论大小。
