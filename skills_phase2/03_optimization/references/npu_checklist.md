# NPU 已知问题 Checklist

## 使用方法

**每次接到优化任务时**，在分析 profiling 之前或同时，对代码做一轮主动扫描。以下每一项都是 NPU（CANN）上已验证会造成性能问题的已知模式。不依赖 profiling 就能发现——grep / 代码审查即可定位。

标记含义：
- 🔍 = 搜索方法
- ⚡ = 影响程度
- 🔧 = 修复方式

---

## H2D / D2H 同步（pipeline 打断）

### `.item()` / `.numpy()` / `.cpu()` 在热路径中

🔍 `grep -rn '\.item()\|\.numpy()\|\.cpu()' --include='*.py'`
⚡ 每次调用 drain 整个 NPU pipeline（毫秒级）
🔧 缓存结果（`first_run` 时做一次）或无条件执行设备侧操作代替分支判断

### CPU 常量每次 `.to(device)`

🔍 `grep -rn '\.to(device\|\.to(self\.device\|\.npu()' --include='*.py'`，检查是否在 forward 内部
⚡ 每次 H2D 传输 + 同步
🔧 首次 `.to()` 后缓存到 dict，热路径直取

### `torch.tensor(scalar, device=npu)` 在循环内

🔍 `grep -rn 'torch\.tensor(' --include='*.py'`，检查 device 参数
⚡ 每次创建触发 H2D
🔧 按 `(value, dtype, device)` 缓存

### 形状元数据存为 device Tensor

🔍 dataclass / NamedTuple 中的 `torch.Tensor` 字段如果只存形状信息
⚡ 后续 `.numpy()` / `.tolist()` 触发 D2H
🔧 改为 Python tuple

---

## CANN 隐式开销（不预期的额外 kernel）

### `F.one_hot`

🔍 `grep -rn 'F\.one_hot\|functional\.one_hot' --include='*.py'`
⚡ CANN 用 sigmoid-based select kernel 实现，比预期多 kernel 且打断流水
🔧 `torch.zeros(..., dtype=target_dtype) + scatter_(-1, idx.unsqueeze(-1), 1)`

### `torch.where` 条件恒为空

🔍 `grep -rn 'torch\.where' --include='*.py'`，检查条件是否可能为常量 False
⚡ 即使条件永不满足，CANN 仍执行完整 select kernel
🔧 Python 层判断条件是否为空操作，直接跳过

### `F.layer_norm` 传 `bias=None`

🔍 模型中 `nn.LayerNorm(... bias=False)` 或 `elementwise_affine=True` 但无 bias
⚡ CANN 内部每次调用分配临时全零 bias buffer
🔧 预创建 `_zero_bias = torch.zeros(shape)` 传入

### `bool_tensor.prod(dim)`

🔍 `grep -rn '\.prod(' --include='*.py'`
⚡ ReduceProd 非 NPU 原生高效算子，拆成多步
🔧 `(tensor.sum(dim) == tensor.shape[dim])` 用 ReduceSum 替代

### 2D advanced indexing `tensor[row_idx, col_idx] = val`

🔍 代码中用两个 index tensor 做赋值
⚡ NPU 逐元素 host 下发，每个赋值一次 dispatch
🔧 `tensor.view(-1).scatter_(0, row_idx * N + col_idx, val)`

---

## 内存/Allocator 问题

### `torch.concatenate` / `torch.cat` 在热路径

🔍 `grep -rn 'torch\.cat\|torch\.concatenate' --include='*.py'`，排除 `__init__`
⚡ 每次内部 `empty()` 可能触发 allocator 同步
🔧 `torch.zeros` 预分配 + `narrow().copy_()`

### `F.one_hot` 的 int64 中间张量

🔍 同上 F.one_hot
⚡ int64 比 float32 大 2 倍，后续 `.to(float)` 又分配一次
🔧 直接在目标 dtype 上 `zeros + scatter_`

### dict/list 持有大张量未释放

🔍 检查 `prev = {}` / `cache = []` 等容器在使用完大张量后是否置 None
⚡ 大张量无法被 allocator 回收
🔧 使用完毕后 `container[key] = None`

### `torch.npu.empty_cache()` 在热路径

🔍 `grep -rn 'empty_cache' --include='*.py'`
⚡ 强制同步 + 回收，打断 pipeline
🔧 仅在 `--memory_saving` 模式的大阶段切换点保留，默认删除

---

## 框架/调度开销

### `transfer_to_npu` 全局 hook

🔍 `grep -rn 'transfer_to_npu' --include='*.py'`
⚡ 每个 tensor 操作套一层 Python 包装
🔧 确认 `.to(device)` 已正确使用后删除该 import

### `tqdm` / `trange` 在推理循环

🔍 `grep -rn 'tqdm\|trange' --include='*.py'`
⚡ 每迭代 Python 开销，污染 profiling trace
🔧 正式推理中替换为 `range()`

### `dropout(p=0)` 未清理

🔍 `grep -rn 'dropout' --include='*.py'`，检查推理路径中 p=0 的 dropout
⚡ NPU 仍会 dispatch 一个空操作 kernel
🔧 推理路径中去除或 monkey-patch

### `einops` / `einsum` 字符串在热循环

🔍 `grep -rn 'einops\|einsum' --include='*.py'`
⚡ 每次调用解析字符串 + 可能拆成多个 kernel
🔧 热路径改为 `view` + `permute` + `bmm` / `matmul`

---

## 多卡并行

### `dist.isend` / `dist.irecv`（HCCL P2P）

🔍 `grep -rn 'dist\.isend\|dist\.irecv' --include='*.py'`
⚡ HCCL P2P tag 匹配不可靠
🔧 用 broadcast 循环替代

### `tensor.chunk()` 直接传给 `all_to_all`

🔍 `all_to_all` 调用前的输入是否 contiguous
⚡ HCCL 要求 contiguous，非连续张量报错
🔧 chunk 后加 `.contiguous()`，或用 `all_to_all_single`

---

## 扫描流程建议

```
1. 先跑一遍上面的 grep 命令，标记所有命中位置
2. 过滤：只保留热路径上的命中（__init__ / 一次性代码中的不管）
3. 按影响程度排序：H2D/D2H 同步 > allocator 同步 > CANN 隐式开销 > 框架开销
4. 逐一修复 + 精度验证
5. 修完后跑 profiling 确认效果
```
