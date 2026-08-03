# 复用——一次获取，反复使用

## 原理

对关键路径上的每一次"创建"操作（内存分配、H2D 传输、计算），问：

**这个结果之后还会被需要吗？** 如果是 → 保留它，后续直取而非重新创建。

两个维度：
- **时间维度复用**：一次计算/分配的结果跨多次调用使用（缓存、预计算）
- **空间维度复用**：同一块内存被不同操作反复写入（buffer 复用、原地操作）

## 判断方法

以下现象提示可以用"复用"手段优化：
- 相同尺寸的 tensor 被反复分配释放 → 预分配 buffer，跨调用复用
- 同一操作（如 `.to(device)`、`.t()`、`torch.tensor(scalar)`）在热路径中多次执行且结果不变 → 缓存结果
- `torch.cat` / `F.one_hot` 等在热路径中分配中间张量 → 用预分配 buffer 替代

## 方案示例

### 时间维度：预计算 + 缓存

对热路径中的每个操作问"这个操作的输出在不同输入下会变吗？"——只依赖权重/模型参数的提到加载期一次性执行；只依赖序列长度等形状的按 shape 做 key 缓存；跨步不变的首次计算后缓存。

**推理权重预拼接（Inference Prepack）**：多个 Linear 共享输入 → 拼接权重做一次 GEMM（合并是去重，但预拼好的权重跨全部 forward 复用是复用）。每个模块实现 `prepack_inference_weights()`，预拼接权重作为普通属性（非 `nn.Parameter`），不参与 `state_dict`。

**常量/标量设备缓存**：CPU 常量每次 `.to(device)` → 首次传输后缓存到 dict，后续直取。标量 `torch.tensor(eps, device=npu)` 同理。推理开始前主动预热缓存。

**D2H 结果缓存**：`mask.all().item()` 的结果如果跨步不变 → 首次做一次，缓存 bool 值。

### 空间维度：同一块内存反复使用

**预分配 + 原地写入**：`torch.concatenate` 内部每次调用 `empty()` 分配输出。替代模式：`torch.zeros` 单次预分配 + `narrow().copy_()` / `scatter_()` 原地写入。如果输出大小是编译期常量，可在计算开始之前分配 buffer，使分配与前序计算流水重叠。

**通信 buffer 复用**：多卡通信原语每次调用分配临时 buffer。全局缓存按 `(shape, dtype, device)` 复用。

**原地操作**：`torch.sigmoid(x)` 分配新张量 → `x.sigmoid_()` 复用输入内存。安全条件：输入后续不再以原始值被使用。适用于所有 unary 操作和标量缩放。

**view 替代 clone**：`weight[:c, :]` 是零拷贝 view，`weight[:c, :].clone()` 是新分配。拆分权重时用 view 可避免显存翻倍。

**expand 替代 repeat**：`tensor.expand(...)` 是零拷贝视图，`tensor.repeat(...)` / `torch.tile(...)` 实际分配新内存。只需广播语义时用 `expand`。

## 约束

### Checkpoint 兼容性

预拼接权重作为普通属性不参与 `state_dict`，旧 checkpoint 可直接加载。但结构性融合（QKV 合并、layout 重排、算子拆并等）会改变权重的 key/形状，此时须提供加载时 remap 函数把旧 checkpoint 转成新结构。remap 代码与 modeling 同处，加载后必须与未优化模型做等价性验证。
