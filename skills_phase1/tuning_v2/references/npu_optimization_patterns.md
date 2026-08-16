# 优化方法论与 NPU Patterns

## 四维度发现框架

对热路径上每个操作，用四维度主动提问（不依赖 profiling）：

| 维度 | 问题 | 信号 |
|------|------|------|
| **去重** | 有多余工作? | 同输入多次调用; 推理时不走的训练分支; 往返 dtype 转换 |
| **复用** | 结果/buffer 能跨调用复用? | 输出只依赖 weights/shape; 同形状反复分配 |
| **掩盖** | 延迟能否被重叠? | 独立计算可并行; 通信可与计算重叠; kernel launch gap |
| **替换** | 有更高效等价写法? | 多步 op 可被单 fused API 替代; 不同 API 映射更优 kernel |

## 优化层级详解

### L0: API 参数 / 环境变量

零代码改动。对每个算子 API 查看完整签名，对比 GPU 侧同功能 API 传了哪些参数。检查运行时环境变量（async dispatch, communication buffer, compile cache）。

**NPU 常用环境变量** (在 run.sh 中 export，训练代码无需改动):

| 变量 | 值 | 作用 | 注意 |
|------|---|------|------|
| `TASK_QUEUE_ENABLE` | `1` 或 `2` | 异步 kernel 派发。减少 host 等待，让 kernel launch 流水化。`2` 更激进但可能引入调度 overhead | 必须在 `import torch_npu` 前设置。不同 workload 最优值不同，需实测对比 |
| `HCCL_BUFFSIZE` | `120` (MB) | 通信 buffer 大小。增大可减少分片次数但占用更多内存 | 默认值可能不是最优，需要 ablation |
| `COMBINED_ENABLE` | `1` | 通信算子合并。将相邻小 collective 合并为一次 | 不一定有效，某些 workload 反而更慢 |
| `ASCEND_LAUNCH_BLOCKING` | `1` | 同步执行（仅 debug 用）。让所有 op 同步执行方便定位报错 | **性能测试时禁用** |

所有环境变量都应作为独立的 L0 候选方案在 Lab 中逐个测试（不要一次全加）。

### L1: 消除冗余

**静态扫描** (grep, 不需 profiling):
- `.item()` / `.cpu()` / `.numpy()` / `.tolist()` 在循环中 → 强制 device→host sync
- `.to(device)` / `torch.tensor(scalar, device=...)` 在 forward 内 → 每步 H2D
- `.float()` / `.type_as()` → 确认 GPU 侧是否真需要（通常是移植遗留）
- 相同输入的重复计算 → 提到循环外

**复用分类**:
- 输出只依赖 weights → 加载时预计算
- 输出只依赖 shape → 按 shape key 缓存
- step 内不变 → 前向开头计算一次

**内存复用**:
- 优先 zero-copy 操作 (`view`, `expand`) 而非 allocating (`clone`, `repeat`)
- `torch.cat` 在循环中 → 预分配 buffer + `narrow().copy_()`
- In-place ops 复用内存 — 仅在原值不再需要时安全

### L2: 平台官方融合 API

**搜索**: `dir(torch_npu)` 按功能关键词过滤，查 docstring 确认签名。

**验证**:
1. Input dtype/shape 兼容
2. Forward 输出正确
3. **Backward 正确传播梯度**（部分 API 无 autograd 注册）
4. 多 shape 测试

**精度对比陷阱**: 测试 NPU API 时，必须和 control 使用**相同的精度路径**。如果 control 有额外的 `.float()` cast 而 NPU API 直接接受 bf16，对比时需要把 control 的 `.float()` 也去掉（先确认这个 cast 是否是移植遗留），否则在测量速度时会把 "NPU API 本身" 和 "去掉 cast" 的增益混在一起，可能得出"API 更慢"的错误结论。

**原则**:
- 平台 fused op 通常内部精度处理更优（accumulated f32 etc）
- 浮点非结合律: 改变计算顺序在低精度下可能累积误差 → 实测验证
- 记录失败的替换尝试避免重复探索

### L3: 等价手动改写

**同一数学运算的多种表达**: PyTorch 中数学等价的表达式可能走完全不同的 dispatch 路径，在 NPU 上性能差异可达 30-50%。对每个热路径中的 op，列出所有等价表达并在 Lab 中对比：
- method 形式: `x.square()`, `x.pow(2)`, `x.mul(x)`, `x * x`
- functional 形式: `torch.square(x)`, `torch.pow(x, 2)`, `torch.mul(x, x)`
- in-place 形式: `x.mul_(x)`, `x.square_()`

哪种最快取决于后端实现——不要假设 "专用方法一定比通用表达快"。总是 benchmark 验证。

**其他常见 pattern**:
- 消除 double-transpose: `F.linear(x, w.T)` → `torch.matmul(x, w)`
- `permute + contiguous` 链 → 上游选择正确 layout
- Flat forward: 从 `nn.Module` 提取 weights 用 `F.*` 直接调用（消除框架 dispatch）
- 结构性权重融合（QKV 合并等）→ 需配套 state_dict remap

### L4: torch.compile

**适合**: 多个 elementwise chain; 无 data-dependent 控制流; tensor 足够大; 纯数学函数。

**不适合**: 单个大算子为主体; 很小 tensor; 含 trainable param scalar 梯度路径。

**scope 策略**: 从纯计算、shape 稳定、高频执行的片段开始，逐步扩大。验证 compile 后确实产生 kernel fusion（对比 pre/post-compile profile）。

**编译器 PATH 排错**: 如果 compile 报 "npuc"、"bishengir" 相关错误（如 "Invalid bishengir path format"），说明 NPU kernel 编译器不在 PATH 中。执行 `find /usr/local/Ascend -name "bishengir-compile"` 找到实际路径，加入 PATH。这个二进制通常在 `ascend-toolkit/*/bisheng_toolkit/bishengir/bin/` 或 `cann-*/tools/bishengir/bin/` 下。

**多卡 compile 崩溃**: 16 卡训练首次 compile 可能因多进程同时写同一个 `kernel_meta/` 目录崩溃（错误信息含 "unable to open output file kernel_meta/..."）。解决: 设置 `export TORCH_NPU_COMPILE_CACHE_DIR=/tmp/npu_compile_cache` 或每 rank 独立 cache 路径。

**多卡注意**: `dynamic=False` 对固定 shape-per-regime 最优（N regime = N 次编译后缓存）。

### L5: Custom Autograd Function

**何时**: API forward 快但 backward 未注册/错误。

**约束**: Python dispatch overhead 在多卡放大 → 必须 ablation。`save_for_backward` tensor 占内存到 backward 完成。

### L6: 自定义 Kernel

L0-L5 全部不足时。

## 掩盖延迟

- 两段无数据依赖的工作 → 不同执行单元重叠
- 双 buffer ping-pong: 预取 block k+1 同时计算 block k
- Graph compile: 消除 host dispatch gap + 合并 kernel launch (profile 显示大量短 kernel + gap 时最有效)

## 验证门禁

**轻量 (~60s)**:
- 等比压缩 total steps + 所有 schedule
- 固定 seed/data/state，对比 val loss

**正确性标准**:
- 前向: max_abs_diff 在 dtype 精度内
- 梯度: cosine_similarity > 0.9999
- 训练: val_loss 在自然波动内

**何时完整验证**: 改 backward/optimizer/communication/state; 新 20% milestone; 最终版。

## 平台陷阱速查

| 现象 | 原因 | 解决 |
|------|------|------|
| `.tolist()` 在循环中慢 | 强制 device→host sync | 预计算/缓存 |
| 移植后多了 `.float()` | 移植遗留 | 确认 GPU 是否需要 |
| API backward grad=None | 未注册 autograd | Custom Function 包装 |
| compile 后多卡崩溃 | kernel_meta 多进程冲突 | 独立 cache dir |
| compile 后 loss 发散 | trainable scalar 参与梯度 | 只 compile 纯数学函数 |
| 单机 2x 多卡无效 | dispatch overhead 被 comm 隐藏 | 必须 ablation |
| `dynamic=True` 比 False 慢 | guard checking overhead | 固定 shape 用 False |
