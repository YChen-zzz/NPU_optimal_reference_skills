# Profiling 特征 → 行动映射

## 使用方法

拿到 profiling 数据后，按下表逐行对照。左列是在 profiling CSV / trace 中看到的**具体特征**，右列是建议的**第一个行动**和对应原语。

不需要从头到尾扫完——找到最显著的特征（占比最大 / 次数最多），执行对应行动，优化后重新 profiling，再回到这张表。

## op_statistic.csv 特征

| 看到什么 | 说明什么 | 第一个行动 | 原语 |
|---------|---------|-----------|------|
| 同一算子（如 Transpose）调用 800+ 次 | layout 选择不当，算子间反复转换 | 追踪这些 Transpose 来自哪些 `.permute().contiguous()` 链，统一 layout 约定使整条链消失 | 去重 |
| Mul + Sigmoid 成对高频出现 | gating 未原地化 | `x.sigmoid_()` 替代 `torch.sigmoid(x)`；如有配套融合算子（如 `npu_swiglu`）直接替换 | 去重+复用 |
| MemSet 1000+ 次 | 大量临时 buffer 被反复分配清零 | 定位高频分配源（通信 buffer / `torch.cat` / `F.one_hot`），改为预分配+复用 | 复用 |
| Slice 大量出现且伴随 Transpose | `chunk` + `permute` 链 | `view` + index 替代 `chunk`+`squeeze`；检查是否可合并上游 Linear 消除 split 需求 | 去重 |
| ConcatD 高频出现 | `torch.cat` / `all_gather` + `cat` | `all_gather_into_tensor` 替代 list-based `all_gather`；`torch.zeros`+`narrow().copy_()` 替代 `cat` | 复用 |
| BatchMatMulV2 次数远超预期 | `einsum` 被 PyTorch 拆成多个 bmm | 改为显式 `bmm` / `matmul` 调用 | 去重 |
| hcom_broadcast / hcom_allgather 占比 > 30% | 通信在关键路径上串行等待 | 检查每个通信操作前后是否有无依赖的计算可放到另一个 stream | 掩盖 |
| Pow + ReduceMean + Rsqrt + Mul 成组出现 | RmsNorm 未融合 | `torch_npu.npu_rms_norm` 替代 | 去重 |
| FlashAttentionScore 少但前后碎片多 | attention 前后处理未优化 | QKV projection 后的 reshape/permute 简化；gating sigmoid 原地化 | 去重 |

## kernel_details.csv 特征

| 看到什么 | 说明什么 | 第一个行动 | 原语 |
|---------|---------|-----------|------|
| stall_ratio > 500% | device 大部分时间在等 host | 转到 operator_details 定位 host 时间花在哪（见下一表） | — |
| 少数 kernel wait time 极大（> 10ms）| 该 kernel 前有同步操作 | trace 中找到该 kernel，检查前面是否有 `.item()` / `.to(device)` / `empty_cache()` | 去重 |
| wait time 分布均匀（每个 kernel 都有 50-100μs 等待）| host dispatch 开销均匀分摊 | 考虑图编译（消除逐算子 dispatch）或 flat forward（绕过 Module.__call__）| 掩盖/去重 |

## operator_details.csv 特征

| 看到什么 | 说明什么 | 第一个行动 | 原语 |
|---------|---------|-----------|------|
| `empty_tensor` Host Self Duration 占比 > 30% | allocator 同步是主要瓶颈 | 定位哪些操作触发 `empty`（`cat` / `one_hot` / 非原地算子），改为预分配 | 复用 |
| `aten::dropout` 出现且 p=0 | 训练遗留代码在推理时白跑 | monkey-patch 或本地化去除 | 去重 |
| `Module.__call__` 相关耗时占比 > 40% | Python 框架调度栈太深 | 考虑 flat forward（提取权重到普通数据结构，绕过 Module）| 去重 |
| `aten::t` / `aten::transpose` Host Self Duration 高 | 权重每次 forward 都做转置 | 初始化时预转置 `.t().contiguous()`，forward 中直取 | 复用 |

## trace_view.json 特征

| 看到什么 | 说明什么 | 第一个行动 | 原语 |
|---------|---------|-----------|------|
| NPU 计算流有明显空泡，空泡前后分别是通信和计算 | 通信-计算未重叠 | 检查通信结果和紧接的计算之间是否有无依赖的其他计算，用 comm_stream 重叠 | 掩盖 |
| NPU 计算流有短空泡，空泡前有 `aclnnInplaceSigmoid` 等不预期算子 | CANN 隐式同步 | 对照 NPU checklist 排查（见 `npu_checklist.md`）| 去重 |
| 某段 NPU 空泡前有 `SynchronizeStream` | 显式 H2D/D2H 同步 | 搜索 `.item()` / `.numpy()` / `.to(device)` 调用，缓存或消除 | 复用/去重 |
| 多个 recycle / diffusion step 间有大空泡 | `empty_cache()` 或阶段切换同步 | 确认是否可去除 `empty_cache()`（仅极端显存压力下保留）| 去重 |

## 多种特征并存时的优先级

```
1. 图编译可行？→ 优先尝试（收益上限最高）
2. 显式同步（.item / H2D / empty_cache）→ 消除（单点修复，收益确定）
3. allocator 同步（empty_tensor 占比高）→ 预分配替代（收益确定）
4. 通信串行（hcom 占比高）→ 通信-计算重叠（收益取决于可重叠的计算量）
5. 碎片算子（Transpose / Slice / MemSet 次数多）→ 合并/消除（单点收益小但累积可观）
6. kernel 本身慢（Compute-Bound）→ 融合算子 / 降精度（需验证精度）
```
