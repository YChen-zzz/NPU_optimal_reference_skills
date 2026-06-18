# 显存 Profiling 与峰值分析

## 核心概念

OOM 由**峰值**决定，不由总量决定。峰值 = 某一时刻所有同时存活的张量大小之和。优化显存的前提是准确定位峰值瞬间。

## 分析方法

### 逐行内存追踪

在 forward 关键位置埋点 `torch.npu.memory_allocated()`：

```python
def log_mem(tag):
    alloc = torch.npu.memory_allocated() / 1024**3
    print(f"[MEM] {tag}: {alloc:.2f} GB")
```

追踪的是**同时存活**的张量总量。一个张量在 `del` 或引用归零后会被释放，后续分配可复用其空间。

### 峰值定位

1. 列出 forward 路径中所有大张量及其**存活区间**（从分配到最后一次被引用）
2. 找叠加最多的时间点——这就是峰值瞬间
3. 在该瞬间逐个审查：哪些张量此刻还必须活着？哪些其实已经可以释放了？

### 张量大小速算

```
shape × sizeof(dtype)
例：[4096, 4096, 128] × fp32 = 4096 × 4096 × 128 × 4 bytes = 8.59 GB
```

## 常见显存陷阱（诊断线索）

### 字典/容器引用持有大张量

Python 的引用计数不会回收仍被容器引用的对象。`del tensor` 不够——如果 dict/list 还持有引用：

```python
prev = {'pair': big_tensor}
# del big_tensor 无效——prev['pair'] 仍持有引用
prev['pair'] = None          # 必须显式置 None
```

**诊断**：如果 `memory_allocated()` 在使用完大张量后没有下降，检查是否有容器仍持有引用。

### `torch.concatenate` / `torch.cat` 的隐藏分配

`cat` 内部调用 `empty()` 分配输出 buffer。NPU 上 memory pool 不足时 `empty()` 触发同步等待——这个同步在 profiling 中表现为 pipeline bubble，而非显式的 OOM。

**诊断**：profiling 中 `MemSet` 算子大量出现 + `empty_tensor` 次数异常多。

### `F.one_hot` 的 int64 中间张量

`F.one_hot` 输出 int64，后续 `.to(float)` 再分配一次。int64 比 float32 大 2 倍。

**诊断**：`memory_allocated()` 在 `one_hot` 调用前后有跳跃，跳跃量 = shape × 8 bytes。

### 同时持有两份权重

预转置权重如果用 `clone()`（新分配）而非 `view`（共享存储），会导致 HBM 中有两份权重副本。不仅浪费显存，还导致 HBM bandwidth 竞争——表现为**所有 kernel 均匀变慢**（不是某个算子变慢）。

**诊断**：`memory_allocated()` 超出"模型参数量 × sizeof(dtype)"的理论值。

### reserved >> allocated

PyTorch memory pool 中有大量空闲碎片但不满足新请求的连续大小要求。

**诊断**：`memory_reserved()` 远大于 `memory_allocated()`。考虑 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 或在大阶段切换点调用 `empty_cache()`。
