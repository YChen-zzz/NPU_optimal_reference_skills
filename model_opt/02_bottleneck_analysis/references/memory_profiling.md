# 内存问题诊断

当 `parse_memory_record` 或 `parse_operator_memory` 发现内存异常时，用本文定位具体原因。

## 核心概念

OOM 由**峰值**决定，不由累计分配量决定。峰值 = 某一时刻所有同时存活张量的大小之和。

关键区分：
- `Reserved`（allocator 池大小）≠ 实际使用——池会预留余量
- `Allocated`（张量实际占用）= 真正的使用量
- `Reserved - Allocated`（gap）= 碎片或预留空间

## 从脚本输出到问题定位

### 看到：Reserved 持续增长

可能原因：
- 动态 shape 导致 allocator 不断申请新 segment
- 问题特征：Reserved 持续上升不回落

### 看到：高频抖动（>50MB 跳变多）

可能原因：
- 每次 forward 都 `torch.cat` / `torch.zeros` 创建大临时 tensor
- 问题特征：反复分配释放同尺寸大块，allocator 池频繁伸缩

### 看到：Reserved - Allocated gap 大且增长

可能原因：
- 碎片化——池中有空闲但不满足新请求的连续大小
- 问题特征：Reserved - Allocated gap 随时间扩大

### 看到：短命大 tensor 量大（operator_memory）

可能原因：
- 中间计算结果每次重新分配又立即释放
- 问题特征：相同 shape 的 tensor 在 forward 中反复创建销毁

### 看到：重复同尺寸分配（operator_memory）

可能原因：
- 同一个 op 每次 forward 都分配相同大小的输出 tensor
- 问题特征：operator_memory 中同尺寸分配次数远大于 1

## 常见陷阱

### 容器引用阻止释放

Python 引用计数不会回收仍被容器引用的对象：
```python
cache = {'pair': big_tensor}
del big_tensor  # 无效——cache 仍持有引用
cache['pair'] = None  # 必须显式置空
```
**症状**：`parse_memory_record` 显示 Allocated 在应释放的时间点没有下降。

### torch.cat 的隐藏分配（尤其是 KV cache）

`cat` 内部调用 `empty()` 分配输出 buffer。NPU 上池不足时 `empty()` 触发同步等待——在 profiling 中表现为 pipeline bubble 而非 OOM。

在自回归 decode 场景中，`DynamicCache.update()` 每步每层都 `torch.cat([old_cache, new_kv])`，产生大量 alloc+copy。

**症状**：`parse_op_statistic` 中 ConcatD/MemSet 大量出现 + `parse_operator_memory` 中相同尺寸的 tensor 反复分配。

**问题特征**：每步每层都产生一次 alloc+copy，累计开销显著。

### 同时持有两份权重

预转置权重如果用 `clone()`（新分配）而非原地操作，HBM 中会有两份权重副本。不仅浪费显存，还导致 HBM bandwidth 竞争——表现为**所有 kernel 均匀变慢**。

**症状**：`parse_memory_record` 中 Allocated 超出"模型参数量 × sizeof(dtype)"的理论值。

### 张量大小速算

```
shape × sizeof(dtype)
例：[4096, 4096, 128] × bf16 = 4096 × 4096 × 128 × 2 bytes = 4.29 GB
```
