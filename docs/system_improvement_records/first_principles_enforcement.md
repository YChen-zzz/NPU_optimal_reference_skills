# 第一性原理倾向强化：增强现有 parse 脚本 + skill 决策规则

> 基于 Ankh3-large NPU 推理优化实战中，agent 未能从第一性原理出发选择 flat 实现的根因分析。
> 核心改进：不新建脚本，而是对 3 个现有 parse 脚本做小量增强 + 在 skill 中增加跨脚本决策规则。
> 归属：`model_opt/02_bottleneck_analysis/`。

## 1. 问题背景

### 1.1 实战暴露的问题

Ankh3-large 三模式 NPU 推理优化中，agent 始终未提出 flat 实现（完全摆脱 nn.Module 框架），而已有 npu_inference 方案通过 flat 获得更好性能。

### 1.2 根因

不是知识缺失，不是数据缺失——**现有 parse 脚本已输出了大部分原始数据**。agent 没有做的是：

1. **没有把数据组合成决策依据**：step_trace 的 "Free 50.7%" + kernel_details 的 "small kernel 46.5%" + trace_view 的 "dispatch latency 12µs" 分散在 3 个脚本输出中，agent 看了各自输出但没有组合推理
2. **缺少关键计算**：现有脚本有 dispatch latency 但没有 dispatch 总开销估算；有 small kernel (<5us) 但没有 <20us 的完整分布；有 utilization 但没有"理论极限"和"可优化空间"的概念标注
3. **缺少决策规则**：skill 没有告诉 agent "当 Free > 30% 且 short kernel > 60% 时，应考虑减少 op 数量而非优化单个 op"

### 1.3 为什么不新建脚本

逐项对比后发现，绝大部分原始数据已在现有脚本中：

| 数据 | 已有脚本 | 已有输出 |
|------|----------|----------|
| Computing/Free/Total | `parse_step_trace.py` | ✅ `Computing: 614.5ms (49.3%), Free: 631.2ms (50.7%)` |
| 利用率 | `parse_step_trace.py` | ✅ `Device Utilization: 49.3%` |
| Host-Bound flag | `parse_step_trace.py` | ✅ `Moderate Host-Bound` |
| Small kernel (<5us) | `parse_kernel_details.py` | ✅ `Count: 28500 (46.5%)` |
| Fusible sequences | `parse_kernel_details.py` | ✅ 连续小 kernel 序列 |
| Data movement overhead | `parse_op_statistic.py` | ✅ `10.2%` |
| Fragmentation signal | `parse_op_statistic.py` | ✅ `High-count low-duration ops` |
| Dispatch latency (avg/p50/p90) | `parse_trace_view.py` | ✅ `avg=12.0us` |
| Gap distribution | `parse_trace_view.py` | ✅ `<10us: ..., >200us: ...` |
| Device kernel count | `parse_trace_view.py` | ✅ `dev_count` |
| Compute stream active time | `parse_trace_view.py` | ✅ `stream_stats[key][0]` |

**真正缺失的只有 3 项**，都可以通过增强现有脚本解决，不需要新脚本：

| 缺失项 | 原因 | 解法 |
|--------|------|------|
| ① "理论极限"和"可优化空间"标注 | `parse_step_trace.py` 有 utilization 但没有"理论极限=Computing, 可优化空间=Free/Total"的概念标注 | 改 `parse_step_trace.py`，加 2 行 |
| ② Kernel 时长全分布 (<5/5-20/20-50/50-200/>200) | `parse_kernel_details.py` 只统计 <5us，缺 20us 阈值 | 改 `parse_kernel_details.py`，扩展 small kernel 章节 |
| ③ Dispatch 总开销估算 + dispatch/kernel ratio | `parse_trace_view.py` 有 dev_count 和 dispatch latency 但没有相乘 | 改 `parse_trace_view.py`，加 1 个乘法 + 1 个除法 |

## 2. 脚本增强方案

### 2.1 `parse_step_trace.py` — 增加理论极限标注

**改动位置**：Overall 章节，在 utilization 输出之后

**改动内容**：增加 2 行

```python
# 现有代码 (L56-57):
lines.append(f"  Computing: {computing_total/1000:.1f} ms ({computing_total/grand_total*100:.1f}%)")
lines.append(f"  Free:      {free_total/1000:.1f} ms ({free_total/grand_total*100:.1f}%)")

# 新增:
lines.append(f"  Theoretical limit (= Computing): {computing_total/1000:.1f} ms")
lines.append(f"  Optimizable space: {free_total/grand_total*100:.1f}% ((Total - Computing) / Total)")
if free_total / grand_total * 100 > threshold("step_trace", "large_optimizable_space", 30):
    lines.append(f"  → 可优化空间大。非计算开销（dispatch/分配/同步）显著，方案排序应按此上限而非实现难度")
```

**新增阈值** (thresholds.py "step_trace" 节):

```python
"large_optimizable_space": 30,  # % — Free/Total above this = large optimizable space
```

**效果**：agent 运行 parse_step_trace 后直接看到 "Optimizable space: 50.7%" 和引导 hint，不需要自己从 utilization 反推。

### 2.2 `parse_kernel_details.py` — 扩展 small kernel 为完整分布

**改动位置**：Small Kernels 章节 (L262-272)

**改动内容**：把单一的 <5us 统计扩展为 5 档分布

```python
# 现有: 只统计 < small_threshold (5us)
# 改为: 5 档分布

# 在 stream_csv 循环中新增统计:
dur_buckets = {"<5us": 0, "5-20us": 0, "20-50us": 0, "50-200us": 0, ">200us": 0}
# 循环内:
if dur > 0:
    if dur < 5: dur_buckets["<5us"] += 1
    elif dur < 20: dur_buckets["5-20us"] += 1
    elif dur < 50: dur_buckets["20-50us"] += 1
    elif dur < 200: dur_buckets["50-200us"] += 1
    else: dur_buckets[">200us"] += 1

# 输出章节改为:
lines.append(f"## {sec_num}. Kernel Duration Distribution")
for bucket, count in dur_buckets.items():
    pct = count / total_rows * 100 if total_rows > 0 else 0
    bar = "█" * int(pct / 3)
    lines.append(f"  {bucket:>8}: {count:>6} ({pct:>5.1f}%) {bar}")
short_ratio = (dur_buckets["<5us"] + dur_buckets["5-20us"]) / total_rows * 100 if total_rows > 0 else 0
lines.append(f"  Short kernel ratio (<20us): {short_ratio:.1f}%")
if short_ratio > threshold("kernel_details", "short_kernel_dominant", 60):
    lines.append(f"  → 大部分 kernel 非常短。减少 op 数量的收益可能 > 优化单个 op")
    lines.append(f"    Cross-validate: op_statistic § fragmentation signal, trace_view § dispatch latency")
# 保留原有 small kernel (<5us) 的 type breakdown
```

**新增阈值** (thresholds.py "kernel_details" 节):

```python
"short_kernel_dominant": 60,  # % — short kernel (<20us) ratio above this = dominant
```

**效果**：agent 看到 "Short kernel ratio (<20us): 80.8%" 和引导 hint，直接知道 dispatch 开销可能占主导。

### 2.3 `parse_trace_view.py` — 增加 dispatch 总开销估算

**改动位置**：Dispatch Latency 章节 (L321-336)

**改动内容**：利用已有的 dev_count 和 dispatch latency，计算总开销和 ratio

```python
# 现有: 输出 count, avg, p50, p90, top dispatch latency
# 新增: dispatch_total 和 dispatch/kernel ratio

# 在 Dispatch Latency 章节末尾新增:
if disp_stats["count"] > 0 and dev_count > 0:
    avg_disp_us = disp_stats["sum"] / disp_stats["count"] / 1000  # ns → us
    disp_total_us = dev_count * avg_disp_us
    # compute stream active time (已有 stream_stats)
    compute_active_us = sum(v[0] for v in compute_streams.values()) / 1000  # ns → us
    if compute_active_us > 0:
        disp_kernel_ratio = disp_total_us / compute_active_us * 100
        lines.append(f"  Estimated dispatch total: {disp_total_us/1000:.1f} ms (dev_count × avg_latency)")
        lines.append(f"  Dispatch / kernel-active ratio: {disp_kernel_ratio:.1f}%")
        if disp_kernel_ratio > threshold("trace_view", "dispatch_kernel_ratio", 50):
            lines.append(f"  → Dispatch 开销估算 > kernel 活跃时间的 {threshold('trace_view', 'dispatch_kernel_ratio', 50)}%")
            lines.append(f"    异步队列下两者可重叠，实际影响取决于 gap 分布:")
            lines.append(f"    若 gap > 50us 占比高: dispatch 未充分重叠，减少 op 数量有收益")
            lines.append(f"    若 gap < 10us 占比高: dispatch 已充分重叠，Free 来自串行依赖")
```

**新增阈值** (thresholds.py "trace_view" 节):

```python
"dispatch_kernel_ratio": 50,  # % — dispatch/kernel above this = significant
```

**效果**：agent 看到 "Dispatch / kernel-active ratio: 119.7%" 和引导 hint，直接知道 dispatch 开销量级。

注意：这个计算完全在 `parse_trace_view.py` 内部完成（dev_count 和 compute_active 都已在脚本中），不需要跨文件。

## 3. skill 决策规则

### 3.1 `model_opt/02_bottleneck_analysis/SKILL.md` — 增加跨脚本组合规则

在分析步骤中增加一段**跨脚本决策规则**：

```markdown
### 跨脚本决策规则

运行 parse 脚本后，按以下规则组合输出做决策：

1. **判断瓶颈类型**（组合 step_trace + kernel_details）:
   - parse_step_trace 输出 "Optimizable space > 30%" + parse_kernel_details 输出 "Short kernel ratio > 60%"
     → 瓶颈可能是框架 dispatch 开销。减少 op 数量的理论收益 > 优化单个 op。
   - parse_step_trace 输出 "Optimizable space > 30%" + parse_kernel_details 输出 "Short kernel ratio < 30%"
     → 瓶颈可能是串行依赖或 sync 等待。减少 op 数量收益有限。

2. **判断 dispatch 重叠程度**（组合 trace_view 的 gap 分布）:
   - trace_view 输出 gap < 10us 占比高 → dispatch 已充分重叠，Free 来自串行依赖
   - trace_view 输出 gap > 50us 占比高 → dispatch 未充分重叠，减少 op 数量有收益

3. **方案排序**（按理论收益上限，而非实现难度）:
   - 每个方案标注"消除的开销类别"和"该类别占总时间的比例"（引用 parse 脚本输出）
   - 按比例降序排列
```

### 3.2 `model_opt/SKILL.md` — 确认节点 A 增加排序要求

在用户确认节点 A 的方案清单要求中增加：

```markdown
方案清单中，每个方案必须标注：
- "消除的开销类别"：引用 parse 脚本输出的开销分类（如 "dispatch 开销" "data movement" "kernel 计算"）
- "理论收益上限"：该开销占总时间的比例（如 "50.7%" "10.2%"）
方案按"理论收益上限"降序排列。
```

## 4. 改动清单

| 文件 | 改动类型 | 改动量 |
|------|----------|--------|
| `02_bottleneck_analysis/scripts/parse_step_trace.py` | 增强 | +5 行（理论极限标注 + hint） |
| `02_bottleneck_analysis/scripts/parse_kernel_details.py` | 增强 | +15 行（5 档分布替换 <5us 统计 + hint） |
| `02_bottleneck_analysis/scripts/parse_trace_view.py` | 增强 | +10 行（dispatch total + ratio + hint） |
| `02_bottleneck_analysis/scripts/thresholds.py` | 增强 | +3 个阈值 |
| `02_bottleneck_analysis/SKILL.md` | 增强 | 增加跨脚本决策规则段落 |
| `model_opt/SKILL.md` | 增强 | 确认节点 A 增加排序要求 |

**不新建任何文件。** 总改动量约 40 行代码 + 2 段 skill 文本。

## 5. 预期效果

以 Ankh3-large Generate 模式为例：

**改进前**：
1. 运行 parse_step_trace → "Moderate Host-Bound, util 49.3%" → 知道有空闲但不知道多少可优化
2. 运行 parse_kernel_details → "Small kernels (5us): 46.5%" → 只看到 <5us，不知道 <20us 占 80.8%
3. 运行 parse_trace_view → "dispatch avg=12us" → 知道延迟但不知道总开销
4. 停止分析，列出融合算子方案（容易但收益低）

**改进后**：
1. 运行 parse_step_trace → "Optimizable space: 50.7% → 可优化空间大" → 知道 50% 可优化
2. 运行 parse_kernel_details → "Short kernel ratio (<20us): 80.8% → 减少 op 数量收益 > 优化单个 op" → 知道方向
3. 运行 parse_trace_view → "Dispatch/kernel ratio: 119.7% → 若 gap < 10us 占比高则 Free 来自串行依赖" → 知道根因
4. 跨脚本决策规则: "Optimizable space > 30%" + "Short kernel > 60%" → "框架 dispatch 开销可能是瓶颈"
5. 方案排序: flat 实现（消除 72 dispatch/step，上限 50%）> 融合算子（上限 ~15%）
6. agent 被数据推向 flat 方案
