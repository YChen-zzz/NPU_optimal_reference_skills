#!/usr/bin/env python3
"""解析 kernel_details.csv — 逐 kernel 执行详情（含硬件单元耗时拆分）。

这是信息最丰富的 profiling 文件之一。包含逐 kernel 的：执行时间、
wait time、硬件单元耗时拆分（mac/mte1/mte2/vec/scalar）、shape、
Block Dim（并行度）以及 cube 利用率。

相比其他 profiling 文件的独特价值：
- 硬件单元利用率（逐 kernel 区分 compute-bound 与 memory-bound）
- 逐 kernel 的 shape 与并行度（Block Dim）
- 顺序 kernel 流与 wait time 模式
- 识别低效 kernel（耗时长但利用率低）

用法:
    python parse_kernel_details.py <profiling_dir> [--rank N] [--top-k 15]
        [--small-threshold 5.0] [--wait-threshold 500]
"""

import argparse
import heapq
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, stream_csv, safe_float, format_duration_ms


def parse(profiling_dir: str, rank=None, top_k: int = 15,
          small_threshold: float = 5.0, wait_threshold: float = 500.0) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "kernel_details.csv"

    if not csv_path.exists():
        return f"[kernel_details] 文件未找到: {csv_path}"

    total_rows = 0
    total_dur_us = 0.0
    total_wait_us = 0.0

    # Accelerator core 拆分
    core_stats = defaultdict(lambda: {"count": 0, "dur_us": 0.0})

    # 硬件单元聚合（针对 AI_CORE）
    aic_kernels = 0
    aic_mac_sum = 0.0
    aic_mte1_sum = 0.0
    aic_mte2_sum = 0.0
    aic_scalar_sum = 0.0
    # duration 加权（少数重型 kernel 占主导；算术平均会掩盖双峰分布）
    aic_dur_sum = 0.0
    aic_mac_wsum = 0.0
    aic_mte_wsum = 0.0
    aic_fixpipe_sum = 0.0
    aic_icache_values = []
    format_counts = defaultdict(int)

    # 硬件单元聚合（针对 AI_VECTOR_CORE）
    aiv_kernels = 0
    aiv_vec_sum = 0.0
    aiv_mte2_sum = 0.0
    aiv_mte3_sum = 0.0
    aiv_scalar_sum = 0.0
    aiv_dur_sum = 0.0
    aiv_vec_wsum = 0.0
    aiv_mte_wsum = 0.0
    aiv_icache_values = []

    # 小 kernel 跟踪
    small_count = 0
    small_dur_total = 0.0
    small_type_count = defaultdict(int)

    # Kernel duration 分布（5 个桶）
    dur_buckets = {"<5us": 0, "5-20us": 0, "20-50us": 0, "50-200us": 0, ">200us": 0}

    # Block Dim 分布
    block_dim_buckets = {"1": 0, "2-8": 0, "9-28": 0, "29+": 0}

    # Cube 利用率跟踪（仅 AI_CORE）
    cube_util_values = []

    # 可疑 kernel：高 duration 但低 compute ratio（两种 core 类型）
    suspect_heap = []
    # 真正的 compute-bound：高 duration 且高 compute ratio（replace/quant 目标）
    compute_bound_heap = []

    # AI_CPU fallback 跟踪（排除 communication 算子 — 它们按设计运行在 AI CPU 上）
    aicpu_kernels = []  # (dur, name, op_type, shapes) — 仅非 comm
    aicpu_comm_count = 0  # AI CPU 上的 communication 算子（预期行为，非问题）
    COMM_KEYWORDS = tuple(threshold("kernel_details", "comm_keywords",
                     ["broadcast", "allgather", "alltoall", "allreduce", "hcom", "send", "recv", "reducescatter"]))

    # Wait time 分布桶
    _wb = threshold("kernel_details", "wait_buckets_us", [100, 500, 2000])
    wait_buckets = {f"<{_wb[0]}us": 0, f"{_wb[0]}-{_wb[1]}us": 0, f"{_wb[1]}-{_wb[2]}us": 0, f">{_wb[2]}us": 0}

    # 高 wait kernel 及上下文
    all_kernels = []

    for row in stream_csv(csv_path):
        total_rows += 1
        dur = safe_float(row.get("Duration(us)", 0))
        wait = safe_float(row.get("Wait Time(us)", 0))
        total_dur_us += dur
        total_wait_us += wait

        name = row.get("Name", "?")
        op_type = row.get("Type", "?")
        core = row.get("Accelerator Core", "?")
        block_dim = int(safe_float(row.get("Block Dim", 0)))
        cube_util = safe_float(row.get("cube_utilization(%)", 0))
        start_time = safe_float(row.get("Start Time(us)", 0))
        stream_id = row.get("Stream ID", "?").strip()
        input_formats = row.get("Input Formats", "").strip()

        core_stats[core]["count"] += 1
        core_stats[core]["dur_us"] += dur

        # Format 分布 (A2)：非 ND format = layout 转换开销
        if input_formats:
            for fmt in input_formats.replace(";", " ").split():
                fmt = fmt.strip().strip('"')
                if fmt:
                    format_counts[fmt] += 1

        # 硬件单元 ratio
        if core == "AI_CORE" and dur > 0:
            aic_kernels += 1
            mac_ratio = safe_float(row.get("aic_mac_ratio", 0))
            mte1_ratio = safe_float(row.get("aic_mte1_ratio", 0))
            mte2_ratio = safe_float(row.get("aic_mte2_ratio", 0))
            scalar_ratio = safe_float(row.get("aic_scalar_ratio", 0))
            fixpipe_ratio = safe_float(row.get("aic_fixpipe_ratio", 0))
            icache_miss = safe_float(row.get("aic_icache_miss_rate", 0))
            aic_mac_sum += mac_ratio
            aic_mte1_sum += mte1_ratio
            aic_mte2_sum += mte2_ratio
            aic_scalar_sum += scalar_ratio
            aic_fixpipe_sum += fixpipe_ratio
            if icache_miss > 0:
                aic_icache_values.append(icache_miss)
            aic_dur_sum += dur
            aic_mac_wsum += mac_ratio * dur
            aic_mte_wsum += (mte1_ratio + mte2_ratio) * dur
            if cube_util > 0:
                cube_util_values.append(cube_util)
            # 可疑：高 duration 但 compute ratio 低
            if dur > threshold("kernel_details", "suspect_min_duration_us", 10) and mac_ratio < threshold("kernel_details", "suspect_mac_ratio", 0.2):
                entry = (dur, total_rows, name, core, mac_ratio,
                         mte1_ratio + mte2_ratio, row.get("Input Shapes", ""), block_dim)
                if len(suspect_heap) < top_k:
                    heapq.heappush(suspect_heap, entry)
                elif dur > suspect_heap[0][0]:
                    heapq.heapreplace(suspect_heap, entry)
            # 真正的 compute-bound：高 duration 且高 mac ratio（replace/quant 目标）
            if dur > threshold("kernel_details", "suspect_min_duration_us", 10) and mac_ratio >= threshold("kernel_details", "compute_bound_mac_ratio", 0.5):
                cb_entry = (dur, total_rows, name, core, mac_ratio, block_dim, row.get("Input Shapes", ""))
                if len(compute_bound_heap) < top_k:
                    heapq.heappush(compute_bound_heap, cb_entry)
                elif dur > compute_bound_heap[0][0]:
                    heapq.heapreplace(compute_bound_heap, cb_entry)

        elif core == "AI_VECTOR_CORE" and dur > 0:
            aiv_kernels += 1
            vec_ratio = safe_float(row.get("aiv_vec_ratio", 0))
            aiv_mte2_ratio = safe_float(row.get("aiv_mte2_ratio", 0))
            aiv_mte3_ratio = safe_float(row.get("aiv_mte3_ratio", 0))
            aiv_scalar_ratio_val = safe_float(row.get("aiv_scalar_ratio", 0))
            aiv_icache_miss = safe_float(row.get("aiv_icache_miss_rate", 0))
            aiv_vec_sum += vec_ratio
            aiv_mte2_sum += aiv_mte2_ratio
            aiv_mte3_sum += aiv_mte3_ratio
            aiv_scalar_sum += aiv_scalar_ratio_val
            if aiv_icache_miss > 0:
                aiv_icache_values.append(aiv_icache_miss)
            aiv_dur_sum += dur
            aiv_vec_wsum += vec_ratio * dur
            aiv_mte_wsum += (aiv_mte2_ratio + aiv_mte3_ratio) * dur
            # 可疑：高 duration 但 vec ratio 低
            if dur > threshold("kernel_details", "suspect_min_duration_us", 10) and vec_ratio < threshold("kernel_details", "suspect_vec_ratio", 0.05):
                entry = (dur, total_rows, name, core, vec_ratio,
                         aiv_mte2_ratio + aiv_mte3_ratio, row.get("Input Shapes", ""), block_dim)
                if len(suspect_heap) < top_k:
                    heapq.heappush(suspect_heap, entry)
                elif dur > suspect_heap[0][0]:
                    heapq.heapreplace(suspect_heap, entry)

        elif "AI_CPU" in core:
            low_type = op_type.lower()
            if any(kw in low_type for kw in COMM_KEYWORDS):
                aicpu_comm_count += 1
            else:
                aicpu_kernels.append((dur, name, op_type, row.get("Input Shapes", "")))

        # 小 kernel
        if dur < small_threshold and dur > 0:
            small_count += 1
            small_dur_total += dur
            small_type_count[op_type] += 1

        # Duration 分布
        if dur > 0:
            if dur < 5: dur_buckets["<5us"] += 1
            elif dur < 20: dur_buckets["5-20us"] += 1
            elif dur < 50: dur_buckets["20-50us"] += 1
            elif dur < 200: dur_buckets["50-200us"] += 1
            else: dur_buckets[">200us"] += 1

        # Block Dim
        if block_dim == 1:
            block_dim_buckets["1"] += 1
        elif block_dim <= threshold("kernel_details", "block_dim_buckets", [8, 28])[0]:
            block_dim_buckets["2-8"] += 1
        elif block_dim <= threshold("kernel_details", "block_dim_buckets", [8, 28])[1]:
            block_dim_buckets["9-28"] += 1
        else:
            block_dim_buckets["29+"] += 1

        # Wait time 桶
        if wait < 100:
            wait_buckets["<100us"] += 1
        elif wait < 500:
            wait_buckets["100-500us"] += 1
        elif wait < 2000:
            wait_buckets["500-2000us"] += 1
        else:
            wait_buckets[">2000us"] += 1

        # 存储用于上下文分析（含 start time + stream，用于时间分组）
        all_kernels.append({"name": name, "type": op_type, "dur": dur, "wait": wait,
                            "start": start_time, "stream": stream_id})

    if total_rows == 0:
        return f"[kernel_details] 空文件: {csv_path}"

    # 检测 fusible 序列：连续的小 kernel（同一 stream，按 start time 排序）
    # 当 stream 交错时，文件行序 ≠ 时间序；按 stream 分组并
    # 按 Start Time 排序，使"连续"指同一 stream 上时间相邻。
    fusible_sequences = []  # (total_dur, count, start_idx, op_types, stream)
    SMALL_THRESH = threshold("kernel_details", "fusible_small_us", 10.0)  # us
    MIN_LEN = threshold("kernel_details", "fusible_min_length", 5)

    # 每 stream 的 all_kernels 原始位置索引，按 start time 排序
    stream_order = defaultdict(list)
    for idx, k in enumerate(all_kernels):
        stream_order[k["stream"]].append(idx)
    for s in stream_order:
        stream_order[s].sort(key=lambda i: all_kernels[i]["start"])

    for s, idxs in stream_order.items():
        i = 0
        while i < len(idxs):
            ki = all_kernels[idxs[i]]
            if ki["dur"] > 0 and ki["dur"] < SMALL_THRESH:
                j = i
                seq_total = 0
                while j < len(idxs) and all_kernels[idxs[j]]["dur"] > 0 and all_kernels[idxs[j]]["dur"] < SMALL_THRESH:
                    seq_total += all_kernels[idxs[j]]["dur"]
                    j += 1
                seq_len = j - i
                if seq_len >= MIN_LEN and seq_total > threshold("kernel_details", "fusible_min_total_us", 100):
                    types = [all_kernels[idxs[k]]["type"] for k in range(i, j)]
                    fusible_sequences.append((seq_total, seq_len, idxs[i], types, s))
                i = j
            else:
                i += 1

    lines = []
    lines.append("# Kernel Details 分析")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"Kernel 总数: {total_rows:,}  |  Compute: {total_dur_us/1000:.1f}ms  |  Wait: {total_wait_us/1000:.1f}ms")
    lines.append("")

    # --- 1. Accelerator Core 分布 ---
    lines.append("## 1. Accelerator Core 分布")
    for core, info in sorted(core_stats.items(), key=lambda x: -x[1]["dur_us"]):
        pct = info["dur_us"] / total_dur_us * 100 if total_dur_us > 0 else 0
        lines.append(f"  {core}: {info['count']:,} 个 kernel, {info['dur_us']/1000:.1f}ms ({pct:.1f}%)")
    if aicpu_kernels:
        lines.append(f"  [!] 检测到非 comm 的 AI_CPU: {len(aicpu_kernels)} 个 kernel — 详见第 3 节")
    if aicpu_comm_count:
        lines.append(f"  ({aicpu_comm_count} 个 communication 算子运行在 AI_CPU 上 — 预期行为，非问题)")
    # Core 切换检测：若 AI_CORE 与 AI_VECTOR_CORE 均显著，频繁切换可能产生 overhead
    aic_dur = core_stats.get("AI_CORE", {}).get("dur_us", 0)
    aiv_dur = core_stats.get("AI_VECTOR_CORE", {}).get("dur_us", 0)
    if aic_dur > 0 and aiv_dur > 0:
        ratio = min(aic_dur, aiv_dur) / max(aic_dur, aiv_dur)
        if ratio > 0.3:
            lines.append(f"  [SIGNAL] AI_CORE ({aic_dur/1000:.1f}ms) 与 AI_VECTOR_CORE ({aiv_dur/1000:.1f}ms) 均显著 (ratio={ratio:.2f})")
    lines.append("")

    # --- 2. 硬件单元利用率 ---
    lines.append("## 2. 硬件单元利用率（平均 ratio）")
    if aic_kernels > 0:
        lines.append(f"  AI_CORE ({aic_kernels} 个 kernel):")
        lines.append(f"    mac (compute):  {aic_mac_sum/aic_kernels:.3f}")
        lines.append(f"    mte1 (load):    {aic_mte1_sum/aic_kernels:.3f}")
        lines.append(f"    mte2 (store):   {aic_mte2_sum/aic_kernels:.3f}")
        lines.append(f"    scalar:         {aic_scalar_sum/aic_kernels:.3f}")
        lines.append(f"    fixpipe:        {aic_fixpipe_sum/aic_kernels:.3f}")
        if aic_icache_values:
            lines.append(f"    icache miss:    avg={sum(aic_icache_values)/len(aic_icache_values):.3f}  max={max(aic_icache_values):.3f}")
        if aic_dur_sum > 0:
            wmac = aic_mac_wsum / aic_dur_sum
            wmte = aic_mte_wsum / aic_dur_sum
            lines.append(f"    [duration-weighted] mac={wmac:.3f}  mte={wmte:.3f}  （重型 kernel 占主导；与上方算术平均对比以观察双峰分布）")
            avg_mac = wmac
            avg_mte = wmte
        else:
            avg_mac = aic_mac_sum / aic_kernels
            avg_mte = (aic_mte1_sum + aic_mte2_sum) / aic_kernels
        if avg_mte > avg_mac * threshold("kernel_details", "hw_dominance_ratio", 1.5):
            lines.append(f"    - Memory 主导: mte ({avg_mte:.3f}) >> mac ({avg_mac:.3f})")
        elif avg_mac > avg_mte * threshold("kernel_details", "hw_dominance_ratio", 1.5):
            lines.append(f"    - Compute 主导: mac ({avg_mac:.3f}) >> mte ({avg_mte:.3f})")
    if aiv_kernels > 0:
        lines.append(f"  AI_VECTOR_CORE ({aiv_kernels} 个 kernel):")
        lines.append(f"    vec (compute):  {aiv_vec_sum/aiv_kernels:.3f}")
        lines.append(f"    mte2 (load):    {aiv_mte2_sum/aiv_kernels:.3f}")
        lines.append(f"    mte3 (store):   {aiv_mte3_sum/aiv_kernels:.3f}")
        lines.append(f"    scalar:         {aiv_scalar_sum/aiv_kernels:.3f}")
        if aiv_icache_values:
            lines.append(f"    icache miss:    avg={sum(aiv_icache_values)/len(aiv_icache_values):.3f}  max={max(aiv_icache_values):.3f}")
        if aiv_dur_sum > 0:
            lines.append(f"    [duration-weighted] vec={aiv_vec_wsum/aiv_dur_sum:.3f}  mte={aiv_mte_wsum/aiv_dur_sum:.3f}")
    # Format 分布 (A2)：非 ND format 表明存在 layout 转换开销
    if format_counts:
        total_fmt = sum(format_counts.values())
        non_nd = sum(c for f, c in format_counts.items() if f != "ND" and f != "N/A")
        lines.append(f"  Input Formats: {dict(sorted(format_counts.items(), key=lambda x: -x[1]))}")
        if non_nd / total_fmt > 0.1 if total_fmt else False:
            lines.append(f"  - {non_nd}/{total_fmt} ({non_nd/total_fmt*100:.0f}%) 非 ND input format — layout 转换开销；交叉验证 op_statistic 的 Transpose/Cast。")
    if cube_util_values:
        avg_cube = sum(cube_util_values) / len(cube_util_values)
        min_cube = min(cube_util_values)
        low_util_count = sum(1 for v in cube_util_values if v < threshold("kernel_details", "cube_low_util", 50))
        lines.append(f"  Cube 利用率: avg={avg_cube:.1f}%, min={min_cube:.1f}%, "
                     f"low(<50%)={low_util_count}/{len(cube_util_values)}")
    if aic_kernels == 0 and aiv_kernels == 0:
        lines.append("  [!] 所有 ratio 均为 0 — 请检查采集时是否使用 aic_metrics=PipeUtilization")
    lines.append("")

    # --- 3. AI CPU Fallback [DEFINITE] ---
    if aicpu_kernels:
        lines.append("## 3. AI CPU Fallback（非 comm）[DEFINITE]")
        lines.append("  运行在 AI CPU 上的非 communication 算子（无 AI Core 实现）。排除 communication 算子 — 它们按设计运行在 AI CPU 上。")
        aicpu_total = sum(d for d, _, _, _ in aicpu_kernels)
        lines.append(f"  数量: {len(aicpu_kernels)}  |  总计: {aicpu_total/1000:.1f}ms")
        aicpu_sorted = sorted(aicpu_kernels, key=lambda x: -x[0])
        lines.append(f"  {'Name':<35} {'Dur(us)':>8} {'Type':<15} {'Shapes'}")
        lines.append(f"  {'-'*35} {'-'*8} {'-'*15} {'-'*20}")
        for dur, name, otype, shapes in aicpu_sorted[:top_k]:
            shapes_clean = shapes.replace("\n", " ").replace(";", "|").replace('"', '')[:30]
            lines.append(f"  {name:<35} {dur:>8.1f} {otype:<15} {shapes_clean}")
        lines.append("")

    # --- 4. Kernel Duration 分布 ---
    sec_num = 4 if aicpu_kernels else 3
    lines.append(f"## {sec_num}. Kernel Duration 分布")
    for bucket, count in dur_buckets.items():
        pct = count / total_rows * 100 if total_rows > 0 else 0
        bar = "█" * int(pct / 3)
        lines.append(f"  {bucket:>8}: {count:>6} ({pct:>5.1f}%) {bar}")
    short_ratio = (dur_buckets["<5us"] + dur_buckets["5-20us"]) / total_rows * 100 if total_rows > 0 else 0
    lines.append(f"  短 kernel 占比 (<20us): {short_ratio:.1f}%")
    if short_ratio > threshold("kernel_details", "short_kernel_dominant", 60):
        lines.append(f"  - 多数 kernel 非常短。减少 op 数量可能比优化单个 op 收益更大")
        lines.append(f"    交叉验证: op_statistic 的 fragmentation signal 部分，trace_view 的 dispatch latency 部分")
    lines.append("")

    sec_num += 1
    lines.append(f"## {sec_num}. 小 kernel（duration < {small_threshold}us）")
    if small_count > 0:
        small_pct = small_count / total_rows * 100
        lines.append(f"  数量: {small_count:,} ({small_pct:.1f}% 占所有 kernel)  |  累计: {small_dur_total/1000:.2f}ms")
        lines.append(f"  Top 类型:")
        for t, c in sorted(small_type_count.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {t}: {c}")
    else:
        lines.append(f"  未发现")
    lines.append("")

    # --- 5. Block Dim 分布 ---
    sec_num += 1
    lines.append(f"## {sec_num}. Block Dim 分布（并行度）")
    for bucket, count in block_dim_buckets.items():
        pct = count / total_rows * 100 if total_rows > 0 else 0
        bar = "█" * int(pct / 3)
        lines.append(f"  Dim {bucket:>5}: {count:>6} ({pct:>5.1f}%) {bar}")
    low_par = block_dim_buckets["1"]
    if low_par / total_rows > threshold("kernel_details", "low_parallelism_ratio", 0.1):
        lines.append(f"  - {low_par} 个 kernel 的 Block Dim=1: shape 可能太小，无法并行")
    lines.append("")

    # --- 6. Wait Time 分布（事实陈述）---
    sec_num += 1
    lines.append(f"## {sec_num}. Wait Time 分布")
    for bucket, count in wait_buckets.items():
        pct = count / total_rows * 100 if total_rows > 0 else 0
        lines.append(f"  {bucket:>10}: {count:>7} ({pct:.1f}%)")
    # TASK_QUEUE 检测：若所有 wait 均匀偏高，async pipeline 可能未生效
    all_waits = [k["wait"] for k in all_kernels]
    if all_waits and len(all_waits) > 20:
        avg_wait = sum(all_waits) / len(all_waits)
        high_wait_count = sum(1 for w in all_waits if w > avg_wait * 0.5)
        if high_wait_count / len(all_waits) > 0.7 and avg_wait > 100:
            lines.append(f"  [SIGNAL] Wait time 普遍偏高 (avg={avg_wait:.0f}us, {high_wait_count}/{len(all_waits)} >50% of avg)")
    lines.append("")

    # --- 7. 可疑信号（诊断）---
    sec_num += 1
    lines.append(f"## {sec_num}. 可疑信号")
    lines.append("  [DEFINITE]=可直接行动  [SIGNAL]=异常，需结合其他维度交叉验证")
    lines.append("")

    # 7a. 可疑 kernel
    suspect_sorted = sorted(suspect_heap, key=lambda x: -x[0])
    if suspect_sorted:
        lines.append("  [SIGNAL] 高 duration、低 compute ratio — 交叉验证: 用 --filter <op> 查看 shape 分布")
        header = f"  {'Name':<42} {'Core':<5} {'Dur(us)':>8} {'Compute':>8} {'Move':>8} {'BDim':>5} {'Shapes'}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for dur, _, name, core, compute_ratio, move_ratio, shapes, bdim in suspect_sorted:
            shapes_clean = shapes.replace("\n", " ").replace(";", "|").replace('"', '')[:30]
            core_short = "AIC" if "AI_CORE" == core else "AIV"
            lines.append(
                f"  {name:<42} {core_short:<5} {dur:>8.1f} "
                f"{compute_ratio:>7.3f} {move_ratio:>7.3f} {bdim:>5} {shapes_clean}")
        lines.append("")

    # 7a-bis. 真正的 compute-bound kernel（高 duration + 高 compute ratio）
    cb_sorted = sorted(compute_bound_heap, key=lambda x: -x[0])
    if cb_sorted:
        lines.append("  [SIGNAL] 真正的 compute-bound（高 duration + 高 compute ratio）— replace/quantize/split 目标")
        header = f"  {'Name':<42} {'Dur(us)':>8} {'mac':>6} {'BDim':>5} {'Shapes'}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for dur, _, name, core, mac_ratio, bdim, shapes in cb_sorted:
            shapes_clean = shapes.replace("\n", " ").replace(";", "|").replace('"', '')[:30]
            lines.append(f"  {name:<42} {dur:>8.1f} {mac_ratio:>5.2f} {bdim:>5} {shapes_clean}")
        lines.append("")

    # 7b. 高 wait 上下文
    high_wait_indices = [i for i, k in enumerate(all_kernels) if k["wait"] > wait_threshold]
    if high_wait_indices:
        lines.append(f"  [SIGNAL] 高 wait kernel (wait > {wait_threshold:.0f}us) — 交叉验证: 在 trace_view 中查找原因")
        lines.append("")
        top_waits = sorted(high_wait_indices, key=lambda i: -all_kernels[i]["wait"])[:min(top_k, 8)]
        for rank_idx, idx in enumerate(top_waits, 1):
            k = all_kernels[idx]
            lines.append(f"  [{rank_idx}] #{idx} {k['name']}  wait={format_duration_ms(k['wait'])}  stream={k['stream']}")
            # 同一 stream 上的时间邻居（非文件顺序）
            s_order = stream_order.get(k["stream"], [])
            pos = s_order.index(idx) if idx in s_order else -1
            context = 2
            if pos >= 0:
                lo = max(0, pos - context)
                hi = min(len(s_order), pos + context + 1)
                for p in range(lo, hi):
                    ci = s_order[p]
                    ck = all_kernels[ci]
                    marker = " <<<" if ci == idx else ""
                    lines.append(f"      [{ci}] {ck['type']:<20} dur={ck['dur']:>7.1f}us  wait={ck['wait']:>7.0f}us{marker}")
            lines.append("")

    # 7c. Fusible 算子序列
    if fusible_sequences:
        fusible_sorted = sorted(fusible_sequences, key=lambda x: -x[0])
        total_fusible = sum(s[0] for s in fusible_sorted)
        lines.append(f"  [SIGNAL] Fusible 序列: {len(fusible_sorted)} 个序列，每个含 ≥{MIN_LEN} 个连续小 kernel (<{SMALL_THRESH}us)")
        lines.append(f"    Fusible 序列总耗时: {total_fusible/1000:.2f}ms ({total_fusible/total_dur_us*100:.1f}% 占 compute)")
        lines.append(f"    按累计耗时排序的 Top {min(top_k, 5)}:")
        for total, count, start_idx, types, s in fusible_sorted[:5]:
            from collections import Counter
            tc = Counter(types).most_common(3)
            type_str = ", ".join(f"{t}:{c}" for t, c in tc)
            lines.append(f"      {total/1000:.2f}ms  {count} 个 kernel  at #{start_idx}  stream={s}  类型: {type_str}")
        lines.append("    - 交叉验证: 检查这些是否能 fuse (equivalent_substitution layer 1) 或 batch。")
        lines.append("")

    if not suspect_sorted and not cb_sorted and not high_wait_indices and not fusible_sequences:
        lines.append("  无")
        lines.append("")

    return "\n".join(lines)


def parse_filtered(profiling_dir: str, filters: list, rank=None, top_k: int = 15) -> str:
    """过滤模式：对特定算子的深入分析。"""
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "kernel_details.csv"

    if not csv_path.exists():
        return f"[kernel_details] 文件未找到: {csv_path}"

    filter_lower = [f.lower() for f in filters]

    matched = []
    all_seq = []  # (index_in_file, is_matched, name, type, dur, wait)
    idx = 0
    for row in stream_csv(csv_path):
        name = row.get("Name", "")
        op_type = row.get("Type", "")
        dur = safe_float(row.get("Duration(us)", 0))
        wait = safe_float(row.get("Wait Time(us)", 0))
        is_match = any(f in name.lower() or f in op_type.lower() for f in filter_lower)
        all_seq.append((idx, is_match, name, op_type, dur, wait))
        if is_match:
            matched.append(row)
        idx += 1

    lines = []
    lines.append(f"# Kernel Details — 过滤式深入分析")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"过滤: {', '.join(filters)}")
    lines.append(f"匹配的 kernel: {len(matched)}")
    lines.append("")

    if not matched:
        lines.append("没有 kernel 匹配该过滤条件。")
        return "\n".join(lines)

    # --- 1. 摘要 ---
    total_dur = sum(safe_float(r.get("Duration(us)", 0)) for r in matched)
    total_wait = sum(safe_float(r.get("Wait Time(us)", 0)) for r in matched)
    lines.append("## 1. 摘要")
    lines.append(f"  数量: {len(matched)}")
    lines.append(f"  总 duration: {total_dur/1000:.2f} ms")
    lines.append(f"  总 wait: {total_wait/1000:.2f} ms")
    lines.append(f"  平均 duration: {total_dur/len(matched):.1f} us")
    lines.append(f"  平均 wait: {total_wait/len(matched):.1f} us")
    lines.append("")

    # --- 2. Shape - Performance 相关性 ---
    lines.append("## 2. Shape - Performance 相关性")
    shape_groups = defaultdict(list)
    for r in matched:
        shape = r.get("Input Shapes", "").replace("\n", " ").replace('"', '').strip()
        if not shape:
            shape = "(空)"
        shape_groups[shape].append(r)

    shape_perf = []
    for shape, rows in shape_groups.items():
        durs = [safe_float(r.get("Duration(us)", 0)) for r in rows]
        waits = [safe_float(r.get("Wait Time(us)", 0)) for r in rows]
        shape_perf.append((sum(durs), shape, len(rows), durs, waits, rows))
    shape_perf.sort(key=lambda x: -x[0])

    header = f"  {'Shape':<40} {'Count':>6} {'Avg Dur':>8} {'Min':>7} {'Max':>7} {'Total(ms)':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for total, shape, count, durs, waits, rows in shape_perf[:top_k]:
        avg_d = sum(durs) / len(durs)
        lines.append(
            f"  {shape[:40]:<40} {count:>6} {avg_d:>7.1f}us "
            f"{min(durs):>6.1f} {max(durs):>6.1f} {total/1000:>10.2f}"
        )
    lines.append("")

    # --- 3. 逐实例硬件拆分（最慢的 N 个）---
    lines.append(f"## 3. 最慢实例 — 硬件拆分")
    matched_sorted = sorted(matched, key=lambda r: -safe_float(r.get("Duration(us)", 0)))
    lines.append(f"  (按 duration 排序的 Top {min(top_k, len(matched_sorted))}，展示逐实例硬件 ratio)")
    lines.append("")

    for i, r in enumerate(matched_sorted[:top_k], 1):
        name = r.get("Name", "?")
        dur = safe_float(r.get("Duration(us)", 0))
        wait = safe_float(r.get("Wait Time(us)", 0))
        core = r.get("Accelerator Core", "?")
        bdim = int(safe_float(r.get("Block Dim", 0)))
        shape = r.get("Input Shapes", "").replace("\n", " ").replace('"', '').strip()[:50]

        lines.append(f"  [{i}] {name}  dur={dur:.1f}us  wait={wait:.0f}us  block_dim={bdim}")
        lines.append(f"      shape: {shape}")

        if core == "AI_CORE":
            mac = safe_float(r.get("aic_mac_ratio", 0))
            mte1 = safe_float(r.get("aic_mte1_ratio", 0))
            mte2 = safe_float(r.get("aic_mte2_ratio", 0))
            scalar = safe_float(r.get("aic_scalar_ratio", 0))
            cube = safe_float(r.get("cube_utilization(%)", 0))
            lines.append(f"      AI_CORE: mac={mac:.3f} mte1={mte1:.3f} mte2={mte2:.3f} scalar={scalar:.3f} cube={cube:.1f}%")
        elif core == "AI_VECTOR_CORE":
            vec = safe_float(r.get("aiv_vec_ratio", 0))
            mte2 = safe_float(r.get("aiv_mte2_ratio", 0))
            mte3 = safe_float(r.get("aiv_mte3_ratio", 0))
            scalar = safe_float(r.get("aiv_scalar_ratio", 0))
            lines.append(f"      AI_VECTOR: vec={vec:.3f} mte2={mte2:.3f} mte3={mte3:.3f} scalar={scalar:.3f}")
        lines.append("")

    # --- 4. Wait time 上下文：该 op 的前后内容 ---
    lines.append("## 4. Wait Time Context")
    matched_waits = [(safe_float(r.get("Wait Time(us)", 0)), i)
                     for i, r in enumerate(matched)]
    avg_wait = total_wait / len(matched)
    max_wait_val = max(w for w, _ in matched_waits) if matched_waits else 0
    lines.append(f"  平均 wait: {avg_wait:.0f}us, 最大 wait: {max_wait_val:.0f}us")
    lines.append("")

    # 在完整序列中找到匹配 kernel 的位置以展示邻居
    high_wait_matched = []
    for seq_idx, (file_idx, is_match, name, op_type, dur, wait) in enumerate(all_seq):
        if is_match and wait > avg_wait * 3 and wait > 200:
            high_wait_matched.append((wait, seq_idx))
    high_wait_matched.sort(key=lambda x: -x[0])

    if high_wait_matched:
        lines.append(f"  wait > 3 倍均值 ({avg_wait*3:.0f}us) 的实例，展示邻居:")
        lines.append("")
        for rank_idx, (wait_val, seq_idx) in enumerate(high_wait_matched[:min(8, top_k)], 1):
            lines.append(f"  [{rank_idx}] wait={format_duration_ms(wait_val)} 位于位置 #{seq_idx}")
            context = 3
            start = max(0, seq_idx - context)
            end = min(len(all_seq), seq_idx + context + 1)
            for i in range(start, end):
                _, is_m, n, t, d, w = all_seq[i]
                marker = " <<<" if i == seq_idx else ""
                match_tag = "*" if is_m else " "
                lines.append(f"      {match_tag}[{i}] {t:<20} dur={d:>7.1f}us  wait={w:>7.0f}us{marker}")
            lines.append("")
    else:
        lines.append("  没有 wait time 显著偏高的实例。")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--filter", nargs="+", default=None,
                        help="按算子名称/类型过滤（子串匹配，大小写不敏感）。"
                             "指定后仅展示匹配算子的逐 kernel 详情。")
    parser.add_argument("--small-threshold", type=float, default=5.0,
                        help="用于识别小 kernel 的 duration 阈值 (us)")
    parser.add_argument("--wait-threshold", type=float, default=500.0,
                        help="用于高 wait 上下文分析的 wait time 阈值 (us)")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    if args.filter:
        result = parse_filtered(args.profiling_dir, args.filter, args.rank, args.top_k)
    else:
        result = parse(args.profiling_dir, args.rank, args.top_k,
                       args.small_threshold, args.wait_threshold)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
