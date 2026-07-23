#!/usr/bin/env python3
"""Parse kernel_details.csv — per-kernel execution details with hardware unit breakdown.

This is one of the richest profiling files. It contains per-kernel: execution time,
wait time, hardware unit time breakdown (mac/mte1/mte2/vec/scalar), shapes,
Block Dim (parallelism), and cube utilization.

Unique value over other profiling files:
- Hardware unit utilization (compute-bound vs memory-bound per kernel)
- Per-kernel shapes and parallelism (Block Dim)
- Sequential kernel flow and wait time patterns
- Identifies inefficient kernels (low utilization despite high duration)

Usage:
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
        return f"[kernel_details] File not found: {csv_path}"

    total_rows = 0
    total_dur_us = 0.0
    total_wait_us = 0.0

    # Accelerator core split
    core_stats = defaultdict(lambda: {"count": 0, "dur_us": 0.0})

    # Hardware unit aggregation (for AI_CORE)
    aic_kernels = 0
    aic_mac_sum = 0.0
    aic_mte1_sum = 0.0
    aic_mte2_sum = 0.0
    aic_scalar_sum = 0.0
    # duration-weighted (a few heavy kernels dominate; arithmetic mean masks bimodal)
    aic_dur_sum = 0.0
    aic_mac_wsum = 0.0
    aic_mte_wsum = 0.0
    aic_fixpipe_sum = 0.0
    aic_icache_values = []
    format_counts = defaultdict(int)

    # Hardware unit aggregation (for AI_VECTOR_CORE)
    aiv_kernels = 0
    aiv_vec_sum = 0.0
    aiv_mte2_sum = 0.0
    aiv_mte3_sum = 0.0
    aiv_scalar_sum = 0.0
    aiv_dur_sum = 0.0
    aiv_vec_wsum = 0.0
    aiv_mte_wsum = 0.0
    aiv_icache_values = []

    # Small kernel tracking
    small_count = 0
    small_dur_total = 0.0
    small_type_count = defaultdict(int)

    # Kernel duration distribution (5 buckets)
    dur_buckets = {"<5us": 0, "5-20us": 0, "20-50us": 0, "50-200us": 0, ">200us": 0}

    # Block Dim distribution
    block_dim_buckets = {"1": 0, "2-8": 0, "9-28": 0, "29+": 0}

    # Cube utilization tracking (for AI_CORE only)
    cube_util_values = []

    # Suspect kernels: high duration but low compute ratio (both core types)
    suspect_heap = []
    # True compute-bound: high duration AND high compute ratio (replace/quant target)
    compute_bound_heap = []

    # AI_CPU fallback tracking (exclude communication ops — they run on AI CPU by design)
    aicpu_kernels = []  # (dur, name, op_type, shapes) — non-comm only
    aicpu_comm_count = 0  # communication ops on AI CPU (expected, not a problem)
    COMM_KEYWORDS = tuple(threshold("kernel_details", "comm_keywords",
                     ["broadcast", "allgather", "alltoall", "allreduce", "hcom", "send", "recv", "reducescatter"]))

    # Wait time distribution buckets
    _wb = threshold("kernel_details", "wait_buckets_us", [100, 500, 2000])
    wait_buckets = {f"<{_wb[0]}us": 0, f"{_wb[0]}-{_wb[1]}us": 0, f"{_wb[1]}-{_wb[2]}us": 0, f">{_wb[2]}us": 0}

    # High-wait kernels with context
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

        # Format distribution (A2): non-ND formats = layout conversion cost
        if input_formats:
            for fmt in input_formats.replace(";", " ").split():
                fmt = fmt.strip().strip('"')
                if fmt:
                    format_counts[fmt] += 1

        # Hardware unit ratios
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
            # Suspect: high duration but compute ratio low
            if dur > threshold("kernel_details", "suspect_min_duration_us", 10) and mac_ratio < threshold("kernel_details", "suspect_mac_ratio", 0.2):
                entry = (dur, total_rows, name, core, mac_ratio,
                         mte1_ratio + mte2_ratio, row.get("Input Shapes", ""), block_dim)
                if len(suspect_heap) < top_k:
                    heapq.heappush(suspect_heap, entry)
                elif dur > suspect_heap[0][0]:
                    heapq.heapreplace(suspect_heap, entry)
            # True compute-bound: high duration AND high mac ratio (replace/quant target)
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
            # Suspect: high duration but vec ratio low
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

        # Small kernel
        if dur < small_threshold and dur > 0:
            small_count += 1
            small_dur_total += dur
            small_type_count[op_type] += 1

        # Duration distribution
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

        # Wait time bucket
        if wait < 100:
            wait_buckets["<100us"] += 1
        elif wait < 500:
            wait_buckets["100-500us"] += 1
        elif wait < 2000:
            wait_buckets["500-2000us"] += 1
        else:
            wait_buckets[">2000us"] += 1

        # Store for context analysis (with start time + stream for temporal grouping)
        all_kernels.append({"name": name, "type": op_type, "dur": dur, "wait": wait,
                            "start": start_time, "stream": stream_id})

    if total_rows == 0:
        return f"[kernel_details] Empty file: {csv_path}"

    # Detect fusible sequences: consecutive small kernels (same stream, by start time)
    # File row order ≠ temporal order when streams interleave; group by stream and
    # sort by Start Time so "consecutive" means temporally adjacent on one stream.
    fusible_sequences = []  # (total_dur, count, start_idx, op_types, stream)
    SMALL_THRESH = threshold("kernel_details", "fusible_small_us", 10.0)  # us
    MIN_LEN = threshold("kernel_details", "fusible_min_length", 5)

    # per-stream index of original all_kernels positions, sorted by start time
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
    lines.append("# Kernel Details Analysis")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Total kernels: {total_rows:,}  |  Compute: {total_dur_us/1000:.1f}ms  |  Wait: {total_wait_us/1000:.1f}ms")
    lines.append("")

    # --- 1. Accelerator Core Distribution ---
    lines.append("## 1. Accelerator Core Distribution")
    for core, info in sorted(core_stats.items(), key=lambda x: -x[1]["dur_us"]):
        pct = info["dur_us"] / total_dur_us * 100 if total_dur_us > 0 else 0
        lines.append(f"  {core}: {info['count']:,} kernels, {info['dur_us']/1000:.1f}ms ({pct:.1f}%)")
    if aicpu_kernels:
        lines.append(f"  ⚠ Non-comm AI_CPU detected: {len(aicpu_kernels)} kernels — see §3 for details")
    if aicpu_comm_count:
        lines.append(f"  ({aicpu_comm_count} communication ops on AI_CPU — expected, not a problem)")
    lines.append("")

    # --- 2. Hardware Unit Utilization ---
    lines.append("## 2. Hardware Unit Utilization (average ratios)")
    if aic_kernels > 0:
        lines.append(f"  AI_CORE ({aic_kernels} kernels):")
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
            lines.append(f"    [duration-weighted] mac={wmac:.3f}  mte={wmte:.3f}  (heavy kernels dominate; compare with arithmetic above for bimodal)")
            avg_mac = wmac
            avg_mte = wmte
        else:
            avg_mac = aic_mac_sum / aic_kernels
            avg_mte = (aic_mte1_sum + aic_mte2_sum) / aic_kernels
        if avg_mte > avg_mac * threshold("kernel_details", "hw_dominance_ratio", 1.5):
            lines.append(f"    → Memory-dominated: mte ({avg_mte:.3f}) >> mac ({avg_mac:.3f})")
        elif avg_mac > avg_mte * threshold("kernel_details", "hw_dominance_ratio", 1.5):
            lines.append(f"    → Compute-dominated: mac ({avg_mac:.3f}) >> mte ({avg_mte:.3f})")
    if aiv_kernels > 0:
        lines.append(f"  AI_VECTOR_CORE ({aiv_kernels} kernels):")
        lines.append(f"    vec (compute):  {aiv_vec_sum/aiv_kernels:.3f}")
        lines.append(f"    mte2 (load):    {aiv_mte2_sum/aiv_kernels:.3f}")
        lines.append(f"    mte3 (store):   {aiv_mte3_sum/aiv_kernels:.3f}")
        lines.append(f"    scalar:         {aiv_scalar_sum/aiv_kernels:.3f}")
        if aiv_icache_values:
            lines.append(f"    icache miss:    avg={sum(aiv_icache_values)/len(aiv_icache_values):.3f}  max={max(aiv_icache_values):.3f}")
        if aiv_dur_sum > 0:
            lines.append(f"    [duration-weighted] vec={aiv_vec_wsum/aiv_dur_sum:.3f}  mte={aiv_mte_wsum/aiv_dur_sum:.3f}")
    # Format distribution (A2): non-ND formats indicate layout conversion cost
    if format_counts:
        total_fmt = sum(format_counts.values())
        non_nd = sum(c for f, c in format_counts.items() if f != "ND" and f != "N/A")
        lines.append(f"  Input Formats: {dict(sorted(format_counts.items(), key=lambda x: -x[1]))}")
        if non_nd / total_fmt > 0.1 if total_fmt else False:
            lines.append(f"  → {non_nd}/{total_fmt} ({non_nd/total_fmt*100:.0f}%) non-ND input formats — layout conversion cost; cross-validate op_statistic Transpose/Cast.")
    if cube_util_values:
        avg_cube = sum(cube_util_values) / len(cube_util_values)
        min_cube = min(cube_util_values)
        low_util_count = sum(1 for v in cube_util_values if v < threshold("kernel_details", "cube_low_util", 50))
        lines.append(f"  Cube utilization: avg={avg_cube:.1f}%, min={min_cube:.1f}%, "
                     f"low(<50%)={low_util_count}/{len(cube_util_values)}")
    if aic_kernels == 0 and aiv_kernels == 0:
        lines.append("  ⚠ All ratios are 0 — check if aic_metrics=PipeUtilization was used during collection")
    lines.append("")

    # --- 3. AI CPU Fallback [DEFINITE] ---
    if aicpu_kernels:
        lines.append("## 3. AI CPU Fallback (non-comm) [DEFINITE]")
        lines.append("  Non-communication ops running on AI CPU (no AI Core impl). Communication ops excluded — they run on AI CPU by design.")
        aicpu_total = sum(d for d, _, _, _ in aicpu_kernels)
        lines.append(f"  Count: {len(aicpu_kernels)}  |  Total: {aicpu_total/1000:.1f}ms")
        aicpu_sorted = sorted(aicpu_kernels, key=lambda x: -x[0])
        lines.append(f"  {'Name':<35} {'Dur(us)':>8} {'Type':<15} {'Shapes'}")
        lines.append(f"  {'-'*35} {'-'*8} {'-'*15} {'-'*20}")
        for dur, name, otype, shapes in aicpu_sorted[:top_k]:
            shapes_clean = shapes.replace("\n", " ").replace(";", "|").replace('"', '')[:30]
            lines.append(f"  {name:<35} {dur:>8.1f} {otype:<15} {shapes_clean}")
        lines.append("")

    # --- 4. Kernel Duration Distribution ---
    sec_num = 4 if aicpu_kernels else 3
    lines.append(f"## {sec_num}. Kernel Duration Distribution")
    for bucket, count in dur_buckets.items():
        pct = count / total_rows * 100 if total_rows > 0 else 0
        bar = "█" * int(pct / 3)
        lines.append(f"  {bucket:>8}: {count:>6} ({pct:>5.1f}%) {bar}")
    short_ratio = (dur_buckets["<5us"] + dur_buckets["5-20us"]) / total_rows * 100 if total_rows > 0 else 0
    lines.append(f"  Short kernel ratio (<20us): {short_ratio:.1f}%")
    if short_ratio > threshold("kernel_details", "short_kernel_dominant", 60):
        lines.append(f"  → Most kernels are very short. Reducing op count may yield more than optimizing individual ops")
        lines.append(f"    Cross-validate: op_statistic § fragmentation signal, trace_view § dispatch latency")
    lines.append("")

    sec_num += 1
    lines.append(f"## {sec_num}. Small Kernels (duration < {small_threshold}us)")
    if small_count > 0:
        small_pct = small_count / total_rows * 100
        lines.append(f"  Count: {small_count:,} ({small_pct:.1f}% of all kernels)  |  Cumulative: {small_dur_total/1000:.2f}ms")
        lines.append(f"  Top types:")
        for t, c in sorted(small_type_count.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {t}: {c}")
    else:
        lines.append(f"  None found")
    lines.append("")

    # --- 5. Block Dim Distribution ---
    sec_num += 1
    lines.append(f"## {sec_num}. Block Dim Distribution (parallelism)")
    for bucket, count in block_dim_buckets.items():
        pct = count / total_rows * 100 if total_rows > 0 else 0
        bar = "█" * int(pct / 3)
        lines.append(f"  Dim {bucket:>5}: {count:>6} ({pct:>5.1f}%) {bar}")
    low_par = block_dim_buckets["1"]
    if low_par / total_rows > threshold("kernel_details", "low_parallelism_ratio", 0.1):
        lines.append(f"  → {low_par} kernels with Block Dim=1: shape may be too small for parallelism")
    lines.append("")

    # --- 6. Wait Time Distribution (factual) ---
    sec_num += 1
    lines.append(f"## {sec_num}. Wait Time Distribution")
    for bucket, count in wait_buckets.items():
        pct = count / total_rows * 100 if total_rows > 0 else 0
        lines.append(f"  {bucket:>10}: {count:>7} ({pct:.1f}%)")
    lines.append("")

    # --- 7. Suspect Signals (diagnostic) ---
    sec_num += 1
    lines.append(f"## {sec_num}. Suspect Signals")
    lines.append("  [DEFINITE]=actionable as-is  [SIGNAL]=anomaly, cross-validate with other dimensions")
    lines.append("")

    # 7a. Suspect Kernels
    suspect_sorted = sorted(suspect_heap, key=lambda x: -x[0])
    if suspect_sorted:
        lines.append("  [SIGNAL] High duration, low compute ratio — cross-validate: --filter <op> for shape distribution")
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

    # 7a-bis. True compute-bound kernels (high duration + high compute ratio)
    cb_sorted = sorted(compute_bound_heap, key=lambda x: -x[0])
    if cb_sorted:
        lines.append("  [SIGNAL] True compute-bound (high duration + high compute ratio) — replace/quantize/split targets")
        header = f"  {'Name':<42} {'Dur(us)':>8} {'mac':>6} {'BDim':>5} {'Shapes'}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for dur, _, name, core, mac_ratio, bdim, shapes in cb_sorted:
            shapes_clean = shapes.replace("\n", " ").replace(";", "|").replace('"', '')[:30]
            lines.append(f"  {name:<42} {dur:>8.1f} {mac_ratio:>5.2f} {bdim:>5} {shapes_clean}")
        lines.append("")

    # 7b. High-wait context
    high_wait_indices = [i for i, k in enumerate(all_kernels) if k["wait"] > wait_threshold]
    if high_wait_indices:
        lines.append(f"  [SIGNAL] High-wait kernels (wait > {wait_threshold:.0f}us) — cross-validate: trace_view for cause")
        lines.append("")
        top_waits = sorted(high_wait_indices, key=lambda i: -all_kernels[i]["wait"])[:min(top_k, 8)]
        for rank_idx, idx in enumerate(top_waits, 1):
            k = all_kernels[idx]
            lines.append(f"  [{rank_idx}] #{idx} {k['name']}  wait={format_duration_ms(k['wait'])}  stream={k['stream']}")
            # temporal neighbors on the SAME stream (not file order)
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

    # 7c. Fusible operator sequences
    if fusible_sequences:
        fusible_sorted = sorted(fusible_sequences, key=lambda x: -x[0])
        total_fusible = sum(s[0] for s in fusible_sorted)
        lines.append(f"  [SIGNAL] Fusible sequences: {len(fusible_sorted)} sequences of ≥{MIN_LEN} consecutive small kernels (<{SMALL_THRESH}us)")
        lines.append(f"    Total time in fusible sequences: {total_fusible/1000:.2f}ms ({total_fusible/total_dur_us*100:.1f}% of compute)")
        lines.append(f"    Top {min(top_k, 5)} by cumulative time:")
        for total, count, start_idx, types, s in fusible_sorted[:5]:
            from collections import Counter
            tc = Counter(types).most_common(3)
            type_str = ", ".join(f"{t}:{c}" for t, c in tc)
            lines.append(f"      {total/1000:.2f}ms  {count} kernels  at #{start_idx}  stream={s}  types: {type_str}")
        lines.append("    → Cross-validate: check if these can be fused (equivalent_substitution layer 1) or batched.")
        lines.append("")

    if not suspect_sorted and not cb_sorted and not high_wait_indices and not fusible_sequences:
        lines.append("  None")
        lines.append("")

    return "\n".join(lines)


def parse_filtered(profiling_dir: str, filters: list, rank=None, top_k: int = 15) -> str:
    """Filtered mode: deep analysis for specific operator(s)."""
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "kernel_details.csv"

    if not csv_path.exists():
        return f"[kernel_details] File not found: {csv_path}"

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
    lines.append(f"# Kernel Details — Filtered Deep Analysis")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Filter: {', '.join(filters)}")
    lines.append(f"Matched kernels: {len(matched)}")
    lines.append("")

    if not matched:
        lines.append("No kernels matched the filter.")
        return "\n".join(lines)

    # --- 1. Summary ---
    total_dur = sum(safe_float(r.get("Duration(us)", 0)) for r in matched)
    total_wait = sum(safe_float(r.get("Wait Time(us)", 0)) for r in matched)
    lines.append("## 1. Summary")
    lines.append(f"  Count: {len(matched)}")
    lines.append(f"  Total duration: {total_dur/1000:.2f} ms")
    lines.append(f"  Total wait: {total_wait/1000:.2f} ms")
    lines.append(f"  Avg duration: {total_dur/len(matched):.1f} us")
    lines.append(f"  Avg wait: {total_wait/len(matched):.1f} us")
    lines.append("")

    # --- 2. Shape → Performance correlation ---
    lines.append("## 2. Shape → Performance Correlation")
    shape_groups = defaultdict(list)
    for r in matched:
        shape = r.get("Input Shapes", "").replace("\n", " ").replace('"', '').strip()
        if not shape:
            shape = "(empty)"
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

    # --- 3. Per-instance hardware breakdown (slowest N) ---
    lines.append(f"## 3. Slowest Instances — Hardware Breakdown")
    matched_sorted = sorted(matched, key=lambda r: -safe_float(r.get("Duration(us)", 0)))
    lines.append(f"  (Top {min(top_k, len(matched_sorted))} by duration, showing per-instance hardware ratios)")
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

    # --- 4. Wait time context: what's before/after this op ---
    lines.append("## 4. Wait Time Context")
    matched_waits = [(safe_float(r.get("Wait Time(us)", 0)), i)
                     for i, r in enumerate(matched)]
    avg_wait = total_wait / len(matched)
    max_wait_val = max(w for w, _ in matched_waits) if matched_waits else 0
    lines.append(f"  Avg wait: {avg_wait:.0f}us, Max wait: {max_wait_val:.0f}us")
    lines.append("")

    # Find matched kernel positions in the full sequence to show neighbors
    high_wait_matched = []
    for seq_idx, (file_idx, is_match, name, op_type, dur, wait) in enumerate(all_seq):
        if is_match and wait > avg_wait * 3 and wait > 200:
            high_wait_matched.append((wait, seq_idx))
    high_wait_matched.sort(key=lambda x: -x[0])

    if high_wait_matched:
        lines.append(f"  Instances with wait > 3x average ({avg_wait*3:.0f}us), showing neighbors:")
        lines.append("")
        for rank_idx, (wait_val, seq_idx) in enumerate(high_wait_matched[:min(8, top_k)], 1):
            lines.append(f"  [{rank_idx}] wait={format_duration_ms(wait_val)} at position #{seq_idx}")
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
        lines.append("  No instances with significantly elevated wait time.")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Filter by operator name/type (substring match, case-insensitive). "
                             "When specified, shows detailed per-kernel info for matched operators only.")
    parser.add_argument("--small-threshold", type=float, default=5.0,
                        help="Duration threshold (us) for small kernel identification")
    parser.add_argument("--wait-threshold", type=float, default=500.0,
                        help="Wait time threshold (us) for high-wait context analysis")
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
