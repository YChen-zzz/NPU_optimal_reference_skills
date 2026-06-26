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
from common import find_ascend_profiler_output, stream_csv, safe_float, format_duration_ms


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

    # Hardware unit aggregation (for AI_VECTOR_CORE)
    aiv_kernels = 0
    aiv_vec_sum = 0.0
    aiv_mte2_sum = 0.0
    aiv_mte3_sum = 0.0
    aiv_scalar_sum = 0.0

    # Small kernel tracking
    small_count = 0
    small_dur_total = 0.0
    small_type_count = defaultdict(int)

    # Block Dim distribution
    block_dim_buckets = {"1": 0, "2-8": 0, "9-28": 0, "29+": 0}

    # Cube utilization tracking (for AI_CORE only)
    cube_util_values = []

    # Suspect kernels: high duration but low compute ratio (both core types)
    suspect_heap = []

    # Wait time distribution buckets
    wait_buckets = {"<100us": 0, "100-500us": 0, "500-2000us": 0, ">2000us": 0}

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

        core_stats[core]["count"] += 1
        core_stats[core]["dur_us"] += dur

        # Hardware unit ratios
        if core == "AI_CORE" and dur > 0:
            aic_kernels += 1
            mac_ratio = safe_float(row.get("aic_mac_ratio", 0))
            mte1_ratio = safe_float(row.get("aic_mte1_ratio", 0))
            mte2_ratio = safe_float(row.get("aic_mte2_ratio", 0))
            scalar_ratio = safe_float(row.get("aic_scalar_ratio", 0))
            aic_mac_sum += mac_ratio
            aic_mte1_sum += mte1_ratio
            aic_mte2_sum += mte2_ratio
            aic_scalar_sum += scalar_ratio
            if cube_util > 0:
                cube_util_values.append(cube_util)
            # Suspect: high duration but compute ratio low
            if dur > 10 and mac_ratio < 0.2:
                entry = (dur, total_rows, name, core, mac_ratio,
                         mte1_ratio + mte2_ratio, row.get("Input Shapes", ""), block_dim)
                if len(suspect_heap) < top_k:
                    heapq.heappush(suspect_heap, entry)
                elif dur > suspect_heap[0][0]:
                    heapq.heapreplace(suspect_heap, entry)

        elif core == "AI_VECTOR_CORE" and dur > 0:
            aiv_kernels += 1
            vec_ratio = safe_float(row.get("aiv_vec_ratio", 0))
            aiv_mte2_ratio = safe_float(row.get("aiv_mte2_ratio", 0))
            aiv_mte3_ratio = safe_float(row.get("aiv_mte3_ratio", 0))
            aiv_scalar_ratio_val = safe_float(row.get("aiv_scalar_ratio", 0))
            aiv_vec_sum += vec_ratio
            aiv_mte2_sum += aiv_mte2_ratio
            aiv_mte3_sum += aiv_mte3_ratio
            aiv_scalar_sum += aiv_scalar_ratio_val
            # Suspect: high duration but vec ratio low
            if dur > 10 and vec_ratio < 0.05:
                entry = (dur, total_rows, name, core, vec_ratio,
                         aiv_mte2_ratio + aiv_mte3_ratio, row.get("Input Shapes", ""), block_dim)
                if len(suspect_heap) < top_k:
                    heapq.heappush(suspect_heap, entry)
                elif dur > suspect_heap[0][0]:
                    heapq.heapreplace(suspect_heap, entry)

        # Small kernel
        if dur < small_threshold and dur > 0:
            small_count += 1
            small_dur_total += dur
            small_type_count[op_type] += 1

        # Block Dim
        if block_dim == 1:
            block_dim_buckets["1"] += 1
        elif block_dim <= 8:
            block_dim_buckets["2-8"] += 1
        elif block_dim <= 28:
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

        # Store for context analysis
        all_kernels.append({"name": name, "type": op_type, "dur": dur, "wait": wait})

    if total_rows == 0:
        return f"[kernel_details] Empty file: {csv_path}"

    lines = []
    lines.append("# Kernel Details Analysis")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Total kernels: {total_rows:,}")
    lines.append(f"Total compute: {total_dur_us/1000:.1f} ms")
    lines.append(f"Total wait: {total_wait_us/1000:.1f} ms")
    lines.append("")

    # --- 1. Accelerator Core Split ---
    lines.append("## 1. Accelerator Core Distribution")
    for core, info in sorted(core_stats.items(), key=lambda x: -x[1]["dur_us"]):
        pct = info["dur_us"] / total_dur_us * 100 if total_dur_us > 0 else 0
        lines.append(f"  {core}: {info['count']:,} kernels, {info['dur_us']/1000:.1f}ms ({pct:.1f}%)")
    lines.append("")

    # --- 2. Hardware Unit Utilization ---
    lines.append("## 2. Hardware Unit Utilization (average ratios)")
    if aic_kernels > 0:
        lines.append(f"  AI_CORE ({aic_kernels} kernels):")
        lines.append(f"    mac (compute):  {aic_mac_sum/aic_kernels:.3f}")
        lines.append(f"    mte1 (load):    {aic_mte1_sum/aic_kernels:.3f}")
        lines.append(f"    mte2 (store):   {aic_mte2_sum/aic_kernels:.3f}")
        lines.append(f"    scalar:         {aic_scalar_sum/aic_kernels:.3f}")
        avg_mac = aic_mac_sum / aic_kernels
        avg_mte = (aic_mte1_sum + aic_mte2_sum) / aic_kernels
        if avg_mte > avg_mac * 1.5:
            lines.append(f"    → Memory-dominated: mte ratio ({avg_mte:.3f}) >> mac ratio ({avg_mac:.3f})")
        elif avg_mac > avg_mte * 1.5:
            lines.append(f"    → Compute-dominated: mac ratio ({avg_mac:.3f}) >> mte ratio ({avg_mte:.3f})")
    if aiv_kernels > 0:
        lines.append(f"  AI_VECTOR_CORE ({aiv_kernels} kernels):")
        lines.append(f"    vec (compute):  {aiv_vec_sum/aiv_kernels:.3f}")
        lines.append(f"    mte2 (load):    {aiv_mte2_sum/aiv_kernels:.3f}")
        lines.append(f"    mte3 (store):   {aiv_mte3_sum/aiv_kernels:.3f}")
        lines.append(f"    scalar:         {aiv_scalar_sum/aiv_kernels:.3f}")
    if cube_util_values:
        avg_cube = sum(cube_util_values) / len(cube_util_values)
        min_cube = min(cube_util_values)
        low_util_count = sum(1 for v in cube_util_values if v < 50)
        lines.append(f"  Cube utilization: avg={avg_cube:.1f}%, min={min_cube:.1f}%, "
                     f"low(<50%)={low_util_count}/{len(cube_util_values)}")
    lines.append("")

    # --- 3. Small Kernel Identification ---
    lines.append(f"## 3. Small Kernels (duration < {small_threshold}us)")
    if small_count > 0:
        small_pct = small_count / total_rows * 100
        lines.append(f"  Count: {small_count:,} ({small_pct:.1f}% of all kernels)")
        lines.append(f"  Cumulative time: {small_dur_total/1000:.2f} ms")
        lines.append(f"  Type distribution:")
        for t, c in sorted(small_type_count.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {t}: {c}")
    else:
        lines.append(f"  None found")
    lines.append("")

    # --- 4. Block Dim (Parallelism) ---
    lines.append("## 4. Block Dim Distribution (parallelism)")
    for bucket, count in block_dim_buckets.items():
        pct = count / total_rows * 100 if total_rows > 0 else 0
        bar = "█" * int(pct / 3)
        lines.append(f"  Dim {bucket:>5}: {count:>6} ({pct:>5.1f}%) {bar}")
    low_par = block_dim_buckets["1"]
    if low_par / total_rows > 0.1:
        lines.append(f"  → {low_par} kernels with Block Dim=1: possibly shape too small for parallelism")
    lines.append("")

    # --- 5. Suspect Kernels (high duration, low compute ratio) ---
    lines.append("## 5. Suspect Kernels (high duration, low compute ratio)")
    lines.append("  Kernels where duration is high but hardware compute units (mac/vec) are barely used.")
    lines.append("  May indicate: shape unfriendly to hardware, excessive data movement, or optimization opportunity.")
    lines.append("  Use --filter to drill into specific types for further investigation.")
    lines.append("")
    suspect_sorted = sorted(suspect_heap, key=lambda x: -x[0])
    if suspect_sorted:
        header = f"  {'Name':<42} {'Core':<10} {'Dur(us)':>8} {'Compute':>8} {'Move':>8} {'BDim':>5} {'Shapes'}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for dur, _, name, core, compute_ratio, move_ratio, shapes, bdim in suspect_sorted:
            shapes_clean = shapes.replace("\n", " ").replace(";", "|").replace('"', '')[:30]
            core_short = "AIC" if "AI_CORE" == core else "AIV"
            lines.append(
                f"  {name:<42} {core_short:<10} {dur:>8.1f} "
                f"{compute_ratio:>7.3f} {move_ratio:>7.3f} {bdim:>5} {shapes_clean}")
    else:
        lines.append("  None found")
    lines.append("")

    # --- 6. Wait Time Distribution ---
    lines.append("## 6. Wait Time Distribution")
    for bucket, count in wait_buckets.items():
        pct = count / total_rows * 100 if total_rows > 0 else 0
        lines.append(f"  {bucket:>10}: {count:>7} kernels ({pct:.1f}%)")
    lines.append("")

    # High-wait context (top few)
    high_wait_indices = [i for i, k in enumerate(all_kernels) if k["wait"] > wait_threshold]
    if high_wait_indices:
        lines.append(f"  Top high-wait points (wait > {wait_threshold:.0f}us), showing surrounding kernels:")
        lines.append("")
        # Pick top-k by wait time
        top_waits = sorted(high_wait_indices, key=lambda i: -all_kernels[i]["wait"])[:min(top_k, 8)]
        for rank_idx, idx in enumerate(top_waits, 1):
            k = all_kernels[idx]
            lines.append(f"  [{rank_idx}] kernel #{idx}: {k['name']}  wait={format_duration_ms(k['wait'])}")
            context = 2
            start = max(0, idx - context)
            end = min(len(all_kernels), idx + context + 1)
            for i in range(start, end):
                ck = all_kernels[i]
                marker = " <<<" if i == idx else ""
                lines.append(f"      [{i}] {ck['type']:<20} dur={ck['dur']:>7.1f}us  wait={ck['wait']:>7.0f}us{marker}")
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
