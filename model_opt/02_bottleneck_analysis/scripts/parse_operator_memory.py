#!/usr/bin/env python3
"""Parse operator_memory.csv — per-tensor allocation lifecycle analysis.

This file records each tensor's: size, allocation/release time, lifetime duration,
and global memory state at alloc/release. Unique value over memory_record.csv:
- Per-tensor granularity (who allocated what, how big, how long it lived)
- Tensor lifetime (short-lived large tensors = buffer reuse candidates)

Usage:
    python parse_operator_memory.py <profiling_dir> [--rank N] [--top-k 20]
"""

import argparse
import heapq
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (find_ascend_profiler_output, stream_csv, safe_float,
                    format_size_mb, format_duration_ms, threshold)


def parse(profiling_dir: str, rank=None, top_k: int = 20,
          hbm_gb: float = 64.0, large_gb: float = 5.0) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "operator_memory.csv"

    if not csv_path.exists():
        return f"[operator_memory] File not found: {csv_path}"

    total_rows = 0
    top_size_heap = []
    short_lived_large = []
    size_op_count = defaultdict(int)
    op_agg = defaultdict(lambda: {"count": 0, "total_kb": 0.0, "durations": []})
    max_alloc_at_alloc = 0.0

    # Large tensor classification for parallelism trigger
    LARGE_KB = large_gb * 1024 * 1024  # 5GB default
    SHORT_LIFE_US = threshold("operator_memory", "short_life_us", 1_000_000)
    SHORT_LIVED_MIN_KB = threshold("operator_memory", "short_lived_min_kb", 100)
    SHORT_LIVED_MAX_LIFE = threshold("operator_memory", "short_lived_max_life_us", 1000)
    SIZE_TRACK_MIN_KB = threshold("operator_memory", "size_track_min_kb", 10)
    REPEATED_COUNT = threshold("operator_memory", "repeated_count", 10)
    CHURN_TOTAL_KB = threshold("operator_memory", "churn_total_kb", 10000)
    DOMINATE_RATIO = threshold("operator_memory", "dominate_ratio", 0.5)
    PARALLELISM_RATIO = threshold("operator_memory", "parallelism_ratio", 0.8)
    large_short = []  # (size_kb, dur_us, name, alloc_us, release_us) — waste
    large_long = []   # (size_kb, dur_us, name) — essential
    peak_alloc_time = 0  # timestamp of peak allocation

    for row in stream_csv(csv_path):
        total_rows += 1
        size_kb = safe_float(row.get("Size(KB)", 0))
        duration_us = safe_float(row.get("Duration(us)", 0))
        name = row.get("Name", "?")
        alloc_total = safe_float(row.get("Allocation Total Allocated(MB)", 0))

        max_alloc_at_alloc = max(max_alloc_at_alloc, alloc_total)
        if alloc_total >= max_alloc_at_alloc:
            peak_alloc_time = safe_float(row.get("Allocation Time(us)", 0))

        if size_kb > 0:
            entry = (size_kb, total_rows, row)
            if len(top_size_heap) < top_k:
                heapq.heappush(top_size_heap, entry)
            elif size_kb > top_size_heap[0][0]:
                heapq.heapreplace(top_size_heap, entry)

        if size_kb > SHORT_LIVED_MIN_KB and 0 < duration_us < SHORT_LIVED_MAX_LIFE:
            short_lived_large.append((size_kb, duration_us, name))

        # Short-lived by ACTIVE duration (true referenced time, not pool lifetime).
        # Caching allocator retains freed tensors in pool → Duration can be long
        # while Active is short. Use Active to catch reuse candidates the pool
        # Duration would mark "long-lived" and miss.
        active_dur = safe_float(row.get("Active Duration(us)", 0))
        if size_kb > SHORT_LIVED_MIN_KB and 0 < active_dur < SHORT_LIVED_MAX_LIFE:
            # dedup against the Duration-based list by (size,name) is unnecessary;
            # we track Active-based separately to surface cache-retained cases.
            if 0 < duration_us >= SHORT_LIVED_MAX_LIFE:
                short_lived_large.append((size_kb, active_dur, name))

        # Classify large tensors for parallelism trigger (use Active Duration:
        # a tensor cached-but-inactive is "waste" only if its active life is short)
        if size_kb > LARGE_KB and duration_us > 0:
            alloc_us = safe_float(row.get("Allocation Time(us)", 0))
            release_us = safe_float(row.get("Release Time(us)", 0))
            life_us = safe_float(row.get("Active Duration(us)", 0)) or duration_us
            if life_us < SHORT_LIFE_US:
                large_short.append((size_kb, life_us, name, alloc_us, release_us))
            else:
                large_long.append((size_kb, life_us, name))

        if size_kb > SIZE_TRACK_MIN_KB:
            size_op_count[f"{name}|{size_kb:.0f}"] += 1

        if size_kb > 0:
            op_agg[name]["count"] += 1
            op_agg[name]["total_kb"] += size_kb
            if len(op_agg[name]["durations"]) < 1000:
                op_agg[name]["durations"].append(duration_us)

    if total_rows == 0:
        return f"[operator_memory] Empty file: {csv_path}"

    lines = []
    lines.append("# Operator Memory Analysis")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Total allocation records: {total_rows:,}")
    lines.append(f"Peak Allocated (at any alloc point): {max_alloc_at_alloc:,.0f} MB")
    lines.append("")

    # --- 1. Top allocations by size ---
    top_entries = sorted(top_size_heap, key=lambda x: -x[0])
    lines.append(f"## 1. Top {min(top_k, len(top_entries))} Allocations by Size")
    header = f"  {'#':>3} {'Op Name':<30} {'Size':>10} {'Lifetime':>12} {'Alloc@(MB)':>11}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for idx, (size_kb, _, row) in enumerate(top_entries, 1):
        name = row.get("Name", "?")
        dur = safe_float(row.get("Duration(us)", 0))
        alloc_at = safe_float(row.get("Allocation Total Allocated(MB)", 0))
        lines.append(
            f"  {idx:>3} {name:<30} {format_size_mb(size_kb):>10} "
            f"{format_duration_ms(dur):>12} {alloc_at:>10,.0f}"
        )
    lines.append("")

    # --- 2. By-Op aggregation ---
    op_sorted = sorted(op_agg.items(), key=lambda x: -x[1]["total_kb"])
    lines.append(f"## 2. Aggregated by Op (Top {min(top_k, len(op_sorted))} by total size)")
    header2 = f"  {'Op Name':<30} {'Count':>7} {'Total':>10} {'Avg Size':>10} {'Avg Life':>10}"
    lines.append(header2)
    lines.append("  " + "-" * (len(header2) - 2))
    for name, info in op_sorted[:top_k]:
        avg_size = info["total_kb"] / info["count"] if info["count"] > 0 else 0
        avg_dur = sum(info["durations"]) / len(info["durations"]) if info["durations"] else 0
        lines.append(
            f"  {name:<30} {info['count']:>7} {format_size_mb(info['total_kb']):>10} "
            f"{format_size_mb(avg_size):>10} {format_duration_ms(avg_dur):>10}"
        )
    lines.append("")

    # --- 3. Short-lived large tensors ---
    lines.append("## 3. Short-Lived Large Tensors (size>100KB, lifetime<1ms)")
    if short_lived_large:
        lines.append(f"  Found: {len(short_lived_large)} tensors")
        lines.append(f"  These are allocated and freed quickly — strong buffer reuse candidates.")
        lines.append("")

        short_by_op = defaultdict(lambda: {"count": 0, "sizes": []})
        for size_kb, dur, name in short_lived_large:
            short_by_op[name]["count"] += 1
            if len(short_by_op[name]["sizes"]) < 10:
                short_by_op[name]["sizes"].append(size_kb)

        short_sorted = sorted(short_by_op.items(), key=lambda x: -x[1]["count"])
        header3 = f"  {'Op Name':<30} {'Count':>7} {'Typical Sizes'}"
        lines.append(header3)
        lines.append("  " + "-" * (len(header3) - 2))
        for name, info in short_sorted[:top_k]:
            sizes_uniq = sorted(set(f"{s:.0f}KB" for s in info["sizes"]))[:5]
            lines.append(f"  {name:<30} {info['count']:>7} {', '.join(sizes_uniq)}")
    else:
        lines.append("  None found")
    lines.append("")

    # --- Suspect Signals ---
    lines.append("## Suspect Signals")
    lines.append("  [DEFINITE]=actionable as-is  [SIGNAL]=anomaly, root cause uncertain — cross-validate with other profiling dimensions")
    suspects_found = False

    repeated = [(key, count) for key, count in size_op_count.items() if count > REPEATED_COUNT]
    repeated.sort(key=lambda x: -x[1])
    if repeated:
        lines.append("  - [DEFINITE] Repeated same-size allocations (buffer reuse opportunity):")
        for key, count in repeated[:8]:
            name_part, size_part = key.rsplit("|", 1)
            lines.append(f"    {name_part}: {size_part}KB × {count} times")
        suspects_found = True

    if short_lived_large:
        total_short_kb = sum(s for s, _, _ in short_lived_large)
        if total_short_kb > CHURN_TOTAL_KB:
            lines.append(f"  - [DEFINITE] Short-lived large tensor churn: {len(short_lived_large)} tensors, "
                         f"cumulative {total_short_kb/1024:.0f}MB")
            lines.append(f"    Allocating and immediately freeing — pre-allocation would eliminate this overhead")
            suspects_found = True

    if op_sorted:
        top_op_total = op_sorted[0][1]["total_kb"]
        total_all = sum(info["total_kb"] for _, info in op_sorted)
        if total_all > 0 and top_op_total / total_all > DOMINATE_RATIO:
            lines.append(f"  - [SIGNAL] Single op dominates memory: {op_sorted[0][0]} "
                         f"({top_op_total/1024:.0f}MB, {top_op_total/total_all*100:.0f}% of total)")
            suspects_found = True

    if not suspects_found:
        lines.append("  None")
    lines.append("")

    # --- Parallelism Trigger (two-stage: waste vs essential) ---
    hbm_mb = hbm_gb * 1024
    peak_mb = max_alloc_at_alloc

    # Waste at peak: sum of short-lived large tensors alive at peak time
    waste_at_peak_mb = 0
    waste_total_mb = 0
    for size_kb, dur, name, alloc_us, release_us in large_short:
        waste_total_mb += size_kb / 1024
        if alloc_us <= peak_alloc_time <= release_us:
            waste_at_peak_mb += size_kb / 1024

    essential_mb = sum(s for s, _, _ in large_long) / 1024
    projected_peak_mb = max(0, peak_mb - waste_at_peak_mb)

    lines.append("## Parallelism Trigger Analysis")
    lines.append(f"  HBM: {hbm_gb:.0f}GB  |  Peak: {peak_mb:,.0f}MB ({peak_mb/hbm_mb*100:.0f}% HBM)")
    lines.append(f"  Large tensors (>{large_gb:.0f}GB):")
    lines.append(f"    Short-lived (<1s, waste):  {len(large_short)} tensors, {waste_total_mb:,.0f}MB total")
    lines.append(f"      ↳ alive at peak moment: {waste_at_peak_mb:,.0f}MB")
    lines.append(f"    Long-lived (>1s, essential): {len(large_long)} tensors, {essential_mb:,.0f}MB")
    lines.append(f"  Projected peak after waste elimination: {projected_peak_mb:,.0f}MB ({projected_peak_mb/hbm_mb*100:.0f}% HBM)")
    lines.append("")

    if projected_peak_mb / hbm_mb > PARALLELISM_RATIO:
        lines.append(f"  [SIGNAL] Projected peak ({projected_peak_mb/hbm_mb*100:.0f}% HBM) exceeds {PARALLELISM_RATIO*100:.0f}% after waste elimination.")
        lines.append("    → Parallelism may be required. Source code analysis needed:")
        lines.append("      1. Read parallel_design.md for splitting principles")
        lines.append("      2. Use operator_details Call Stack to locate large tensors in source code")
        lines.append("      3. Identify shardable dimensions from computation structure")
        lines.append("      4. Re-profile after splitting to verify")
        lines.append("    → Profiling only triggers; source code determines the splitting strategy.")
    elif waste_at_peak_mb > 0:
        lines.append(f"  [DEFINITE] Waste at peak = {waste_at_peak_mb:,.0f}MB. "
                     f"Projected peak after elimination: {projected_peak_mb/hbm_mb*100:.0f}% HBM — within single-card capacity.")
        lines.append("    → Parallelism not required. Prioritize memory optimization (eliminate/reuse).")
    else:
        lines.append("  No large short-lived tensors at peak. Memory is not a parallelism trigger.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hbm-gb", type=float, default=64.0,
                        help="HBM capacity in GB (default 64 for Ascend 910B)")
    parser.add_argument("--large-gb", type=float, default=5.0,
                        help="Threshold for 'large tensor' classification in GB (default 5)")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank, args.top_k, args.hbm_gb, args.large_gb)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
