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
                    format_size_mb, format_duration_ms)


def parse(profiling_dir: str, rank=None, top_k: int = 20) -> str:
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

    for row in stream_csv(csv_path):
        total_rows += 1
        size_kb = safe_float(row.get("Size(KB)", 0))
        duration_us = safe_float(row.get("Duration(us)", 0))
        name = row.get("Name", "?")
        alloc_total = safe_float(row.get("Allocation Total Allocated(MB)", 0))

        max_alloc_at_alloc = max(max_alloc_at_alloc, alloc_total)

        if size_kb > 0:
            entry = (size_kb, total_rows, row)
            if len(top_size_heap) < top_k:
                heapq.heappush(top_size_heap, entry)
            elif size_kb > top_size_heap[0][0]:
                heapq.heapreplace(top_size_heap, entry)

        if size_kb > 100 and 0 < duration_us < 1000:
            short_lived_large.append((size_kb, duration_us, name))

        if size_kb > 10:
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

    repeated = [(key, count) for key, count in size_op_count.items() if count > 10]
    repeated.sort(key=lambda x: -x[1])
    if repeated:
        lines.append("  - [DEFINITE] Repeated same-size allocations (buffer reuse opportunity):")
        for key, count in repeated[:8]:
            name_part, size_part = key.rsplit("|", 1)
            lines.append(f"    {name_part}: {size_part}KB × {count} times")
        suspects_found = True

    if short_lived_large:
        total_short_kb = sum(s for s, _, _ in short_lived_large)
        if total_short_kb > 10000:
            lines.append(f"  - [DEFINITE] Short-lived large tensor churn: {len(short_lived_large)} tensors, "
                         f"cumulative {total_short_kb/1024:.0f}MB")
            lines.append(f"    Allocating and immediately freeing — pre-allocation would eliminate this overhead")
            suspects_found = True

    if op_sorted:
        top_op_total = op_sorted[0][1]["total_kb"]
        total_all = sum(info["total_kb"] for _, info in op_sorted)
        if total_all > 0 and top_op_total / total_all > 0.5:
            lines.append(f"  - [SIGNAL] Single op dominates memory: {op_sorted[0][0]} "
                         f"({top_op_total/1024:.0f}MB, {top_op_total/total_all*100:.0f}% of total)")
            suspects_found = True

    if not suspects_found:
        lines.append("  None")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank, args.top_k)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
