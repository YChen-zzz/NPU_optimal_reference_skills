#!/usr/bin/env python3
"""Parse memory_record.csv — memory usage timeline.

This file records memory state over time from two sources:
- APP rows: periodic sampling (every ~20ms), only Total Reserved
- PTA/PTA+GE rows: event-triggered on each alloc/dealloc, has Allocated + Reserved + Active

Key metrics:
- Total Reserved: allocator pool size (grows in steps, rarely shrinks)
- Total Allocated: actual tensor memory usage (fluctuates with alloc/free)
- Reserved - Allocated: free pool space (fragmentation / headroom)

Usage:
    python parse_memory_record.py <profiling_dir> [--rank N] [--buckets 20] [--top-k 10]
"""

import argparse
import heapq
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, stream_csv, safe_float


def parse(profiling_dir: str, rank=None, num_buckets: int = 20, top_k: int = 10) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "memory_record.csv"

    if not csv_path.exists():
        return f"[memory_record] File not found: {csv_path}"

    # Collect records, separating by what data is available
    reserved_records = []  # (ts, reserved) - all rows
    allocated_records = []  # (ts, allocated, reserved) - PTA rows only
    component_counts = defaultdict(int)

    for row in stream_csv(csv_path):
        ts = safe_float(row.get("Timestamp(us)", 0))
        reserved = safe_float(row.get("Total Reserved(MB)", 0))
        allocated = safe_float(row.get("Total Allocated(MB)", 0))
        component = row.get("Component", "").strip()

        if ts <= 0:
            continue

        component_counts[component] += 1
        reserved_records.append((ts, reserved))

        if allocated > 0:
            allocated_records.append((ts, allocated, reserved))

    if not reserved_records:
        return f"[memory_record] No valid records in {csv_path}"

    reserved_records.sort(key=lambda x: x[0])
    t0 = reserved_records[0][0]
    t_end = reserved_records[-1][0]
    duration_s = (t_end - t0) / 1e6

    lines = []
    lines.append("# Memory Record Summary")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Total records: {len(reserved_records):,}")
    lines.append(f"Duration: {duration_s:.2f}s")
    lines.append(f"Components: {dict(component_counts)}")
    lines.append("")

    # --- 1. Reserved memory (pool size) ---
    res_values = [r[1] for r in reserved_records]
    max_res = max(res_values)
    min_res = min(res_values)
    max_res_idx = res_values.index(max_res)
    max_res_time = (reserved_records[max_res_idx][0] - t0) / 1e6

    lines.append("## 1. Reserved Memory (allocator pool)")
    lines.append(f"  Min: {min_res:,.0f} MB")
    lines.append(f"  Max: {max_res:,.0f} MB  (at {max_res_time:.3f}s)")
    lines.append(f"  Range: {max_res - min_res:,.0f} MB")
    lines.append("")

    # Bucketed timeline for Reserved
    if num_buckets > 0 and len(reserved_records) > 1:
        lines.append("  Timeline (max Reserved per bucket):")
        bucket_width = (t_end - t0) / num_buckets
        buckets = [0.0] * num_buckets
        for ts, mem in reserved_records:
            idx = min(int((ts - t0) / bucket_width), num_buckets - 1)
            buckets[idx] = max(buckets[idx], mem)

        bar_width = 30
        scale = max_res if max_res > 0 else 1
        for i, bmax in enumerate(buckets):
            t_s = (i * bucket_width) / 1e6
            bar_len = int(bmax / scale * bar_width)
            bar = "█" * bar_len
            lines.append(f"    {t_s:>7.3f}s {bar:<{bar_width}} {bmax:>8,.0f} MB")
        lines.append("")

    # --- 2. Allocated memory (actual tensor usage) ---
    if allocated_records:
        allocated_records.sort(key=lambda x: x[0])
        alloc_values = [r[1] for r in allocated_records]
        max_alloc = max(alloc_values)
        min_alloc = min(alloc_values)
        max_alloc_idx = alloc_values.index(max_alloc)
        max_alloc_time = (allocated_records[max_alloc_idx][0] - t0) / 1e6

        lines.append("## 2. Allocated Memory (actual tensor usage)")
        lines.append(f"  Records with Allocated data: {len(allocated_records):,}")
        lines.append(f"  Min: {min_alloc:,.0f} MB")
        lines.append(f"  Max: {max_alloc:,.0f} MB  (at {max_alloc_time:.3f}s)")
        lines.append(f"  Range: {max_alloc - min_alloc:,.0f} MB")
        lines.append("")

        # --- 3. Fragmentation: Reserved - Allocated ---
        frag_values = [r[2] - r[1] for r in allocated_records]
        max_frag = max(frag_values)
        min_frag = min(frag_values)
        avg_frag = sum(frag_values) / len(frag_values)

        lines.append("## 3. Pool Fragmentation (Reserved - Allocated)")
        lines.append(f"  Min gap: {min_frag:,.0f} MB")
        lines.append(f"  Max gap: {max_frag:,.0f} MB")
        lines.append(f"  Avg gap: {avg_frag:,.0f} MB")
        if max_frag > threshold("memory_record", "frag_gap_mb", 1000):
            lines.append(f"  → Large gap ({max_frag:,.0f}MB) suggests significant pool fragmentation or over-reservation")
        lines.append("")

        # Bucketed fragmentation timeline
        if num_buckets > 0 and len(allocated_records) > 1:
            alloc_t0 = allocated_records[0][0]
            alloc_tend = allocated_records[-1][0]
            if alloc_tend > alloc_t0:
                lines.append("  Fragmentation timeline (max gap per bucket):")
                bw = (alloc_tend - alloc_t0) / num_buckets
                frag_buckets = [0.0] * num_buckets
                for ts, alloc, res in allocated_records:
                    bidx = min(int((ts - alloc_t0) / bw), num_buckets - 1)
                    frag_buckets[bidx] = max(frag_buckets[bidx], res - alloc)

                frag_scale = max_frag if max_frag > 0 else 1
                for i, fmax in enumerate(frag_buckets):
                    t_s = (i * bw) / 1e6
                    bar_len = int(fmax / frag_scale * bar_width) if frag_scale > 0 else 0
                    bar = "█" * bar_len
                    lines.append(f"    {t_s:>7.3f}s {bar:<{bar_width}} {fmax:>8,.0f} MB")
                lines.append("")
    else:
        lines.append("## 2. Allocated Memory")
        lines.append("  (No Allocated data available — only APP sampling rows present)")
        lines.append("")

    # --- 4. Top jumps ---
    jumps = []
    for i in range(1, len(reserved_records)):
        delta = reserved_records[i][1] - reserved_records[i - 1][1]
        if delta != 0:
            jumps.append((abs(delta), delta, reserved_records[i][0], reserved_records[i][1]))

    top_allocs = heapq.nlargest(top_k, (j for j in jumps if j[1] > 0), key=lambda x: x[0])
    top_deallocs = heapq.nlargest(top_k, (j for j in jumps if j[1] < 0), key=lambda x: x[0])

    lines.append(f"## 4. Top {top_k} Reserved Jumps")
    if top_allocs:
        lines.append("  Largest increases (pool growth):")
        header = f"    {'Time(s)':>9} {'Delta(MB)':>12} {'After(MB)':>12}"
        lines.append(header)
        for _, delta, ts, mem in top_allocs:
            rel = (ts - t0) / 1e6
            lines.append(f"    {rel:>9.3f} {delta:>+12,.0f} {mem:>12,.0f}")
    if top_deallocs:
        lines.append("  Largest decreases (pool shrink):")
        for _, delta, ts, mem in top_deallocs:
            rel = (ts - t0) / 1e6
            lines.append(f"    {rel:>9.3f} {delta:>+12,.0f} {mem:>12,.0f}")
    lines.append("")

    # --- Suspect Signals ---
    lines.append("## Suspect Signals")
    lines.append("  [DEFINITE]=actionable as-is  [SIGNAL]=anomaly, root cause uncertain — cross-validate with other profiling dimensions")
    suspects_found = False

    # Growth trend
    n_records = len(reserved_records)
    if n_records > threshold("memory_record", "growth_min_records", 20):
        early = res_values[:n_records // 10]
        late = res_values[-(n_records // 10):]
        early_avg = sum(early) / len(early)
        late_avg = sum(late) / len(late)
        growth = late_avg - early_avg
        if growth > threshold("memory_record", "growth_mb", 100):
            lines.append(f"  - [SIGNAL] Reserved growth trend: early avg {early_avg:,.0f}MB → late avg {late_avg:,.0f}MB (+{growth:,.0f}MB)")
            lines.append(f"    Cross-validate: check operator_memory for unreleased tensor accumulation")
            suspects_found = True

    # High churn
    if jumps:
        large_jumps = [j for j in jumps if j[0] > threshold("memory_record", "churn_jump_mb", 50)]
        if len(large_jumps) > threshold("memory_record", "churn_count", 20):
            lines.append(f"  - [SIGNAL] High memory churn: {len(large_jumps)} jumps > 50MB")
            lines.append(f"    Cross-validate: check operator_memory for repeated same-size alloc → buffer reuse opportunity")
            suspects_found = True

    # Fragmentation growing over time
    if allocated_records and len(allocated_records) > 20:
        n_alloc = len(allocated_records)
        early_frag = [r[2] - r[1] for r in allocated_records[:n_alloc // 10]]
        late_frag = [r[2] - r[1] for r in allocated_records[-(n_alloc // 10):]]
        early_frag_avg = sum(early_frag) / len(early_frag)
        late_frag_avg = sum(late_frag) / len(late_frag)
        frag_growth = late_frag_avg - early_frag_avg
        if frag_growth > threshold("memory_record", "frag_growth_mb", 50):
            lines.append(f"  - [SIGNAL] Fragmentation growing: early gap avg {early_frag_avg:,.0f}MB → late {late_frag_avg:,.0f}MB (+{frag_growth:,.0f}MB)")
            suspects_found = True

    # OOM risk
    if max_res > threshold("memory_record", "oom_risk_mb", 60000):
        lines.append(f"  - [DEFINITE] Peak reserved {max_res:,.0f}MB — close to HBM capacity, OOM risk")
        suspects_found = True

    if not suspects_found:
        lines.append("  None — memory usage appears stable")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--buckets", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank, args.buckets, args.top_k)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
