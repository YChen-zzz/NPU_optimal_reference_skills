#!/usr/bin/env python3
"""Parse step_trace_time.csv — device utilization per step.

Shows Computing vs Free vs Communication time ratio per step.
Key for determining if the workload is Host-Bound, Compute-Bound, or Comm-Bound.

Usage:
    python parse_step_trace.py <profiling_dir> [--rank N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, read_csv_all, safe_float


def parse(profiling_dir: str, rank=None) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "step_trace_time.csv"
    rows = read_csv_all(csv_path)

    if not rows:
        return f"[step_trace_time] File not found: {csv_path}"

    lines = []
    lines.append("# Step Trace Time Summary")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Steps: {len(rows)}")
    lines.append("")

    computing_total = 0.0
    free_total = 0.0
    comm_total = 0.0

    step_data = []
    for row in rows:
        computing = safe_float(row.get("Computing", 0))
        free = safe_float(row.get("Free", 0))
        communication = safe_float(row.get("Communication", 0))
        preparing = safe_float(row.get("Preparing", 0))
        step_total = computing + free + communication

        computing_total += computing
        free_total += free
        comm_total += communication
        step_data.append((computing, free, communication, preparing, step_total))

    grand_total = computing_total + free_total + comm_total

    if grand_total > 0:
        util = computing_total / (computing_total + free_total) * 100 if (computing_total + free_total) > 0 else 0
        lines.append("## Overall")
        lines.append(f"  Device Utilization: {util:.1f}%  (Computing / (Computing + Free))")
        lines.append(f"  Computing: {computing_total/1000:.1f} ms ({computing_total/grand_total*100:.1f}%)")
        lines.append(f"  Free:      {free_total/1000:.1f} ms ({free_total/grand_total*100:.1f}%)")
        if comm_total > 0:
            lines.append(f"  Comm:      {comm_total/1000:.1f} ms ({comm_total/grand_total*100:.1f}%)")
        lines.append("")

        if util < threshold("step_trace", "severe_host_bound_util", 20):
            lines.append("  ** SEVERE Host-Bound: device idle >80% of time, bottleneck is on host side **")
        elif util < threshold("step_trace", "moderate_host_bound_util", 50):
            lines.append("  ** Moderate Host-Bound: device idle >50%, significant host-side overhead **")
        else:
            lines.append(f"  Bottleneck is on device side (utilization {util:.0f}%)")
            lines.append(f"  → Need kernel-level analysis to distinguish compute-bound vs memory-bound")
        lines.append("")

        optimizable = free_total / grand_total * 100 if grand_total > 0 else 0
        lines.append(f"  Theoretical limit (= Computing): {computing_total/1000:.1f} ms")
        lines.append(f"  Optimizable space: {optimizable:.1f}% ((Total - Computing) / Total)")
        if optimizable > threshold("step_trace", "large_optimizable_space", 30):
            lines.append(f"  → 可优化空间大。非计算开销（dispatch/分配/同步）显著，方案排序应按此上限而非实现难度")
        lines.append("")

    if len(step_data) > 1:
        lines.append("## Per-Step Breakdown")
        has_preparing = any(p > 0 for _, _, _, p, _ in step_data)
        if has_preparing:
            header = f"{'Step':>5} {'Computing(ms)':>13} {'Free(ms)':>10} {'Comm(ms)':>10} {'Preparing(ms)':>14} {'Util%':>7}"
        else:
            header = f"{'Step':>5} {'Computing(ms)':>13} {'Free(ms)':>10} {'Comm(ms)':>10} {'Util%':>7}"
        lines.append(header)
        lines.append("-" * len(header))
        for idx, (comp, free, comm, prep, total) in enumerate(step_data):
            u = comp / (comp + free) * 100 if (comp + free) > 0 else 0
            if has_preparing:
                lines.append(f"{idx:>5} {comp/1000:>13.1f} {free/1000:>10.1f} {comm/1000:>10.1f} {prep/1000:>14.1f} {u:>6.1f}%")
            else:
                lines.append(f"{idx:>5} {comp/1000:>13.1f} {free/1000:>10.1f} {comm/1000:>10.1f} {u:>6.1f}%")
        lines.append("")

        if has_preparing:
            avg_prep = sum(p for _, _, _, p, _ in step_data) / len(step_data)
            avg_comp = computing_total / len(step_data)
            lines.append("## Preparing Analysis")
            lines.append(f"  Avg Preparing per step: {avg_prep/1000:.1f} ms")
            lines.append(f"  Avg Computing per step: {avg_comp/1000:.1f} ms")
            if avg_prep > avg_comp:
                lines.append(f"  [SIGNAL] Preparing > Computing: may be real host bottleneck or profiler trace-writing overhead")
                lines.append(f"    Preparing includes Level1 profiler trace-writing cost — cannot determine from this alone.")
                lines.append(f"    Cross-validate: re-collect with L0 — if Preparing still high under L0 then real host gap, otherwise profiler injection.")
            lines.append("")

    # --- Suspect signals ---
    if len(step_data) > 1:
        step_utils = [c / (c + f) * 100 if (c + f) > 0 else 0 for c, f, _, _, _ in step_data]
        step_totals = [c + f + cm for c, f, cm, _, _ in step_data]
        avg_total = sum(step_totals) / len(step_totals)

        lines.append("## Suspect Signals")
        lines.append("  [DEFINITE]=actionable as-is  [SIGNAL]=anomaly, root cause uncertain — cross-validate with other profiling dimensions")
        suspects_found = False

        # Variance between steps
        if len(step_utils) > 1:
            util_min = min(step_utils)
            util_max = max(step_utils)
            if util_max - util_min > threshold("step_trace", "step_util_variance", 20):
                lines.append(f"  [SIGNAL] Step utilization variance: {util_min:.1f}% ~ {util_max:.1f}%")
                lines.append(f"    Some steps significantly less efficient — cross-validate: check for warmup/compilation/dynamic shape")
                suspects_found = True

        # Step duration outliers
        if len(step_totals) > 1:
            total_min = min(step_totals)
            total_max = max(step_totals)
            if total_max > total_min * threshold("step_trace", "step_duration_spread", 2.0):
                lines.append(f"  [SIGNAL] Step duration spread: {total_min/1000:.1f}ms ~ {total_max/1000:.1f}ms ({total_max/total_min:.1f}x)")
                lines.append(f"    Large variation — cross-validate: check trace_view whether compile events cluster in outlier steps")
                suspects_found = True

        if not suspects_found:
            lines.append("  None — steps appear consistent")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
