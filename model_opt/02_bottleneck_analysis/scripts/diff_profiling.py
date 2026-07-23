#!/usr/bin/env python3
"""Diff two profiling runs — compare op times and memory peaks.

Usage:
    python diff_profiling.py <before_dir> <after_dir> [--rank N] [--top-k 20]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (find_ascend_profiler_output, read_csv_all, stream_csv,
                    safe_float, safe_int)


def _load_op_stats(ascend_dir):
    rows = read_csv_all(ascend_dir / "op_statistic.csv")
    result = {}
    for row in rows:
        op_type = row.get("OP Type", "?")
        result[op_type] = {
            "count": safe_int(row.get("Count", 0)),
            "total_us": safe_float(row.get("Total Time(us)", 0)),
        }
    return result


def _get_memory_peak(ascend_dir):
    """Peak memory. Prefer Total Active (true live set) over Reserved
    (Reserved includes pool retention and can stay high after optimization
    reduces actual usage). Falls back to Reserved when Active absent (L0)."""
    csv_path = ascend_dir / "memory_record.csv"
    peak_active = 0.0
    peak_reserved = 0.0
    has_active = False
    if not csv_path.exists():
        return 0.0, 0.0, False
    for row in stream_csv(csv_path):
        reserved = safe_float(row.get("Total Reserved(MB)", 0))
        active = safe_float(row.get("Total Active(MB)", 0))
        peak_reserved = max(peak_reserved, reserved)
        if active > 0:
            has_active = True
            peak_active = max(peak_active, active)
    return (peak_active if has_active else peak_reserved), peak_reserved, has_active


def _get_step_info(ascend_dir):
    """Return (step_count, computing_us, free_us, comm_us, has_operator_details)
    for normalization + L0/L1 口径 guard + utilization diff."""
    step_count = 0
    computing = free = comm = 0.0
    st = ascend_dir / "step_trace_time.csv"
    if st.exists():
        for row in read_csv_all(st):
            step_count += 1
            computing += safe_float(row.get("Computing", 0))
            free += safe_float(row.get("Free", 0))
            comm += safe_float(row.get("Communication", 0))
    has_od = (ascend_dir / "operator_details.csv").exists()  # L1+ has it, L0 doesn't
    return step_count, computing, free, comm, has_od


def parse(dir_before: str, dir_after: str, rank=None, top_k: int = 20) -> str:
    ascend_before = find_ascend_profiler_output(dir_before, rank)
    ascend_after = find_ascend_profiler_output(dir_after, rank)

    lines = []
    lines.append("# Profiling Diff")
    lines.append(f"Before: {dir_before}")
    lines.append(f"After:  {dir_after}")
    lines.append("")

    # Op time diff
    ops_before = _load_op_stats(ascend_before)
    ops_after = _load_op_stats(ascend_after)

    all_ops = set(ops_before.keys()) | set(ops_after.keys())
    op_diffs = []
    for op in all_ops:
        before_us = ops_before.get(op, {}).get("total_us", 0)
        after_us = ops_after.get(op, {}).get("total_us", 0)
        delta = after_us - before_us
        op_diffs.append((op, before_us, after_us, delta))

    op_diffs.sort(key=lambda x: abs(x[3]), reverse=True)

    total_before = sum(v["total_us"] for v in ops_before.values())
    total_after = sum(v["total_us"] for v in ops_after.values())
    total_delta = total_after - total_before
    pct = (total_delta / total_before * 100) if total_before > 0 else 0

    lines.append(f"## Op Time Diff (Top {top_k})")
    lines.append(f"Total: {total_before/1000:.1f}ms → {total_after/1000:.1f}ms ({total_delta/1000:+.1f}ms, {pct:+.1f}%)")
    lines.append("")
    header = f"  {'OP Type':<30} {'Before(ms)':>10} {'After(ms)':>10} {'Delta(ms)':>10} {'Change':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for op, before_us, after_us, delta in op_diffs[:top_k]:
        if before_us > 0:
            change = f"{delta/before_us*100:+.1f}%"
        elif delta > 0:
            change = "NEW"
        else:
            change = "GONE"
        lines.append(
            f"  {op:<30} {before_us/1000:>10.1f} {after_us/1000:>10.1f} "
            f"{delta/1000:>+10.1f} {change:>8}"
        )

    # Highlight disappeared and new ops
    disappeared = [(op, b) for op, b, a, d in op_diffs if a == 0 and b > 0]
    new_ops = [(op, a) for op, b, a, d in op_diffs if b == 0 and a > 0]
    if disappeared:
        lines.append("")
        lines.append("  Eliminated ops: " + ", ".join(f"{op}({b/1000:.1f}ms)" for op, b in disappeared[:10]))
    if new_ops:
        lines.append("  New ops: " + ", ".join(f"{op}({a/1000:.1f}ms)" for op, a in new_ops[:10]))
    lines.append("")

    # Memory peak diff (D6: Active preferred over Reserved)
    peak_before, reserved_before, has_active_b = _get_memory_peak(ascend_before)
    peak_after, reserved_after, has_active_a = _get_memory_peak(ascend_after)
    delta_peak = peak_after - peak_before
    lines.append("## Memory Peak")
    metric = "Active" if (has_active_b or has_active_a) else "Reserved"
    lines.append(f"  Metric: {metric} (Active = true live set; Reserved inflated by pool retention)")
    lines.append(f"  Before: {peak_before:,.0f} MB  (Reserved {reserved_before:,.0f})")
    lines.append(f"  After:  {peak_after:,.0f} MB  (Reserved {reserved_after:,.0f})")
    lines.append(f"  Delta:  {delta_peak:+,.0f} MB")
    lines.append("")

    # Normalization + L0/L1 口径 guard (D7)
    sb, comp_b, free_b, comm_b, odb = _get_step_info(ascend_before)
    sa, comp_a, free_a, comm_a, oda = _get_step_info(ascend_after)
    lines.append("## Comparability Guard")
    warn = False
    if odb != oda:
        lines.append(f"  ⚠ L0/L1 口径不一致 (operator_details.csv: before={odb} after={oda}) — L1 含 profiler 注入开销，对比可能失真。建议同口径重采。")
        warn = True
    if sb > 0 and sa > 0 and sb != sa:
        lines.append(f"  ⚠ Step 数不同 (before={sb} after={sa}) — 总量不可直接比，下方已按 per-step 归一化。")
        warn = True
    if not warn:
        lines.append("  OK — 同口径、同 step 数。")
    lines.append("")

    # Utilization diff (D1) — the key inference metric "did host-bound improve"
    if sb > 0 and sa > 0:
        util_b = comp_b / (comp_b + free_b) * 100 if (comp_b + free_b) > 0 else 0
        util_a = comp_a / (comp_a + free_a) * 100 if (comp_a + free_a) > 0 else 0
        norm = sb  # per-step normalization factor
        lines.append("## Utilization / Free Diff (per-step)")
        lines.append(f"  Utilization: {util_b:.1f}% → {util_a:.1f}% ({util_a-util_b:+.1f}%)")
        lines.append(f"  Free/step:   {free_b/norm/1000:.1f}ms → {free_a/sa/1000:.1f}ms")
        lines.append(f"  Computing/step: {comp_b/norm/1000:.1f}ms → {comp_a/sa/1000:.1f}ms")
        if comm_b + comm_a > 0:
            lines.append(f"  Comm/step:   {comm_b/norm/1000:.1f}ms → {comm_a/sa/1000:.1f}ms")
        if util_a <= util_b and (free_b - free_a) < 0:
            lines.append("  ⚠ Utilization did not improve despite op time change — possible bottleneck shift; cross-validate with operator_details / trace_view diff.")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.before, args.after, args.rank, args.top_k)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
