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
    csv_path = ascend_dir / "memory_record.csv"
    peak = 0.0
    for row in stream_csv(csv_path):
        mem = safe_float(row.get("Total Reserved(MB)", 0))
        peak = max(peak, mem)
    return peak


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

    # Memory peak diff
    lines.append("## Memory Peak")
    peak_before = _get_memory_peak(ascend_before)
    peak_after = _get_memory_peak(ascend_after)
    delta_peak = peak_after - peak_before
    lines.append(f"  Before: {peak_before:,.0f} MB")
    lines.append(f"  After:  {peak_after:,.0f} MB")
    lines.append(f"  Delta:  {delta_peak:+,.0f} MB")
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
