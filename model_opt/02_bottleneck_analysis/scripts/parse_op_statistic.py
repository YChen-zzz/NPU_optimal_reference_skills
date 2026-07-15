#!/usr/bin/env python3
"""Parse op_statistic.csv — global operator time distribution.

This file is always small (~100 rows) and gives the highest-level view of
where device time is spent. Key for identifying bottleneck TYPE.

Usage:
    python parse_op_statistic.py <profiling_dir> [--rank N] [--top-k 30]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import find_ascend_profiler_output, read_csv_all, safe_float, safe_int


def parse(profiling_dir: str, rank=None, top_k: int = 30) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "op_statistic.csv"
    rows = read_csv_all(csv_path)

    if not rows:
        return f"[op_statistic] File not found: {csv_path}"

    for row in rows:
        row["_total_us"] = safe_float(row.get("Total Time(us)", 0))
        row["_count"] = safe_int(row.get("Count", 0))
        row["_avg_us"] = row["_total_us"] / row["_count"] if row["_count"] > 0 else 0

    rows_sorted = sorted(rows, key=lambda r: r["_total_us"], reverse=True)
    total_us = sum(r["_total_us"] for r in rows_sorted)
    total_count = sum(r["_count"] for r in rows_sorted)

    lines = []
    lines.append(f"# Op Statistic Summary")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Total op types: {len(rows_sorted)}")
    lines.append(f"Total device time: {total_us/1000:.1f} ms")
    lines.append(f"Total kernel count: {total_count}")
    lines.append("")

    header = f"{'#':>3} {'OP Type':<32} {'Count':>7} {'Total(ms)':>10} {'Avg(us)':>9} {'Ratio%':>7} {'Cumul%':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    cumul = 0.0
    for idx, row in enumerate(rows_sorted[:top_k]):
        ratio = (row["_total_us"] / total_us * 100) if total_us > 0 else 0
        cumul += ratio
        lines.append(
            f"{idx+1:>3} {row.get('OP Type', '?'):<32} "
            f"{row['_count']:>7} {row['_total_us']/1000:>10.1f} "
            f"{row['_avg_us']:>9.1f} {ratio:>6.1f}% {cumul:>6.1f}%"
        )

    lines.append("")

    # Statistical analysis
    lines.append("## Suspect Signals")
    lines.append("  [DEFINITE]=actionable as-is  [SIGNAL]=anomaly, root cause uncertain — cross-validate with other profiling dimensions")

    # 1. Concentration: how focused is the bottleneck
    top3_us = sum(r["_total_us"] for r in rows_sorted[:3])
    top3_ratio = top3_us / total_us * 100 if total_us > 0 else 0
    lines.append(f"- [DEFINITE] Top-3 concentration: {top3_ratio:.1f}% of total device time")
    if top3_ratio > 80:
        lines.append(f"  → Bottleneck highly concentrated — optimizing top ops has strong leverage")

    # 2. Data movement overhead (non-compute ops)
    move_keywords = ("Transpose", "Cast", "Copy", "Contiguous", "Reshape", "MemSet", "Format")
    move_us = sum(r["_total_us"] for r in rows_sorted
                  if any(kw.lower() in r.get("OP Type", "").lower() for kw in move_keywords))
    move_ratio = move_us / total_us * 100 if total_us > 0 else 0
    if move_ratio > 3:
        move_ops = [r.get("OP Type", "") for r in rows_sorted
                    if any(kw.lower() in r.get("OP Type", "").lower() for kw in move_keywords)
                    and r["_total_us"] > 0]
        lines.append(f"- [SIGNAL] Data movement overhead: {move_ratio:.1f}% ({move_us/1000:.1f}ms)")
        lines.append(f"  ops: {', '.join(move_ops[:8])}")
        lines.append(f"  → Layout/format conversion cost. Cross-validate: kernel_details for mte dominance, operator_details for source location")

    # 3. High-count low-avg ops (fragmentation signal)
    if total_count > 0:
        avg_count_per_type = total_count / len(rows_sorted)
        fragmented = [(r.get("OP Type", ""), r["_count"], r["_avg_us"])
                      for r in rows_sorted
                      if r["_count"] > avg_count_per_type * 3 and r["_avg_us"] < 10]
        if fragmented:
            lines.append(f"- [SIGNAL] High-count low-duration ops (fragmentation signal):")
            for name, count, avg in fragmented[:5]:
                lines.append(f"  {name}: count={count}, avg={avg:.1f}us — cross-validate: potential for fusion/batching")

    # 4. Low-count high-avg ops (heavy single ops)
    heavy = [(r.get("OP Type", ""), r["_count"], r["_avg_us"], r["_total_us"])
             for r in rows_sorted
             if r["_count"] <= 10 and r["_avg_us"] > 100 and r["_total_us"] / total_us > 0.01]
    if heavy:
        lines.append(f"- [SIGNAL] Heavy single-invocation ops:")
        for name, count, avg, total in heavy[:5]:
            lines.append(f"  {name}: count={count}, avg={avg:.0f}us — large shape or expensive kernel")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank, args.top_k)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
