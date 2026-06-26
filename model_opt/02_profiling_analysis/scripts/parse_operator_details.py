#!/usr/bin/env python3
"""Parse operator_details.csv — source code localization tool.

This file's unique value is the Call Stack column, which links operations back to
Python source code lines. Use it AFTER other scripts identify suspects — come here
to find WHERE in source code those operations are triggered.

Also uniquely provides per-operator Host Self Duration (host-side overhead),
which kernel_details.csv does not have.

Two modes:
- Default: lightweight host overhead overview (what's unique to this file)
- Filter (--filter): given operator name(s), show all Call Stacks + host/device
  breakdown for source code localization

Usage:
    python parse_operator_details.py <profiling_dir> [--rank N] [--top-k 15]
    python parse_operator_details.py <profiling_dir> --filter aclnnMatmul
    python parse_operator_details.py <profiling_dir> --filter empty_tensor aten::view
"""

import argparse
import heapq
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (find_ascend_profiler_output, stream_csv, safe_float,
                    clean_multiline_field, format_duration_ms)


def _parse_call_stack(raw: str) -> list:
    """Parse Call Stack field into list of frame strings."""
    frames = [f.strip() for f in raw.replace("\r\n", "\n").replace("\r", "\n").split(";")
              if f.strip()]
    return [f.replace("\n", " ").strip() for f in frames if f.replace("\n", "").strip()]


def parse_overview(profiling_dir: str, rank=None, top_k: int = 15) -> str:
    """Default mode: host overhead overview — the unique info this file provides."""
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "operator_details.csv"

    if not csv_path.exists():
        return f"[operator_details] File not found: {csv_path}"

    total_rows = 0
    total_host_us = 0.0
    total_device_us = 0.0

    # Aggregate by op name: host time focus
    op_agg = defaultdict(lambda: {"count": 0, "host_us": 0.0, "device_us": 0.0})

    for row in stream_csv(csv_path):
        total_rows += 1
        host_dur = safe_float(row.get("Host Self Duration(us)", 0))
        device_dur = safe_float(row.get("Device Self Duration(us)", 0))
        total_host_us += host_dur
        total_device_us += device_dur

        name = row.get("Name", "?")
        op_agg[name]["count"] += 1
        op_agg[name]["host_us"] += host_dur
        op_agg[name]["device_us"] += device_dur

    if total_rows == 0:
        return f"[operator_details] Empty file: {csv_path}"

    lines = []
    lines.append("# Operator Details — Host Overhead Overview")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Total rows: {total_rows:,}")
    lines.append(f"Total host time: {total_host_us/1000:.1f} ms")
    lines.append(f"Total device time: {total_device_us/1000:.1f} ms")
    lines.append(f"Host/Device ratio: {total_host_us/total_device_us:.1f}x" if total_device_us > 0 else "")
    lines.append("")

    # Top ops by HOST time (this is what's unique here)
    lines.append(f"## Top {top_k} Ops by Host Self Duration")
    lines.append("  (Focus on host-side overhead — device time analysis use kernel_details instead)")
    lines.append("")
    op_sorted_host = sorted(op_agg.items(), key=lambda x: -x[1]["host_us"])
    header = f"  {'Op Name':<35} {'Count':>8} {'Host(ms)':>10} {'Device(ms)':>10} {'H/D Ratio':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, info in op_sorted_host[:top_k]:
        ratio = info["host_us"] / info["device_us"] if info["device_us"] > 0 else float('inf')
        ratio_str = f"{ratio:.1f}x" if ratio < 10000 else "∞"
        lines.append(
            f"  {name:<35} {info['count']:>8} "
            f"{info['host_us']/1000:>10.1f} {info['device_us']/1000:>10.1f} "
            f"{ratio_str:>10}"
        )
    lines.append("")

    # Pure host ops (no device time at all)
    pure_host = [(name, info) for name, info in op_sorted_host
                 if info["device_us"] == 0 and info["host_us"] > 0]
    pure_host_total = sum(info["host_us"] for _, info in pure_host)
    pure_host_pct = pure_host_total / total_host_us * 100 if total_host_us > 0 else 0

    lines.append(f"## Pure Host Ops (no device kernel triggered)")
    lines.append(f"  Total pure-host time: {pure_host_total/1000:.1f} ms ({pure_host_pct:.1f}% of all host time)")
    lines.append(f"  These are metadata/framework operations with zero device work.")
    lines.append("")
    header2 = f"  {'Op Name':<35} {'Count':>8} {'Host(ms)':>10}"
    lines.append(header2)
    lines.append("  " + "-" * (len(header2) - 2))
    for name, info in pure_host[:top_k]:
        lines.append(f"  {name:<35} {info['count']:>8} {info['host_us']/1000:>10.1f}")
    lines.append("")

    # Suspect signals
    lines.append("## Suspect Signals")
    suspects_found = False

    if pure_host_pct > 50:
        lines.append(f"  - Pure host ops dominate: {pure_host_pct:.0f}% of host time has no device work")
        lines.append(f"    → Framework/dispatch overhead is the primary host bottleneck")
        suspects_found = True

    high_ratio = [(name, info) for name, info in op_sorted_host
                  if info["host_us"] > info["device_us"] * 10 and info["host_us"] > 5000
                  and info["device_us"] > 0]
    if high_ratio:
        lines.append(f"  - Ops with extreme host/device ratio (host > 10x device, host > 5ms):")
        for name, info in high_ratio[:5]:
            lines.append(f"    {name}: host={info['host_us']/1000:.1f}ms vs device={info['device_us']/1000:.1f}ms")
        lines.append(f"    → Use --filter to check Call Stack and locate source code")
        suspects_found = True

    if not suspects_found:
        lines.append("  None")
    lines.append("")

    lines.append("## Next Step")
    lines.append("  Use --filter <op_name> to drill into specific operators and see their Call Stacks.")
    lines.append("")

    return "\n".join(lines)


def parse_filtered(profiling_dir: str, filters: list, rank=None, top_k: int = 20) -> str:
    """Filter mode: show Call Stacks for specific ops — source code localization."""
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "operator_details.csv"

    if not csv_path.exists():
        return f"[operator_details] File not found: {csv_path}"

    filter_lower = [f.lower() for f in filters]
    matched = []

    for row in stream_csv(csv_path):
        name = row.get("Name", "")
        if any(f in name.lower() for f in filter_lower):
            matched.append(row)

    lines = []
    lines.append("# Operator Details — Source Code Localization")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Filter: {', '.join(filters)}")
    lines.append(f"Matched rows: {len(matched)}")
    lines.append("")

    if not matched:
        lines.append("No operators matched the filter.")
        return "\n".join(lines)

    # Summary
    total_host = sum(safe_float(r.get("Host Self Duration(us)", 0)) for r in matched)
    total_device = sum(safe_float(r.get("Device Self Duration(us)", 0)) for r in matched)
    lines.append("## Summary")
    lines.append(f"  Count: {len(matched)}")
    lines.append(f"  Total host: {total_host/1000:.2f} ms")
    lines.append(f"  Total device: {total_device/1000:.2f} ms")
    lines.append(f"  Avg host: {total_host/len(matched):.1f} us")
    lines.append(f"  Avg device: {total_device/len(matched):.1f} us")
    lines.append("")

    # Call Stack aggregation: group by unique stack, show frequency
    stack_groups = defaultdict(lambda: {"count": 0, "host_us": 0.0, "device_us": 0.0,
                                         "example_shapes": []})
    for r in matched:
        raw_stack = r.get("Call Stack", "")
        frames = _parse_call_stack(raw_stack)
        # Use first 3 frames as grouping key (enough to identify unique call sites)
        key = " | ".join(frames[:3]) if frames else "(no stack)"
        stack_groups[key]["count"] += 1
        stack_groups[key]["host_us"] += safe_float(r.get("Host Self Duration(us)", 0))
        stack_groups[key]["device_us"] += safe_float(r.get("Device Self Duration(us)", 0))
        shapes = clean_multiline_field(r.get("Input Shapes", ""))
        if shapes and len(stack_groups[key]["example_shapes"]) < 3:
            if shapes not in stack_groups[key]["example_shapes"]:
                stack_groups[key]["example_shapes"].append(shapes)

    sorted_stacks = sorted(stack_groups.items(), key=lambda x: -x[1]["host_us"])

    lines.append(f"## Call Sites (grouped by stack, top {min(top_k, len(sorted_stacks))} by host time)")
    lines.append("")

    for idx, (stack_key, info) in enumerate(sorted_stacks[:top_k], 1):
        lines.append(f"  [{idx}] count={info['count']}  host={info['host_us']/1000:.2f}ms  device={info['device_us']/1000:.2f}ms")
        lines.append(f"      stack: {stack_key[:200]}")
        if info["example_shapes"]:
            lines.append(f"      shapes: {', '.join(s[:50] for s in info['example_shapes'])}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Filter by operator name (substring match). "
                             "Shows Call Stacks grouped by call site for source localization.")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    if args.filter:
        result = parse_filtered(args.profiling_dir, args.filter, args.rank, args.top_k)
    else:
        result = parse_overview(args.profiling_dir, args.rank, args.top_k)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
