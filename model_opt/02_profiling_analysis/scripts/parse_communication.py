#!/usr/bin/env python3
"""Parse communication.json + communication_matrix.json — HCCL communication analysis.

These files are produced when profiler_level >= Level1 in multi-card scenarios.
They contain per-HCCL-op timing breakdown (Transit/Wait/Sync/Idle) and per-link
bandwidth info — the only source for "why is communication slow" analysis.

Usage:
    python parse_communication.py <profiling_dir> [--rank N] [--top-k 15]
        [--output out.txt]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_op_type(opname):
    """hcom_allGather__612_0_1@xxx -> allGather"""
    base = opname.split("@")[0].split("__")[0]
    return base.replace("hcom_", "")


def parse(comm_path: Path, matrix_path: Path, top_k: int) -> str:
    comm_data = json.loads(comm_path.read_text(encoding="utf-8"))

    # Also load matrix if available (per-link bandwidth)
    matrix_data = None
    if matrix_path.exists():
        matrix_data = json.loads(matrix_path.read_text(encoding="utf-8"))

    L = []
    L.append("# Communication Analysis")
    L.append(f"Source: {comm_path}")
    if matrix_data:
        L.append(f"Matrix: {matrix_path}")
    L.append("")

    # Aggregate across all steps
    total_elapse = 0.0
    total_transit = 0.0
    total_wait = 0.0
    total_sync = 0.0
    total_idle = 0.0
    by_type = defaultdict(lambda: {"count": 0, "elapse": 0, "transit": 0, "wait": 0, "sync": 0, "idle": 0})
    op_list = []  # (elapse, opname, op_type, time_info, bw_info, step)
    p2p_count = 0

    for step, step_data in comm_data.items():
        # Collective
        for opname, info in step_data.get("collective", {}).items():
            if opname == "Total Op Info":
                continue
            ti = info.get("Communication Time Info", {})
            elapse = safe_float(ti.get("Elapse Time(ms)", 0))
            transit = safe_float(ti.get("Transit Time(ms)", 0))
            wait = safe_float(ti.get("Wait Time(ms)", 0))
            sync = safe_float(ti.get("Synchronization Time(ms)", 0))
            idle = safe_float(ti.get("Idle Time(ms)", 0))
            op_type = extract_op_type(opname)
            by_type[op_type]["count"] += 1
            by_type[op_type]["elapse"] += elapse
            by_type[op_type]["transit"] += transit
            by_type[op_type]["wait"] += wait
            by_type[op_type]["sync"] += sync
            by_type[op_type]["idle"] += idle
            total_elapse += elapse
            total_transit += transit
            total_wait += wait
            total_sync += sync
            total_idle += idle
            op_list.append((elapse, opname, op_type, ti, info.get("Communication Bandwidth Info", {}), step))
        # P2P
        p2p_count += len(step_data.get("p2p", {}))

    if not op_list:
        return f"[communication] No collective ops found in {comm_path}"

    # --- 1. Summary ---
    L.append("## 1. Summary")
    L.append(f"  Total ops: {len(op_list)}  |  P2P ops: {p2p_count}")
    L.append(f"  Total Elapse: {total_elapse:.1f}ms  |  Transit: {total_transit:.1f}ms  |  Wait: {total_wait:.1f}ms  |  Sync: {total_sync:.1f}ms  |  Idle: {total_idle:.1f}ms")
    if total_elapse > 0:
        L.append(f"  Time breakdown: Transit {total_transit/total_elapse*100:.1f}%  Wait {total_wait/total_elapse*100:.1f}%  Sync {total_sync/total_elapse*100:.1f}%  Idle {total_idle/total_elapse*100:.1f}%")
    L.append("")

    # --- 2. By Op Type ---
    L.append("## 2. By Op Type")
    L.append(f"  {'Type':<15} {'Count':>6} {'Elapse(ms)':>11} {'Transit(ms)':>12} {'Wait(ms)':>10} {'Wait%':>6} {'Transit%':>9}")
    L.append(f"  {'-'*15} {'-'*6} {'-'*11} {'-'*12} {'-'*10} {'-'*6} {'-'*9}")
    for t, agg in sorted(by_type.items(), key=lambda x: -x[1]["elapse"]):
        wait_pct = agg["wait"] / agg["elapse"] * 100 if agg["elapse"] > 0 else 0
        transit_pct = agg["transit"] / agg["elapse"] * 100 if agg["elapse"] > 0 else 0
        L.append(f"  {t:<15} {agg['count']:>6} {agg['elapse']:>11.1f} {agg['transit']:>12.1f} {agg['wait']:>10.1f} {wait_pct:>5.1f}% {transit_pct:>8.1f}%")
    L.append("")

    # --- 3. Top Ops by Elapse ---
    L.append(f"## 3. Top {min(top_k, len(op_list))} Ops by Elapse Time")
    op_list.sort(key=lambda x: -x[0])
    L.append(f"  {'Op':<50} {'Type':<12} {'Elapse(ms)':>10} {'Wait%':>6}")
    L.append(f"  {'-'*50} {'-'*12} {'-'*10} {'-'*6}")
    for elapse, opname, op_type, ti, bw, step in op_list[:top_k]:
        wait_pct = safe_float(ti.get("Wait Time(ms)", 0)) / elapse * 100 if elapse > 0 else 0
        L.append(f"  {opname[:50]:<50} {op_type:<12} {elapse:>10.2f} {wait_pct:>5.1f}%")
    L.append("")

    # --- 4. Bandwidth Analysis (from matrix) ---
    if matrix_data:
        L.append("## 4. Per-Link Bandwidth (from communication_matrix)")
        link_bw = []  # (bw, size, time, transport, link, opname)
        for step, step_data in matrix_data.items():
            for opname, links in step_data.get("collective", {}).items():
                for link, lv in links.items():
                    bw = safe_float(lv.get("Bandwidth(GB/s)", 0))
                    size = safe_float(lv.get("Transit Size(MB)", 0))
                    ttime = safe_float(lv.get("Transit Time(ms)", 0))
                    transport = lv.get("Transport Type", "?")
                    if size > 0:
                        link_bw.append((bw, size, ttime, transport, link, opname[:40]))

        if link_bw:
            link_bw.sort(key=lambda x: -x[0])
            bws = [x[0] for x in link_bw]
            avg_bw = sum(bws) / len(bws)
            min_bw = min(bws)
            max_bw = max(bws)
            L.append(f"  Links with data: {len(link_bw)}  |  Bandwidth: min={min_bw:.1f} avg={avg_bw:.1f} max={max_bw:.1f} GB/s")
            L.append("")
            L.append(f"  Top {min(top_k, 5)} highest bandwidth links:")
            for bw, size, ttime, transport, link, opname in link_bw[:5]:
                L.append(f"    {bw:>7.1f} GB/s  {size:>10.1f}MB  {transport:<6} {link:<6} {opname}")
            L.append("")
            L.append(f"  Top {min(top_k, 5)} lowest bandwidth links (potential bottleneck):")
            for bw, size, ttime, transport, link, opname in reversed(link_bw[-5:]):
                L.append(f"    {bw:>7.1f} GB/s  {size:>10.1f}MB  {transport:<6} {link:<6} {opname}")
        L.append("")

    # --- 5. Suspect Signals ---
    L.append("## 5. Suspect Signals")
    L.append("  [DEFINITE]=actionable as-is  [SIGNAL]=anomaly, cross-validate with other dimensions")
    suspects = False

    # Wait ratio high
    if total_elapse > 0 and total_wait / total_elapse > threshold("communication", "wait_dominant_ratio", 0.8):
        L.append(f"  [DEFINITE] Wait time dominates: {total_wait/total_elapse*100:.0f}% of communication time is waiting, not transmitting.")
        L.append("    → Communication is synchronization-bound, not bandwidth-bound. Check: communication-computation overlap, "
                 "rank straggler (some ranks slow → others wait), or excessive sync points.")
        suspects = True

    # Per-type wait ratio
    for t, agg in sorted(by_type.items(), key=lambda x: -x[1]["elapse"]):
        if agg["elapse"] > 0 and agg["wait"] / agg["elapse"] > threshold("communication", "per_type_wait_ratio", 0.9) and agg["count"] > threshold("communication", "per_type_min_count", 10):
            L.append(f"  [SIGNAL] {t}: {agg['count']} ops, {agg['wait']/agg['elapse']*100:.0f}% wait time — "
                     f"cross-validate: trace_view for compute-comm overlap, check straggler ranks")
            suspects = True

    # Low bandwidth links
    if matrix_data and link_bw:
        low_bw = [x for x in link_bw if x[0] < avg_bw * threshold("communication", "low_bw_ratio", 0.3) and x[1] > threshold("communication", "low_bw_min_size_mb", 1)]
        if low_bw:
            L.append(f"  [SIGNAL] {len(low_bw)} links with bandwidth < 30% of average ({avg_bw:.1f} GB/s) — "
                     f"potential bottleneck links")
            suspects = True

    # Small packet ratio (from Size Distribution)
    for step, step_data in comm_data.items():
        total_info = step_data.get("collective", {}).get("Total Op Info", {})
        bw_info = total_info.get("Communication Bandwidth Info", {})
        for link_type, bw in bw_info.items():
            dist = bw.get("Size Distribution", {})
            if dist:
                total_count = sum(v[0] for v in dist.values() if isinstance(v, list) and len(v) >= 1)
                small_count = sum(v[0] for k, v in dist.items()
                                  if isinstance(v, list) and len(v) >= 1 and safe_float(k) < 1.0)
                if total_count > 0 and small_count / total_count > threshold("communication", "small_packet_ratio", 0.3):
                    L.append(f"  [SIGNAL] {link_type}: {small_count/total_count*100:.0f}% small packets (<1MB) — "
                             f"cross-validate: consider batching to reduce small-packet overhead")
                    suspects = True
        break

    if not suspects:
        L.append("  None")
    L.append("")

    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    ascend_dir = find_ascend_profiler_output(args.profiling_dir, args.rank)
    comm_path = ascend_dir / "communication.json"
    matrix_path = ascend_dir / "communication_matrix.json"

    if not comm_path.exists():
        result = (f"[communication] File not found: {comm_path}\n"
                  "Communication data requires profiler_level >= Level1 in multi-card scenarios.")
    else:
        result = parse(comm_path, matrix_path, args.top_k)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
