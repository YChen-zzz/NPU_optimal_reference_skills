#!/usr/bin/env python3
"""Parse api_statistic.csv — CANN runtime (ACL) API call duration statistics.

This file is produced at L1 (profiler_level=Level1). It aggregates wall-clock
durations of CANN runtime API calls at three levels:
- acl: AscendCL runtime API (dominated by *_Tiling — host-side tiling per aclnn op)
- communication: HCCL communication API (Notify_*, etc.)
- node: node-level launch (count == Node@launch in trace_view)

Unique value over other files: host dispatch overhead broken down to the
**CANN runtime API layer** (tiling / launch / comm notify), complementing
operator_details (op-name layer) and trace_view (per-instance timeline).
Answers "how much host time is tiling vs launch vs comm-notify".

Usage:
    python parse_api_statistic.py <profiling_dir> [--rank N] [--top-k 20]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, read_csv_all, safe_float, safe_int, format_duration_ms


def parse(profiling_dir: str, rank=None, top_k: int = 20) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "api_statistic.csv"
    rows = read_csv_all(csv_path)

    if not rows:
        return f"[api_statistic] File not found or empty: {csv_path}\n(api_statistic.csv is produced at L1; absent at L0.)"

    # aggregate by Level
    by_level = defaultdict(lambda: {"count": 0, "time_us": 0.0, "apis": []})
    total_time = 0.0
    for r in rows:
        lv = r.get("Level", "?").strip()
        cnt = safe_int(r.get("Count", 0))
        t = safe_float(r.get("Time(us)", 0))
        avg = safe_float(r.get("Avg(us)", 0))
        mx = safe_float(r.get("Max(us)", 0))
        name = r.get("API Name", "?")
        by_level[lv]["count"] += cnt
        by_level[lv]["time_us"] += t
        by_level[lv]["apis"].append((t, name, cnt, avg, mx))
        total_time += t

    lines = []
    lines.append("# API Statistic Summary (CANN runtime API overhead)")
    lines.append(f"Source: {csv_path}")
    lines.append(f"Total API time: {total_time/1000:.1f} ms")
    lines.append("")

    # Per-level summary
    lines.append("## By Level")
    for lv, info in sorted(by_level.items()):
        pct = info["time_us"] / total_time * 100 if total_time > 0 else 0
        lines.append(f"  {lv:<14} calls={info['count']:>7,}  time={info['time_us']/1000:>9.1f} ms ({pct:>5.1f}%)")
    lines.append("")

    # acl level: tiling breakdown (the host-side dispatch cost per op type)
    acl = by_level.get("acl")
    if acl:
        lines.append(f"## ACL API Top {min(top_k, len(acl['apis']))} (host runtime, by total time)")
        lines.append("  Tiling = host-side param computation per aclnn op (cacheable; dynamic shape re-tiling is waste).")
        header = f"  {'API Name':<32} {'Count':>7} {'Total(ms)':>10} {'Avg(us)':>9} {'Max(us)':>9}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for t, name, cnt, avg, mx in sorted(acl["apis"], key=lambda x: -x[0])[:top_k]:
            lines.append(f"  {name:<32} {cnt:>7} {t/1000:>10.2f} {avg:>9.2f} {mx:>9.2f}")
        # tiling subtotal
        tiling_us = sum(t for t, n, *_ in acl["apis"] if n.endswith("_Tiling"))
        if tiling_us > 0:
            lines.append("")
            lines.append(f"  Tiling subtotal: {tiling_us/1000:.1f} ms ({tiling_us/total_time*100:.1f}% of all API time)")
            if tiling_us / total_time > 0.3:
                lines.append("  → Tiling dominates ACL API time. If shapes are static/repeated, tiling is re-computed per call — cache it (graph compile / op cache).")
        lines.append("")

    # node launch (count should match trace_view Node@launch)
    node = by_level.get("node")
    if node:
        lines.append("## Node-level Launch")
        for t, name, cnt, avg, mx in node["apis"]:
            lines.append(f"  {name}: count={cnt:,}  total={t/1000:.1f}ms  avg={avg:.1f}us")
        lines.append(f"  (count should match trace_view Node@launch events.)")
        lines.append("")

    # communication API
    comm = by_level.get("communication")
    if comm:
        lines.append(f"## Communication API Top {min(top_k, len(comm['apis']))}")
        for t, name, cnt, avg, mx in sorted(comm["apis"], key=lambda x: -x[0])[:top_k]:
            lines.append(f"  {name:<24} count={cnt:>6,}  total={t/1000:>8.1f}ms  avg={avg:.2f}us")
        lines.append("")

    lines.append("## Suspect Signals")
    # Categorize acl APIs to find the dominant host-API cost driver
    if acl:
        cat_us = defaultdict(float)
        for t, name, *_ in acl["apis"]:
            nl = name.lower()
            if any(k in nl for k in ("free", "malloc", "mapphysical", "unmapmem", "alloc", "freephysical")):
                cat_us["memory mgmt"] += t
            elif "sync" in nl:
                cat_us["stream/device sync"] += t
            elif nl.endswith("_tiling"):
                cat_us["tiling"] += t
            elif "launch" in nl:
                cat_us["launch"] += t
            else:
                cat_us["other"] += t
        if cat_us:
            dom_cat, dom_us = max(cat_us.items(), key=lambda x: x[1])
            dom_pct = dom_us / total_time * 100 if total_time > 0 else 0
            for c, u in sorted(cat_us.items(), key=lambda x: -x[1]):
                lines.append(f"  {c:<16} {u/1000:>9.1f} ms ({u/total_time*100 if total_time>0 else 0:>5.1f}%)")
            if dom_pct > 20:
                hint = {
                    "memory mgmt": "→ Memory mgmt dominates ACL API time (Free/Malloc/Unmap). High churn = frequent alloc/free; cross-validate operator_memory repeated allocs → buffer reuse.",
                    "stream/device sync": "→ Sync dominates ACL API time (SynchronizeStream/Device). Explicit syncs or .item() forcing D→H; cross-validate operator_details sync category.",
                    "tiling": "→ Tiling dominates; cache tiling for static/repeated shapes (graph compile / op cache).",
                    "launch": "→ Launch overhead; reduce op count (fusion / graph compile).",
                }.get(dom_cat, "")
                if hint:
                    lines.append(f"  [SIGNAL] {hint}")
    else:
        lines.append("  None")
    lines.append("")
    lines.append("## Cross-validation")
    lines.append("  - vs operator_details: tiling/launch here is the CANN-API layer of the host time shown there (by op name).")
    lines.append("  - vs trace_view: node launch count matches Node@launch; tiling precedes each launch on the host thread.")
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
