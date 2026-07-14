#!/usr/bin/env python3
"""Parse trace_view.json — timeline / dispatch-chain analysis.

trace_view.json (Chrome Trace format) is the only profiling file carrying the
temporal relationship between host dispatch and device execution. Its available
content is determined by collection switches, NOT by train/inference:

- Always present: device kernels (ph=X with args["Task Type"]) + HostToDevice flow.
- With NPU-only / stack-off: host side appears as CANN "AscendCL@..." events.
- With CPU activity + with_stack: host side appears as cpu_op + python_function
  lanes, and cpu_op args carry "Call stack" (source bridge).

This script probes which layers actually exist in the file, then extracts
structured timeline facts an agent can reason over (a human would read the
visual timeline instead). It streams the array so multi-GB files are safe.

Output: (1) compute-stream timeline, (2) stalls aggregated by kernel pair,
(3) dispatch latency, (4) online-compile classified warmup(A) vs per-step(B),
(5) prefetch/prealloc candidates (H2D/alloc ops) with condensed call stack.

Usage:
    python parse_trace_view.py <profiling_dir> [--rank N] [--top-k 15]
        [--gap-threshold 50] [--filter NAME ...] [--output out.txt]
"""

import argparse
import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import find_ascend_profiler_output, format_duration_ms


def stream_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024):
    """Incrementally yield objects from a top-level JSON array without loading
    the whole file. Uses an index pointer into the buffer (no per-object slicing)
    so multi-GB files stay near O(n)."""
    dec = json.JSONDecoder()
    ws = " \t\r\n,"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = ""
        idx = 0
        started = False
        eof = False
        while True:
            if not eof:
                piece = f.read(chunk_size)
                if piece:
                    if idx:                       # drop already-consumed prefix
                        buf = buf[idx:]
                        idx = 0
                    buf += piece
                else:
                    eof = True
            # consume as many objects as possible from current buffer
            while True:
                n = len(buf)
                while idx < n and buf[idx] in ws:
                    idx += 1
                if idx >= n:
                    break
                if buf[idx] == "[" and not started:
                    started = True
                    idx += 1
                    continue
                if buf[idx] == "]":
                    return
                try:
                    obj, end = dec.raw_decode(buf, idx)
                except ValueError:
                    break                         # incomplete tail; need more data
                idx = end
                yield obj
            if eof:
                return


def parse_ts_ns(val) -> int:
    """Parse a high-precision microsecond timestamp string to integer ns,
    avoiding float precision loss on ~1e15 values."""
    try:
        s = str(val)
        if "." in s:
            whole, frac = s.split(".", 1)
            frac = (frac + "000")[:3]
            return int(whole) * 1000 + int(frac)
        return int(s) * 1000
    except (ValueError, TypeError):
        return 0


def dur_ns(val) -> int:
    try:
        return int(round(float(val) * 1000))
    except (ValueError, TypeError):
        return 0


def is_device(e):
    return e.get("ph") == "X" and isinstance(e.get("args"), dict) and "Task Type" in e["args"]


def condense_stack(cs: str, max_frames: int = 6):
    """Condense a call stack: keep project frames (drop site-packages / std libs),
    cap the count. Returns list of frame strings. Falls back to top frames if all
    frames are library frames."""
    frames = [f.strip() for f in str(cs).replace("\r", "").split(";") if f.strip()]
    lib_markers = ("site-packages", "dist-packages", "/lib/python", "torch/nn/modules",
                   "torch/_ops", "autograd/profiler", "torch_npu/profiler")
    proj = [f for f in frames if not any(m in f for m in lib_markers)]
    picked = proj if proj else frames
    return picked[:max_frames]


def parse(csv_path: Path, top_k: int, gap_threshold_us: float) -> str:
    gap_threshold_ns = int(gap_threshold_us * 1000)

    caps = Counter()                       # detected event categories
    dev_count = 0
    # key -> [active_ns, min_ns, max_ns, compute_count, total_count]
    stream_stats = defaultdict(lambda: [0, None, None, 0, 0])
    last_end = {}                          # (pid,tid) -> (end_ns, name) for compute tasks
    noncompute_count = 0
    gap_buckets = {"<10us": 0, "10-50us": 0, "50-200us": 0, ">200us": 0}
    stall_agg = defaultdict(lambda: [0, 0, 0])  # (prev,cur) -> [count, sum_gap, max_gap]

    flow_ts = {}                           # HostToDevice id -> [s_ts, f_ts, dev_name]
    last_device_name = None                # nearest preceding device kernel (for flow labeling)
    disp_lat = []                          # dispatch latencies (ns), capped sample
    disp_stats = {"count": 0, "sum": 0, "max": 0}
    disp_top = []                          # heap: (lat_ns, dev_name)

    acl_compile = Counter()                # opCompile-like host events
    acl_compile_dur = 0
    compile_ts = []                        # timestamps of compile events (bounded)
    host_prefetch = []                     # heap: (dur_ns, name, callstack) for H2D/alloc ops
    has_callstack = False
    PREFETCH_KEYS = ("aten::to", "copy_", "aten::copy", "::empty", "empty_",
                     "aten::empty", "memcpy", "to_copy", "_to_copy", "pin_memory")

    # AI Core frequency, GC, stream sync, Node@launch tracking
    freq_values = []
    freq_max = 0
    gc_events = []                        # (dur_ns, name)
    sync_stream_count = 0
    node_launch_count = 0
    sync_stream_dur = 0

    for e in stream_json_array(csv_path):
        if not isinstance(e, dict):
            continue
        ph = e.get("ph")
        cat = e.get("cat")
        name = e.get("name", "")
        caps[cat if cat is not None else ("acl" if str(name).startswith(("AscendCL@", "Node@")) else "_other")] += 1

        if is_device(e):
            dev_count += 1
            tt = e["args"].get("Task Type", "")
            compute = any(k in tt for k in ("AI_CORE", "AI_VECTOR", "AICORE", "AIVEC", "MIX", "VECTOR"))
            ts = parse_ts_ns(e.get("ts"))
            dn = dur_ns(e.get("dur"))
            end = ts + dn
            key = (e.get("pid"), e.get("tid"))
            st = stream_stats[key]
            st[0] += dn
            st[1] = ts if st[1] is None else min(st[1], ts)
            st[2] = end if st[2] is None else max(st[2], end)
            st[4] += 1
            if not compute:
                noncompute_count += 1
            else:
                st[3] += 1
                last_device_name = name
                if key in last_end:
                    prev_end, prev_name = last_end[key]
                    gap = ts - prev_end
                    if gap > 0:
                        if gap < 10000:
                            gap_buckets["<10us"] += 1
                        elif gap < 50000:
                            gap_buckets["10-50us"] += 1
                        elif gap < 200000:
                            gap_buckets["50-200us"] += 1
                        else:
                            gap_buckets[">200us"] += 1
                        if gap >= gap_threshold_ns:
                            a = stall_agg[(prev_name, name)]
                            a[0] += 1
                            a[1] += gap
                            a[2] = max(a[2], gap)
                last_end[key] = (end, name)

        elif ph in ("s", "f") and cat == "HostToDevice":
            fid = e.get("id")
            slot = flow_ts.get(fid)
            if slot is None:
                slot = [None, None, None]
                flow_ts[fid] = slot
            if ph == "s":
                slot[0] = parse_ts_ns(e.get("ts"))
            else:
                slot[1] = parse_ts_ns(e.get("ts"))
                slot[2] = last_device_name
            if slot[0] is not None and slot[1] is not None:
                lat = slot[1] - slot[0]
                dev_name = slot[2]
                del flow_ts[fid]
                if lat >= 0:
                    disp_stats["count"] += 1
                    disp_stats["sum"] += lat
                    disp_stats["max"] = max(disp_stats["max"], lat)
                    if len(disp_lat) < 200000:
                        disp_lat.append(lat)
                    entry = (lat, dev_name)
                    if len(disp_top) < top_k:
                        heapq.heappush(disp_top, entry)
                    elif lat > disp_top[0][0]:
                        heapq.heapreplace(disp_top, entry)

        elif cat == "cpu_op":
            if str(name).startswith("ProfilerStep"):
                continue
            cs = e.get("args", {}).get("Call stack") if isinstance(e.get("args"), dict) else None
            if cs:
                has_callstack = True
            low = str(name).lower()
            if any(k in low for k in PREFETCH_KEYS):
                dn = dur_ns(e.get("dur"))
                entry = (dn, name, cs or "")
                if len(host_prefetch) < top_k:
                    heapq.heappush(host_prefetch, entry)
                elif dn > host_prefetch[0][0]:
                    heapq.heapreplace(host_prefetch, entry)
        elif ph == "X" and str(name).startswith("AscendCL@") and "ompile" in name:
            acl_compile[name] += 1
            acl_compile_dur += dur_ns(e.get("dur"))
            if len(compile_ts) < 500000:
                compile_ts.append(parse_ts_ns(e.get("ts")))

        elif ph == "C" and isinstance(e.get("args"), dict) and "MHz" in e["args"]:
            mhz = e["args"]["MHz"]
            freq_values.append(mhz)
            freq_max = max(freq_max, mhz)

        elif cat == "GC" and ph == "X":
            gc_events.append((dur_ns(e.get("dur")), name))

        elif ph == "X" and "SynchronizeStream" in str(name):
            sync_stream_count += 1
            sync_stream_dur += dur_ns(e.get("dur"))

        elif ph == "X" and str(name) == "Node@launch":
            node_launch_count += 1

    if dev_count == 0 and not caps:
        return f"[trace_view] Empty or unrecognized file: {csv_path}"

    L = []
    L.append("# Trace View Timeline Analysis")
    L.append(f"Source: {csv_path}")
    L.append("")

    # --- 0. Capability probe ---
    L.append("## 0. Detected Layers (driven by collection switches)")
    L.append(f"  device kernels (Task Type):   {dev_count:,}")
    L.append(f"  HostToDevice dispatch flow:   {disp_stats['count']:,}")
    L.append(f"  CANN AscendCL@ host events:   {caps.get('acl', 0):,}")
    L.append(f"  torch cpu_op events:          {caps.get('cpu_op', 0):,}")
    L.append(f"  python_function frames:       {caps.get('python_function', 0):,}")
    L.append(f"  fwdbwd links:                 {caps.get('fwdbwd', 0):,}")
    if caps.get("cpu_op", 0) == 0:
        L.append("  ⚠ No cpu_op/Call stack detected (CPU activity + with_stack not enabled) —")
        L.append("    Only host→device dispatch timeline available; source mapping requires re-collection with with_stack enabled.")
    elif not has_callstack:
        L.append("  ⚠ cpu_op present but Call stack empty (with_stack not enabled) — host op timing only, no source stack.")
    L.append("")

    # --- 1. Device timeline (compute streams only; fold non-compute) ---
    L.append("## 1. Device Timeline (compute streams)")
    compute_streams = {k: v for k, v in stream_stats.items() if v[3] > 0}
    other_streams = {k: v for k, v in stream_stats.items() if v[3] == 0}
    if compute_streams:
        L.append(f"  {'Stream (pid,tid)':<22} {'Span':>10} {'Active':>10} {'Busy%':>7} {'Kernels':>8}")
        for key, (active, mn, mx, cc, tc) in sorted(compute_streams.items(), key=lambda x: -x[1][0]):
            span = (mx - mn) if (mn is not None and mx is not None) else 0
            busy = active / span * 100 if span > 0 else 0
            L.append(f"  {str(key):<22} {format_duration_ms(span/1000):>10} "
                     f"{format_duration_ms(active/1000):>10} {busy:>6.1f}% {cc:>8,}")
    else:
        L.append("  (no compute stream detected)")
    if other_streams:
        tot = sum(v[4] for v in other_streams.values())
        L.append(f"  + {len(other_streams)} non-compute streams folded "
                 f"({tot:,} tasks: comm/sync/DMA — NOTIFY/SDMA/EVENT/AI_CPU)")
    L.append("  Gap distribution (between consecutive COMPUTE tasks per stream):")
    for b, c in gap_buckets.items():
        L.append(f"    {b:>9}: {c:,}")
    L.append("")

    # --- 2. Stall points aggregated by kernel pair ---
    L.append(f"## 2. Device Stalls (gap >= {gap_threshold_us:.0f}us, aggregated by kernel pair)")
    L.append("  Stalls at the same kernel pair aggregated; sorted by cumulative gap — repeated and large ones deserve optimization.")
    if stall_agg:
        pairs = sorted(stall_agg.items(), key=lambda x: -x[1][1])
        L.append(f"  {'Count':>6} {'SumGap':>10} {'AvgGap':>9} {'MaxGap':>9}  {'After -> Before kernel'}")
        for (prev, cur), (cnt, s, mx) in pairs[:top_k]:
            L.append(f"  {cnt:>6} {format_duration_ms(s/1000):>10} "
                     f"{format_duration_ms(s/cnt/1000):>9} {format_duration_ms(mx/1000):>9}  "
                     f"{str(prev)[:26]} -> {str(cur)[:26]}")
        L.append(f"  distinct stall pairs: {len(stall_agg):,}")
    else:
        L.append("  None above threshold.")
    L.append("")

    # --- 3. Dispatch latency ---
    L.append("## 3. Host→Device Dispatch Latency (HostToDevice flow)")
    if disp_stats["count"]:
        avg = disp_stats["sum"] / disp_stats["count"]
        L.append(f"  count={disp_stats['count']:,}  avg={avg/1000:.1f}us  max={disp_stats['max']/1000:.1f}us")
        if disp_lat:
            disp_lat.sort()
            p50 = disp_lat[len(disp_lat)//2]
            p90 = disp_lat[int(len(disp_lat)*0.9)]
            L.append(f"  p50={p50/1000:.1f}us  p90={p90/1000:.1f}us  (p90>>p50 indicates uneven dispatch backlog)")
        if disp_top:
            L.append("  Top dispatch latency (nearest device kernel name for locating dispatch point):")
            for lat, dev_name in sorted(disp_top, key=lambda x: -x[0]):
                L.append(f"    {lat/1000:>8.1f}us  ≈ {str(dev_name)[:40]}")
    else:
        L.append("  No HostToDevice flow found.")
    L.append("")

    # --- 4. Suspect Signals (diagnostic: compile classification + prefetch candidates) ---
    sec_num = 4
    has_suspects = bool(acl_compile) or bool(host_prefetch)
    if has_suspects or (caps.get("cpu_op", 0) and has_callstack):
        L.append(f"## {sec_num}. Suspect Signals")
        L.append("  [DEFINITE]=actionable as-is  [SIGNAL]=anomaly, cross-validate with other dimensions")
        L.append("")

    if acl_compile:
        L.append(f"  [DEFINITE] Online-Compile ({sum(acl_compile.values()):,} events, "
                 f"total {format_duration_ms(acl_compile_dur/1000)})")
        dev_min = min((v[1] for v in compute_streams.values() if v[1] is not None), default=None)
        dev_max = max((v[2] for v in compute_streams.values() if v[2] is not None), default=None)
        early_frac = None
        if compile_ts and dev_min is not None and dev_max is not None and dev_max > dev_min:
            span = dev_max - dev_min
            early = sum(1 for t in compile_ts if (t - dev_min) / span < 0.2)
            early_frac = early / len(compile_ts)
        if early_frac is not None and early_frac >= 0.8:
            L.append(f"    Distribution: {early_frac*100:.0f}% in first 20% of timeline → **Type A: warmup compilation**.")
            L.append("    → Collection issue: increase schedule skip_first to skip warmup — not a real bottleneck.")
        elif early_frac is not None:
            L.append(f"    Distribution: only {early_frac*100:.0f}% in first 20%, rest spans entire timeline → **Type B: per-step online compilation**.")
            L.append("    → Real bottleneck, not solvable by collection params: check if jit_compile is off, whether dynamic shapes cause re-compilation,"
                     "or switch to graph compilation; aclop per-op online compile path should be avoided.")
        else:
            L.append("    → Cannot determine distribution (missing device time baseline).")
        L.append("")

    if host_prefetch:
        L.append("  [SIGNAL] Prefetch / Prealloc Candidates (H2D copy & alloc ops)")
        L.append("    These ops can be optimized without replacing operators (prefetch / pre-allocate / buffer reuse); Call stack points to code location.")
        for dn, name, cs in sorted(host_prefetch, key=lambda x: -x[0]):
            L.append(f"    - {name}  host={dn/1000:.1f}us")
            for frame in condense_stack(cs):
                L.append(f"        {frame[:110]}")
        L.append("")
    elif caps.get("cpu_op", 0) and has_callstack and not acl_compile:
        L.append("  No significant H2D copy or repeated allocation ops found (aten::to / copy_ / empty etc.).")
        L.append("")

    # AI Core frequency degradation
    if freq_values and freq_max > 0:
        decrease_ratio = sum(freq_max - f for f in freq_values) / (freq_max * len(freq_values))
        if decrease_ratio >= 0.05:
            L.append(f"  [DEFINITE] AI Core frequency degradation: {decrease_ratio*100:.1f}% "
                     f"(max={freq_max}MHz, min={min(freq_values)}MHz)")
            L.append("    → Thermal/power throttling. Check: cooling, power limit, or reduce compute intensity.")
            L.append("")

    # GC events
    if gc_events:
        gc_total = sum(d for d, _ in gc_events)
        gc_sorted = sorted(gc_events, key=lambda x: -x[0])
        L.append(f"  [SIGNAL] Python GC: {len(gc_events)} events, total {format_duration_ms(gc_total/1000)}")
        L.append("    Top GC events:")
        for dur, name in gc_sorted[:3]:
            L.append(f"      {format_duration_ms(dur/1000)}  {name[:60]}")
        L.append("    → Cross-validate: if GC frequent, check for excessive small tensor allocations (operator_memory).")
        L.append("")

    # Stream synchronization
    if sync_stream_count > 0 and node_launch_count > 0:
        co_ratio = sync_stream_count / node_launch_count * 100
        if co_ratio > 10:
            L.append(f"  [DEFINITE] Frequent stream synchronization: {sync_stream_count} sync vs {node_launch_count} launches ({co_ratio:.1f}%)")
            L.append("    → Likely ASCEND_LAUNCH_BLOCKING=1 or explicit syncs. Check env vars and .item()/.numpy() usage.")
            L.append("")

    return "\n".join(L)


def parse_filtered(csv_path: Path, filters: list, top_k: int) -> str:
    fl = [f.lower() for f in filters]
    matched = []          # (dur_ns, name, callstack, input_dims)
    for e in stream_json_array(csv_path):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", ""))
        if e.get("cat") in ("cpu_op", "python_function") or is_device(e):
            if any(f in name.lower() for f in fl):
                a = e.get("args", {}) if isinstance(e.get("args"), dict) else {}
                matched.append((dur_ns(e.get("dur")), name,
                                a.get("Call stack", ""), a.get("Input Dims", "")))

    L = [f"# Trace View — Filtered", f"Source: {csv_path}",
         f"Filter: {', '.join(filters)}", f"Matched: {len(matched)}", ""]
    if not matched:
        L.append("No events matched (check filter, or file lacks cpu_op layer).")
        return "\n".join(L)
    matched.sort(key=lambda x: -x[0])
    for dn, name, cs, dims in matched[:top_k]:
        L.append(f"- {name}  dur={dn/1000:.1f}us  dims={str(dims)[:40]}")
        for frame in condense_stack(cs):
            L.append(f"    {frame[:110]}")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--gap-threshold", type=float, default=50.0,
                        help="Device gap threshold (us) to report as a stall point")
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Show timeline neighbors + call stack for matched op name (substring)")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    ascend_dir = find_ascend_profiler_output(args.profiling_dir, args.rank)
    csv_path = ascend_dir / "trace_view.json"
    if not csv_path.exists():
        result = f"[trace_view] File not found: {csv_path}"
    elif args.filter:
        result = parse_filtered(csv_path, args.filter, args.top_k)
    else:
        result = parse(csv_path, args.top_k, args.gap_threshold)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
