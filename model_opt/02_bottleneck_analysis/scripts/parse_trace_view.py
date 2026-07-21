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
import bisect
import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, format_duration_ms


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


def condense_stack(cs: str):
    """Condense a call stack: keep project frames (drop site-packages / std libs),
    cap the count. Returns list of frame strings. Falls back to top frames if all
    frames are library frames."""
    frames = [f.strip() for f in str(cs).replace("\r", "").split(";") if f.strip()]
    lib_markers = tuple(threshold("trace_view", "stack_lib_markers",
                   ["site-packages", "dist-packages", "/lib/python", "torch/nn/modules", "torch/_ops", "autograd/profiler", "torch_npu/profiler"]))
    proj = [f for f in frames if not any(m in f for m in lib_markers)]
    picked = proj if proj else frames
    return picked[:threshold("trace_view", "stack_max_frames", 6)]


def _detect_h2d_runs(launch_by_cid, dev_by_cid, gap_thr_ns, min_run_len):
    """Detect host2device-bound regions: consecutive launch→device pairs on the same
    device stream whose gap (device_start - launch_start) is below threshold.

    A small/negative gap means the device starts right when (or before, due to
    host/device clock skew) the host launches it — the device queue was empty and
    starving for host dispatch. A run of such ops = a host2device-bound region.

    Returns a list of regions (each a list of pair dicts), sorted by device idle
    time (span - dev_active) descending."""
    pairs = []
    for cid, (lts, ldur, item) in launch_by_cid.items():
        dl = dev_by_cid.get(cid)
        if not dl:
            continue
        # earliest device kernel start = the op's device execution start
        dk = min(dl, key=lambda x: x[0])
        pairs.append({
            "cid": cid, "item": item, "lts": lts, "ldur": ldur,
            "dts": dk[0], "ddur": dk[1], "stream": dk[3], "gap": dk[0] - lts,
        })
    by_stream = defaultdict(list)
    for p in pairs:
        by_stream[p["stream"]].append(p)
    runs = []
    for ps in by_stream.values():
        ps.sort(key=lambda x: x["dts"])
        cur = []
        for p in ps:
            if p["gap"] < gap_thr_ns:
                cur.append(p)
            else:
                if len(cur) >= min_run_len:
                    runs.append(cur)
                cur = []
        if len(cur) >= min_run_len:
            runs.append(cur)
    runs.sort(key=lambda r: ((r[-1]["dts"] + r[-1]["ddur"]) - r[0]["lts"]
                             - sum(x["ddur"] for x in r)), reverse=True)
    return runs


def _chain_summary(run):
    """Compact op-chain string for a run, deduping adjacent identical op types."""
    chain = []
    for p in run:
        short = p["item"].split("_", 1)[0] if p["item"] else "?"
        if not chain or chain[-1][0] != short:
            chain.append([short, 1])
        else:
            chain[-1][1] += 1
    return " → ".join(f"{n}×{c}" if c > 1 else n for n, c in chain)


def parse(csv_path: Path, top_k: int, gap_threshold_us: float) -> str:
    gap_threshold_ns = int(gap_threshold_us * 1000)

    caps = Counter()                       # detected event categories
    dev_count = 0
    # key -> [active_ns, min_ns, max_ns, compute_count, total_count]
    stream_stats = defaultdict(lambda: [0, None, None, 0, 0])
    last_end = {}                          # (pid,tid) -> (end_ns, name) for compute tasks
    noncompute_count = 0
    _gb = threshold("trace_view", "gap_buckets_us", [10, 50, 200])
    gap_buckets = {f"<{_gb[0]}us": 0, f"{_gb[0]}-{_gb[1]}us": 0, f"{_gb[1]}-{_gb[2]}us": 0, f">{_gb[2]}us": 0}
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
    PREFETCH_KEYS = tuple(threshold("trace_view", "prefetch_keywords",
                     ["aten::to", "copy_", "aten::copy", "::empty", "empty_", "aten::empty", "memcpy", "to_copy", "_to_copy", "pin_memory"]))

    # AI Core frequency, GC, stream sync, Node@launch tracking
    freq_values = []
    freq_max = 0
    gc_events = []                        # (dur_ns, name)
    sync_stream_count = 0
    node_launch_count = 0
    sync_stream_dur = 0

    # Host2Device-bound detection: launch↔device pairs via connection_id, plus the
    # async_npu(torch_to_npu) flow and cpu_op timeline needed to resolve call stacks.
    launch_by_cid = {}                    # cid -> (lts_ns, ldur_ns, item_id)
    dev_by_cid = defaultdict(list)        # cid -> [(dts_ns, ddur_ns, name, stream, task_type)]
    t2n_s = {}                            # async_npu flow id (== device_start_ns) -> torch host ts ns
    cpuop_timeline = []                   # (start_ns, end_ns, name, callstack)

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
            compute = any(k in tt for k in threshold("trace_view", "compute_task_types", ["AI_CORE", "AI_VECTOR", "AICORE", "AIVEC", "MIX", "VECTOR"]))
            ts = parse_ts_ns(e.get("ts"))
            dn = dur_ns(e.get("dur"))
            end = ts + dn
            key = (e.get("pid"), e.get("tid"))
            st = stream_stats[key]
            st[0] += dn
            st[1] = ts if st[1] is None else min(st[1], ts)
            st[2] = end if st[2] is None else max(st[2], end)
            st[4] += 1
            cid_dev = e["args"].get("connection_id")
            if cid_dev is not None:
                dev_by_cid[cid_dev].append((ts, dn, name, key, tt))
            if not compute:
                noncompute_count += 1
            else:
                st[3] += 1
                last_device_name = name
                if key in last_end:
                    prev_end, prev_name = last_end[key]
                    gap = ts - prev_end
                    if gap > 0:
                        if gap < _gb[0] * 1000:
                            gap_buckets["<10us"] += 1
                        elif gap < _gb[1] * 1000:
                            gap_buckets["10-50us"] += 1
                        elif gap < _gb[2] * 1000:
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
                    if len(disp_lat) < threshold("trace_view", "disp_lat_sample_cap", 200000):
                        disp_lat.append(lat)
                    entry = (lat, dev_name)
                    if len(disp_top) < top_k:
                        heapq.heappush(disp_top, entry)
                    elif lat > disp_top[0][0]:
                        heapq.heapreplace(disp_top, entry)

        elif cat == "async_npu" and ph == "s":
            # torch_to_npu flow: id == device kernel start (ns); s.ts == torch host
            # enqueue time. Used to link a device kernel back to its host cpu_op.
            t2n_s[e.get("id")] = parse_ts_ns(e.get("ts"))

        elif cat == "cpu_op":
            if str(name).startswith("ProfilerStep"):
                continue
            cs = e.get("args", {}).get("Call stack") if isinstance(e.get("args"), dict) else None
            if cs:
                has_callstack = True
                cts = parse_ts_ns(e.get("ts"))
                cpuop_timeline.append((cts, cts + dur_ns(e.get("dur")), str(name), cs))
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
            if len(compile_ts) < threshold("trace_view", "compile_ts_cap", 500000):
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
            lcid = e["args"].get("connection_id") if isinstance(e.get("args"), dict) else None
            if lcid is not None and lcid not in launch_by_cid:
                launch_by_cid[lcid] = (parse_ts_ns(e.get("ts")),
                                       dur_ns(e.get("dur")),
                                       e["args"].get("item_id", ""))

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
        if dev_count > 0:
            # Use p50 for estimation (avg is skewed by queue stall outliers)
            est_per_op_us = (p50 / 1000) if disp_lat else (avg / 1000)
            est_source = "p50" if disp_lat else "avg"
            disp_total_us = dev_count * est_per_op_us
            compute_active_us = sum(v[0] for v in compute_streams.values()) / 1000
            L.append(f"  Estimated dispatch total: {disp_total_us/1000:.1f} ms (dev_count × {est_source}_latency)")
            if compute_active_us > 0:
                disp_kernel_ratio = disp_total_us / compute_active_us * 100
                L.append(f"  Dispatch / kernel-active ratio: {disp_kernel_ratio:.1f}%")
                if disp_kernel_ratio > threshold("trace_view", "dispatch_kernel_ratio", 50):
                    L.append(f"  → Dispatch overhead est. > {threshold('trace_view', 'dispatch_kernel_ratio', 50)}% of kernel-active time")
                    L.append(f"    Under async queue both overlap; actual impact depends on gap distribution:")
                    L.append(f"    High ratio of gap > 50us: dispatch not fully overlapped — reducing op count helps")
                    L.append(f"    High ratio of gap < 10us: dispatch fully overlapped — Free comes from serial dependency")
    else:
        L.append("  No HostToDevice flow found.")
    L.append("")

    # --- 4. Host2Device Bound Regions (temporal runs of dispatch-starved ops) ---
    L.append("## 4. Host2Device Bound Regions")
    runs = []
    if launch_by_cid and dev_by_cid:
        h2d_thr_ns = int(threshold("trace_view", "h2d_gap_threshold_us", 50) * 1000)
        h2d_min_run = threshold("trace_view", "h2d_min_run_len", 3)
        h2d_max_runs = threshold("trace_view", "h2d_max_runs", 15)
        h2d_cs_per_run = threshold("trace_view", "h2d_callstack_per_run", 2)
        runs = _detect_h2d_runs(launch_by_cid, dev_by_cid, h2d_thr_ns, h2d_min_run)
        total_pairs = sum(len(r) for r in runs)
        L.append(f"  Launch→device gap < {h2d_thr_ns/1000:.0f}us = device starts right at launch "
                 f"(queue empty, starving for host dispatch).")
        L.append(f"  Runs of >= {h2d_min_run} consecutive such ops on the same stream = host2device-bound region.")
        L.append(f"  {total_pairs:,} host-bound ops in {len(runs)} region(s)"
                 + (f" (showing top {min(len(runs), h2d_max_runs)} by device idle time)" if len(runs) > h2d_max_runs else "")
                 + ".")
        L.append("")
        # callstack lookup index: cpu_op timeline sorted by start, plus async_npu link
        cpuop_timeline.sort(key=lambda x: x[0])
        cstarts = [c[0] for c in cpuop_timeline]
        def resolve_cs(dts_ns):
            torch_ts = t2n_s.get(dts_ns)
            if torch_ts is None:
                return []
            i = bisect.bisect_right(cstarts, torch_ts) - 1
            out = []
            for j in range(max(0, i - 1), min(len(cpuop_timeline), i + 3)):
                s, e_, nm, cs = cpuop_timeline[j]
                if s <= torch_ts <= e_:
                    out.append((nm, cs))
            return out
        shown = 0
        for run in runs[:h2d_max_runs]:
            span = (run[-1]["dts"] + run[-1]["ddur"]) - run[0]["lts"]
            dev_active = sum(x["ddur"] for x in run)
            idle_ns = span - dev_active
            idle_pct = idle_ns / span * 100 if span > 0 else 0
            t0_s = run[0]["lts"] / 1e9
            L.append(f"  Region @~{t0_s:.3f}s  stream={run[0]['stream']}  ops={len(run)}  "
                     f"span={format_duration_ms(span/1000)}  devActive={format_duration_ms(dev_active/1000)}  "
                     f"idle={idle_pct:.0f}%")
            chain_str = _chain_summary(run)
            L.append(f"    chain: {chain_str[:120]}")
            # call stacks for the tightest (smallest gap) distinct ops
            seen_items = set()
            cs_shown = 0
            for p in sorted(run, key=lambda x: x["gap"]):
                if p["item"] in seen_items or cs_shown >= h2d_cs_per_run:
                    continue
                cands = resolve_cs(p["dts"])
                if not cands:
                    continue
                seen_items.add(p["item"])
                cs_shown += 1
                nm, cs = cands[0]
                L.append(f"    [{p['item'][:40]}] gap={p['gap']/1000:.1f}us  ← {nm}")
                for frame in condense_stack(cs):
                    L.append(f"        {frame[:110]}")
            L.append("")
        if not runs:
            L.append("  No sustained host2device-bound region found (pipeline dispatches run far ahead of device).")
            L.append("")
        if not cpuop_timeline:
            L.append("  ⚠ No cpu_op Call stack in file — source mapping unavailable; re-collect with with_stack enabled.")
            L.append("")
    else:
        L.append("  No Node@launch↔device pairs (connection_id) found — cannot assess host2device bound.")
        L.append("")
    L.append("")

    # --- 5. Suspect Signals (diagnostic: compile classification + prefetch candidates) ---
    sec_num = 5
    has_suspects = bool(acl_compile) or bool(host_prefetch) or bool(runs)
    if has_suspects or (caps.get("cpu_op", 0) and has_callstack):
        L.append(f"## {sec_num}. Suspect Signals")
        L.append("  [DEFINITE]=actionable as-is  [SIGNAL]=anomaly, cross-validate with other dimensions")
        L.append("")

    if runs:
        total_h2d = sum(len(r) for r in runs)
        worst = max(runs, key=lambda r: ((r[-1]["dts"] + r[-1]["ddur"]) - r[0]["lts"]
                                         - sum(x["ddur"] for x in r)))
        wspan = (worst[-1]["dts"] + worst[-1]["ddur"]) - worst[0]["lts"]
        widle_pct = (wspan - sum(x["ddur"] for x in worst)) / wspan * 100 if wspan > 0 else 0
        L.append(f"  [SIGNAL] Host2Device-bound: {len(runs)} region(s), {total_h2d:,} ops where device "
                 f"starts right at launch (queue starving for host dispatch).")
        L.append(f"    Worst region: {len(worst)} ops, device idle {widle_pct:.0f}%, "
                 f"chain: {_chain_summary(worst)[:100]}")
        L.append("    → See §4 for full chains + call stacks; cross-validate with operator_details / step_trace.")
        L.append("")

    if acl_compile:
        L.append(f"  [DEFINITE] Online-Compile ({sum(acl_compile.values()):,} events, "
                 f"total {format_duration_ms(acl_compile_dur/1000)})")
        dev_min = min((v[1] for v in compute_streams.values() if v[1] is not None), default=None)
        dev_max = max((v[2] for v in compute_streams.values() if v[2] is not None), default=None)
        early_frac = None
        if compile_ts and dev_min is not None and dev_max is not None and dev_max > dev_min:
            span = dev_max - dev_min
            early = sum(1 for t in compile_ts if (t - dev_min) / span < threshold("trace_view", "compile_early_window", 0.2))
            early_frac = early / len(compile_ts)
        if early_frac is not None and early_frac >= threshold("trace_view", "compile_early_frac", 0.8):
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
        if decrease_ratio >= threshold("trace_view", "freq_decrease_ratio", 0.05):
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
        if co_ratio > threshold("trace_view", "sync_co_ratio", 10):
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
