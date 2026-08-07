#!/usr/bin/env python3
"""解析 trace_view.json — 时间线 / dispatch 链分析。

trace_view.json（Chrome Trace 格式）是唯一携带 host dispatch 与 device 执行之间
时间关系的 profiling 文件。其可用内容由采集开关决定，而非由训练/推理决定：

- 始终存在：device kernel（ph=X 且 args["Task Type"]）+ HostToDevice flow。
- 仅 NPU / 关闭 stack 时：host 侧以 CANN "AscendCL@..." 事件呈现。
- 开启 CPU activity + with_stack 时：host 侧以 cpu_op + python_function
  通道呈现，cpu_op args 携带 "Call stack"（源码桥梁）。

本脚本探测文件中实际存在的 layer，然后提取可供 agent 推理的结构化时间线事实
（人类则会直接阅读可视化时间线）。以流式读取数组，因此多 GB 文件也安全。

输出：(1) compute-stream 时间线，(2) 按 kernel pair 聚合的 stall，
(3) dispatch latency，(4) 在线编译分类为 warmup(A) 与 per-step(B)，
(5) prefetch/prealloc 候选（H2D/alloc op）及精简 call stack。

用法:
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
    """增量式地从顶层 JSON 数组中 yield 对象，无需加载整个文件。
    使用指向 buffer 的索引指针（无逐对象切片），使多 GB 文件保持近 O(n)。"""
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
                    if idx:                       # 丢弃已消费的前缀
                        buf = buf[idx:]
                        idx = 0
                    buf += piece
                else:
                    eof = True
            # 从当前 buffer 中尽可能多地消费对象
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
                    break                         # 不完整的尾部；需要更多数据
                idx = end
                yield obj
            if eof:
                return


def parse_ts_ns(val) -> int:
    """将高精度微秒时间戳字符串解析为整数 ns，
    避免 ~1e15 量级值的浮点精度损失。"""
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
    """精简 call stack：保留 project frame（丢弃 site-packages / std 库），
    限制数量。返回 frame 字符串列表。若所有 frame 均为库 frame，则回退到顶部 frame。"""
    frames = [f.strip() for f in str(cs).replace("\r", "").split(";") if f.strip()]
    lib_markers = tuple(threshold("trace_view", "stack_lib_markers",
                   ["site-packages", "dist-packages", "/lib/python", "torch/nn/modules", "torch/_ops", "autograd/profiler", "torch_npu/profiler"]))
    proj = [f for f in frames if not any(m in f for m in lib_markers)]
    picked = proj if proj else frames
    return picked[:threshold("trace_view", "stack_max_frames", 6)]


def _detect_h2d_runs(launch_by_cid, dev_by_cid, gap_thr_ns, min_run_len):
    """检测 host2device-bound 区域：同一 device stream 上连续的 launch-device pair，
    其 gap（device_start - launch_start）低于阈值。

    小/负 gap 意味着 device 在 host launch 时（或因 host/device 时钟偏移而更早）立即启动
    — device 队列为空、等待 host dispatch。此类 op 的连续序列即 host2device-bound 区域。

    返回区域列表（每个为 pair dict 列表），按 device idle 时间（span - dev_active）降序排列。"""
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
    """为 run 生成紧凑的 op-chain 字符串，对相邻相同的 op 类型去重。"""
    chain = []
    for p in run:
        short = p["item"].split("_", 1)[0] if p["item"] else "?"
        if not chain or chain[-1][0] != short:
            chain.append([short, 1])
        else:
            chain[-1][1] += 1
    return " - ".join(f"{n}x{c}" if c > 1 else n for n, c in chain)


def parse(csv_path: Path, top_k: int, gap_threshold_us: float) -> str:
    gap_threshold_ns = int(gap_threshold_us * 1000)

    caps = Counter()                       # 检测到的事件类别
    dev_count = 0
    # key -> [active_ns, min_ns, max_ns, compute_count, total_count]
    stream_stats = defaultdict(lambda: [0, None, None, 0, 0])
    last_end = {}                          # (pid,tid) -> (end_ns, name) 用于 compute task
    noncompute_count = 0
    _gb = threshold("trace_view", "gap_buckets_us", [10, 50, 200])
    gap_buckets = {f"<{_gb[0]}us": 0, f"{_gb[0]}-{_gb[1]}us": 0, f"{_gb[1]}-{_gb[2]}us": 0, f">{_gb[2]}us": 0}
    stall_agg = defaultdict(lambda: [0, 0, 0])  # (prev,cur) -> [count, sum_gap, max_gap]

    flow_ts = {}                           # HostToDevice id -> [s_ts, f_ts, dev_name]
    last_device_name = None                # 最近的先前 device kernel（用于 flow 标注）
    disp_lat = []                          # dispatch latency（ns），有上限采样
    disp_stats = {"count": 0, "sum": 0, "max": 0}
    disp_top = []                          # heap: (lat_ns, dev_name)

    acl_compile = Counter()                # 类 opCompile 的 host 事件
    acl_compile_dur = 0
    compile_ts = []                        # compile 事件的时间戳（有上限）
    host_prefetch = []                     # heap: (dur_ns, name, callstack) 用于 H2D/alloc op
    has_callstack = False
    PREFETCH_KEYS = tuple(threshold("trace_view", "prefetch_keywords",
                     ["aten::to", "copy_", "aten::copy", "::empty", "empty_", "aten::empty", "memcpy", "to_copy", "_to_copy", "pin_memory"]))

    # AI Core 频率、GC、stream sync、Node@launch 跟踪
    freq_values = []
    freq_max = 0
    # Resource 利用率计数器 (A1)：HBM bw、LLC hit rate/throughput、
    # L2/MAC bw level、APP/HBM 占用率。按计数器名聚合。
    # counters[name] = [count, sum, min, max]
    counters = defaultdict(lambda: [0, 0.0, None, None])
    _COUNTER_CATS = ("Read(MB/s)", "Write(MB/s)", "Hit Rate(%)", "Throughput(MB/s)",
                     "L2 Buffer Bw Level", "Mata Bw Level", "KB", "value")
    gc_events = []                        # (dur_ns, name)
    sync_stream_count = 0
    node_launch_count = 0
    sync_stream_dur = 0

    # Host2Device-bound 检测：通过 connection_id 的 launch↔device pair，加上
    # 解析 call stack 所需的 async_npu(torch_to_npu) flow 与 cpu_op 时间线。
    launch_by_cid = {}                    # cid -> (lts_ns, ldur_ns, item_id)
    dev_by_cid = defaultdict(list)        # cid -> [(dts_ns, ddur_ns, name, stream, task_type)]
    t2n_s = {}                            # async_npu flow id (== device_start_ns) -> torch host ts ns
    cpuop_timeline = []                   # (start_ns, end_ns, name, callstack)
    # C3: 用于 overlap + idle 归因的 compute-stream 区间（C2 v2 需要完整时间线）
    compute_intervals = []                # (start_ns, end_ns, stream_key)
    _CI_CAP = 2000000
    # C2 v2: 按类别分组的 host 事件区间，用于 idle 归因
    host_ev_intervals = defaultdict(list)  # cat -> [(start_ns, end_ns)]

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
                if len(compute_intervals) < _CI_CAP:
                    compute_intervals.append((ts, end, key))
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
            # torch_to_npu flow：id == device kernel 启动时间（ns）；s.ts == torch host
            # 入队时间。用于将 device kernel 关联回其 host cpu_op。
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

        # C2 v2: capture host AscendCL@/sync/launch event intervals for idle attribution
        elif ph == "X" and (str(name).startswith("AscendCL@") or "Synchronize" in str(name) or str(name) == "Node@launch"):
            nl = str(name).lower()
            if "ompile" in nl:
                hcat = "compile"
            elif "freephysical" in nl or "unmapmem" in nl or "mallocphysical" in nl or "mapmem" in nl:
                hcat = "mem-mgmt"
            elif "synchronize" in nl:
                hcat = "sync"
            elif str(name) == "Node@launch":
                hcat = "launch"
            else:
                hcat = "other-aclrt"
            if len(host_ev_intervals[hcat]) < 200000:
                host_ev_intervals[hcat].append((parse_ts_ns(e.get("ts")),
                                                parse_ts_ns(e.get("ts")) + dur_ns(e.get("dur"))))

        elif ph == "C" and isinstance(e.get("args"), dict):
            args = e["args"]
            if "MHz" in args:
                mhz = args["MHz"]
                freq_values.append(mhz)
                freq_max = max(freq_max, mhz)
            # 按名称聚合所有非 MHz 计数器
            cname = str(e.get("name", ""))
            for k in _COUNTER_CATS:
                if k in args:
                    try:
                        v = float(args[k])
                    except (ValueError, TypeError):
                        continue
                    st = counters[cname]
                    st[0] += 1
                    st[1] += v
                    st[2] = v if st[2] is None else min(st[2], v)
                    st[3] = v if st[3] is None else max(st[3], v)
                    break

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
        return f"[trace_view] 空文件或无法识别: {csv_path}"

    L = []
    L.append("# Trace View 时间线分析")
    L.append(f"数据来源: {csv_path}")
    L.append("")

    # --- 0. 能力探测 ---
    L.append("## 0. 检测到的 Layer (由采集开关决定)")
    L.append(f"  device kernels (Task Type):   {dev_count:,}")
    L.append(f"  HostToDevice dispatch flow:   {disp_stats['count']:,}")
    L.append(f"  CANN AscendCL@ host events:   {caps.get('acl', 0):,}")
    L.append(f"  torch cpu_op events:          {caps.get('cpu_op', 0):,}")
    L.append(f"  python_function frames:       {caps.get('python_function', 0):,}")
    L.append(f"  fwdbwd links:                 {caps.get('fwdbwd', 0):,}")
    if caps.get("cpu_op", 0) == 0:
        L.append("  [!] 未检测到 cpu_op/Call stack (未开启 CPU activity + with_stack) —")
        L.append("    仅有 host-device dispatch 时间线可用；源码映射需重新采集并开启 with_stack。")
    elif not has_callstack:
        L.append("  [!] 存在 cpu_op 但 Call stack 为空 (未开启 with_stack) — 仅 host op 耗时，无源码 stack。")
    L.append("")

    # --- 1. Device 时间线（仅 compute stream；折叠非 compute）---
    L.append("## 1. Device 时间线 (compute streams)")
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
        L.append("  (未检测到 compute stream)")
    if other_streams:
        tot = sum(v[4] for v in other_streams.values())
        L.append(f"  + 折叠了 {len(other_streams)} 个非 compute stream "
                 f"（{tot:,} 个 task: comm/sync/DMA — NOTIFY/SDMA/EVENT/AI_CPU）")
    L.append("  Gap 分布 (每 stream 内连续 COMPUTE task 之间):")
    for b, c in gap_buckets.items():
        L.append(f"    {b:>9}: {c:,}")
    L.append("")

    # --- 2. 按 kernel pair 聚合的 stall 点 ---
    L.append(f"## 2. Device Stall (gap >= {gap_threshold_us:.0f}us, 按 kernel pair 聚合)")
    L.append("  相同 kernel pair 的 stall 已聚合；按累计 gap 排序 — 重复且大的 stall 值得优化。")
    if stall_agg:
        pairs = sorted(stall_agg.items(), key=lambda x: -x[1][1])
        L.append(f"  {'Count':>6} {'SumGap':>10} {'AvgGap':>9} {'MaxGap':>9}  {'After -> Before kernel'}")
        for (prev, cur), (cnt, s, mx) in pairs[:top_k]:
            L.append(f"  {cnt:>6} {format_duration_ms(s/1000):>10} "
                     f"{format_duration_ms(s/cnt/1000):>9} {format_duration_ms(mx/1000):>9}  "
                     f"{str(prev)[:26]} -> {str(cur)[:26]}")
        L.append(f"  不同 stall pair 数: {len(stall_agg):,}")
    else:
        L.append("  无超过阈值的。")
    L.append("")

    # --- 3. Dispatch latency ---
    L.append("## 3. Host-Device Dispatch Latency (HostToDevice flow)")
    if disp_stats["count"]:
        avg = disp_stats["sum"] / disp_stats["count"]
        L.append(f"  count={disp_stats['count']:,}  avg={avg/1000:.1f}us  max={disp_stats['max']/1000:.1f}us")
        if disp_lat:
            disp_lat.sort()
            p50 = disp_lat[len(disp_lat)//2]
            p90 = disp_lat[int(len(disp_lat)*0.9)]
            L.append(f"  p50={p50/1000:.1f}us  p90={p90/1000:.1f}us  （p90>>p50 表明 dispatch backlog 不均匀）")
        if disp_top:
            L.append("  Top dispatch latency (最近的 device kernel 名, 用于定位 dispatch 点):")
            for lat, dev_name in sorted(disp_top, key=lambda x: -x[0]):
                L.append(f"    {lat/1000:>8.1f}us  ≈ {str(dev_name)[:40]}")
        if dev_count > 0:
            # 用 p50 估算（avg 受 queue stall 异常值影响）
            est_per_op_us = (p50 / 1000) if disp_lat else (avg / 1000)
            est_source = "p50" if disp_lat else "avg"
            disp_total_us = dev_count * est_per_op_us
            compute_active_us = sum(v[0] for v in compute_streams.values()) / 1000
            L.append(f"  预估 dispatch 总量: {disp_total_us/1000:.1f} ms (dev_count x {est_source}_latency)")
            if compute_active_us > 0:
                disp_kernel_ratio = disp_total_us / compute_active_us * 100
                L.append(f"  Dispatch / kernel-active ratio: {disp_kernel_ratio:.1f}%")
                if disp_kernel_ratio > threshold("trace_view", "dispatch_kernel_ratio", 50):
                    L.append(f"  - Dispatch overhead 预估 > {threshold('trace_view', 'dispatch_kernel_ratio', 50)}% 的 kernel-active 时间")
                    L.append(f"    在 async queue 下二者 overlap；实际影响取决于 gap 分布:")
                    L.append(f"    gap > 50us 占比高: dispatch 未完全 overlap — 减少 op 数量有帮助")
                    L.append(f"    gap < 10us 占比高: dispatch 已完全 overlap — Free 源自串行依赖")
    else:
        L.append("  未发现 HostToDevice flow。")
    L.append("")

    # --- 4. Host2Device Bound 区域（dispatch 不足的 op 的时间序列）---
    L.append("## 4. Host2Device Bound 区域")
    runs = []
    if launch_by_cid and dev_by_cid:
        h2d_thr_ns = int(threshold("trace_view", "h2d_gap_threshold_us", 50) * 1000)
        h2d_min_run = threshold("trace_view", "h2d_min_run_len", 3)
        h2d_max_runs = threshold("trace_view", "h2d_max_runs", 15)
        h2d_cs_per_run = threshold("trace_view", "h2d_callstack_per_run", 2)
        runs = _detect_h2d_runs(launch_by_cid, dev_by_cid, h2d_thr_ns, h2d_min_run)
        total_pairs = sum(len(r) for r in runs)
        L.append(f"  Launch-device gap < {h2d_thr_ns/1000:.0f}us = device 在 launch 时立即启动 "
                 f"（队列空，等待 host dispatch）。")
        L.append(f"  同一 stream 上 >= {h2d_min_run} 个连续此类 op = host2device-bound region。")
        L.append(f"  {total_pairs:,} 个 host-bound op，分布在 {len(runs)} 个区域"
                 + (f" （展示按 device idle 时间排序的 top {min(len(runs), h2d_max_runs)}）" if len(runs) > h2d_max_runs else "")
                 + "。")
        L.append("")
        # callstack 查找索引：cpu_op 时间线按 start 排序，加上 async_npu 链接
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
            L.append(f"  Region @~{t0_s:.3f}s  stream={run[0]['stream']}  op={len(run)}  "
                     f"span={format_duration_ms(span/1000)}  devActive={format_duration_ms(dev_active/1000)}  "
                     f"idle={idle_pct:.0f}%")
            chain_str = _chain_summary(run)
            L.append(f"    chain: {chain_str[:120]}")
            # 最紧凑（gap 最小）的去重 op 的 call stack
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
            L.append("  未发现持续的 host2device-bound 区域 (pipeline dispatch 远超 device)。")
            L.append("")
        if not cpuop_timeline:
            L.append("  [!] 文件中无 cpu_op Call stack — 源码映射不可用；请开启 with_stack 重新采集。")
            L.append("")
    else:
        L.append("  未发现 Node@launch↔device pairs (connection_id) — 无法评估 host2device bound。")
        L.append("")
    L.append("")

    # --- 5. Resource 利用率时间线（计数器，A1）---
    # HBM bandwidth / LLC hit rate / throughput / L2-MAC bw level / 占用率
    # 随时间变化。是 kernel_details 静态逐 kernel ratio 的动态对应物 —
    # ratio 的动态对应物 — 区分"memory-bound 阶段"与"compute-bound 阶段"。
    L.append("## 5. Resource 利用率时间线 (counters)")
    if counters:
        # 按类别对计数器名分组以提升可读性
        def _cat(nm):
            if "/Read" in nm or "read_bandwidth" in nm: return "HBM Read BW"
            if "/Write" in nm or "write_bandwidth" in nm: return "HBM Write BW"
            if "Hit Rate" in nm: return "LLC Hit Rate(%)"
            if "Throughput" in nm: return "LLC Throughput(MB/s)"
            if "L2 Buffer" in nm: return "L2 Buffer Bw Level"
            if "Mata Bw" in nm: return "MAC Bw Level"
            if "HBM" in nm or "DDR" in nm: return "Memory Occupancy(KB)"
            return "utilization"
        from collections import defaultdict as _dd
        grouped = _dd(lambda: [0, 0.0, None, None])
        for nm, st in counters.items():
            c = _cat(nm)
            g = grouped[c]
            g[0] += st[0]; g[1] += st[1]
            g[2] = st[2] if g[2] is None else min(g[2], st[2])
            g[3] = st[3] if g[3] is None else max(g[3], st[3])
        order = ["HBM Read BW", "HBM Write BW", "LLC Hit Rate(%)",
                 "LLC Throughput(MB/s)", "L2 Buffer Bw Level", "MAC Bw Level",
                 "Memory Occupancy(KB)", "utilization"]
        for c in order:
            if c not in grouped:
                continue
            cnt, s, mn, mx = grouped[c]
            avg = s / cnt if cnt > 0 else 0
            unit = "" if c in ("LLC Hit Rate(%)", "L2 Buffer Bw Level", "MAC Bw Level") else ""
            L.append(f"  {c:<26} samples={cnt:>7,}  avg={avg:>10.1f}  min={mn:>10.1f}  max={mx:>10.1f}")
        # 饱和度提示
        hbm_read = grouped.get("HBM Read BW")
        if hbm_read and hbm_read[3] > 0:
            L.append(f"  - HBM Read BW peak={hbm_read[3]:.0f} MB/s — 交叉验证 HBM peak BW 以判断内存饱和。")
        llc = grouped.get("LLC Hit Rate(%)")
        if llc and llc[1] / (llc[0] or 1) < 0.5:
            L.append(f"  - LLC Hit Rate 均值偏低 ({llc[1]/(llc[0] or 1)*100:.0f}%) — cache-unfriendly 的访问模式；交叉验证 kernel_details 的 mte 占比。")
        L.append("")
    else:
        L.append("  未发现 resource 计数器 (HBM bw / LLC / 利用率时间线不可用).")
        L.append("")

    # --- 5b. Stream 并发 / Overlap（C3，overlap 维度）---
    # 量化同时有多少 compute stream 在忙 — 这是"hide latency / overlap"
    # 优化维度的唯一输出。
    L.append("## 5b. Stream Concurrency (overlap 维度)")
    if compute_intervals and len(compute_streams) > 1:
        # 事件扫描：在每个转折点统计活跃 stream 数
        events = []
        for s, e_, _ in compute_intervals:
            events.append((s, 1))
            events.append((e_, -1))
        events.sort()
        cur = 0
        prev_t = events[0][0] if events else 0
        concur = defaultdict(int)  # n_active_streams -> time_ns
        for t, delta in events:
            if t > prev_t and cur >= 0:
                concur[cur] += (t - prev_t)
            cur += delta
            prev_t = t
        total_span = sum(concur.values())
        if total_span > 0:
            L.append(f"  Compute stream: {len(compute_streams)} | 采样区间数: {len(compute_intervals):,}")
            L.append(f"  并发 busy-streams 分布 (时间占比):")
            for n in sorted(concur.keys()):
                if n <= 4 or n == max(concur.keys()):
                    pct = concur[n] / total_span * 100
                    L.append(f"    {n} 个 stream busy: {concur[n]/1e9:>8.3f}s ({pct:>5.1f}%)")
            one = concur.get(1, 0) + concur.get(0, 0)
            multi = sum(v for k, v in concur.items() if k >= 2)
            if multi / total_span < 0.2:
                L.append(f"  - 仅 {multi/total_span*100:.0f}% 的时间有 ≥2 个 stream 并发 busy — overlap 未充分利用 (通过 multi-stream hide latency 的机会).")
            else:
                L.append(f"  - {multi/total_span*100:.0f}% 的时间有 ≥2 个 stream busy — overlap 已充分利用。")
            L.append("  交叉验证: 某 stream 上的 gap (见第 1 节 gap 分布) 与另一 stream 的 busy 时间 overlap = 可覆盖的气泡。")
        L.append("")
    else:
        L.append("  单个 compute stream 或无区间 — 无 multi-stream overlap 可利用 (stream 级 latency-hiding N/A).")
        L.append("")

    # --- 5c. Idle 原因拆解（C2 v2：通过 host AscendCL@ 事件做时间对齐）---
    # 对 device idle 的每个时刻，归因到 host 线程正在做的事
    # （mem-mgmt / sync / compile / launch / other-aclrt）。剩余 = Python/无工作。
    L.append("## 5c. Idle 时间原因拆解 (time-aligned: device idle 时 host 在做什么)")
    if compute_intervals:
        # 构建统一事件扫描：device-busy（+1/-1）+ host 类别进入/退出。
        # "device idle" = 无任何 compute stream 在忙的 wall-clock 时间（真实 idle，
        # 非逐 stream 求和——后者会重复计算 overlap）。与 step_trace Free 对齐。
        PRIO = {"sync": 0, "mem-mgmt": 1, "compile": 2, "launch": 3, "other-aclrt": 4}
        events = []
        for s, e_, _ in compute_intervals:
            events.append((s, 0, +1, None))   # device busy start
            events.append((e_, 0, -1, None))
        for cat, ivs in host_ev_intervals.items():
            for s, e_ in ivs:
                if e_ > s:
                    events.append((s, 1, 0, (cat, +1)))  # host cat enter
                    events.append((e_, 1, 0, (cat, -1)))
        events.sort(key=lambda x: (x[0], x[1], x[2] if x[2] is not None else 0))
        dev_busy = 0
        active_cats = {}
        prev_t = events[0][0] if events else 0
        span_start = prev_t
        span_end = prev_t
        idle_attr = defaultdict(int)
        residual_idle = 0
        for t, _, dev_delta, cat_evt in events:
            if t > prev_t:
                if dev_busy == 0:  # all streams idle during [prev_t, t]
                    live = [c for c in active_cats if active_cats[c] > 0]
                    if live:
                        best = min(live, key=lambda c: PRIO.get(c, 99))
                        idle_attr[best] += (t - prev_t)
                    else:
                        residual_idle += (t - prev_t)
            if dev_delta:
                dev_busy += dev_delta
            if cat_evt:
                cat, d = cat_evt
                active_cats[cat] = active_cats.get(cat, 0) + d
            prev_t = t
            span_end = t
        sweep_idle = sum(idle_attr.values()) + residual_idle
        span_wc = span_end - span_start
        if sweep_idle > 0:
            L.append(f"  Device idle (所有 stream idle): {format_duration_ms(sweep_idle/1000)} / wall-clock span {format_duration_ms(span_wc/1000)} ({sweep_idle/span_wc*100:.0f}%)")
            L.append("  Idle 归因到该时刻 host 正在做的事:")
            order = ["mem-mgmt", "sync", "compile", "launch", "other-aclrt"]
            for c in order:
                v = idle_attr.get(c, 0)
                if v > 0:
                    L.append(f"    {c:<12} {format_duration_ms(v/1000)} ({v/sweep_idle*100:>5.1f}%)")
            L.append(f"    {'residual':<12} {format_duration_ms(residual_idle/1000)} ({residual_idle/sweep_idle*100:>5.1f}%)  (Python framework / 无工作；交叉验证 operator_details host category)")
            all_attr = list(idle_attr.items()) + [("residual", residual_idle)]
            dom = max(all_attr, key=lambda x: x[1])
            L.append(f"  - 主导 idle 原因: {dom[0]} ({dom[1]/sweep_idle*100:.0f}% 占 idle)")
            if dom[0] == "mem-mgmt":
                L.append("    Host 阻塞在 aclrt memory APIs (Free/Unmap/Malloc/Map) - device starves. 交叉验证 api_statistic (memory-mgmt 类别).")
            L.append("")

    # --- 6. 可疑信号（诊断：compile 分类 + prefetch 候选）---
    sec_num = 6
    has_suspects = bool(acl_compile) or bool(host_prefetch) or bool(runs)
    if has_suspects or (caps.get("cpu_op", 0) and has_callstack):
        L.append(f"## {sec_num}. 可疑信号")
        L.append("  [DEFINITE]=可直接行动  [SIGNAL]=异常，需结合其他维度交叉验证")
        L.append("")

    if runs:
        total_h2d = sum(len(r) for r in runs)
        worst = max(runs, key=lambda r: ((r[-1]["dts"] + r[-1]["ddur"]) - r[0]["lts"]
                                         - sum(x["ddur"] for x in r)))
        wspan = (worst[-1]["dts"] + worst[-1]["ddur"]) - worst[0]["lts"]
        widle_pct = (wspan - sum(x["ddur"] for x in worst)) / wspan * 100 if wspan > 0 else 0
        L.append(f"  [SIGNAL] Host2Device-bound: {len(runs)} 个区域，{total_h2d:,} 个 op 的 device "
                 f"在 launch 时立即启动（队列等待 host dispatch）。")
        L.append(f"    最差区域: {len(worst)} 个 op，device idle {widle_pct:.0f}%, "
                 f"chain: {_chain_summary(worst)[:100]}")
        L.append("    - 完整 chain + call stack 见第 4 节；交叉验证 operator_details / step_trace。")
        L.append("")

    if acl_compile:
        L.append(f"  [DEFINITE] Online-Compile（{sum(acl_compile.values()):,} 个事件，"
                 f"总计 {format_duration_ms(acl_compile_dur/1000)})")
        dev_min = min((v[1] for v in compute_streams.values() if v[1] is not None), default=None)
        dev_max = max((v[2] for v in compute_streams.values() if v[2] is not None), default=None)
        early_frac = None
        if compile_ts and dev_min is not None and dev_max is not None and dev_max > dev_min:
            span = dev_max - dev_min
            early = sum(1 for t in compile_ts if (t - dev_min) / span < threshold("trace_view", "compile_early_window", 0.2))
            early_frac = early / len(compile_ts)
        if early_frac is not None and early_frac >= threshold("trace_view", "compile_early_frac", 0.8):
            L.append(f"    分布: {early_frac*100:.0f}% 集中在前 20% 时间线 - **Type A: warmup compilation**。")
            L.append("    - 采集问题: 增大 schedule skip_first 以跳过 warmup — 非真实瓶颈。")
        elif early_frac is not None:
            L.append(f"    分布: 仅 {early_frac*100:.0f}% 在前 20%, 其余贯穿整个时间线 - **Type B: per-step online compilation**。")
            L.append("    - 真实瓶颈，无法通过采集参数解决: 检查 jit_compile 是否关闭、dynamic shape 是否导致重复编译，"
                     "或改用 graph compilation；应避免 aclop 的逐 op 在线编译路径。")
        else:
            L.append("    - 无法确定分布 (缺少 device 时间基线).")
        L.append("")

    if host_prefetch:
        L.append("  [SIGNAL] Prefetch / Prealloc 候选 (H2D copy & alloc ops)")
        L.append("    这些 op 无需替换算子即可优化 (prefetch / pre-allocate / buffer 复用); Call stack 指向代码位置。")
        for dn, name, cs in sorted(host_prefetch, key=lambda x: -x[0]):
            L.append(f"    - {name}  host={dn/1000:.1f}us")
            for frame in condense_stack(cs):
                L.append(f"        {frame[:110]}")
        L.append("")
    elif caps.get("cpu_op", 0) and has_callstack and not acl_compile:
        L.append("  未发现显著的 H2D copy 或重复 alloc op (aten::to / copy_ / empty 等).")
        L.append("")

    # AI Core frequency degradation
    if freq_values and freq_max > 0:
        decrease_ratio = sum(freq_max - f for f in freq_values) / (freq_max * len(freq_values))
        if decrease_ratio >= threshold("trace_view", "freq_decrease_ratio", 0.05):
            L.append(f"  [DEFINITE] AI Core 频率降频: {decrease_ratio*100:.1f}% "
                     f"(max={freq_max}MHz, min={min(freq_values)}MHz)")
            L.append("    - 散热/功耗 throttling。检查: 散热、功耗限制，或降低 compute 强度。")
            L.append("")

    # GC events
    if gc_events:
        gc_total = sum(d for d, _ in gc_events)
        gc_sorted = sorted(gc_events, key=lambda x: -x[0])
        L.append(f"  [SIGNAL] Python GC: {len(gc_events)} 个事件，总计 {format_duration_ms(gc_total/1000)}")
        L.append("    Top GC 事件:")
        for dur, name in gc_sorted[:3]:
            L.append(f"      {format_duration_ms(dur/1000)}  {name[:60]}")
        L.append("    - 交叉验证: 若 GC 频繁，检查是否存在过多小 tensor 分配 (operator_memory).")
        L.append("")

    # Stream synchronization
    if sync_stream_count > 0 and node_launch_count > 0:
        co_ratio = sync_stream_count / node_launch_count * 100
        if co_ratio > threshold("trace_view", "sync_co_ratio", 10):
            L.append(f"  [DEFINITE] 频繁的 stream synchronization: {sync_stream_count} 次 sync vs {node_launch_count} 次 launch ({co_ratio:.1f}%)")
            L.append("    - 可能是 ASCEND_LAUNCH_BLOCKING=1 或显式 sync。检查环境变量与 .item()/.numpy() 用法。")
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

    L = [f"# Trace View — 过滤", f"数据来源: {csv_path}",
         f"过滤: {', '.join(filters)}", f"匹配数: {len(matched)}", ""]
    if not matched:
        L.append("没有事件匹配 (检查 filter, 或文件缺少 cpu_op layer).")
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
                        help="Device gap 阈值（us），超过则报告为 stall 点")
    parser.add_argument("--filter", nargs="+", default=None,
                        help="为匹配的 op 名（子串）展示时间线邻居 + call stack")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    ascend_dir = find_ascend_profiler_output(args.profiling_dir, args.rank)
    csv_path = ascend_dir / "trace_view.json"
    if not csv_path.exists():
        result = f"[trace_view] 文件未找到: {csv_path}"
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
