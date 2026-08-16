#!/usr/bin/env python3
"""解析 operator_memory.csv — 逐 tensor 分配生命周期分析。

该文件记录每个 tensor 的：大小、分配/释放时间、生命周期时长，
以及分配/释放时的全局内存状态。相比 memory_record.csv 的独特价值：
- 逐 tensor 粒度（谁分配了什么、多大、存活多久）
- Tensor 生命周期（短生命周期的大 tensor = buffer 复用候选）

用法:
    python parse_operator_memory.py <profiling_dir> [--rank N] [--top-k 20]
"""

import argparse
import heapq
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (find_ascend_profiler_output, stream_csv, safe_float,
                    format_size_mb, format_duration_ms, threshold)


def parse(profiling_dir: str, rank=None, top_k: int = 20,
          hbm_gb: float = 64.0, large_gb: float = 5.0) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "operator_memory.csv"

    if not csv_path.exists():
        return f"[operator_memory] 文件未找到: {csv_path}"

    total_rows = 0
    top_size_heap = []
    short_lived_large = []
    size_op_count = defaultdict(int)
    op_agg = defaultdict(lambda: {"count": 0, "total_kb": 0.0, "durations": []})
    max_alloc_at_alloc = 0.0

    # 大 tensor 分类，用于 parallelism 触发判断
    LARGE_KB = large_gb * 1024 * 1024  # 5GB default
    SHORT_LIFE_US = threshold("operator_memory", "short_life_us", 1_000_000)
    SHORT_LIVED_MIN_KB = threshold("operator_memory", "short_lived_min_kb", 100)
    SHORT_LIVED_MAX_LIFE = threshold("operator_memory", "short_lived_max_life_us", 1000)
    SIZE_TRACK_MIN_KB = threshold("operator_memory", "size_track_min_kb", 10)
    REPEATED_COUNT = threshold("operator_memory", "repeated_count", 10)
    CHURN_TOTAL_KB = threshold("operator_memory", "churn_total_kb", 10000)
    DOMINATE_RATIO = threshold("operator_memory", "dominate_ratio", 0.5)
    PARALLELISM_RATIO = threshold("operator_memory", "parallelism_ratio", 0.8)
    large_short = []  # (size_kb, dur_us, name, alloc_us, release_us) — 浪费
    large_long = []   # (size_kb, dur_us, name) — 必要
    all_tensors = []  # (size_kb, name, alloc_us, release_us) 用于 peak 归因 (C10)
    peak_alloc_time = 0  # peak 分配的时间戳

    for row in stream_csv(csv_path):
        total_rows += 1
        size_kb = safe_float(row.get("Size(KB)", 0))
        duration_us = safe_float(row.get("Duration(us)", 0))
        name = row.get("Name", "?")
        alloc_total = safe_float(row.get("Allocation Total Allocated(MB)", 0))

        max_alloc_at_alloc = max(max_alloc_at_alloc, alloc_total)
        if alloc_total >= max_alloc_at_alloc:
            peak_alloc_time = safe_float(row.get("Allocation Time(us)", 0))

        # C10: 收集所有 tensor 用于 peak 归因（peak 时刻仍存活）
        if size_kb > 0:
            all_tensors.append((size_kb, name, safe_float(row.get("Allocation Time(us)", 0)),
                                 safe_float(row.get("Release Time(us)", 0))))

        if size_kb > 0:
            entry = (size_kb, total_rows, row)
            if len(top_size_heap) < top_k:
                heapq.heappush(top_size_heap, entry)
            elif size_kb > top_size_heap[0][0]:
                heapq.heapreplace(top_size_heap, entry)

        if size_kb > SHORT_LIVED_MIN_KB and 0 < duration_us < SHORT_LIVED_MAX_LIFE:
            short_lived_large.append((size_kb, duration_us, name))

        # 按 ACTIVE duration 判定短生命周期（真实引用时间，非 pool 生命周期）。
        # Caching allocator 会将已释放的 tensor 保留在 pool 中 - Duration 可能很长
        # 而 Active 很短。用 Active 来捕捉那些 pool
        # Duration 会标记为"长生命周期"而漏掉的复用候选。
        active_dur = safe_float(row.get("Active Duration(us)", 0))
        if size_kb > SHORT_LIVED_MIN_KB and 0 < active_dur < SHORT_LIVED_MAX_LIFE:
            # 按 (size,name) 对基于 Duration 的列表去重没有必要；
            # 我们单独跟踪基于 Active 的列表以呈现 cache 保留的情况。
            if 0 < duration_us >= SHORT_LIVED_MAX_LIFE:
                short_lived_large.append((size_kb, active_dur, name))

        # 对大 tensor 分类用于 parallelism 触发判断（使用 Active Duration：
        # cached-but-inactive 的 tensor 仅当其 active 生命周期短时才算"浪费"）
        if size_kb > LARGE_KB and duration_us > 0:
            alloc_us = safe_float(row.get("Allocation Time(us)", 0))
            release_us = safe_float(row.get("Release Time(us)", 0))
            life_us = safe_float(row.get("Active Duration(us)", 0)) or duration_us
            if life_us < SHORT_LIFE_US:
                large_short.append((size_kb, life_us, name, alloc_us, release_us))
            else:
                large_long.append((size_kb, life_us, name))

        if size_kb > SIZE_TRACK_MIN_KB:
            size_op_count[f"{name}|{size_kb:.0f}"] += 1

        if size_kb > 0:
            op_agg[name]["count"] += 1
            op_agg[name]["total_kb"] += size_kb
            if len(op_agg[name]["durations"]) < 1000:
                op_agg[name]["durations"].append(duration_us)

    if total_rows == 0:
        return f"[operator_memory] 空文件: {csv_path}"

    lines = []
    lines.append("# Operator Memory 分析")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"总分配记录数: {total_rows:,}")
    lines.append(f"Peak Allocated (任意分配点): {max_alloc_at_alloc:,.0f} MB")
    lines.append("")

    # --- 1. 按大小排序的 Top 分配 ---
    top_entries = sorted(top_size_heap, key=lambda x: -x[0])
    lines.append(f"## 1. 按大小排序的 Top {min(top_k, len(top_entries))} 分配")
    header = f"  {'#':>3} {'Op Name':<30} {'Size':>10} {'Lifetime':>12} {'Alloc@(MB)':>11}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for idx, (size_kb, _, row) in enumerate(top_entries, 1):
        name = row.get("Name", "?")
        dur = safe_float(row.get("Duration(us)", 0))
        alloc_at = safe_float(row.get("Allocation Total Allocated(MB)", 0))
        lines.append(
            f"  {idx:>3} {name:<30} {format_size_mb(size_kb):>10} "
            f"{format_duration_ms(dur):>12} {alloc_at:>10,.0f}"
        )
    lines.append("")

    # --- 2. 按算子聚合 ---
    op_sorted = sorted(op_agg.items(), key=lambda x: -x[1]["total_kb"])
    lines.append(f"## 2. 按算子聚合 (按总大小排序的 Top {min(top_k, len(op_sorted))})")
    header2 = f"  {'Op Name':<30} {'Count':>7} {'Total':>10} {'Avg Size':>10} {'Avg Life':>10}"
    lines.append(header2)
    lines.append("  " + "-" * (len(header2) - 2))
    for name, info in op_sorted[:top_k]:
        avg_size = info["total_kb"] / info["count"] if info["count"] > 0 else 0
        avg_dur = sum(info["durations"]) / len(info["durations"]) if info["durations"] else 0
        lines.append(
            f"  {name:<30} {info['count']:>7} {format_size_mb(info['total_kb']):>10} "
            f"{format_size_mb(avg_size):>10} {format_duration_ms(avg_dur):>10}"
        )
    lines.append("")

    # --- 3. 短生命周期大 tensor ---
    lines.append("## 3. 短生命周期大 Tensor (size>100KB, lifetime<1ms)")
    if short_lived_large:
        lines.append(f"  发现: {len(short_lived_large)} 个 tensor")
        lines.append(f"  这些 tensor 快速分配并释放 — 适合 buffer 复用。")
        lines.append("")

        short_by_op = defaultdict(lambda: {"count": 0, "sizes": []})
        for size_kb, dur, name in short_lived_large:
            short_by_op[name]["count"] += 1
            if len(short_by_op[name]["sizes"]) < 10:
                short_by_op[name]["sizes"].append(size_kb)

        short_sorted = sorted(short_by_op.items(), key=lambda x: -x[1]["count"])
        header3 = f"  {'Op Name':<30} {'Count':>7} {'Typical Sizes'}"
        lines.append(header3)
        lines.append("  " + "-" * (len(header3) - 2))
        for name, info in short_sorted[:top_k]:
            sizes_uniq = sorted(set(f"{s:.0f}KB" for s in info["sizes"]))[:5]
            lines.append(f"  {name:<30} {info['count']:>7} {', '.join(sizes_uniq)}")
    else:
        lines.append("  未发现")
    lines.append("")

    # --- 可疑信号 ---
    lines.append("## 可疑信号")
    lines.append("  [DEFINITE]=可直接行动  [SIGNAL]=异常，根因未定 — 需结合其他 profiling 维度交叉验证")
    suspects_found = False

    repeated = [(key, count) for key, count in size_op_count.items() if count > REPEATED_COUNT]
    repeated.sort(key=lambda x: -x[1])
    if repeated:
        lines.append("  - [DEFINITE] 重复的等大小分配 (buffer 复用机会):")
        for key, count in repeated[:8]:
            name_part, size_part = key.rsplit("|", 1)
            lines.append(f"    {name_part}: {size_part}KB x {count} 次")
        suspects_found = True

    if short_lived_large:
        total_short_kb = sum(s for s, _, _ in short_lived_large)
        if total_short_kb > CHURN_TOTAL_KB:
            lines.append(f"  - [DEFINITE] 短生命周期大 tensor churn: {len(short_lived_large)} 个 tensor，"
                         f"累计 {total_short_kb/1024:.0f}MB")
            lines.append(f"    分配后立即释放 — 预分配可消除此 overhead")
            suspects_found = True

    if op_sorted:
        top_op_total = op_sorted[0][1]["total_kb"]
        total_all = sum(info["total_kb"] for _, info in op_sorted)
        if total_all > 0 and top_op_total / total_all > DOMINATE_RATIO:
            lines.append(f"  - [SIGNAL] 单个 op 主导内存: {op_sorted[0][0]} "
                         f"({top_op_total/1024:.0f}MB, {top_op_total/total_all*100:.0f}% 占 total)")
            suspects_found = True

    if not suspects_found:
        lines.append("  无")
    lines.append("")

    # --- Parallelism 触发判断（两阶段：浪费 vs 必要）---
    hbm_mb = hbm_gb * 1024
    peak_mb = max_alloc_at_alloc

    # Peak 时的浪费：peak 时刻仍存活的短生命周期大 tensor 之和
    waste_at_peak_mb = 0
    waste_total_mb = 0
    for size_kb, dur, name, alloc_us, release_us in large_short:
        waste_total_mb += size_kb / 1024
        if alloc_us <= peak_alloc_time <= release_us:
            waste_at_peak_mb += size_kb / 1024

    essential_mb = sum(s for s, _, _ in large_long) / 1024
    projected_peak_mb = max(0, peak_mb - waste_at_peak_mb)

    lines.append("## Parallelism 触发分析")
    lines.append(f"  HBM: {hbm_gb:.0f}GB  |  Peak: {peak_mb:,.0f}MB ({peak_mb/hbm_mb*100:.0f}% HBM)")
    lines.append(f"  大 tensor (>{large_gb:.0f}GB):")
    lines.append(f"    短生命周期 (<1s, 浪费):  {len(large_short)} 个 tensor, 共 {waste_total_mb:,.0f}MB")
    lines.append(f"      ↳ peak 时刻仍存活: {waste_at_peak_mb:,.0f}MB")
    lines.append(f"    长生命周期 (>1s, 必要): {len(large_long)} 个 tensor, {essential_mb:,.0f}MB")
    lines.append(f"  消除浪费后的预估 peak: {projected_peak_mb:,.0f}MB ({projected_peak_mb/hbm_mb*100:.0f}% HBM)")
    lines.append("")

    if projected_peak_mb / hbm_mb > PARALLELISM_RATIO:
        lines.append(f"  [SIGNAL] 预估 peak ({projected_peak_mb/hbm_mb*100:.0f}% HBM) 在消除浪费后仍超过 {PARALLELISM_RATIO*100:.0f}%。")
        lines.append("    - 可能需要 parallelism。需进行源码分析:")
        lines.append("      1. 阅读 parallel_design.md 了解拆分原则")
        lines.append("      2. 用 operator_details 的 Call Stack 在源码中定位大 tensor")
        lines.append("      3. 从计算结构中识别可 shard 的维度")
        lines.append("      4. 拆分后重新 profile 以验证")
        lines.append("    - Profiling 仅用于触发；拆分策略由源码决定。")
    elif waste_at_peak_mb > 0:
        lines.append(f"  [DEFINITE] Peak 时的浪费 = {waste_at_peak_mb:,.0f}MB。 "
                     f"消除后的预估 peak: {projected_peak_mb/hbm_mb*100:.0f}% HBM — 在单卡容量范围内。")
        lines.append("    - 无需 parallelism。优先进行内存优化（消除/复用）。")
    else:
        lines.append("  Peak 时无短生命周期大 tensor。内存不是 parallelism 触发因素。")
    lines.append("")

    # --- Peak 归因 (C10)：peak 时刻哪些 tensor 仍存活 ---
    if peak_alloc_time > 0 and all_tensors:
        alive = [(sz, nm) for sz, nm, a, r in all_tensors
                 if a <= peak_alloc_time <= r and sz > 0]
        alive.sort(key=lambda x: -x[0])
        alive_total_mb = sum(sz for sz, _ in alive) / 1024
        lines.append("## Peak 归因 (peak 时刻存活的 tensor)")
        lines.append(f"  Peak at ts={peak_alloc_time:.0f}us | {len(alive):,} 个 tensor 存活 | sum={alive_total_mb:,.0f}MB")
        lines.append(f"  按大小排序的 Top {min(top_k, len(alive))} (这些驱动 peak — 优先减少/复用它们):")
        for sz, nm in alive[:top_k]:
            lines.append(f"    {format_size_mb(sz):>10}  {nm[:40]}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hbm-gb", type=float, default=64.0,
                        help="HBM 容量，单位 GB（默认 64，对应 Ascend 910B）")
    parser.add_argument("--large-gb", type=float, default=5.0,
                        help="'大 tensor' 分类阈值，单位 GB（默认 5）")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank, args.top_k, args.hbm_gb, args.large_gb)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
