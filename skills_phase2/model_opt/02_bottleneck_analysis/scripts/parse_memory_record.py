#!/usr/bin/env python3
"""解析 memory_record.csv — 内存使用时间线。

该文件记录随时间变化的内存状态，数据来自两个来源：
- APP 行：周期性采样（约每 20ms），仅含 Total Reserved
- PTA/PTA+GE 行：由每次 alloc/dealloc 事件触发，含 Allocated + Reserved + Active

关键指标：
- Total Reserved: allocator 池大小（阶梯式增长，很少收缩）
- Total Allocated: 实际 tensor 内存使用（随 alloc/free 波动）
- Reserved - Allocated: 池内空闲空间（fragmentation / headroom）

用法:
    python parse_memory_record.py <profiling_dir> [--rank N] [--buckets 20] [--top-k 10]
"""

import argparse
import heapq
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, stream_csv, safe_float


def parse(profiling_dir: str, rank=None, num_buckets: int = 20, top_k: int = 10) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "memory_record.csv"

    if not csv_path.exists():
        return f"[memory_record] 文件未找到: {csv_path}"

    # 收集记录，按可用数据分类
    reserved_records = []  # (ts, reserved) - 所有行
    allocated_records = []  # (ts, allocated, reserved) - 仅 PTA 行
    active_records = []     # (ts, active) - 含 Total Active 的行
    component_counts = defaultdict(int)
    component_peak = defaultdict(float)  # component -> max reserved (WORKSPACE vs tensor)

    for row in stream_csv(csv_path):
        ts = safe_float(row.get("Timestamp(us)", 0))
        reserved = safe_float(row.get("Total Reserved(MB)", 0))
        allocated = safe_float(row.get("Total Allocated(MB)", 0))
        active = safe_float(row.get("Total Active(MB)", 0))
        component = row.get("Component", "").strip()

        if ts <= 0:
            continue

        component_counts[component] += 1
        if reserved > component_peak[component]:
            component_peak[component] = reserved
        reserved_records.append((ts, reserved))

        if allocated > 0:
            allocated_records.append((ts, allocated, reserved))
        if active > 0:
            active_records.append((ts, active))

    if not reserved_records:
        return f"[memory_record] {csv_path} 中无有效记录"

    reserved_records.sort(key=lambda x: x[0])
    t0 = reserved_records[0][0]
    t_end = reserved_records[-1][0]
    duration_s = (t_end - t0) / 1e6

    lines = []
    lines.append("# Memory Record 摘要")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"总记录数: {len(reserved_records):,}")
    lines.append(f"时长: {duration_s:.2f}s")
    lines.append(f"Components: {dict(component_counts)}")
    lines.append("")

    # --- 0. 按 Component（WORKSPACE vs tensor 内存）---
    # WORKSPACE 是 operator workspace（可通过 tiling/env 控制），
    # 与 APP/PTA tensor 内存不同。分段以使可控部分可见。
    if len(component_peak) > 1:
        lines.append("## 0. 各 Component 的 Peak Reserved")
        lines.append("  WORKSPACE = operator workspace（可通过 tiling/env 控制）；APP/PTA(+GE) = tensor 内存。")
        for comp, peak in sorted(component_peak.items(), key=lambda x: -x[1]):
            lines.append(f"  {comp:<12} peak={peak:>10,.0f} MB  ({component_counts[comp]:,} records)")
        ws = component_peak.get("WORKSPACE", 0)
        if ws > 0:
            lines.append(f"  - WORKSPACE peak {ws:,.0f} MB 可独立控制（非 tensor 分配）。")
        lines.append("")

    # Active 内存（真实活跃集，与含 cache 的 Allocated 不同）
    if active_records:
        active_values = [a for _, a in active_records]
        lines.append(f"## 0a. Active 内存（真实活跃集）")
        lines.append(f"  记录数: {len(active_records):,}  |  Min: {min(active_values):,.0f} MB  |  Max: {max(active_values):,.0f} MB")
        lines.append(f"  Active < Allocated = 已缓存但可复用的 headroom。应使用 Active (而非 Allocated) 确定 batch-size 上限。")
        lines.append("")

    # --- 1. Reserved 内存（池大小）---
    res_values = [r[1] for r in reserved_records]
    max_res = max(res_values)
    min_res = min(res_values)
    max_res_idx = res_values.index(max_res)
    max_res_time = (reserved_records[max_res_idx][0] - t0) / 1e6

    lines.append("## 1. Reserved 内存 (allocator pool)")
    lines.append(f"  Min: {min_res:,.0f} MB")
    lines.append(f"  Max: {max_res:,.0f} MB  (at {max_res_time:.3f}s)")
    lines.append(f"  Range: {max_res - min_res:,.0f} MB")
    lines.append("")

    # Reserved 的分桶时间线
    if num_buckets > 0 and len(reserved_records) > 1:
        lines.append("  时间线（每桶最大 Reserved）:")
        bucket_width = (t_end - t0) / num_buckets
        buckets = [0.0] * num_buckets
        for ts, mem in reserved_records:
            idx = min(int((ts - t0) / bucket_width), num_buckets - 1)
            buckets[idx] = max(buckets[idx], mem)

        bar_width = 30
        scale = max_res if max_res > 0 else 1
        for i, bmax in enumerate(buckets):
            t_s = (i * bucket_width) / 1e6
            bar_len = int(bmax / scale * bar_width)
            bar = "█" * bar_len
            lines.append(f"    {t_s:>7.3f}s {bar:<{bar_width}} {bmax:>8,.0f} MB")
        lines.append("")

    # --- 2. Allocated 内存（实际 tensor 使用）---
    if allocated_records:
        allocated_records.sort(key=lambda x: x[0])
        alloc_values = [r[1] for r in allocated_records]
        max_alloc = max(alloc_values)
        min_alloc = min(alloc_values)
        max_alloc_idx = alloc_values.index(max_alloc)
        max_alloc_time = (allocated_records[max_alloc_idx][0] - t0) / 1e6

        lines.append("## 2. Allocated 内存（实际 tensor 使用）")
        lines.append(f"  含 Allocated 数据的记录数: {len(allocated_records):,}")
        lines.append(f"  Min: {min_alloc:,.0f} MB")
        lines.append(f"  Max: {max_alloc:,.0f} MB  (at {max_alloc_time:.3f}s)")
        lines.append(f"  Range: {max_alloc - min_alloc:,.0f} MB")
        lines.append("")

        # --- 3. Fragmentation: Reserved - Allocated ---
        frag_values = [r[2] - r[1] for r in allocated_records]
        max_frag = max(frag_values)
        min_frag = min(frag_values)
        avg_frag = sum(frag_values) / len(frag_values)

        lines.append("## 3. 池 Fragmentation (Reserved - Allocated)")
        lines.append(f"  Min gap: {min_frag:,.0f} MB")
        lines.append(f"  Max gap: {max_frag:,.0f} MB")
        lines.append(f"  Avg gap: {avg_frag:,.0f} MB")
        if max_frag > threshold("memory_record", "frag_gap_mb", 1000):
            lines.append(f"  - 大 gap ({max_frag:,.0f}MB) 表明存在显著的池 fragmentation 或过度预留")
        lines.append("")

        # 分桶 fragmentation 时间线
        if num_buckets > 0 and len(allocated_records) > 1:
            alloc_t0 = allocated_records[0][0]
            alloc_tend = allocated_records[-1][0]
            if alloc_tend > alloc_t0:
                lines.append("  Fragmentation 时间线（每桶最大 gap）:")
                bw = (alloc_tend - alloc_t0) / num_buckets
                frag_buckets = [0.0] * num_buckets
                for ts, alloc, res in allocated_records:
                    bidx = min(int((ts - alloc_t0) / bw), num_buckets - 1)
                    frag_buckets[bidx] = max(frag_buckets[bidx], res - alloc)

                frag_scale = max_frag if max_frag > 0 else 1
                for i, fmax in enumerate(frag_buckets):
                    t_s = (i * bw) / 1e6
                    bar_len = int(fmax / frag_scale * bar_width) if frag_scale > 0 else 0
                    bar = "█" * bar_len
                    lines.append(f"    {t_s:>7.3f}s {bar:<{bar_width}} {fmax:>8,.0f} MB")
                lines.append("")
    else:
        lines.append("## 2. Allocated 内存")
        lines.append("  (无 Allocated 数据 — 仅存在 APP 采样行)")
        lines.append("")

    # --- 4. Top 跳变 ---
    jumps = []
    for i in range(1, len(reserved_records)):
        delta = reserved_records[i][1] - reserved_records[i - 1][1]
        if delta != 0:
            jumps.append((abs(delta), delta, reserved_records[i][0], reserved_records[i][1]))

    top_allocs = heapq.nlargest(top_k, (j for j in jumps if j[1] > 0), key=lambda x: x[0])
    top_deallocs = heapq.nlargest(top_k, (j for j in jumps if j[1] < 0), key=lambda x: x[0])

    lines.append(f"## 4. Top {top_k} Reserved 跳变")
    if top_allocs:
        lines.append("  最大增长（池扩张）:")
        header = f"    {'Time(s)':>9} {'Delta(MB)':>12} {'After(MB)':>12}"
        lines.append(header)
        for _, delta, ts, mem in top_allocs:
            rel = (ts - t0) / 1e6
            lines.append(f"    {rel:>9.3f} {delta:>+12,.0f} {mem:>12,.0f}")
    if top_deallocs:
        lines.append("  最大减少（池收缩）:")
        for _, delta, ts, mem in top_deallocs:
            rel = (ts - t0) / 1e6
            lines.append(f"    {rel:>9.3f} {delta:>+12,.0f} {mem:>12,.0f}")
    lines.append("")

    # --- 可疑信号 ---
    lines.append("## 可疑信号")
    lines.append("  [DEFINITE]=可直接行动  [SIGNAL]=异常，根因未定 — 需结合其他 profiling 维度交叉验证")
    suspects_found = False

    # 增长趋势
    n_records = len(reserved_records)
    if n_records > threshold("memory_record", "growth_min_records", 20):
        early = res_values[:n_records // 10]
        late = res_values[-(n_records // 10):]
        early_avg = sum(early) / len(early)
        late_avg = sum(late) / len(late)
        growth = late_avg - early_avg
        if growth > threshold("memory_record", "growth_mb", 100):
            lines.append(f"  - [SIGNAL] Reserved 增长趋势：早期均值 {early_avg:,.0f}MB - 末期均值 {late_avg:,.0f}MB (+{growth:,.0f}MB)")
            lines.append(f"    交叉验证: 检查 operator_memory 是否有未释放的 tensor 累积")
            suspects_found = True

    # 高 churn
    if jumps:
        large_jumps = [j for j in jumps if j[0] > threshold("memory_record", "churn_jump_mb", 50)]
        if len(large_jumps) > threshold("memory_record", "churn_count", 20):
            lines.append(f"  - [SIGNAL] 高内存 churn: {len(large_jumps)} 次跳变 > 50MB")
            lines.append(f"    交叉验证: 在 operator_memory 中查找重复的等大小 alloc - buffer 复用机会")
            suspects_found = True

    # Fragmentation 随时间增长
    if allocated_records and len(allocated_records) > 20:
        n_alloc = len(allocated_records)
        early_frag = [r[2] - r[1] for r in allocated_records[:n_alloc // 10]]
        late_frag = [r[2] - r[1] for r in allocated_records[-(n_alloc // 10):]]
        early_frag_avg = sum(early_frag) / len(early_frag)
        late_frag_avg = sum(late_frag) / len(late_frag)
        frag_growth = late_frag_avg - early_frag_avg
        if frag_growth > threshold("memory_record", "frag_growth_mb", 50):
            lines.append(f"  - [SIGNAL] Fragmentation 增长：早期 gap 均值 {early_frag_avg:,.0f}MB - 末期 {late_frag_avg:,.0f}MB (+{frag_growth:,.0f}MB)")
            suspects_found = True

    # OOM 风险
    if max_res > threshold("memory_record", "oom_risk_mb", 60000):
        lines.append(f"  - [DEFINITE] Peak reserved {max_res:,.0f}MB — 接近 HBM 容量，存在 OOM 风险")
        suspects_found = True

    if not suspects_found:
        lines.append("  无 — 内存使用表现稳定")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--buckets", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank, args.buckets, args.top_k)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
