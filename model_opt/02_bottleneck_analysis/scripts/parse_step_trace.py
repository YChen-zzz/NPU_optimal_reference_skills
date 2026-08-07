#!/usr/bin/env python3
"""解析 step_trace_time.csv — 每 step 的 device 利用率。

展示每 step 的 Computing / Free / Communication 时间占比。
用于判断 workload 是 Host-Bound、Compute-Bound 还是 Comm-Bound。

用法:
    python parse_step_trace.py <profiling_dir> [--rank N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, read_csv_all, safe_float


def parse(profiling_dir: str, rank=None) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "step_trace_time.csv"
    rows = read_csv_all(csv_path)

    if not rows:
        return f"[step_trace_time] 文件未找到: {csv_path}"

    lines = []
    lines.append("# Step Trace 耗时摘要")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"步数: {len(rows)}")
    lines.append("")

    computing_total = 0.0
    free_total = 0.0
    comm_total = 0.0

    step_data = []
    for row in rows:
        computing = safe_float(row.get("Computing", 0))
        free = safe_float(row.get("Free", 0))
        communication = safe_float(row.get("Communication", 0))
        preparing = safe_float(row.get("Preparing", 0))
        step_total = computing + free + communication

        computing_total += computing
        free_total += free
        comm_total += communication
        step_data.append((computing, free, communication, preparing, step_total))

    grand_total = computing_total + free_total + comm_total

    if grand_total > 0:
        util = computing_total / (computing_total + free_total) * 100 if (computing_total + free_total) > 0 else 0
        lines.append("## 总体")
        lines.append(f"  Device 利用率: {util:.1f}%  (Computing / (Computing + Free))")
        lines.append(f"  Computing: {computing_total/1000:.1f} ms ({computing_total/grand_total*100:.1f}%)")
        lines.append(f"  Free:      {free_total/1000:.1f} ms ({free_total/grand_total*100:.1f}%)")
        if comm_total > 0:
            lines.append(f"  Comm:      {comm_total/1000:.1f} ms ({comm_total/grand_total*100:.1f}%)")
        lines.append("")

        if util < threshold("step_trace", "severe_host_bound_util", 20):
            lines.append("  ** 严重 Host-Bound: device 空闲 >80%，瓶颈在 host 侧 **")
        elif util < threshold("step_trace", "moderate_host_bound_util", 50):
            lines.append("  ** 中度 Host-Bound: device 空闲 >50%，host 侧 overhead 显著 **")
        else:
            lines.append(f"  瓶颈在 device 侧（利用率 {util:.0f}%）")
            lines.append(f"  - 需 kernel 级分析区分 compute-bound 和 memory-bound")
        lines.append("")

        optimizable = free_total / grand_total * 100 if grand_total > 0 else 0
        lines.append(f"  理论下限 (= Computing): {computing_total/1000:.1f} ms")
        lines.append(f"  可优化空间: {optimizable:.1f}% ((Total - Computing) / Total)")
        if optimizable > threshold("step_trace", "large_optimizable_space", 30):
            lines.append(f"  - 可优化空间大。非 compute 的 overhead（dispatch/alloc/sync）显著；按此上限而非实现难度对候选排序")
        lines.append("")

        # 优化上限（C4, Amdahl 式，基于本 step 的时间拆分）
        lines.append("## 优化上限（按这些对候选排序）")
        lines.append(f"  Compute 下限（不可低于）: {computing_total/1000:.1f} ms ({computing_total/grand_total*100:.1f}%)")
        lines.append(f"  Host/dispatch 上限（可回收 Free）: {free_total/1000:.1f} ms ({free_total/grand_total*100:.1f}%)")
        if comm_total > 0:
            lines.append(f"  Communication 上限（可 overlap/消除）: {comm_total/1000:.1f} ms ({comm_total/grand_total*100:.1f}%)")
        if free_total >= comm_total and free_total > 0:
            lines.append(f"  - 最大上限 = host/dispatch (Free)。优先处理 host 侧问题。")
        elif comm_total > 0:
            lines.append(f"  - 最大上限 = communication。优先 comm-compute overlap / comm 减少。")
        lines.append("  子类别上限（更细拆分）:")
        lines.append("    sync vs alloc vs dispatch - operator_details 中 Host Time by Category 部分")
        lines.append("    fusible small-op 节省  - kernel_details 中 Fusible sequences 部分")
        lines.append("")

    if len(step_data) > 1:
        lines.append("## 每 step 拆分")
        has_preparing = any(p > 0 for _, _, _, p, _ in step_data)
        if has_preparing:
            header = f"{'Step':>5} {'Computing(ms)':>13} {'Free(ms)':>10} {'Comm(ms)':>10} {'Preparing(ms)':>14} {'Util%':>7}"
        else:
            header = f"{'Step':>5} {'Computing(ms)':>13} {'Free(ms)':>10} {'Comm(ms)':>10} {'Util%':>7}"
        lines.append(header)
        lines.append("-" * len(header))
        for idx, (comp, free, comm, prep, total) in enumerate(step_data):
            u = comp / (comp + free) * 100 if (comp + free) > 0 else 0
            if has_preparing:
                lines.append(f"{idx:>5} {comp/1000:>13.1f} {free/1000:>10.1f} {comm/1000:>10.1f} {prep/1000:>14.1f} {u:>6.1f}%")
            else:
                lines.append(f"{idx:>5} {comp/1000:>13.1f} {free/1000:>10.1f} {comm/1000:>10.1f} {u:>6.1f}%")
        lines.append("")

        if has_preparing:
            avg_prep = sum(p for _, _, _, p, _ in step_data) / len(step_data)
            avg_comp = computing_total / len(step_data)
            lines.append("## Preparing 分析")
            lines.append(f"  平均每 step Preparing: {avg_prep/1000:.1f} ms")
            lines.append(f"  平均每 step Computing: {avg_comp/1000:.1f} ms")
            if avg_prep > avg_comp:
                lines.append(f"  [SIGNAL] Preparing > Computing: 可能是真实 host 瓶颈或 profiler trace-writing overhead")
                lines.append(f"    Preparing 包含 Level1 profiler trace-writing 开销 — 仅凭此项无法确定。")
                lines.append(f"    交叉验证: 用 L0 重新采集 — 若 L0 下 Preparing 仍高则为真实 host 缺口，否则为 profiler 注入。")
            lines.append("")

    # --- 可疑信号 ---
    # 始终输出（单步推理仍会输出信号）；variance/spread
    # 在内部以 >1 step 为前提。
    step_utils = [c / (c + f) * 100 if (c + f) > 0 else 0 for c, f, _, _, _ in step_data]
    step_totals = [c + f + cm for c, f, cm, _, _ in step_data]

    lines.append("## 可疑信号")
    lines.append("  [DEFINITE]=可直接行动  [SIGNAL]=异常，根因未定 — 需结合其他 profiling 维度交叉验证")
    suspects_found = False

    # 单步推理提示（variance/spread 信号需要 >1 step）
    if len(step_data) == 1:
        lines.append(f"  [INFO] 单步推理 profile（{len(step_data)} step）— step variance/spread 信号未启用")
        lines.append(f"    使用上方的总体利用率与可优化空间；通过 trace_view / operator_details 交叉验证 host-bound 成因")
        suspects_found = True

    # step 间 variance
    if len(step_utils) > 1:
        util_min = min(step_utils)
        util_max = max(step_utils)
        if util_max - util_min > threshold("step_trace", "step_util_variance", 20):
            lines.append(f"  [SIGNAL] step 利用率 variance: {util_min:.1f}% ~ {util_max:.1f}%")
            lines.append(f"    某些 step 效率明显偏低 — 交叉验证: 检查 warmup/compilation/dynamic shape")
            suspects_found = True

    # step duration 异常值
    if len(step_totals) > 1:
        total_min = min(step_totals)
        total_max = max(step_totals)
        if total_max > total_min * threshold("step_trace", "step_duration_spread", 2.0):
            lines.append(f"  [SIGNAL] step duration spread: {total_min/1000:.1f}ms ~ {total_max/1000:.1f}ms ({total_max/total_min:.1f}x)")
            lines.append(f"    波动较大 — 交叉验证: 检查 trace_view 中 compile 事件是否集中在异常 step")
            suspects_found = True

    if not suspects_found:
        lines.append("  无 — 各 step 表现一致")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
