#!/usr/bin/env python3
"""解析 operator_details.csv — 源码定位工具。

该文件的独特价值在于 Call Stack 列，可将操作回溯到 Python 源码行。在其他脚本
识别出可疑项之后使用本脚本 — 来定位这些操作在源码中的触发位置。

此外还独特地提供逐算子的 Host Self Duration（host 侧 overhead），
这是 kernel_details.csv 所没有的。

两种模式：
- 默认：轻量 host overhead 概览（本文件独有的信息）
- Filter（--filter）：给定算子名，展示所有 Call Stack + host/device
  拆分，用于源码定位

用法:
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
from common import (threshold, find_ascend_profiler_output, stream_csv, safe_float,
                    clean_multiline_field, format_duration_ms)


def _parse_call_stack(raw: str) -> list:
    """将 Call Stack 字段解析为 frame 字符串列表。"""
    frames = [f.strip() for f in raw.replace("\r\n", "\n").replace("\r", "\n").split(";")
              if f.strip()]
    return [f.replace("\n", " ").strip() for f in frames if f.replace("\n", "").strip()]


def _host_category(name: str) -> str:
    """将 op 名分类为 host-time category（C1 拆解）。
    sync、dispatch、alloc 的优化方向相反，因此
    按 category 拆解总 host 时间可决定优化方向。
    规则为 thresholds.py 中的框架默认模式 — 按模型调整。"""
    rules = threshold("operator_details", "host_category_rules", {})
    if not rules:
        return "other"
    low = str(name).lower()
    for cat, patterns in rules.items():
        if any(p.lower() in low for p in patterns):
            return cat
    return "other"


def parse_overview(profiling_dir: str, rank=None, top_k: int = 15) -> str:
    """默认模式：host overhead 概览 — 本文件提供的独有信息。"""
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "operator_details.csv"

    if not csv_path.exists():
        return f"[operator_details] 文件未找到: {csv_path}"

    total_rows = 0
    total_host_us = 0.0
    total_device_us = 0.0
    total_host_total_us = 0.0  # inclusive host (self + children)

    # 按算子名聚合：聚焦 host 时间
    op_agg = defaultdict(lambda: {"count": 0, "host_us": 0.0, "device_us": 0.0,
                                   "device_total": 0.0, "device_aicpu": 0.0})

    # 按 category 拆解 host 时间 (C1): sync / alloc / H2D-copy / dispatch / framework / other
    cat_agg = defaultdict(float)
    # Layer 归因 (B4/C6): 按首个 project call-stack frame 的 inclusive Host Total
    layer_agg = defaultdict(lambda: {"host_total": 0.0, "count": 0})
    _LIB_MARKERS = ("site-packages", "dist-packages", "/lib/python", "torch/nn/modules",
                    "torch/_ops", "autograd/profiler", "torch_npu/profiler")

    for row in stream_csv(csv_path):
        total_rows += 1
        host_dur = safe_float(row.get("Host Self Duration(us)", 0))
        device_dur = safe_float(row.get("Device Self Duration(us)", 0))
        host_total = safe_float(row.get("Host Total Duration(us)", 0))
        device_total = safe_float(row.get("Device Total Duration(us)", 0))
        device_aicpu = safe_float(row.get("Device Self Duration With AICore(us)", 0))
        total_host_us += host_dur
        total_device_us += device_dur
        total_host_total_us += host_total

        name = row.get("Name", "?")
        op_agg[name]["count"] += 1
        op_agg[name]["host_us"] += host_dur
        op_agg[name]["device_us"] += device_dur
        op_agg[name]["device_total"] += device_total
        op_agg[name]["device_aicpu"] += device_aicpu

        # C1: 按算子名的 host category
        cat_agg[_host_category(name)] += host_dur

        # B4/C6: 通过首个 project frame 进行 layer 归因（inclusive Host Total）
        frames = _parse_call_stack(row.get("Call Stack", ""))
        proj_frame = next((f for f in frames if not any(m in f for m in _LIB_MARKERS)), None)
        if proj_frame:
            key = proj_frame[:90]
            layer_agg[key]["host_total"] += host_total if host_total > 0 else host_dur
            layer_agg[key]["count"] += 1

    if total_rows == 0:
        return f"[operator_details] 空文件: {csv_path}"

    lines = []
    lines.append("# Operator Details — Host Overhead 概览")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"总行数: {total_rows:,}")
    lines.append(f"总 host 时间: {total_host_us/1000:.1f} ms")
    lines.append(f"Device 总耗时: {total_device_us/1000:.1f} ms")
    lines.append(f"Host/Device 比例: {total_host_us/total_device_us:.1f}x" if total_device_us > 0 else "")
    lines.append("")

    # 按 HOST 时间排序的 Top op（此处独有）
    lines.append(f"## Top {top_k} 个 op (按 Host Self Duration)")
    lines.append("  (聚焦 host 侧 overhead — device 时间分析请用 kernel_details)")
    lines.append("")
    op_sorted_host = sorted(op_agg.items(), key=lambda x: -x[1]["host_us"])
    header = f"  {'Op Name':<35} {'Count':>8} {'Host(ms)':>10} {'Device(ms)':>10} {'H/D Ratio':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, info in op_sorted_host[:top_k]:
        ratio = info["host_us"] / info["device_us"] if info["device_us"] > 0 else float('inf')
        ratio_str = f"{ratio:.1f}x" if ratio < 10000 else "inf"
        lines.append(
            f"  {name:<35} {info['count']:>8} "
            f"{info['host_us']/1000:>10.1f} {info['device_us']/1000:>10.1f} "
            f"{ratio_str:>10}"
        )
    lines.append("")

    # 纯 host op（完全无 device 时间）
    pure_host = [(name, info) for name, info in op_sorted_host
                 if info["device_us"] == 0 and info["host_us"] > 0]
    pure_host_total = sum(info["host_us"] for _, info in pure_host)
    pure_host_pct = pure_host_total / total_host_us * 100 if total_host_us > 0 else 0

    lines.append(f"## 纯 Host Op (未触发 device kernel)")
    lines.append(f"  纯 host 总耗时: {pure_host_total/1000:.1f} ms ({pure_host_pct:.1f}% 占全部 host 时间)")
    lines.append(f"  这些是 metadata/framework 操作，不产生任何 device 工作。")
    lines.append("")
    header2 = f"  {'Op Name':<35} {'Count':>8} {'Host(ms)':>10}"
    lines.append(header2)
    lines.append("  " + "-" * (len(header2) - 2))
    for name, info in pure_host[:top_k]:
        lines.append(f"  {name:<35} {info['count']:>8} {info['host_us']/1000:>10.1f}")
    lines.append("")

    # --- 按 category 拆解 host 时间 (C1) ---
    # sync、dispatch、alloc 的修复方向相反；拆解以确定方向。
    if total_host_us > 0 and cat_agg:
        lines.append("## 按 Category 拆解 Host 时间")
        lines.append("  按算子 category 拆解总 host Self 时间 - 决定优化方向。")
        lines.append(f"  (total host self = {total_host_us/1000:.1f} ms)")
        for cat, us in sorted(cat_agg.items(), key=lambda x: -x[1]):
            pct = us / total_host_us * 100
            lines.append(f"  {cat:<24} {us/1000:>9.1f} ms  ({pct:>5.1f}%)")
        sync_us = sum(v for k, v in cat_agg.items() if k.startswith("sync"))
        if sync_us / total_host_us > 0.2:
            lines.append(f"  - sync (D-H) 占主导 ({sync_us/total_host_us*100:.0f}%): 消除 .item()/.numpy()，缓存/延迟 sync")
        lines.append("")

    # --- Layer 归因 (B4/C6) ---
    # 按首个 project call-stack frame 的 inclusive Host Total — 供 Line A 使用
    # layer 归因门槛（任何 layer 占 host 时间 >10% 都需有候选）。
    if layer_agg:
        lines.append("## 按 Call-Chain Layer 拆解 Host 时间 (inclusive Host Total)")
        lines.append("  每个 layer 的 inclusive host 开销 (Host Total, self+children)。Line A 门槛: layer 占 total >10% - 必须有候选。")
        layers_sorted = sorted(layer_agg.items(), key=lambda x: -x[1]["host_total"])
        denom = total_host_total_us if total_host_total_us > 0 else total_host_us
        for frame, info in layers_sorted[:top_k]:
            pct = info["host_total"] / denom * 100 if denom > 0 else 0
            lines.append(f"  {pct:>5.1f}%  {info['host_total']/1000:>9.1f} ms  ({info['count']:>5} 个 op)  {frame}")
        lines.append("")

    # 可疑信号
    lines.append("## 可疑信号")
    lines.append("  [DEFINITE]=可直接行动  [SIGNAL]=异常，根因未定 — 需结合其他 profiling 维度交叉验证")
    suspects_found = False

    if pure_host_pct > threshold("operator_details", "pure_host_pct", 50):
        lines.append(f"  - [DEFINITE] 纯 host op 占主导: {pure_host_pct:.0f}% 的 host 时间无 device 工作")
        lines.append(f"    - Framework/dispatch overhead 是主要 host 瓶颈")
        suspects_found = True

    high_ratio = [(name, info) for name, info in op_sorted_host
                  if info["host_us"] > info["device_us"] * threshold("operator_details", "extreme_hd_ratio", 10) and info["host_us"] > threshold("operator_details", "extreme_host_us", 5000)
                  and info["device_us"] > 0]
    if high_ratio:
        lines.append(f"  - [SIGNAL] host/device 比例极端的 op (host > 10x device, host > 5ms):")
        for name, info in high_ratio[:5]:
            lines.append(f"    {name}: host={info['host_us']/1000:.1f}ms vs device={info['device_us']/1000:.1f}ms")
        lines.append(f"    交叉验证: 用 --filter <op> 查看 Call Stack 源码位置，用 trace_view 查看 host dispatch 积压")
        suspects_found = True

    # A5: AI_CPU device 归因 — device 时间在 AI_CPU 上的 op（fallback）
    aicpu_ops = [(name, info) for name, info in op_sorted_host
                 if info["device_us"] > 0 and info["device_aicpu"] / info["device_us"] > 0.5
                 and info["device_aicpu"] > 1000]
    if aicpu_ops:
        lines.append(f"  - [DEFINITE] device 时间在 AI_CPU 上的 op (>50% 归因 AICore, fallback):")
        for name, info in aicpu_ops[:5]:
            lines.append(f"    {name}: device={info['device_us']/1000:.1f}ms  aicpu={info['device_aicpu']/1000:.1f}ms "
                         f"({info['device_aicpu']/info['device_us']*100:.0f}%)")
        lines.append(f"    - 用 AI Core impl 替换 / 更改 dtype。交叉验证 kernel_details 第 3 节 AI CPU Fallback。")
        suspects_found = True

    if not suspects_found:
        lines.append("  无")
    lines.append("")

    lines.append("## 下一步")
    lines.append("  使用 --filter <op_name> 深入特定算子并查看其 Call Stack。")
    lines.append("")

    return "\n".join(lines)


def parse_filtered(profiling_dir: str, filters: list, rank=None, top_k: int = 20) -> str:
    """过滤模式：展示特定 op 的 Call Stack — 源码定位。"""
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "operator_details.csv"

    if not csv_path.exists():
        return f"[operator_details] 文件未找到: {csv_path}"

    filter_lower = [f.lower() for f in filters]
    matched = []

    for row in stream_csv(csv_path):
        name = row.get("Name", "")
        if any(f in name.lower() for f in filter_lower):
            matched.append(row)

    lines = []
    lines.append("# Operator Details — 源码定位")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"过滤: {', '.join(filters)}")
    lines.append(f"匹配行数: {len(matched)}")
    lines.append("")

    if not matched:
        lines.append("没有算子匹配该过滤条件。")
        return "\n".join(lines)

    # 摘要
    total_host = sum(safe_float(r.get("Host Self Duration(us)", 0)) for r in matched)
    total_device = sum(safe_float(r.get("Device Self Duration(us)", 0)) for r in matched)
    lines.append("## 摘要")
    lines.append(f"  数量: {len(matched)}")
    lines.append(f"  总 host: {total_host/1000:.2f} ms")
    lines.append(f"  总 device: {total_device/1000:.2f} ms")
    lines.append(f"  平均 host: {total_host/len(matched):.1f} us")
    lines.append(f"  平均 device: {total_device/len(matched):.1f} us")
    lines.append("")

    # Call Stack 聚合：按唯一 stack 分组，展示频率
    stack_groups = defaultdict(lambda: {"count": 0, "host_us": 0.0, "device_us": 0.0,
                                         "example_shapes": []})
    for r in matched:
        raw_stack = r.get("Call Stack", "")
        frames = _parse_call_stack(raw_stack)
        # 用前 3 个 frame 作为分组键（足以识别唯一 call site）
        key = " | ".join(frames[:3]) if frames else "(无 stack)"
        stack_groups[key]["count"] += 1
        stack_groups[key]["host_us"] += safe_float(r.get("Host Self Duration(us)", 0))
        stack_groups[key]["device_us"] += safe_float(r.get("Device Self Duration(us)", 0))
        shapes = clean_multiline_field(r.get("Input Shapes", ""))
        if shapes and len(stack_groups[key]["example_shapes"]) < 3:
            if shapes not in stack_groups[key]["example_shapes"]:
                stack_groups[key]["example_shapes"].append(shapes)

    sorted_stacks = sorted(stack_groups.items(), key=lambda x: -x[1]["host_us"])

    lines.append(f"## Call Site (按 stack 分组，按 host 时间排序的 Top {min(top_k, len(sorted_stacks))})")
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
                        help="按算子名过滤（子串匹配）。"
                             "展示按 call site 分组的 Call Stack，用于源码定位。")
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
