#!/usr/bin/env python3
"""Unified profiling analysis entry point.

Runs all parse scripts in sequence and outputs a single organized report.
Does NOT combine, filter, or pattern-match across script outputs — each
script's raw output is preserved verbatim, only grouped into logical sections.

Usage:
    python run_analysis.py <l1_profiling_dir> [--l0-dir <l0_dir>] [--rank N] [--output report.txt]

The report is organized into sections by analysis perspective:
    A. 全局视角 (step_trace, + L0 cross-check if --l0-dir provided)
    B. 设备侧：算子分布 (op_statistic)
    C. 设备侧：Kernel 级详情 (kernel_details)
    D. Host-Device 交互 (trace_view)
    E. 源码定位 (operator_details)
    F. 内存 (memory_record, operator_memory)
    G. CANN 运行时 (api_statistic)
    H. 通信 (communication, only if communication.json exists)
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import find_ascend_profiler_output

import parse_step_trace
import parse_op_statistic
import parse_kernel_details
import parse_trace_view
import parse_operator_details
import parse_memory_record
import parse_operator_memory
import parse_api_statistic
import parse_communication


DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


def run_section(title: str, func, *args, **kwargs) -> str:
    """Run a parse function and wrap its output in a section header."""
    lines = [SUB_DIVIDER, f"--- {title} ---", ""]
    try:
        result = func(*args, **kwargs)
        lines.append(result.rstrip())
    except Exception as e:
        lines.append(f"[ERROR] {func.__module__}.{func.__name__} failed: {e}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("profiling_dir", help="L1 profiling output directory")
    parser.add_argument("--l0-dir", default=None,
                        help="L0 profiling directory for L0/L1 cross-validation")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output", "-o", default=None,
                        help="Write report to file (default: stdout)")
    args = parser.parse_args()

    l1_dir = args.profiling_dir
    rank = args.rank

    sections = []
    sections.append(DIVIDER)
    sections.append("=== Phase 2 Profiling 分析报告 ===")
    sections.append(f"L1 目录: {l1_dir}")
    if args.l0_dir:
        sections.append(f"L0 参考目录: {args.l0_dir}")
    else:
        sections.append("L0 参考目录: 未提供（L1 结论未经交叉验证，须谨慎）")
    if rank is not None:
        sections.append(f"Rank: {rank}")
    sections.append("")

    # --- A. 全局视角 ---
    section_a = [SUB_DIVIDER, "--- A. 全局视角 ---", ""]

    section_a.append("[L1] " + parse_step_trace.parse(l1_dir, rank).rstrip())

    if args.l0_dir:
        section_a.append("")
        section_a.append("[L0] " + parse_step_trace.parse(args.l0_dir, rank).rstrip())
        section_a.append("")
        section_a.append("⚠ L0/L1 交叉验证：对比上方 L0 和 L1 的 Computing%/Free%/Utilization。")
        section_a.append("  若 L1 Utilization 显著低于 L0（差 >20pp），L1 的低利用率可能由")
        section_a.append("  profiler barrier 注入导致。瓶颈类型判定以 L0 为准；L1 的算子级")
        section_a.append("  数据（op_statistic、kernel_details 等）仍然有效。")

    section_a.append("")
    sections.append("\n".join(section_a))

    # --- B. 设备侧：算子分布 ---
    sections.append(run_section(
        "B. 设备侧：算子分布",
        parse_op_statistic.parse, l1_dir, rank,
    ))

    # --- C. 设备侧：Kernel 级详情 ---
    sections.append(run_section(
        "C. 设备侧：Kernel 级详情",
        parse_kernel_details.parse, l1_dir, rank,
    ))

    # --- D. Host-Device 交互 ---
    ascend_dir = find_ascend_profiler_output(l1_dir, rank)
    trace_view_path = ascend_dir / "trace_view.json"
    if trace_view_path.exists():
        sections.append(run_section(
            "D. Host-Device 交互",
            parse_trace_view.parse, trace_view_path, 15, 50.0,
        ))
    else:
        sections.append(f"{SUB_DIVIDER}\n--- D. Host-Device 交互 ---\n\n[trace_view] File not found: {trace_view_path}\n")

    # --- E. 源码定位 ---
    sections.append(run_section(
        "E. 源码定位",
        parse_operator_details.parse_overview, l1_dir, rank,
    ))

    # --- F. 内存 ---
    section_f = [SUB_DIVIDER, "--- F. 内存 ---", ""]
    try:
        section_f.append(parse_memory_record.parse(l1_dir, rank).rstrip())
    except Exception as e:
        section_f.append(f"[ERROR] parse_memory_record failed: {e}")
    section_f.append("")
    try:
        section_f.append(parse_operator_memory.parse(l1_dir, rank).rstrip())
    except Exception as e:
        section_f.append(f"[ERROR] parse_operator_memory failed: {e}")
    section_f.append("")
    sections.append("\n".join(section_f))

    # --- G. CANN 运行时 ---
    sections.append(run_section(
        "G. CANN 运行时",
        parse_api_statistic.parse, l1_dir, rank,
    ))

    # --- H. 通信（仅多卡）---
    comm_path = ascend_dir / "communication.json"
    matrix_path = ascend_dir / "communication_matrix.json"
    if comm_path.exists():
        sections.append(run_section(
            "H. 通信（多卡）",
            parse_communication.parse, comm_path, matrix_path, 15,
        ))

    sections.append(DIVIDER)

    report = "\n".join(sections)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
