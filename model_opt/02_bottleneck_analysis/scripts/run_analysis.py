#!/usr/bin/env python3
"""Profiling 统一分析入口。

按顺序执行所有 parse 脚本，输出一份完整的分析报告。
各脚本的原始输出逐字保留，仅按逻辑分组。

用法:
    python run_analysis.py <L1 profiling 目录> [--l0-dir <L0目录>] [--rank N] [--output 报告路径]

报告结构:
    A. 全局视角 (step_trace, + L0 交叉验证)
    B. 设备侧：算子分布 (op_statistic)
    C. 设备侧：Kernel 级详情 (kernel_details)
    D. Host-Device 交互 (trace_view)
    E. 源码定位 (operator_details)
    F. 内存 (memory_record, operator_memory)
    G. CANN 运行时 (api_statistic)
    H. 通信 (communication, 仅多卡时存在)

默认行为: 报告自动保存到 L1 profiling 目录下的 analysis_report.txt。
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
    """执行单个 parse 函数，用章节标题包裹输出。"""
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
    parser.add_argument("profiling_dir", help="L1 profiling 输出目录")
    parser.add_argument("--l0-dir", default=None,
                        help="L0 profiling 目录（用于 L0/L1 交叉验证）")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output", "-o", default=None,
                        help="报告输出路径（默认: 保存到 L1 目录下 analysis_report.txt）")
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

    # 确定输出路径: 指定了 --output 则用指定路径，否则默认保存到 L1 目录下
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(l1_dir) / "analysis_report.txt"

    output_path.write_text(report, encoding="utf-8")
    print(f"分析报告已保存: {output_path}")
    print(report)


if __name__ == "__main__":
    main()
