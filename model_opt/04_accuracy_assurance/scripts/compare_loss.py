#!/usr/bin/env python3
"""比较两份文本日志中的标量训练信号。

脚本会从自由格式日志中提取 `loss`、`val_loss`、`test_loss` 等指标，
汇总两侧结果，并按 step 或索引对齐后进行比较。

示例
----
python compare_loss.py baseline.log ascend.log
python compare_loss.py baseline.log ascend.log --metric loss --metric val_loss
python compare_loss.py baseline.log ascend.log --metric train_loss \
  --pattern train_loss='train_loss\\s*=\\s*(?P<value>[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)'
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
STEP_PATTERNS = (
    re.compile(r"(?i)\bstep\s*[:=]\s*(\d+)"),
    re.compile(r"(?i)\biter(?:ation)?\s*[:=]\s*(\d+)"),
    re.compile(r"(?i)\bglobal[_ ]step\s*[:=]\s*(\d+)"),
    re.compile(r"(?i)\bepoch\s*[:=]\s*(\d+)"),
)


@dataclass
class Sample:
    index: int
    line: int
    value: float
    step: Optional[int]


def build_default_patterns(metric: str) -> List[re.Pattern[str]]:
    escaped = re.escape(metric)
    boundary = rf"(?<![\w/]){escaped}(?![\w/])"
    return [
        re.compile(rf"{boundary}\s*[:=]\s*(?P<value>{NUMBER_PATTERN})", re.IGNORECASE),
        re.compile(rf"{boundary}\s+(?P<value>{NUMBER_PATTERN})", re.IGNORECASE),
    ]


def parse_custom_patterns(raw_patterns: Sequence[str]) -> Dict[str, List[re.Pattern[str]]]:
    custom_patterns: Dict[str, List[re.Pattern[str]]] = {}
    for raw_pattern in raw_patterns:
        if "=" not in raw_pattern:
            raise ValueError(
                f"无效的 --pattern 参数 '{raw_pattern}'，格式应为 metric=regex。"
            )
        metric, regex = raw_pattern.split("=", 1)
        metric = metric.strip()
        regex = regex.strip()
        if not metric or not regex:
            raise ValueError(
                f"无效的 --pattern 参数 '{raw_pattern}'，格式应为 metric=regex。"
            )
        try:
            compiled = re.compile(regex, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"指标 '{metric}' 的正则表达式无效：{exc}") from exc
        custom_patterns.setdefault(metric, []).append(compiled)
    return custom_patterns


def extract_step(line: str) -> Optional[int]:
    for pattern in STEP_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def extract_value(match: re.Match[str]) -> float:
    if "value" in match.re.groupindex:
        return float(match.group("value"))
    if match.lastindex:
        return float(match.group(1))
    raise ValueError("正则必须包含名为 'value' 的分组，或至少包含一个捕获分组。")


def load_series(
    path: Path,
    metrics: Sequence[str],
    patterns: Dict[str, List[re.Pattern[str]]],
    encoding: str,
) -> Dict[str, List[Sample]]:
    text = path.read_text(encoding=encoding, errors="ignore")
    metric_series: Dict[str, List[Sample]] = {metric: [] for metric in metrics}
    next_index: Dict[str, int] = {metric: 1 for metric in metrics}

    for line_number, line in enumerate(text.splitlines(), start=1):
        step = extract_step(line)
        for metric in metrics:
            for pattern in patterns[metric]:
                match = pattern.search(line)
                if not match:
                    continue
                metric_series[metric].append(
                    Sample(
                        index=next_index[metric],
                        line=line_number,
                        value=extract_value(match),
                        step=step,
                    )
                )
                next_index[metric] += 1
                break

    return metric_series


def summarize(series: Sequence[Sample]) -> Dict[str, float]:
    values = [sample.value for sample in series]
    if not values:
        return {"count": 0}

    summary = {
        "count": len(values),
        "final": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }
    summary["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def relative_diff(reference: float, other: float) -> Optional[float]:
    scale = abs(reference)
    if scale < 1e-12:
        return None
    return abs(other - reference) / scale


def align_series(
    baseline: Sequence[Sample],
    candidate: Sequence[Sample],
) -> Tuple[str, List[Tuple[object, float, float]]]:
    baseline_by_step = {sample.step: sample.value for sample in baseline if sample.step is not None}
    candidate_by_step = {sample.step: sample.value for sample in candidate if sample.step is not None}
    shared_steps = sorted(set(baseline_by_step) & set(candidate_by_step))

    if len(shared_steps) >= 3:
        return (
            "shared_step",
            [
                (step, baseline_by_step[step], candidate_by_step[step])
                for step in shared_steps
            ],
        )

    overlap = min(len(baseline), len(candidate))
    return (
        "index",
        [
            (index + 1, baseline[index].value, candidate[index].value)
            for index in range(overlap)
        ],
    )


def compare_aligned(
    aligned: Sequence[Tuple[object, float, float]]
) -> Dict[str, Optional[float]]:
    if not aligned:
        return {
            "aligned_count": 0,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "tail_mean_abs_diff": None,
        }

    abs_diffs = [abs(candidate - baseline) for _, baseline, candidate in aligned]
    tail_size = max(3, min(20, len(aligned) // 5 or 1))
    tail = aligned[-tail_size:]
    tail_abs_diffs = [abs(candidate - baseline) for _, baseline, candidate in tail]
    return {
        "aligned_count": len(aligned),
        "mean_abs_diff": statistics.fmean(abs_diffs),
        "max_abs_diff": max(abs_diffs),
        "tail_mean_abs_diff": statistics.fmean(tail_abs_diffs),
    }


def format_alignment_mode(mode: str) -> str:
    mapping = {
        "shared_step": "共享 step 对齐",
        "index": "按出现顺序对齐",
    }
    return mapping.get(mode, mode)


def write_csv_exports(
    output_dir: Path,
    side_name: str,
    metric_series: Dict[str, List[Sample]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, series in metric_series.items():
        output_path = output_dir / f"{side_name}.{metric}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "step", "line", "value"])
            for sample in series:
                writer.writerow([sample.index, sample.step, sample.line, sample.value])


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if math.isfinite(value):
        return f"{value:.8g}"
    return str(value)


def build_report(
    metrics: Sequence[str],
    baseline_series: Dict[str, List[Sample]],
    candidate_series: Dict[str, List[Sample]],
) -> Dict[str, object]:
    report: Dict[str, object] = {"metrics": {}}
    for metric in metrics:
        baseline_summary = summarize(baseline_series[metric])
        candidate_summary = summarize(candidate_series[metric])
        alignment_mode, aligned = align_series(baseline_series[metric], candidate_series[metric])
        aligned_summary = compare_aligned(aligned)

        baseline_final = baseline_summary.get("final")
        candidate_final = candidate_summary.get("final")
        final_abs_diff = None
        final_rel_diff = None
        if baseline_final is not None and candidate_final is not None:
            final_abs_diff = abs(candidate_final - baseline_final)
            final_rel_diff = relative_diff(baseline_final, candidate_final)

        report["metrics"][metric] = {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "alignment_mode": alignment_mode,
            "aligned": aligned_summary,
            "final_abs_diff": final_abs_diff,
            "final_rel_diff": final_rel_diff,
        }
    return report


def render_text_report(report: Dict[str, object]) -> str:
    lines: List[str] = []
    metrics = report["metrics"]
    for metric, details in metrics.items():
        baseline = details["baseline"]
        candidate = details["candidate"]
        aligned = details["aligned"]

        lines.append(f"指标: {metric}")
        if baseline["count"] == 0 and candidate["count"] == 0:
            lines.append("  两份日志中都没有找到该指标。")
            lines.append("")
            continue

        lines.append(
            "  基线侧: "
            f"count={baseline['count']} final={format_float(baseline.get('final'))} "
            f"mean={format_float(baseline.get('mean'))} stdev={format_float(baseline.get('stdev'))} "
            f"min={format_float(baseline.get('min'))} max={format_float(baseline.get('max'))}"
        )
        lines.append(
            "  待比对侧: "
            f"count={candidate['count']} final={format_float(candidate.get('final'))} "
            f"mean={format_float(candidate.get('mean'))} stdev={format_float(candidate.get('stdev'))} "
            f"min={format_float(candidate.get('min'))} max={format_float(candidate.get('max'))}"
        )
        lines.append(
            "  最终值差异: "
            f"abs={format_float(details['final_abs_diff'])} "
            f"rel={format_float(details['final_rel_diff'])}"
        )
        lines.append(
            "  对齐点统计: "
            f"mode={format_alignment_mode(details['alignment_mode'])} "
            f"count={aligned['aligned_count']} "
            f"mean_abs_diff={format_float(aligned['mean_abs_diff'])} "
            f"tail_mean_abs_diff={format_float(aligned['tail_mean_abs_diff'])} "
            f"max_abs_diff={format_float(aligned['max_abs_diff'])}"
        )
        lines.append("")

    return "\n".join(lines).rstrip()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="可信基线日志路径")
    parser.add_argument("candidate", type=Path, help="昇腾侧或待比较日志路径")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="要提取的指标名。可重复传入多个，默认是 loss。",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="自定义指标匹配规则，格式为 metric=regex。正则需包含 'value' 分组或至少一个捕获分组。",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="读取日志文件时使用的编码，默认 utf-8。",
    )
    parser.add_argument(
        "--export-csv-dir",
        type=Path,
        help="可选。将提取后的序列导出为 CSV 的目录。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出完整报告。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    metrics = args.metric or ["loss"]
    try:
        custom_patterns = parse_custom_patterns(args.pattern)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    patterns: Dict[str, List[re.Pattern[str]]] = {}
    for metric in metrics:
        patterns[metric] = custom_patterns.get(metric, build_default_patterns(metric))

    try:
        baseline_series = load_series(args.baseline, metrics, patterns, args.encoding)
        candidate_series = load_series(args.candidate, metrics, patterns, args.encoding)
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.export_csv_dir:
        write_csv_exports(args.export_csv_dir, "baseline", baseline_series)
        write_csv_exports(args.export_csv_dir, "candidate", candidate_series)

    report = build_report(metrics, baseline_series, candidate_series)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text_report(report))

    metrics_with_values = [
        metric
        for metric in metrics
        if baseline_series[metric] or candidate_series[metric]
    ]
    if not metrics_with_values:
        print(
            "警告: 没有提取到任何指标值，请检查 --metric 或 --pattern 参数",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
