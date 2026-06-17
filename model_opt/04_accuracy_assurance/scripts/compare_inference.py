#!/usr/bin/env python3
"""比较推理输出、指标文件或结果目录。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_FILE_EXTS = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".log"}


@dataclass
class CompareStats:
    numeric_total: int = 0
    numeric_match: int = 0
    numeric_mismatch: int = 0
    exact_total: int = 0
    exact_match: int = 0
    exact_mismatch: int = 0
    missing_left: int = 0
    missing_right: int = 0
    length_mismatch: int = 0
    mismatches: List[Dict[str, object]] = field(default_factory=list)

    def add_mismatch(self, entry: Dict[str, object], max_mismatches: int) -> None:
        if len(self.mismatches) < max_mismatches:
            self.mismatches.append(entry)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_text_lines(text: str) -> List[str]:
    return [line.rstrip() for line in text.splitlines()]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def load_json(path: Path) -> object:
    return json.loads(read_text_file(path))


def load_jsonl(path: Path) -> List[object]:
    items: List[object] = []
    for line in read_text_file(path).splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def load_csv_like(path: Path) -> object:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    lower_fields = [field.lower() for field in fieldnames]
    if len(fieldnames) == 2 and lower_fields[0] in {"metric", "name", "key"} and lower_fields[1] in {"value", "score"}:
        metrics: Dict[str, object] = {}
        for row in rows:
            metrics[str(row[fieldnames[0]])] = coerce_scalar(row[fieldnames[1]])
        return metrics

    if len(rows) == 1:
        metrics = {}
        numeric_fields = 0
        for field in fieldnames:
            value = coerce_scalar(rows[0][field])
            if is_number(value):
                numeric_fields += 1
            metrics[field] = value
        if numeric_fields > 0:
            return metrics

    return [{field: coerce_scalar(row[field]) for field in fieldnames} for row in rows]


def load_text(path: Path) -> List[str]:
    return normalize_text_lines(read_text_file(path))


def coerce_scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(marker in text for marker in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def detect_mode(path: Path) -> str:
    if path.is_dir():
        return "dir"
    ext = path.suffix.lower()
    if ext == ".json":
        return "json"
    if ext == ".jsonl":
        return "jsonl"
    if ext in {".csv", ".tsv"}:
        return "csv"
    return "text"


def load_by_mode(path: Path, mode: str) -> object:
    if mode == "json":
        return load_json(path)
    if mode == "jsonl":
        return load_jsonl(path)
    if mode == "csv":
        return load_csv_like(path)
    if mode == "text":
        return load_text(path)
    raise ValueError(f"不支持的比较模式: {mode}")


def compare_values(
    left: object,
    right: object,
    path: str,
    stats: CompareStats,
    atol: float,
    rtol: float,
    max_mismatches: int,
) -> None:
    if is_number(left) and is_number(right):
        stats.numeric_total += 1
        abs_diff = abs(float(left) - float(right))
        rel_base = abs(float(left))
        rel_diff = None if rel_base < 1e-12 else abs_diff / rel_base
        if math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol):
            stats.numeric_match += 1
        else:
            stats.numeric_mismatch += 1
            stats.add_mismatch(
                {
                    "path": path,
                    "reason": "numeric_mismatch",
                    "left": left,
                    "right": right,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                },
                max_mismatches,
            )
        return

    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        for key in sorted(left_keys - right_keys):
            stats.missing_right += 1
            stats.add_mismatch(
                {"path": f"{path}.{key}" if path else str(key), "reason": "missing_right", "left": left[key], "right": None},
                max_mismatches,
            )
        for key in sorted(right_keys - left_keys):
            stats.missing_left += 1
            stats.add_mismatch(
                {"path": f"{path}.{key}" if path else str(key), "reason": "missing_left", "left": None, "right": right[key]},
                max_mismatches,
            )
        for key in sorted(left_keys & right_keys):
            child_path = f"{path}.{key}" if path else str(key)
            compare_values(left[key], right[key], child_path, stats, atol, rtol, max_mismatches)
        return

    if isinstance(left, list) and isinstance(right, list):
        min_len = min(len(left), len(right))
        if len(left) != len(right):
            stats.length_mismatch += 1
            stats.add_mismatch(
                {"path": path or "$", "reason": "length_mismatch", "left": len(left), "right": len(right)},
                max_mismatches,
            )
        for index in range(min_len):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            compare_values(left[index], right[index], child_path, stats, atol, rtol, max_mismatches)
        for index in range(min_len, len(left)):
            stats.missing_right += 1
            stats.add_mismatch(
                {"path": f"{path}[{index}]" if path else f"[{index}]", "reason": "missing_right", "left": left[index], "right": None},
                max_mismatches,
            )
        for index in range(min_len, len(right)):
            stats.missing_left += 1
            stats.add_mismatch(
                {"path": f"{path}[{index}]" if path else f"[{index}]", "reason": "missing_left", "left": None, "right": right[index]},
                max_mismatches,
            )
        return

    stats.exact_total += 1
    if left == right:
        stats.exact_match += 1
    else:
        stats.exact_mismatch += 1
        stats.add_mismatch(
            {"path": path or "$", "reason": "exact_mismatch", "left": left, "right": right},
            max_mismatches,
        )


def summarize_stats(kind: str, stats: CompareStats) -> Dict[str, object]:
    return {
        "kind": kind,
        "numeric_total": stats.numeric_total,
        "numeric_match": stats.numeric_match,
        "numeric_mismatch": stats.numeric_mismatch,
        "exact_total": stats.exact_total,
        "exact_match": stats.exact_match,
        "exact_mismatch": stats.exact_mismatch,
        "missing_left": stats.missing_left,
        "missing_right": stats.missing_right,
        "length_mismatch": stats.length_mismatch,
        "mismatches": stats.mismatches,
    }


def compare_file(
    left_path: Path,
    right_path: Path,
    forced_mode: str,
    atol: float,
    rtol: float,
    max_mismatches: int,
) -> Dict[str, object]:
    mode = forced_mode if forced_mode != "auto" else detect_mode(left_path)
    left = load_by_mode(left_path, mode)
    right = load_by_mode(right_path, mode)
    stats = CompareStats()
    compare_values(left, right, "", stats, atol, rtol, max_mismatches)
    return {
        "left": str(left_path),
        "right": str(right_path),
        "summary": summarize_stats(mode, stats),
    }


def collect_supported_files(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_EXTS:
            rel = str(path.relative_to(root)).replace("\\", "/")
            mapping[rel] = path
    return mapping


def compare_directories(
    left_root: Path,
    right_root: Path,
    atol: float,
    rtol: float,
    max_mismatches: int,
) -> Dict[str, object]:
    left_files = collect_supported_files(left_root)
    right_files = collect_supported_files(right_root)

    shared = sorted(set(left_files) & set(right_files))
    left_only = sorted(set(left_files) - set(right_files))
    right_only = sorted(set(right_files) - set(left_files))

    reports: List[Dict[str, object]] = []
    aggregate = CompareStats()
    identical_files = 0

    for rel in shared:
        report = compare_file(left_files[rel], right_files[rel], "auto", atol, rtol, max_mismatches)
        summary = report["summary"]
        reports.append({"path": rel, **summary})
        aggregate.numeric_total += summary["numeric_total"]
        aggregate.numeric_match += summary["numeric_match"]
        aggregate.numeric_mismatch += summary["numeric_mismatch"]
        aggregate.exact_total += summary["exact_total"]
        aggregate.exact_match += summary["exact_match"]
        aggregate.exact_mismatch += summary["exact_mismatch"]
        aggregate.missing_left += summary["missing_left"]
        aggregate.missing_right += summary["missing_right"]
        aggregate.length_mismatch += summary["length_mismatch"]
        for mismatch in summary["mismatches"]:
            aggregate.add_mismatch({"file": rel, **mismatch}, max_mismatches)
        if (
            summary["numeric_mismatch"] == 0
            and summary["exact_mismatch"] == 0
            and summary["missing_left"] == 0
            and summary["missing_right"] == 0
            and summary["length_mismatch"] == 0
        ):
            identical_files += 1

    return {
        "kind": "dir",
        "left": str(left_root),
        "right": str(right_root),
        "shared_files": len(shared),
        "identical_files": identical_files,
        "left_only": left_only,
        "right_only": right_only,
        "files": reports,
        "summary": summarize_stats("dir", aggregate),
    }


def render_file_report(report: Dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        f"左侧: {report['left']}",
        f"右侧: {report['right']}",
        f"比较类型: {summary['kind']}",
        f"数值项: total={summary['numeric_total']} match={summary['numeric_match']} mismatch={summary['numeric_mismatch']}",
        f"精确项: total={summary['exact_total']} match={summary['exact_match']} mismatch={summary['exact_mismatch']}",
        f"缺失项: missing_left={summary['missing_left']} missing_right={summary['missing_right']} length_mismatch={summary['length_mismatch']}",
    ]
    if summary["mismatches"]:
        lines.append("差异样例:")
        for mismatch in summary["mismatches"]:
            detail = f"  - path={mismatch['path']} reason={mismatch['reason']} left={mismatch.get('left')} right={mismatch.get('right')}"
            if "abs_diff" in mismatch:
                detail += f" abs_diff={mismatch.get('abs_diff')} rel_diff={mismatch.get('rel_diff')}"
            lines.append(detail)
    return "\n".join(lines)


def render_dir_report(report: Dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        f"左目录: {report['left']}",
        f"右目录: {report['right']}",
        f"共同文件: {report['shared_files']}",
        f"完全一致文件: {report['identical_files']}",
        f"仅左侧文件: {len(report['left_only'])}",
        f"仅右侧文件: {len(report['right_only'])}",
        f"聚合数值项: total={summary['numeric_total']} match={summary['numeric_match']} mismatch={summary['numeric_mismatch']}",
        f"聚合精确项: total={summary['exact_total']} match={summary['exact_match']} mismatch={summary['exact_mismatch']}",
        f"聚合缺失项: missing_left={summary['missing_left']} missing_right={summary['missing_right']} length_mismatch={summary['length_mismatch']}",
    ]
    if report["left_only"]:
        lines.append("仅左侧文件示例:")
        for path in report["left_only"][:10]:
            lines.append(f"  - {path}")
    if report["right_only"]:
        lines.append("仅右侧文件示例:")
        for path in report["right_only"][:10]:
            lines.append(f"  - {path}")
    differing_files = [
        item
        for item in report["files"]
        if item["numeric_mismatch"] or item["exact_mismatch"] or item["missing_left"] or item["missing_right"] or item["length_mismatch"]
    ]
    if differing_files:
        lines.append("存在差异的文件:")
        for item in differing_files[:20]:
            lines.append(
                f"  - {item['path']} | kind={item['kind']} | numeric_mismatch={item['numeric_mismatch']} | exact_mismatch={item['exact_mismatch']} | missing_left={item['missing_left']} | missing_right={item['missing_right']}"
            )
    if summary["mismatches"]:
        lines.append("差异样例:")
        for mismatch in summary["mismatches"]:
            detail = (
                f"  - file={mismatch['file']} path={mismatch['path']} reason={mismatch['reason']} "
                f"left={mismatch.get('left')} right={mismatch.get('right')}"
            )
            if "abs_diff" in mismatch:
                detail += f" abs_diff={mismatch.get('abs_diff')} rel_diff={mismatch.get('rel_diff')}"
            lines.append(detail)
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="左侧文件或目录，通常是 GPU 结果")
    parser.add_argument("right", type=Path, help="右侧文件或目录，通常是昇腾结果")
    parser.add_argument(
        "--mode",
        choices=("auto", "json", "jsonl", "csv", "text"),
        default="auto",
        help="文件模式下强制指定比较方式，默认 auto。",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="绝对误差阈值，默认 1e-6。")
    parser.add_argument("--rtol", type=float, default=1e-5, help="相对误差阈值，默认 1e-5。")
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=20,
        help="输出中最多保留多少条差异样例，默认 20。",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果。")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if not args.left.exists():
        raise SystemExit(f"错误: 左侧路径不存在: {args.left}")
    if not args.right.exists():
        raise SystemExit(f"错误: 右侧路径不存在: {args.right}")

    if args.left.is_dir() and args.right.is_dir():
        report = compare_directories(args.left, args.right, args.atol, args.rtol, args.max_mismatches)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(render_dir_report(report))
        return 0

    if args.left.is_dir() != args.right.is_dir():
        raise SystemExit("错误: 左右路径类型不一致，必须同时是文件或同时是目录。")

    report = compare_file(args.left, args.right, args.mode, args.atol, args.rtol, args.max_mismatches)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_file_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main([] if False else __import__("sys").argv[1:]))
