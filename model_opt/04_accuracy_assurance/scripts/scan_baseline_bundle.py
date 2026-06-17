#!/usr/bin/env python3
"""扫描用户提供的 GPU 基线或对比目录，识别可用于精度对齐的文件。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__"}
TEXT_EXTS = {".txt", ".log", ".out", ".err", ".md"}
METRIC_EXTS = {".json", ".jsonl", ".csv", ".tsv", ".txt"}
CONFIG_EXTS = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".json"}
DUMP_EXTS = {".npy", ".npz", ".bin", ".pt", ".pth", ".pkl"}
CHECKPOINT_EXTS = {".ckpt", ".safetensors"}

SIDE_KEYWORDS = {
    "gpu": "gpu",
    "cuda": "gpu",
    "baseline": "gpu",
    "ascend": "ascend",
    "npu": "ascend",
    "target": "ascend",
}

SCENARIO_KEYWORDS = {
    "training": {"train", "training", "epoch", "loss", "optimizer", "checkpoint"},
    "inference": {"infer", "inference", "predict", "prediction", "output", "decode", "sample"},
    "dump": {"dump", "tensor", "activation", "hidden", "logits", "attn", "msprobe"},
}


@dataclass
class FileRecord:
    path: str
    size: int
    category: str
    side: str
    scenarios: List[str]


def collect_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def infer_side(path: Path) -> str:
    for part in path.parts:
        keyword = SIDE_KEYWORDS.get(part.lower())
        if keyword:
            return keyword
    return "unknown"


def infer_scenarios(path: Path, category: str) -> Set[str]:
    scenarios: Set[str] = set()
    lower_parts = {part.lower() for part in path.parts}
    lower_name = path.name.lower()

    if category == "dump":
        scenarios.add("dump")
    if category in {"log", "checkpoint"}:
        scenarios.add("training")
    if category in {"output", "input"}:
        scenarios.add("inference")

    for scenario, keywords in SCENARIO_KEYWORDS.items():
        if any(keyword in lower_name for keyword in keywords):
            scenarios.add(scenario)
            continue
        if lower_parts & keywords:
            scenarios.add(scenario)

    return scenarios or {"unknown"}


def classify_category(path: Path) -> str:
    lower_name = path.name.lower()
    lower_parts = [part.lower() for part in path.parts]
    ext = path.suffix.lower()

    if lower_name == "manifest.json":
        return "manifest"
    if "dump" in lower_parts or ext in DUMP_EXTS:
        if any(keyword in lower_name for keyword in ("checkpoint", "model", "weight")) and "dump" not in lower_parts:
            return "checkpoint"
        return "dump"
    if ext in CHECKPOINT_EXTS or any(keyword in lower_name for keyword in ("checkpoint", "weights")):
        return "checkpoint"
    if any(keyword in lower_name for keyword in ("command", "cmd", "launch")):
        return "run"
    if ext in {".sh", ".bat", ".ps1"}:
        return "run"
    if ext in CONFIG_EXTS and any(keyword in lower_name for keyword in ("config", "args", "train", "infer")):
        return "config"
    if ext in CONFIG_EXTS and any(part in {"config", "configs"} for part in lower_parts):
        return "config"
    if ext in METRIC_EXTS and any(keyword in lower_name for keyword in ("metric", "score", "result", "eval", "benchmark")):
        return "metrics"
    if ext in {".json", ".jsonl", ".csv", ".tsv", ".txt"} and any(
        keyword in lower_name for keyword in ("predict", "prediction", "output", "answer", "generation")
    ):
        return "output"
    if ext in {".json", ".jsonl", ".csv", ".tsv", ".txt"} and any(
        keyword in lower_name for keyword in ("input", "sample", "prompt", "query")
    ):
        return "input"
    if ext in TEXT_EXTS and any(keyword in lower_name for keyword in ("log", "stdout", "stderr", "train", "loss", "eval", "test")):
        return "log"
    if ext in CONFIG_EXTS:
        return "config"
    return "other"


def scan_bundle(root: Path) -> Dict[str, object]:
    records: List[FileRecord] = []
    by_category: Counter[str] = Counter()
    by_side: Counter[str] = Counter()
    by_scenario: Counter[str] = Counter()
    by_side_category: Dict[str, Counter[str]] = defaultdict(Counter)

    for path in sorted(collect_files(root)):
        category = classify_category(path)
        side = infer_side(path.relative_to(root))
        scenarios = sorted(infer_scenarios(path.relative_to(root), category))
        record = FileRecord(
            path=str(path.relative_to(root)).replace("\\", "/"),
            size=path.stat().st_size,
            category=category,
            side=side,
            scenarios=scenarios,
        )
        records.append(record)
        by_category[category] += 1
        by_side[side] += 1
        by_side_category[side][category] += 1
        for scenario in scenarios:
            by_scenario[scenario] += 1

    manifest_present = any(record.category == "manifest" for record in records)
    detected_scenarios = [
        scenario for scenario, count in by_scenario.items() if scenario != "unknown" and count > 0
    ]

    suggestions: List[str] = []
    if not manifest_present:
        suggestions.append("未发现 manifest.json，建议补一份目录清单。")
    if by_side["gpu"] == 0:
        suggestions.append("未识别到 GPU 侧目录关键词，请确认基线文件路径或目录命名。")
    if by_category["log"] == 0 and by_category["metrics"] == 0 and by_category["output"] == 0 and by_category["dump"] == 0:
        suggestions.append("未识别到训练日志、指标文件、推理输出或 dump 文件，当前目录可能不是精度对齐结果目录。")
    if by_category["dump"] > 0 and by_category["manifest"] == 0:
        suggestions.append("发现 dump 文件但未发现 dump manifest，建议补记录文件。")

    return {
        "bundle_root": str(root),
        "total_files": len(records),
        "manifest_present": manifest_present,
        "detected_scenarios": sorted(detected_scenarios),
        "counts": {
            "by_category": dict(sorted(by_category.items())),
            "by_side": dict(sorted(by_side.items())),
            "by_side_category": {
                side: dict(sorted(counter.items()))
                for side, counter in sorted(by_side_category.items())
            },
        },
        "suggestions": suggestions,
        "files": [asdict(record) for record in records],
    }


def validate_expectation(summary: Dict[str, object], expected: str) -> Optional[str]:
    counts = summary["counts"]["by_category"]
    if expected == "training" and counts.get("log", 0) == 0 and counts.get("metrics", 0) == 0:
        return "期望训练目录，但未发现训练日志或指标文件。"
    if expected == "inference" and counts.get("output", 0) == 0 and counts.get("metrics", 0) == 0:
        return "期望推理目录，但未发现推理输出或指标文件。"
    if expected == "dump" and counts.get("dump", 0) == 0:
        return "期望 dump 目录，但未发现 dump 文件。"
    return None


def build_detected_manifest(summary: Dict[str, object]) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for record in summary["files"]:
        grouped[record["side"]][record["category"]].append(record["path"])

    return {
        "bundle_version": "detected-1.0",
        "detected_scenarios": summary["detected_scenarios"],
        "sides": sorted(grouped.keys()),
        "files": {
            side: {category: paths for category, paths in sorted(categories.items())}
            for side, categories in sorted(grouped.items())
        },
        "notes": [
            "该文件由 scan_baseline_bundle.py 自动生成，仅作为目录索引，不代表所有路径都适合直接比较。"
        ],
    }


def render_text(summary: Dict[str, object], max_files: int) -> str:
    lines: List[str] = []
    lines.append(f"根目录: {summary['bundle_root']}")
    lines.append(f"总文件数: {summary['total_files']}")
    lines.append(f"已发现 manifest: {'是' if summary['manifest_present'] else '否'}")
    scenarios = summary["detected_scenarios"] or ["未识别"]
    lines.append(f"场景线索: {', '.join(scenarios)}")
    lines.append("")
    lines.append("分类统计:")
    for category, count in summary["counts"]["by_category"].items():
        lines.append(f"  - {category}: {count}")
    lines.append("")
    lines.append("按侧统计:")
    for side, count in summary["counts"]["by_side"].items():
        lines.append(f"  - {side}: {count}")
    if summary["suggestions"]:
        lines.append("")
        lines.append("建议:")
        for suggestion in summary["suggestions"]:
            lines.append(f"  - {suggestion}")
    lines.append("")
    lines.append("示例文件:")
    for record in summary["files"][:max_files]:
        scenarios_text = ",".join(record["scenarios"])
        lines.append(
            f"  - {record['path']} | category={record['category']} | side={record['side']} | scenarios={scenarios_text}"
        )
    remaining = summary["total_files"] - min(summary["total_files"], max_files)
    if remaining > 0:
        lines.append(f"  - 其余 {remaining} 个文件请用 --json 查看完整列表。")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path, help="用户提供的本地目录路径")
    parser.add_argument(
        "--expect",
        choices=("training", "inference", "dump"),
        help="可选。声明期望的目录类型，用于额外校验。",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="可选。把自动识别出的目录清单写成 JSON 文件。",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="文本输出时展示的文件数，默认 20。",
    )
    parser.add_argument("--json", action="store_true", help="输出完整 JSON 结果。")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    root = args.bundle_root
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"错误: 目录不存在或不可读取: {root}")

    summary = scan_bundle(root)
    if args.expect:
        error = validate_expectation(summary, args.expect)
        if error:
            print(f"警告: {error}")

    if args.write_manifest:
        detected = build_detected_manifest(summary)
        args.write_manifest.write_text(
            json.dumps(detected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(render_text(summary, args.max_files))

    return 0


if __name__ == "__main__":
    raise SystemExit(main([] if False else __import__("sys").argv[1:]))
