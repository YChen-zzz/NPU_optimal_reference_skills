"""Shared utilities for CANN profiling data parsing.

These utilities handle the fixed file formats produced by the Ascend CANN profiler.
They are independent of any specific model or project — only the CANN CSV schema matters.

Typical profiling directory structure:
    <profiling_dir>/
    └── profiling_data/
        └── <host>_<pid>_<ts>_ascend_pt/
            ├── ASCEND_PROFILER_OUTPUT/   ← target directory
            │   ├── op_statistic.csv
            │   ├── step_trace_time.csv
            │   ├── kernel_details.csv
            │   ├── memory_record.csv
            │   ├── operator_details.csv  (can be very large: 1M-20M rows)
            │   ├── operator_memory.csv
            │   └── trace_view.json
            └── mindstudio_profiler_output/  (optional)
"""

import csv
import io
import os
from pathlib import Path
from typing import List, Dict, Optional, Iterator

_THRESHOLDS_CACHE = None


def load_thresholds() -> dict:
    """Load thresholds.py once and cache. Returns nested dict."""
    global _THRESHOLDS_CACHE
    if _THRESHOLDS_CACHE is not None:
        return _THRESHOLDS_CACHE
    try:
        from thresholds import THRESHOLDS
        _THRESHOLDS_CACHE = THRESHOLDS
    except ImportError:
        _THRESHOLDS_CACHE = {}
    return _THRESHOLDS_CACHE


def threshold(script: str, key: str, default=None):
    """Get a threshold value: thresholds.py THRESHOLDS[script][key], with fallback default."""
    cfg = load_thresholds()
    return cfg.get(script, {}).get(key, default)


def find_ascend_profiler_output(profiling_dir: str, rank: Optional[int] = None) -> Path:
    """Locate the ASCEND_PROFILER_OUTPUT directory within a profiling tree."""
    base = Path(profiling_dir)
    if rank is not None:
        rank_dir = base / f"rank_{rank}"
        if rank_dir.exists():
            base = rank_dir

    for candidate in base.rglob("ASCEND_PROFILER_OUTPUT"):
        if candidate.is_dir():
            return candidate
    return base


def safe_float(val, default: float = 0.0) -> float:
    """Parse a value to float, handling whitespace and tab chars."""
    try:
        return float(str(val).strip().replace("\t", ""))
    except (ValueError, TypeError):
        return default


def safe_int(val, default: int = 0) -> int:
    try:
        return int(str(val).strip().replace("\t", ""))
    except (ValueError, TypeError):
        return default


def read_csv_all(filepath: Path) -> List[Dict[str, str]]:
    """Read a small CSV file fully into memory. Use for files < 100K rows."""
    if not filepath.exists():
        return []
    text = filepath.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.DictReader(io.StringIO(text)))


def stream_csv(filepath: Path, chunk_size: int = 8192) -> Iterator[Dict[str, str]]:
    """Stream a large CSV file row by row without loading everything into memory."""
    if not filepath.exists():
        return

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def format_duration_ms(us: float) -> str:
    """Format microseconds as milliseconds string."""
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    if us >= 1000:
        return f"{us / 1000:.1f}ms"
    return f"{us:.1f}us"


def format_size_mb(kb: float) -> str:
    """Format KB as readable size."""
    mb = kb / 1024
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def clean_multiline_field(value: str) -> str:
    """Clean a CSV field that may contain embedded newlines/semicolons."""
    return value.replace("\n", " ").replace("\r", " ").replace(";", " | ").strip()
