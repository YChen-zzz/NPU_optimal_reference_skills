#!/usr/bin/env python3
"""验证 torch_npu profiler 环境是否就绪。"""

import argparse
import os
import sys


def emit(status: str, label: str, message: str) -> None:
    print(f"[{status}] {label}: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profiling 环境预检")
    parser.add_argument("--device", default="npu:0", help="目标设备")
    parser.add_argument(
        "--output-dir", default="./profiling",
        help="Profiling 输出根目录（实际输出在其下的时间戳子目录中）",
    )
    args = parser.parse_args()

    failures = 0

    parent = os.path.dirname(os.path.abspath(args.output_dir))
    if not os.path.isdir(parent):
        emit("FAIL", "output-dir", f"父路径不存在: {parent}")
        failures += 1
    else:
        emit("PASS", "output-dir", f"父路径存在: {parent}")

    try:
        import torch
        import torch_npu

        emit("PASS", "torch", torch.__version__)
        emit("PASS", "torch_npu", torch_npu.__version__)

        profiler = getattr(torch_npu, "profiler", None)
        if profiler is None:
            emit("FAIL", "profiler", "torch_npu.profiler 不可用")
            failures += 1
        else:
            required = ["profile", "schedule", "tensorboard_trace_handler"]
            missing = [n for n in required if not hasattr(profiler, n)]
            if missing:
                emit("FAIL", "profiler-api", f"缺少接口: {', '.join(missing)}")
                failures += 1
            else:
                emit("PASS", "profiler-api", "所有必需 profiler API 就绪")

        if not torch.npu.is_available():
            emit("FAIL", "npu", "torch.npu.is_available() 返回 False")
            failures += 1
        else:
            tensor = torch.randn(4, 4, device=args.device)
            value = (tensor @ tensor).mean().item()
            emit("PASS", "npu-runtime", f"设备 {args.device} 可用 (mean={value:.4f})")
    except Exception as exc:
        emit("FAIL", "runtime", str(exc))
        failures += 1

    print(f"\n{'全部通过' if failures == 0 else f'存在 {failures} 项失败'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
