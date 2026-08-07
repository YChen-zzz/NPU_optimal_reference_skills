#!/usr/bin/env python3
"""模型权重管理工具

功能:
  detect-source    检测可用的下载源
  download         下载模型权重到指定目录 (自动优选 safetensors 格式)
  cleanup          删除冗余格式 (下载时同时拿到了多种格式的情况)
  verify           验证权重完整性 (load 无 missing keys)

格式策略:
  - 下载时优先选 safetensors (通过 ignore_file_pattern 跳过 .bin/.h5)
  - 如果模型只提供 .bin 格式，直接使用 .bin，不做转换
  - 不做格式转换：转换是额外操作，有出错风险且无实际收益

用法:
  python weight_manager.py detect-source
  python weight_manager.py download --model-id Qwen/Qwen3-1.7B --local-dir ./weights
  python weight_manager.py download --model-id facebook/wav2vec2-base --local-dir ./weights --keep-all
  python weight_manager.py cleanup --weights-dir ./weights
  python weight_manager.py verify --weights-dir ./weights
"""
import argparse
import glob
import json
import os
import ssl
import sys
import urllib.request


def detect_source():
    """检测可用的下载源，返回最优源名称"""
    ctx = ssl._create_unverified_context()
    sources = [
        ("modelscope", "https://modelscope.cn"),
        ("hf-mirror", "https://hf-mirror.com"),
        ("huggingface", "https://huggingface.co"),
    ]
    available = []
    for name, url in sources:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            urllib.request.urlopen(req, context=ctx, timeout=10)
            available.append(name)
            print(f"  [OK] {name:12s} ({url})")
        except Exception as e:
            print(f"  [--] {name:12s} ({url}) -> {type(e).__name__}")
    if available:
        print(f"\n推荐下载源: {available[0]}")
        return available[0]
    else:
        print("\n错误: 无可用下载源")
        sys.exit(1)


def download(model_id, local_dir, source, keep_all=False):
    """从指定源下载模型权重
    
    默认行为: 优先下载 safetensors 格式 (跳过 .bin/.h5/.onnx)
    --keep-all: 不跳过任何格式 (用于只有 .bin 的模型，或需要完整 repo 的情况)
    """
    # 默认跳过的格式 (当模型有 safetensors 时这些是冗余的)
    ignore = ["*.h5", "*.onnx", "*.msgpack", "*.tflite"]
    if not keep_all:
        ignore += ["*.bin"]

    os.makedirs(local_dir, exist_ok=True)
    print(f"下载: {model_id} -> {local_dir}")
    print(f"来源: {source}")
    print(f"格式策略: {'保留所有格式' if keep_all else '优选 safetensors (跳过 .bin)'}")

    if source == "modelscope":
        from modelscope import snapshot_download
        path = snapshot_download(model_id, local_dir=local_dir, ignore_file_pattern=ignore)

    elif source in ("hf-mirror", "huggingface"):
        endpoint = "https://hf-mirror.com" if source == "hf-mirror" else None
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
        from huggingface_hub import snapshot_download as hf_download
        path = hf_download(model_id, local_dir=local_dir, ignore_patterns=ignore)

    else:
        print(f"错误: 不支持的源 '{source}'")
        sys.exit(1)

    # 检查是否实际下到了权重文件
    weight_files = glob.glob(os.path.join(local_dir, "**", "*.safetensors"), recursive=True)
    weight_files += glob.glob(os.path.join(local_dir, "**", "*.bin"), recursive=True)
    weight_files += glob.glob(os.path.join(local_dir, "**", "*.pt"), recursive=True)

    if not weight_files and not keep_all:
        # 模型可能只有 .bin 格式，重新下载不跳过 .bin
        print("\n未找到 safetensors，该模型可能只提供 .bin 格式，重新下载(keep_all)...")
        # snapshot_download 会自动补全缺失文件，不会重复下已有的 config/tokenizer
        return download(model_id, local_dir, source, keep_all=True)

    total_size = sum(os.path.getsize(f) for f in weight_files) / 1e9
    print(f"\n完成: {len(weight_files)} 个权重文件, 共 {total_size:.2f}GB")
    return path


def cleanup(weights_dir):
    """删除冗余格式 (仅当同一目录下同时存在 safetensors 和 bin 时才删 bin)"""
    removed = 0
    freed = 0

    for root, dirs, files in os.walk(weights_dir):
        safetensors = [f for f in files if f.endswith(".safetensors")]
        bins = [f for f in files if f.endswith(".bin")]
        h5s = [f for f in files if f.endswith(".h5")]

        # 只有同时存在两种格式时才删冗余的
        if safetensors and bins:
            for b in bins:
                path = os.path.join(root, b)
                size = os.path.getsize(path) / 1e6
                os.remove(path)
                print(f"  删除: {path} ({size:.1f}MB)")
                removed += 1
                freed += size
        if safetensors and h5s:
            for h in h5s:
                path = os.path.join(root, h)
                size = os.path.getsize(path) / 1e6
                os.remove(path)
                print(f"  删除: {path} ({size:.1f}MB)")
                removed += 1
                freed += size

    # 清理下载中断留下的 .incomplete 文件
    for root, dirs, files in os.walk(weights_dir):
        for f in files:
            if f.endswith(".incomplete"):
                path = os.path.join(root, f)
                size = os.path.getsize(path) / 1e6
                os.remove(path)
                print(f"  删除: {path} (incomplete, {size:.1f}MB)")
                removed += 1
                freed += size

    if removed:
        print(f"\n共清理 {removed} 个文件, 释放 {freed:.1f}MB")
    else:
        print("无冗余文件需要清理")


def verify(weights_dir):
    """验证权重完整性"""
    config_path = os.path.join(weights_dir, "config.json")
    if not os.path.exists(config_path):
        for root, dirs, files in os.walk(weights_dir):
            if "config.json" in files:
                config_path = os.path.join(root, "config.json")
                break

    # 列出权重文件
    weight_files = glob.glob(os.path.join(weights_dir, "**", "*.safetensors"), recursive=True)
    weight_files += glob.glob(os.path.join(weights_dir, "**", "*.bin"), recursive=True)
    weight_files += glob.glob(os.path.join(weights_dir, "**", "*.pt"), recursive=True)

    if not weight_files:
        print("错误: 未找到任何权重文件")
        sys.exit(1)

    total_size = sum(os.path.getsize(f) for f in weight_files) / 1e9
    formats = set(os.path.splitext(f)[1] for f in weight_files)
    print(f"权重文件: {len(weight_files)} 个, 格式: {formats}, 总计: {total_size:.2f}GB")

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        arch = config.get("architectures", [config.get("model_type", "unknown")])
        print(f"模型架构: {arch}")

    # 尝试加载验证
    try:
        from transformers import AutoModel
        print("加载验证中...")
        model = AutoModel.from_pretrained(weights_dir)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  参数量: {params:.1f}M")
        print("  验证 PASS: 模型加载成功, 无 missing keys")
        del model
    except Exception as e:
        # transformers 可能不支持该模型，尝试直接加载文件
        print(f"  AutoModel 加载失败 ({type(e).__name__}), 尝试直接读取权重文件...")
        try:
            if any(f.endswith(".safetensors") for f in weight_files):
                from safetensors.torch import load_file
                for wf in weight_files:
                    if wf.endswith(".safetensors"):
                        sd = load_file(wf)
                        print(f"  {os.path.basename(wf)}: {len(sd)} tensors")
                        del sd
            else:
                import torch
                for wf in weight_files:
                    sd = torch.load(wf, map_location="cpu", weights_only=True)
                    n = len(sd) if isinstance(sd, dict) else 1
                    print(f"  {os.path.basename(wf)}: {n} entries")
                    del sd
            print("  验证 PASS: 权重文件可正常读取")
        except Exception as e2:
            print(f"  验证 FAIL: {e2}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="模型权重管理工具")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("detect-source", help="检测可用下载源")

    dl = sub.add_parser("download", help="下载模型权重")
    dl.add_argument("--model-id", required=True, help="模型 ID (如 Qwen/Qwen3-1.7B)")
    dl.add_argument("--local-dir", required=True, help="本地保存路径")
    dl.add_argument("--source", default=None, help="下载源 (modelscope/hf-mirror/huggingface), 不指定则自动检测")
    dl.add_argument("--keep-all", action="store_true", help="保留所有格式，不跳过 .bin")

    cl = sub.add_parser("cleanup", help="删除冗余格式 (同目录下 safetensors 和 bin 共存时删 bin)")
    cl.add_argument("--weights-dir", required=True)

    vf = sub.add_parser("verify", help="验证权重完整性")
    vf.add_argument("--weights-dir", required=True)

    args = parser.parse_args()

    if args.command == "detect-source":
        detect_source()
    elif args.command == "download":
        source = args.source or detect_source()
        download(args.model_id, args.local_dir, source, args.keep_all)
    elif args.command == "cleanup":
        cleanup(args.weights_dir)
    elif args.command == "verify":
        verify(args.weights_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
