"""NPU vs CPU 精度对比工具库

提供对比函数和逐层 hook 工具，供 agent 在具体模型的验证脚本中 import 使用。

用法示例:
    import sys
    sys.path.insert(0, "/path/to/skills/model_adaptation/scripts")
    from compare_baseline import compare_tensors, compare_layerwise

    # 端到端对比
    result = compare_tensors(cpu_output, npu_output, name="logits")

    # 逐层对比 (定位偏差)
    compare_layerwise(model_cpu, model_npu, input_dict, device_npu="npu:0")
"""
import torch
import torch.nn.functional as F


def compare_tensors(baseline, target, name="output", threshold_cosine=0.999, threshold_max_abs=1e-2):
    """对比两个 tensor，返回对比结果 dict。

    Args:
        baseline: 基准 tensor (CPU, 任意 dtype)
        target: 待对比 tensor (CPU, 任意 dtype)
        name: 显示名称
        threshold_cosine: cosine 通过阈值 (默认 0.999 对应 fp16 单步)
        threshold_max_abs: max_abs 通过阈值

    Returns:
        dict with keys: cosine, max_abs, mean_abs, passed
    """
    b = baseline.detach().float().flatten()
    t = target.detach().float().flatten()

    if b.shape != t.shape:
        print(f"[FAIL] {name}: shape mismatch {b.shape} vs {t.shape}")
        return {"cosine": 0, "max_abs": float("inf"), "mean_abs": float("inf"), "passed": False}

    cos = F.cosine_similarity(b.unsqueeze(0), t.unsqueeze(0)).item()
    max_abs = (b - t).abs().max().item()
    mean_abs = (b - t).abs().mean().item()
    passed = cos >= threshold_cosine and max_abs <= threshold_max_abs

    status = "PASS" if passed else "WARN" if cos >= 0.99 else "FAIL"
    print(f"[{status}] {name:40s} cosine={cos:.6f}  max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}")
    return {"cosine": cos, "max_abs": max_abs, "mean_abs": mean_abs, "passed": passed}


def register_hooks(model):
    """为模型的 Linear/Conv2d/LayerNorm 层注册 hook，采集输出。

    Returns:
        dict: name -> tensor (在 forward 之后填充)
    """
    outputs = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                outputs[name] = out.detach().cpu().float()
            elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                outputs[name] = out[0].detach().cpu().float()
        return hook

    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
            m.register_forward_hook(make_hook(name))
    return outputs


def compare_layerwise(model_cpu, model_npu, inputs_cpu, device_npu="npu:0",
                      threshold_cosine=0.999, stop_on_first=False):
    """逐层对比 CPU 和 NPU 模型的中间输出，定位首个偏差层。

    Args:
        model_cpu: CPU 上的模型 (eval mode)
        model_npu: NPU 上的模型 (eval mode)
        inputs_cpu: CPU 上的输入 dict (如 {"input_ids": tensor, "attention_mask": tensor})
        device_npu: NPU 设备
        threshold_cosine: 判定偏差的 cosine 阈值
        stop_on_first: 是否在找到第一个偏差层后停止

    Returns:
        list of (layer_name, result_dict)，按顺序排列
    """
    cpu_hooks = register_hooks(model_cpu)
    npu_hooks = register_hooks(model_npu)

    inputs_npu = {k: v.to(device_npu) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs_cpu.items()}

    with torch.no_grad():
        model_cpu(**inputs_cpu)
        model_npu(**inputs_npu)

    results = []
    for name in cpu_hooks:
        if name not in npu_hooks:
            continue
        result = compare_tensors(cpu_hooks[name], npu_hooks[name], name, threshold_cosine)
        results.append((name, result))
        if stop_on_first and not result["passed"]:
            print(f"\n  首个显著偏差: {name}")
            break

    return results
