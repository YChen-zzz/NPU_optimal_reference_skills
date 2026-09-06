# Lab 骨架模板
#
# 使用方法：复制此文件为 benchmarks/supernodes/sn_<name>.py
# 填写所有 GATE 区域后才能实现候选方案

"""
SN-<NAME> Supernode Lab
========================
GPU fusion groups: <从 ir_post_fusion.txt 提取>
GPU pre-compile ops:
  - <op1>: <dtype> <shape>
  - <op2>: ...
GPU post-compile result:
  - <几个 kernel, 哪些 ops 被融合/消除>
NPU current implementation:
  - <file>:<line> <op1>
  - <file>:<line> <op2>
"""

# =====================================================================
# GATE 1: 精度对齐审计（必填，否则 assert 失败）
#
# Part A: 记录该 SN 在 GPU 和 NPU 上的完整精度路径
# Part B: 对每个显式 cast 做 GPU 对照判定
#
# 即使用户要求"不改数学语义"，此审计也不能跳过
# =====================================================================

# Part A: GPU vs NPU dtype 路径（必填）
DTYPE_PATH = {
    "gpu_input_dtype": None,       # GPU source 中该 SN 输入的 dtype (如 bf16)
    "gpu_compute_dtype": None,     # GPU 计算过程中的 dtype (如 bf16, f32 accumulator)
    "gpu_output_dtype": None,      # GPU 输出的 dtype
    "npu_input_dtype": None,       # NPU 当前输入的 dtype
    "npu_compute_dtype": None,     # NPU 计算过程中的 dtype
    "npu_output_dtype": None,      # NPU 输出的 dtype
    "weight_dtype": None,          # 参数存储 dtype (f32 weight + bf16 activation = 隐式 cast)
    "mismatches": [],              # GPU 和 NPU 不一致的地方
}
for key in ("gpu_input_dtype", "npu_input_dtype", "gpu_compute_dtype", "npu_compute_dtype"):
    assert DTYPE_PATH[key] is not None, f"GATE 1A: {key} 未填写"

# Part B: 显式 cast 审计（对每个 .float()/.type_as()/.to(torch.float32)）
#   1. 读 GPU source 同一位置，确认 GPU 是否有这个 cast
#   2. GPU 没有 → "移植遗留"，归入 L1 候选
#   3. GPU 也有 → "原始设计"，不动
CAST_AUDIT = {
    # 格式: "位置描述": {"gpu_has_cast": True/False, "verdict": "移植遗留"/"原始设计"}
    # 示例: "loss.py:42 logits.float()": {"gpu_has_cast": False, "verdict": "移植遗留"},
    # 如果该 SN 无显式 cast，填: "无显式cast": {"gpu_has_cast": None, "verdict": "该SN无cast"}
}
assert len(CAST_AUDIT) > 0, \
    "GATE 1B 未完成：必须审计该 SN 中所有显式 cast，无 cast 也要标注"

# =====================================================================
# GATE 2: 真实 Shape（必填，从训练代码 print 获取，禁止手动推断）
# 在训练代码中插入 print(x.shape, x.dtype) 实际运行一次获取
# =====================================================================
SHAPES = {
    # "regime_0": {"input": (B, T, D), "dtype": torch.bfloat16},
    # "regime_1": ...,
    # "regime_2": ...,
}
assert len(SHAPES) > 0, "GATE 2 未完成：必须从训练代码 print 获取真实 shape"

# =====================================================================
# GATE 3: L4 Compile 候选覆盖审计（进入 L4 时必填）
# L4 必须覆盖以下 4 类候选，缺一不可（不适用的标注原因）
# =====================================================================
COMPILE_AUDIT = {
    "compile_cumulative_winner": None,     # 编译当前累计 winner（来自 L0/L1/L2/L3）
    "compile_gpu_fusion_expr": None,       # 编译 GPU fusion 对应的代数表达族
    "compile_scope_variants": None,        # 不同 scope（最小chain / +linear / 完整SN）
    "all_shapes_tested": None,             # 是否覆盖全部真实 shape
}
# 每项填 True（已测试）或 "skip: <原因>"（不适用）
# 进入 L4 时取消下面的注释来激活检查:
# for key, val in COMPILE_AUDIT.items():
#     assert val is not None, f"GATE 3: {key} 未填写"

# =====================================================================
# B0: Control（当前 NPU 实现）
# =====================================================================
def B0_control():
    pass  # TODO: 从训练代码提取

# =====================================================================
# 候选方案
# =====================================================================
# L0: <从 PRECISION_AUDIT 和 API 参数 gap 推导>
# L1: <从 PRECISION_AUDIT 中的移植遗留 + 冗余操作推导>
# L2: <搜索 dir(torch_npu) 找等价融合算子>
# L3: <等价手动改写>
# L4: <compile 候选，必须满足 COMPILE_AUDIT>

# =====================================================================
# Benchmark Harness
# =====================================================================
# warmup + timing (median/p95) + grad cosine similarity
# 所有 regime shape 都测
