# 图编译

图编译可彻底消除 Host dispatch 开销（属于"掩盖"维度的极端形态），是 host-bound 场景的终极手段。

## 前置条件检查

在尝试图编译前，先检查可用性：

```python
# 方式 1: torch.compile (依赖 triton)
try:
    import triton
    print("triton 可用, torch.compile 可用")
except ImportError:
    print("triton 不可用, torch.compile 不可用")

# 方式 2: torchair (NPU 专用图编译)
try:
    import torchair
    print(f"torchair 可用: {torchair.__version__}")
except ImportError:
    print("torchair 不可用")

# 方式 3: NPU JIT 编译 (不需要 triton, 但可能触发 tiling error)
import torch_npu
torch.npu.set_compile_mode(jit_compile=True)
# 用小输入测试是否正常
```

> 三种方式按优先级排序：torchair > torch.compile > jit_compile。
> 若都不可用，回退 eager 模式，通过减少 kernel 数量（融合算子、flat forward）缓解 host-bound。

## torch.compile

### 基本用法

```python
compiled_layer = torch.compile(model.encoder.layers[0], mode="reduce-overhead")

# 验证精度
with torch.no_grad():
    original_output = model.encoder.layers[0](x)
    compiled_output = compiled_layer(x)
    cos = torch.cosine_similarity(original_output.flatten(), compiled_output.flatten(), dim=0)
    assert cos > 0.999, f"精度不达标: {cos}"
```

### 模式选择

| 模式 | 适用场景 | 注意 |
|------|---------|------|
| `reduce-overhead` | 形状固定、Host 调度开销显著 | 使用 ACLGraph，消除 Python dispatch |
| `max-autotune` | 可接受较长编译时间，追求极致吞吐 | 使用 GE 后端，会尝试 kernel 自动调优 |
| `default` | 通用 | 仅做图捕获，不做激进的调度优化 |

### 编译范围策略

**核心原则**：挑"碎而密"的地方编，不挑"大而炸"的地方编。

- ✅ 从纯计算子模块开始（如单层 Transformer Block、LayerNorm + Linear 链路）逐渐扩大范围
- ✅ 形状固定的子图（encode 路径，非自回归 decode）
- ❌ 不要直接 `torch.compile(model)`——控制流、side-effect 会切图
- ❌ 含动态 shape 的循环（每步 shape 变化导致反复重编译）

## torchair

torchair 是昇腾专用的图编译工具，不依赖 triton，对 NPU 兼容性更好。

```python
import torchair
from torchair.ge_concrete_graph import ge_apis

# 配置图编译选项
config = torchair.CompilerConfig()
config.experimental_config.frozen_parameter = True  # 推理时冻结参数
config.debug.graphdump.type = "json"  # 可选：导出图结构用于调试

# 编译模型
compiled_model = torchair.compile(model, config=config)
```

> torchair 的具体 API 随版本变化，使用前查阅对应版本的[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0001.html)。

## 放弃条件

满足任一即回退 eager：

- 算子不兼容（编译报错且无法绕过）
- 图太大导致编译期 OOM
- 触发框架 bug（如 tiling error、段错误）
- 精度不达标（cosine < 0.999）且无法通过调整编译选项修复
- 编译时间过长（> 30 分钟）且无法缓存

> 回退后记录失败原因到 evidence_db，避免重复尝试。

## 环境配置

图编译的环境变量与通用 CANN 环境配置一致，见 [environment_reference.md](../../01_preparation/references/environment_reference.md)「环境变量配置清单」。

图编译与 Python 级设置的关系：
- `jit_compile=False` + 图编译：图编译独立于 JIT，不冲突
- `allow_internal_format`：图编译模式下建议 `True`（GE 后端内部会处理格式转换）
- 图编译后不需要 `TASK_QUEUE_ENABLE`（图模式下没有逐算子 dispatch）
