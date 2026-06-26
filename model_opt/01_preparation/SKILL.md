---
name: NPU 适配前期准备
description: 模型 NPU 适配的前期准备工作，包括模型代码理解、CANN 环境搭建、测试数据准备、Profiling 脚本和精度验证脚本的构建。当用户需要从零开始在 NPU 上跑通模型推理时触发。
---

# NPU 适配前期准备

详见 [environment_reference.md](references/environment_reference.md) 获取详细的 CANN 环境配置参考。

---

## 一、模型代码理解

**目标**：在动手适配前，摸清推理路径，避免在错误位置改代码。

### 系统性探索顺序
```bash
# 1. 看全局结构
find . -name "*.py" | head -60

# 2. 看包入口，确认对外暴露了什么
cat src/<package>/__init__.py

# 3. 定位模型类定义（通常含 forward / __call__）
grep -rn "class.*Model\|class.*Encoder\|class.*Decoder" src/ --include="*.py"

# 4. 定位推理入口（脚本 / CLI / pipeline）
grep -rn "def main\|if __name__" *.py scripts/*.py
```

### 从 config.json 快速确认架构
```python
import json
cfg = json.load(open("config.json"))
# 关注：d_model / hidden_size, num_layers, num_heads, vocab_size, model_type
```

### 参考文档对比
- 若有已有适配文档（如其他版本、同系列模型），先列出差异点：**可复用 vs 需纠偏**。
- 重点关注：自定义算子、特殊注意力变体、非标准位置编码。

### 特殊输入识别
- 非文本输入（图像、蛋白质序列、音频等）需确认预处理路径和 tokenizer/encoder 是否独立。
- 检查 `collate_fn` / `DataCollator` 是否有设备绑定逻辑。

---

## 二、环境准备

### CANN 环境诊断
```bash
npu-smi info                          # 确认芯片型号、驱动版本、卡数
ls $ASCEND_HOME_PATH/opp/             # 检查 OPP 算子包
ls ~/Ascend/ascend-toolkit/           # 用户目录（优先级高于系统目录）
source ~/Ascend/ascend-toolkit/latest/set_env.sh  # 激活工具链
```

### 多卡管理
```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3   # 控制可见卡（对应 npu:0,1,2,3）
# 注意：逻辑编号从 0 开始，与 npu-smi 的物理编号可能不一致
```

### 版本配套确认
```python
import torch, torch_npu, torchair
print(torch.__version__, torch_npu.__version__, torchair.__version__)
# CANN 版本：cat $ASCEND_HOME_PATH/version.cfg
```
> 版本不配套是最常见的环境问题，优先确认 torch_npu 与 CANN 版本对应表。

### 设备兼容层设计（init_device 模板）
在任何推理脚本中统一用一个 `init_device` 函数屏蔽平台差异：

```python
def init_device(device_str: str):
    if device_str.startswith("npu"):
        import torch_npu
        # NPU 适配三件套：关闭 JIT 编译 + 关闭内部格式（保证确定性和兼容性）
        torch.npu.set_compile_mode(jit_compile=False)
        torch_npu.npu.config.allow_internal_format = False
    return torch.device(device_str)
```

---

## 三、测试数据准备

**原则**：小而代表性，能快速复现问题，不浪费时间跑全量。

### 轻量测试集
- 按长度/关键特征分桶（如序列长度：短/中/长），每桶抽若干条。
- 总量控制在可在 1–2 分钟内完成推理的规模（通常 50–200 条）。

### Profiling 专用子集
- 仅需 **5–10 条**，覆盖不同规模档位（如最短、中位、最长各一条）。
- 单独保存为 `profiling_subset.json` / `.pkl`，与主测试集分开。

### 代表性数据选取
```python
import numpy as np
lengths = [len(s) for s in dataset]
median_idx = np.argsort(lengths)[len(lengths) // 2]
# 取中位数附近 +-5% 的样本作为代表集
```

---

## 四、Profiling 采集体系构建

**目标**：构建标准化的性能数据采集流程，为后续瓶颈分析提供可靠数据。

详见 [profiling_collection.md](references/profiling_collection.md) 获取完整代码模板和框架适配方案。

### Profiling 输出路径规范

所有 profiling 输出**必须**保存在用户工作目录下，禁止使用 `/tmp` 或其他临时目录：

```
<workspace>/profiling/
├── YYYYMMDD_HHMMSS/          # 每次采集带时间戳
│   ├── *.csv
│   ├── trace_view.json
│   └── ...
├── YYYYMMDD_HHMMSS/
└── latest -> YYYYMMDD_HHMMSS  # 软链接指向最新一次
```

**路径构造模板**：
```python
import os, datetime

PROFILING_BASE = os.path.join(os.getcwd(), "profiling")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
profiling_dir = os.path.join(PROFILING_BASE, timestamp)
os.makedirs(profiling_dir, exist_ok=True)

# 更新 latest 软链接
latest_link = os.path.join(PROFILING_BASE, "latest")
if os.path.islink(latest_link):
    os.remove(latest_link)
os.symlink(timestamp, latest_link)
```

**强制规则**：
- `tensorboard_trace_handler` 的路径参数必须指向 `<workspace>/profiling/<timestamp>/`
- 禁止使用 `/tmp`、系统临时目录或无时间戳的固定路径
- 每次采集后更新 `latest` 软链接，方便用户直接查看最新结果
- `profiling/` 目录已在 `.gitignore` 中排除（详见 05_engineering）

### 采集级别选择

| 级别 | 内容 | 数据量 | 适用场景 |
|------|------|--------|----------|
| **L0**（默认） | 仅采集 NPU 活动，最小膨胀 | 小 | 快速定位热点、整体耗时分布 |
| **L1** | CPU + NPU + 算子详情 | 中 | 分析 Host/Device 比例、算子级耗时 |
| **L2** | L1 + 调用栈 + 内存 | 大 | 深度分析调用链、内存瓶颈 |

> 用户未指定级别时默认使用 L0；仅当明确需要算子分析或调用栈时才升级。

### 采集流程

```
0. 环境校验（运行本 skill 的 scripts/validate_profiling_env.py）
→ 1. 确定采集场景（训练 / 推理）
→ 2. 选择采集级别（L0 / L1 / L2）
→ 3. 植入 Profiling 代码（按框架选择模板）
→ 4. 设置环境变量（TASK_QUEUE_ENABLE, CPU_AFFINITY_CONF）
→ 5. 执行采集，产出 trace 文件
→ 6. 进入 Phase 2 分析
```

### 采集前必设环境变量

```bash
# 必须在启动脚本前设置
export TASK_QUEUE_ENABLE=2    # Host-Device 异步流水，获得接近生产环境的真实性能
export CPU_AFFINITY_CONF=1    # CPU 绑核，减少调度抖动，使采集数据稳定可复现
```

### 重要默认行为

1. **不要修改业务代码中的 CUDA 写法**：通过 `import torch_npu` + `transfer_to_npu`，业务代码中的 `.cuda()` 会自动转为 NPU 调用。Profiling 代码使用 `torch_npu.profiler` 是必要的，但业务脚本保持原样。
2. **多推理路径分离采集**：不同推理模式（encoder-only / generate / scoring）应分别建立独立 profiling 段，不混合采集。
3. **GPU/NPU 对称采集**：当需要跨平台对比时，在 GPU 上使用相同 schedule 配置 + `torch.profiler.ProfilerActivity.CUDA`。

### 框架适配指引

| 训练框架 | 接入方式 | 模板位置 |
|---------|---------|----------|
| 原生 PyTorch 循环 | `with torch_npu.profiler.profile(...) as prof` | profiling_collection.md §3.1–3.3 |
| PyTorch Lightning | 自定义 Callback | profiling_collection.md §4.1 |
| HuggingFace Trainer | 自定义 TrainerCallback | profiling_collection.md §4.2 |
| DeepSpeed | rank 0 采集 | profiling_collection.md §4.3 |

### 环境预检

采集前运行本 skill 提供的环境校验脚本（位于本 skill 的 `scripts/validate_profiling_env.py`）：
```bash
python <skill_path>/01_preparation/scripts/validate_profiling_env.py --device npu:0 --output-dir ./profiling
```
> 注意：`<skill_path>` 是本 skill 所在目录的实际路径，agent 执行时需替换为真实路径。

---

## 五、精度验证脚本构建

**目标**：GPU 和 NPU 分别离线保存结果，事后纯 numpy 对比，不依赖双卡同时在线。

### 输出保存约定
```python
import numpy as np, json

# 连续输出（logits、embedding）-> npy
np.save(f"outputs/{sample_id}_logits.npy", logits.cpu().float().numpy())

# 离散输出（token ids、标签）-> json
with open(f"outputs/{sample_id}_tokens.json", "w") as f:
    json.dump({"tokens": token_ids}, f)
```

### 对比指标设计
| 输出类型 | 主指标 | 辅助指标 |
|---|---|---|
| 连续向量（logits/embedding） | cosine similarity | mean abs diff |
| 离散序列（token ids） | 完全匹配率 | top-1 token 匹配率 |
| 聚合标量（loss/score） | 相对误差 |a-b|/|a| | -- |

### 离线对比脚本
```python
# compare_outputs.py -- 不依赖任何设备，纯 numpy
gpu_logits = np.load("gpu_outputs/logits.npy")
npu_logits = np.load("npu_outputs/logits.npy")

cos_sim = np.dot(gpu_logits.flatten(), npu_logits.flatten()) / (
    np.linalg.norm(gpu_logits) * np.linalg.norm(npu_logits)
)
abs_diff = np.abs(gpu_logits - npu_logits).mean()
print(f"cosine={cos_sim:.6f}  mean_abs_diff={abs_diff:.6f}")
```

### NPU 确定性验证条件
```python
model.eval()
with torch.no_grad():
    # 确保 jit_compile=False + allow_internal_format=False 已在 init_device 中设置
    outputs = model(inputs)
```
> 同一输入的 NPU 结果应完全可复现 (bit-exact)。若不可复现，排查：dropout 未关闭、随机算子、或 `allow_internal_format` 未禁用。
