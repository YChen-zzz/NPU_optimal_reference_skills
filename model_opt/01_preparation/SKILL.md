---
name: npu-adaptation-preparation
description: NPU 适配前期准备：代码理解、CANN 环境搭建、测试数据、profiling 采集与精度验证脚本构建。当用户需要把模型在 NPU 上跑通、搭建/诊断 CANN 环境、采集基线 profiling、或构建精度对比脚本时触发。
---

# NPU 适配前期准备

---

## 一、模型代码理解

**目标**：在动手适配前，摸清推理路径，避免在错误位置改代码。

### 系统性探索顺序
从全局到局部：包入口（`__init__.py`）→ 模型类定义（含 forward）→ 推理入口（main/CLI/pipeline）。关注 config.json 中的架构参数（hidden_size, num_layers, num_heads 等）。

### 已有文档阅读
- 阅读项目中的Readme、设计文档等
- 若有已有适配文档（如其他版本、同系列模型），先列出差异点：**可复用 vs 需纠偏**。

### 特殊输入识别
- 非文本输入（图像、蛋白质序列、音频等）需确认预处理路径和 tokenizer/encoder 是否独立。
- 检查 `collate_fn` / `DataCollator` 是否有设备绑定逻辑。

---

## 二、环境准备

完整的环境变量清单（含性能优化变量和 Python 级设置）见 [environment_reference.md](references/environment_reference.md)「环境变量配置清单」。

### CANN 环境诊断
```bash
npu-smi info                          # 确认芯片型号、驱动版本、卡数、确定空闲卡
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
        # 默认关闭以保确定性和兼容性——这是起点，不是不可协商的前提。
        # 验证后可逐个开启：jit_compile=True 可能提速但触发 tiling error；
        # allow_internal_format=True 可能提速但引入格式转换偏差。
        torch.npu.set_compile_mode(jit_compile=False)
        torch_npu.npu.config.allow_internal_format = False
    return torch.device(device_str)
```

> 这两个设置是**确定性与性能的权衡**，不是通用最优。默认关闭确保结果可复现和兼容性；若 profiling 显示 format 转换或 JIT 编译是瓶颈，可逐个开启并验证精度无退化。开/关的选择应基于 profiling 数据，不是默认值。

---

## 三、测试数据准备

**原则**：小而代表性，能快速复现问题，不浪费时间跑全量。

### 数据集构造
如果有现成的测试集，进行**轻量化**和**代表性抽样**；如果没有，则按以下原则构造

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

**路径构造模板**见 [profiling_collection.md](references/profiling_collection.md) §0。

**强制规则**：
- `tensorboard_trace_handler` 的路径参数必须指向 `<workspace>/profiling/<timestamp>/`
- 禁止使用 `/tmp`、系统临时目录或无时间戳的固定路径
- 每次采集后更新 `latest` 软链接，方便用户直接查看最新结果
- `profiling/` 目录已在 `.gitignore` 中排除（详见 05_engineering）

### 采集级别

| 级别 | 内容 | 数据量 |
|------|------|--------|
| **L0** | 仅采集 NPU 活动，最小膨胀 | 小 |
| **L1** | CPU + NPU + 算子详情 + 调用栈 + 内存 + AI Core 指标 + CANN 运行时 API 统计（覆盖全部 8 个解析脚本） | 大 |

### 采集流程

```
0. 环境校验（运行本 skill 的 scripts/validate_profiling_env.py）
→ 1. 确定采集场景（训练 / 推理）
→ 2. 选择采集级别（L0 /  L1 ）
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
2. **多推理路径分离采集**：不同推理模式（例如LLM中的prefill/decode, 蛋白质结构预测中的generate/scoring等）应分别建立独立 profiling 段，不混合采集。
3. **GPU/NPU 对称采集**：当需要跨平台对比时，在 GPU 上使用相同 schedule 配置 + `torch.profiler.ProfilerActivity.CUDA`。

### 框架适配指引

| 训练框架 | 接入方式 | 模板位置 |
|---------|---------|----------|
| 原生 PyTorch 循环 | `with torch_npu.profiler.profile(...) as prof` | [profiling_collection.md](references/profiling_collection.md) §1 |
| PyTorch Lightning | 自定义 Callback | [profiling_collection.md](references/profiling_collection.md) §2.1 |
| HuggingFace Trainer | 自定义 TrainerCallback | [profiling_collection.md](references/profiling_collection.md) §2.2 |
| DeepSpeed | 全卡采集，按 rank 分目录 | [profiling_collection.md](references/profiling_collection.md) §2.3 |

### 环境预检

采集前运行本 skill 提供的环境校验脚本（位于本 skill 的 `scripts/validate_profiling_env.py`）：
```bash
python <skill_path>/01_preparation/scripts/validate_profiling_env.py --device npu:x --output-dir ./profiling
```
> 注意：`<skill_path>` 是本 skill 所在目录的实际路径，agent 执行时需替换为真实路径。`npu:x` 指定要采集的 NPU 卡号，`--output-dir` 指定输出目录。

---

## 五、精度验证脚本构建

**目标**：在优化开始前，保存一份可信 baseline 输出，并构建可一键运行的对比脚本，使后续每次优化都能快速验证精度是否退化。

### Baseline 来源与对比策略

Baseline 来源、对比指标、阈值的确定**不在本阶段定义**——遵循 [04_accuracy_assurance/SKILL.md](../04_accuracy_assurance/SKILL.md) 的方法论：

- **Baseline 来源**：按优先级（官方基线 → 用户指定 → NPU 优化前自身输出），详见 [baseline_policy.md](../04_accuracy_assurance/references/baseline_policy.md)。不假定必须是 GPU 输出——取决于项目实际情况。
- **对比指标**：按输出类型选择（连续向量 → cosine + max_abs；离散序列 → 匹配率；聚合标量 → 相对误差），详见 04_accuracy_assurance「指标选择原则」。不硬编码特定指标。
- **阈值**：必须在比较前声明，不可事后调整。参考阈值见 04_accuracy_assurance，最终以项目实际情况为准。

### 输出保存约定

agent 应根据项目的输出类型选择保存格式，原则是：离线可加载、不依赖设备、可复现。

```python
import numpy as np, json

# 示例：连续输出 -> npy，离散输出 -> json
np.save(f"baseline/{sample_id}_output.npy", output.cpu().float().numpy())
with open(f"baseline/{sample_id}_tokens.json", "w") as f:
    json.dump({"tokens": token_ids}, f)
```

### 对比脚本

对比脚本必须是自包含的（给定 baseline 目录和当前输出目录即可独立运行），不依赖任何设备。指标和阈值在脚本中显式声明，运行后输出判定结论并保存为文件。

### 确定性验证

优化前的 baseline 采集和优化后的输出采集，必须在相同的确定性条件下进行。推理场景：`model.eval()` + `torch.no_grad()` + 固定输入 + 关闭随机性。详见 [04_accuracy_assurance/SKILL.md](../04_accuracy_assurance/SKILL.md)「确定性保证」。
