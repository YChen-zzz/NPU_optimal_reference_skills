---
name: npu-model-adaptation
description: NPU 模型适配全流程：环境检测与隔离配置 → 模型权重下载与格式管理 → 严格按原始文档实现推理脚本 → 分层精度验证。当用户需要把新模型跑通在 NPU 上、下载和管理模型权重、编写推理脚本、或验证适配后精度正确性时触发。
---

# NPU 模型适配

## 核心原则

- **原始文档为准**：推理配置必须来自模型 config.json / README / paper，不凭经验猜测
- **精度可验证**：完成标准不是"能跑"，而是"输出与原始实现等价"
- **环境可复现**：隔离、锁版本、一键激活
- **最小改动**：仅 `"cuda" → "npu"`，不改模型逻辑

## 工作流

```
Phase 1  环境检测与隔离  →  Phase 2  权重获取  →  Phase 3  推理实现  →  Phase 4  精度验证
```

### Phase 1: 环境检测与隔离

1. 硬件检测 (`npu-smi info`, CANN 版本, 架构, 磁盘空间)
2. 版本配套确认 → [references/environment.md](references/environment.md)
3. 向用户确认隔离方案 (Docker / venv / 系统直装)
4. 安装依赖，验证 `torch.npu.is_available()`
5. 生成一键环境启动脚本 (`source set_env.sh` + activate venv)

### Phase 2: 模型权重获取

使用 [scripts/weight_manager.py](scripts/weight_manager.py)：

```bash
# 探测当前网络能访问哪个源 (ModelScope > hf-mirror > HuggingFace)
python scripts/weight_manager.py detect-source

# 下载权重到标准目录。自动优选 safetensors 格式；若模型只提供 .bin 则 fallback 使用原格式，不做转换
python scripts/weight_manager.py download --model-id {id} --local-dir {model}/weights/

# 如果下载源同时提供了 safetensors 和 bin (冗余)，删除 bin 节省空间
python scripts/weight_manager.py cleanup --weights-dir {model}/weights/

# 加载权重验证完整性：确认无 missing keys、文件未损坏
python scripts/weight_manager.py verify --weights-dir {model}/weights/
```

目录约定：`{model}/weights/` 放权重和配置，`{model}/script/` 放推理代码和测试数据。

### Phase 3: 推理脚本实现

**配置溯源**：按优先级从模型自带文件中提取每个参数——

```
config.json / preprocessor_config.json  (最权威)
  → README / Model Card
    → 原始论文
      → 官方示例代码
```

优先使用官方封装 (`AutoImageProcessor`, `AutoTokenizer`, `Pipeline`)。手动实现时逐项交叉验证。

**NPU 适配**：设备迁移后检查精度。已知平台问题见 [references/environment.md](references/environment.md)，推理踩坑案例见 [references/known_issues.md](references/known_issues.md)。

**禁止**：不查文档凭经验设参数、随意减步数、忽略 dtype。

### Phase 4: 精度验证

两步走：**冒烟** → **按输出性质选择验证策略**。

**冒烟**（所有模型必做）：输出非 None、无 nan/inf、shape 正确。

**按输出性质选择验证策略**：

| 输出性质 | 典型模型 | 验证策略 |
|---------|---------|---------|
| 确定性 + 可枚举 | 分类、匹配、QA | 构造有标准答案的输入，精确断言 (assert top1=="cat") |
| 确定性 + 连续值 | 特征提取、嵌入 | 跳过语义检查，直接做数值对齐 (CPU vs NPU cosine/max_abs) |
| 随机性 + 可感知 | 图像/视频生成 | 合法性检查 (分辨率/像素范围/方差) + 感知指标 (CLIP-score 对 prompt 的匹配度) |
| 随机性 + 领域约束 | 蛋白质设计、分子模拟 | 领域合法性约束 (合法残基/能量有限/守恒律) + 统计基线 (论文报告的恢复率/RMSD 范围) |

数值对齐工具（适用于确定性+连续值策略）→ [scripts/compare_baseline.py](scripts/compare_baseline.py)

**决策逻辑**：先判断模型输出属于哪一类，再选对应策略。不要对所有模型套同一套验证方法。
