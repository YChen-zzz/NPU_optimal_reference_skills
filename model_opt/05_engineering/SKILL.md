---
name: npu-engineering-practice
description: 工程化实践：目录规划、一键脚本、版本管理、文档维护。当用户需要组织代码目录、设计一键脚本、管理 git 分支与提交、或维护优化日志时触发。
---

# NPU 适配工程化实践

## 工程目录规划

### 命名原则
- 目录命名按功能语义：
- GPU/NPU 推理目录结构对称，方便对比

### 推荐结构
```
project/
├── gpu_inference/          # GPU baseline 推理
│   ├── run.sh
│   └── infer.py
├── npu_inference/          # NPU 优化推理
│   ├── run.sh
│   ├── infer.py
│   └── patches/           # NPU 适配补丁
├── accuracy/               # 精度验证
│   ├── compare.py
│   └── baselines/          # baseline 数据
├── profiling/              # 性能分析
│   ├── output/
│   │   ├── 20260615_143022/
│   │   └── latest -> 20260615_143022
│   └── analyze.py
├── docs/                   # 优化报告与记录
└── scripts/                # 通用工具脚本
```

### 卫生习惯
- profiling 输出带时间戳 + `latest` 软链接
- 定期清理 `__pycache__`、`.pyc` 文件
- `.gitignore` 覆盖大文件（模型权重、profiling trace）

## 一键脚本设计（run.sh）

### 核心原则
- 一个脚本产出全部结果（推理输出 + profiling trace）
- 时间戳防覆盖，软链接指向最新

### 模板
```bash
#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# 软链接指向最新结果
ln -sfn "${TIMESTAMP}" output/latest

echo "=== 开始推理 (预计耗时: ~2min) ==="
python infer.py \\
    --output_dir "${OUTPUT_DIR}" \\
    --num_samples 50 \\
    2>&1 | tee "${OUTPUT_DIR}/run.log"

echo "=== 完成，结果保存在 ${OUTPUT_DIR} ==="
```

### 设计要点
- GPU/NPU 两边脚本步骤对称（相同参数、相同数据）
- 注释耗时预期（帮助判断是否卡住）
- `set -euo pipefail` 确保错误立即暴露
- 日志同时输出到终端和文件（`tee`）

## Git 版本管理

### 分支策略

```
main 分支（稳定版本，不直接修改）
  └── optimize/<任务名>（主工作分支）
        ├── 逐批实施优化，每批 commit
        ├── explore/<方案A>（子分支，探索有冲突的优化路径）
        └── explore/<方案B>
```

**工作分支规则**：
- 所有优化工作均在 `optimize/` 主工作分支进行，不在 main 上直接修改
- 仅当用户确认优化效果后，才将工作分支合入 main

**子分支探索**（`explore/` 分支）：对互相冲突但都可能带来收益的优化方案，从工作分支开子分支并行评估，选定方案后合并回工作分支。

### .gitignore 配置

profiling trace 文件大、临时文件多，不应纳入 git。详见 [templates/gitignore_template.md](templates/gitignore_template.md)。

应当纳入 git 的：源代码、脚本、文档、对比结果的摘要（非原始大文件）。

### 提交纪律

- **提交前必须完成以下步骤**（缺一不可）：
  1. 全量精度验证通过（Level 2，与原始 baseline 对比）
  2. Profiling 确认有收益（重新采集，对比优化前后数据）
  3. evidence_db 案例已按 [schema](../06_evidence_db/schema.md) 记录到项目工作目录的 `evidence_db/` 下
  4. 用户确认提交（展示总结 + evidence_db 路径，等待用户同意）
- 每批优化一个 commit，不要多批合并
- 实验性尝试前确认有干净的回退点（以当前工作分支 HEAD 为基点）

### 提交前用户确认流程

git commit 前，**必须**向用户展示本批总结并使用 `ask_user_question` 等待确认：

1. **总结内容**（必须包含以下信息）：
   - 本批实施的优化点列表及简述
   - 性能数据对比（优化前 vs 优化后，含数据来源）
   - 精度验证结果（指标 + 通过/未通过）
   - 修改的文件列表
   - 未采纳的方案及原因（如有）
2. **询问用户**：
   - 是否确认提交本批优化
   - 是否需要回退某些改动再提交
3. **用户确认后**才执行 git commit
4. 如用户要求调整，完成调整后重新验证精度和 profiling，再次走确认流程

commit message 格式：
```bash
# 格式: [类型] 内容: 简要数据
git commit -m "[perf] 融合 RmsNorm: encoder 延迟 -30%, cos=0.9999"
git commit -m "[perf] Flat Forward: 绕过 Module.__call__, throughput +45%"
git commit -m "[perf] Prefetch Buffer: 消除 empty_tensor, latency -20%"
git commit -m "[fix] 修复 causal mask 方向: 生成结果与 baseline 全量一致"
git commit -m "[revert] 回退 StaticCache: 在 NPU 上比 DynamicCache 更慢"
```

### 回退原则
- 回退要干净——不留 `if use_xxx` 条件分支
- 删除相关代码，而非注释掉
- 回退后重新运行快速验证

## 优化日志维护

**触发时机**：每批优化全量验证通过、git commit 完成后，必须更新一次优化日志。

日志条目格式和示例详见 [templates/optimization_log_template.md](templates/optimization_log_template.md)。

## 文档维护

### 优化报告结构

```
1. 环境信息（硬件、固件、框架版本）
2. 总体对比（GPU vs NPU 关键指标）
3. 分项优化技术（每项含 what + why + 效果）
4. 算子级分析（Top-N 耗时算子对比）
5. 精度验证结果
6. 未采纳方案（及原因）
7. 后续优化方向
```

### Why 描述模板

不同类型的优化需要不同的解释方式：

- **算子融合类**：
  "原始 N 个 kernel → 1 个融合 kernel，消除 N-1 次 dispatch gap"
- **内存管理类**：
  "原始每步动态分配 → 预分配 buffer，消除分配开销"
- **格式优化类**：
  "原始 4D matmul 触发物理 Transpose → 3D bmm 消除 runtime transpose"
- **调度优化类**：
  "原始串行执行 → 流水并行，提高硬件利用率"

### 维护纪律
- 代码迁移/重命名后同步更新文档中的路径引用
- 每个优化点描述必须包含 what（做了什么）+ why（为什么有效）
- 性能数字必须标注来源（哪次 profiling 的结果）

## Baseline 脚本生成

### 原则
- 逐一禁用所有优化点（不设置环境变量，不调用 patch，用原始 HF 模型）
- 保持其他条件完全一致（数据集、profiler 配置、warmup 步数、推理参数）
- 脚本自包含（不 import 有 side effect 的模块）

### 验证 baseline 有效性
```python
# 确认 baseline 脚本没有意外引入优化
import sys
assert 'torch_npu' not in sys.modules, "Baseline 不应加载 torch_npu"
# 或根据实际情况检查其他优化模块
```

## 性能数据校验

- 每个性能数字都要能追溯到具体的 profiling 输出文件或 perf 报告
- profiling 目录可能是 symlink → 检查实际内容（`readlink -f`）
- baseline 不一定是最早的目录 → 按 commit 或标注确认
- 对比时注明 warmup 策略（首次运行通常偏慢）

## 常见交互模式

根据用户反馈选择对应的响应策略：

| 用户反馈 | 响应策略 |
|---------|----------|
| "不对" | 重新审视分析，沿用户方向验证 |
| "过于复杂" | 识别最小完成单元，直接执行 |
| "精度能保证么" | 扩大验证范围（维度 + baseline） |
| "不要简单归结为 XX" | 做精细定位（逐层/逐步 diff） |
| "不要改计算逻辑" | 只改执行方式，不改计算流程 |
| "回退吧" | 立即回退 + 清理残留 + 记录原因 |

### 方案探索顺序

按风险递增排列，优先尝试低风险方案：

```
低风险（eager 优化）
  → 代码重构、冗余消除、参数调优
  → 不改变计算逻辑，回退成本低

中风险（融合算子）
  → 替换为 NPU 融合算子
  → 需精度验证，回退需恢复原始实现

高风险（dtype 变更）
  → FP32 → FP16/BF16、混合精度
  → 可能影响精度，需全量验证
```
