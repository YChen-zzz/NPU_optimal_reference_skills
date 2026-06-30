# NPU 模型适配优化 Skills

面向 AI Agent 的昇腾 NPU 模型适配与性能优化技能包。基于 [Anthropic Agent Skills](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills) 规范构建。

## 快速使用

### Kerminal CLI

克隆本仓库后，将 `model_opt` 目录软链接到 Kerminal 的 skills 目录：

```bash
git clone <本仓库地址>
ln -sfn $(pwd)/model_opt ~/.kerminal/skills/model_opt
```
或者直接下载到 Kerminal 的 skills 目录

Kerminal 会在启动时扫描 `~/.kerminal/skills/` 下的所有 `SKILL.md`，根据 `description` 字段自动匹配用户意图并加载。

### 其他 Agent 框架

本 Skills 遵循通用的 Agent Skills 目录规范：

- 每个 skill 是一个包含 `SKILL.md` 的目录
- `SKILL.md` 以 YAML frontmatter 开头（`name` + `description`）
- `references/` 存放按需加载的参考文档
- `scripts/` 存放可执行的确定性工具脚本
- `templates/` 存放输出模板

放置到对应框架的 skills 目录即可（如 `~/.claude/skills/`、`.agents/skills/` 等）。

## 目录结构

```
model_opt/
├── SKILL.md                          # 主 Skill：全流程、确认节点、子 skill 索引
├── references/
│   └── standardized_operations.md    # Profiling 采集与精度对比规范
│
├── 01_preparation/                   # Phase 1：环境搭建、数据准备、脚本构建
│   ├── SKILL.md
│   ├── references/
│   └── scripts/
│
├── 02_profiling_analysis/            # Phase 2：Profiling 数据分析 + 源码根因定位
│   ├── SKILL.md
│   ├── references/                   # 脚本指南、推理指南、诊断方法、源码分析
│   └── scripts/                      # 7 个 CANN CSV 解析脚本
│
├── 03_optimization/                  # Phase 3：基于三原语的优化实施
│   ├── SKILL.md
│   └── references/                   # 三原语详解、NPU checklist、算子参考...
│
├── 04_accuracy_assurance/            # Phase 4：推理/训练精度验证
│   ├── SKILL.md
│   ├── references/
│   └── scripts/                      # 精度对比工具
│
└── 05_engineering/                   # Phase 5：Git 管理、日志、文档
    ├── SKILL.md
    └── templates/
```

## Profiling 解析脚本

`02_profiling_analysis/scripts/` 提供 7 个 CANN profiling CSV 解析工具：

| 脚本 | 对应文件 | 用途 |
|------|---------|------|
| `parse_step_trace.py` | step_trace_time.csv | 设备利用率，判断瓶颈侧 |
| `parse_op_statistic.py` | op_statistic.csv | 算子耗时分布 + 异常检测 |
| `parse_kernel_details.py` | kernel_details.csv | 硬件单元、小算子、流水 stall |
| `parse_operator_details.py` | operator_details.csv | Host 开销 + Call Stack 源码定位 |
| `parse_memory_record.py` | memory_record.csv | 内存时间线、碎片化 |
| `parse_operator_memory.py` | operator_memory.csv | 逐 tensor 生命周期 |
| `diff_profiling.py` | 两份 profiling 对比 | 优化前后效果验证 |

所有脚本的统一接口：

```bash
python <script>.py <profiling_dir> [--rank N] [--top-k K] [--output file.txt]
```

`parse_kernel_details.py` 和 `parse_operator_details.py` 支持 `--filter` 模式对特定算子深入分析。

## 贡献指南

### 原则

1. **SKILL.md 保持精简**：作为调度中心，不超过 150 行。详细内容放 references/
2. **渐进式加载**：只在 `description` 中声明触发条件，不在 SKILL.md 中堆叠所有知识
3. **脚本做确定性工作**：可重复执行、输出稳定的操作用脚本；需要判断力的工作留给 Agent
4. **不做项目特定绑定**：references 和 scripts 中不硬编码项目路径或正则匹配特定代码
5. **经验可积累**：`npu_checklist.md`、`npu_operator_reference.md` 等文件可以持续增加条目

### 修改规范

**修改 SKILL.md**：
- 确保 frontmatter 的 `name` 和 `description` 准确反映触发条件
- 保持"调度中心"定位——只含流程/框架/索引，细节下沉到 references

**修改 references**：
- 每个文件聚焦一个主题，不超过 200 行
- 不重复其他文件已有的内容（用链接引用）
- 描述现象时不绑定到具体脚本名——用"当看到 X 时"而非"当 parse_xxx 输出 Y 时"

**修改 scripts**：
- 只依赖 CANN profiler 的固定 CSV 格式，不依赖项目代码
- 大文件用流式处理（heapq Top-K），不全量加载
- 输出结构：先数据事实，最后 Suspect Signals（疑点标记，不做最终判定）
- 所有脚本支持 `--output` 参数写文件

**新增文件**：
- 新增 reference 时在对应 SKILL.md 中添加索引和加载触发条件
- 新增 script 时同步更新 `profiling_scripts_guide.md`

### 测试

修改或新增脚本后，用实际 profiling 数据验证。例如：

```bash
# profiling 解析脚本示例
python 02_profiling_analysis/scripts/parse_op_statistic.py /path/to/profiling
python 02_profiling_analysis/scripts/parse_kernel_details.py /path/to/profiling --filter MatMul --top-k 5
```

确保：
- 脚本无报错，输出格式完整
- 大文件场景（如 >10M 行的 CSV）在合理时间内完成
