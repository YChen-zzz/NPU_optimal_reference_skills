---
name: NPU Profiling 分析
description: 分析昇腾 NPU profiling 数据以定位性能瓶颈并提出优化方向。当用户要求分析 profiling、定位瓶颈、查看性能数据时触发。
---

# NPU Profiling 分析

## 核心方法论：Profiling + 源码双驱动

Profiling 数据给出的是**现象**（哪个操作慢、耗时多少），不是原因。定位根因必须结合**功能性源码的具体实现**：

```
Profiling 定位现象        源码定位根因
      ↓                      ↓
"aten::matmul 调用 500 次"  → 为什么调用 500 次？→ 阅读 forward 实现发现循环中逐 token 调用
"empty_tensor 大量出现"     → 谁在分配？→ 追踪到某个 layer 每次 forward 都 new 一个 buffer
"Transpose 算子占比 30%"    → 为什么需要 transpose？→ 源码中 weight 形状与算子期望布局不一致
```

**关键原则**：

1. **不要停在调用层**：profiling 告诉你 `aten::mm` 慢，但原因可能在上层——是谁调用的、为什么这样调用、能否换一种方式
2. **向下追溯实现**：找到 profiling 热点对应的 Python 代码后，继续深入该函数/模块的内部实现
3. **向上追溯调用链**：某操作调用次数异常时，追溯是哪个循环/递归产生的
4. **区分必要 vs 冗余**：相同操作可能既有必要调用也有冗余调用，源码分析才能区分

### 源码分析（根因定位）

Profiling 定位到热点后，需要在源码中回答"为什么"：
- **向上追溯**：谁调用了这个操作、循环了多少次、能否在更高层消除
- **向下深入**：操作内部有没有不必要的分支、分配、转换
- **全局认知**：理解模型层级结构和框架交互模式

详见 [source_code_analysis.md](references/source_code_analysis.md) 获取系统性的源码分析方法。

## Profiling 文件索引

CANN profiler 输出以下文件，每个文件提供不同维度的信息：

| 文件 | 信息维度 | 典型大小 | 关键用途 |
|------|---------|---------|---------|
| `step_trace_time.csv` | 每 step 的 Computing/Free/Comm 时间 | 几行 | 判断瓶颈在 host 侧还是 device 侧 |
| `op_statistic.csv` | 按算子类型聚合的 count + 总耗时 | ~100 行 | 全局视图：哪类算子最耗时 |
| `kernel_details.csv` | 每个 kernel 的执行时间、等待时间、硬件单元、shape | 1K-100K 行 | 最丰富的文件：硬件利用、小算子、并行度、流水 stall |
| `operator_details.csv` | 每次算子调用的 host/device 时间 + Call Stack | 100K-20M 行 | 唯一能关联到 Python 源码行的文件 |
| `memory_record.csv` | 按时间采样的 Reserved/Allocated 内存 | 30K-1M 行 | 内存时间线、峰值定位 |
| `operator_memory.csv` | 每个 tensor 的 size、lifetime、分配时全局状态 | ~10K 行 | 逐 tensor 生命周期，buffer 复用分析 |

## 瓶颈分类

| 类型 | Profiling 表现 | 核心问题 |
|------|---------------|---------|
| **Host-Bound** | 设备利用率低（Free >> Computing） | host dispatch、Python 开销、同步 |
| **Compute-Bound** | 利用率高，kernel 耗时大，mac_ratio 高 | 算子本身计算密集 |
| **Memory-Bound** | 利用率高但 mte_ratio >> mac_ratio | HBM 带宽瓶颈 |
| **Allocator-Bound** | 类似 Host-Bound 但 empty_tensor 占比高 | allocator 同步阻塞 |

> Host-Bound 与 Allocator-Bound 易混淆——都表现为"设备空闲"。区分：operator_details 中 `empty_tensor` Host Duration 占比高 → Allocator-Bound；Python dispatch wrapper 占比高 → Host-Bound。

详见 [host_bound_patterns.md](references/host_bound_patterns.md) 获取 Host-Bound 深度诊断方法。
详见 [memory_profiling.md](references/memory_profiling.md) 获取显存峰值分析和常见陷阱。
详见 [profiling_to_action.md](references/profiling_to_action.md) 获取"profiling 特征 → 具体行动"映射表。

## 解析脚本

本 skill 的 `scripts/` 目录提供 Profiling CSV 解析工具。详细使用说明见 [profiling_scripts_guide.md](references/profiling_scripts_guide.md)。

> 脚本位于本 skill 的 `scripts/` 目录，执行时需使用实际路径。以下 `$S` 代表该目录。

### 典型工作流

**流程 1：首次分析新 profiling**
```
$S/parse_step_trace.py <dir>         → 判断瓶颈侧（host or device）
$S/parse_op_statistic.py <dir>       → 哪类算子最耗时
$S/parse_kernel_details.py <dir>     → 硬件单元、小算子、流水 stall
$S/parse_memory_record.py <dir>      → 内存峰值、碎片化、分配趋势
$S/parse_operator_memory.py <dir>    → 逐 tensor 生命周期，buffer 复用机会
```

**流程 2：深入某个可疑算子**
```
$S/parse_kernel_details.py <dir> --filter Transpose   → 该算子的 shape、硬件利用、性能分布
$S/parse_operator_details.py <dir> --filter Transpose → 从哪行源码触发的（Call Stack）
```

**流程 3：优化效果验证**
```
$S/diff_profiling.py <before> <after> → 算子耗时 diff + 内存峰值变化
```

### 注意

- 脚本只做**数据提取和疑点标记**，不做优化决策
- 拿到脚本输出后，结合源码分析定位根因（参见上方「核心方法论」）
- `kernel_details.csv` 是信息量最大的文件——其他文件信息不够时回到它做深入分析
- `operator_details.csv` 是唯一有 Call Stack 的文件——需要定位源码时用它

## 下一步

分析完成后，整理优化建议清单，进入主 SKILL.md 的**确认节点 A**——向用户展示方案并等待确认后，再进入 03_optimization 实施。
