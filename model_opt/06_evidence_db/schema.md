# 案例库 Schema 说明

## 定位

本文件是案例库的**构造语法说明**,agent 据此在项目工作目录的 `evidence_db/` 下记录优化案例。

> **路径约定**: 案例数据存在**项目工作目录** `<workspace>/evidence_db/` 下(与 `profiling/` 同级),**不在 skill 目录中**。skill 目录只存本 schema 定义。

## 当前阶段目标

现阶段案例库的唯一目的是**把优化过程中的信息尽可能完整地记录下来**。消费端(检索、匹配、统计)是后续的事——先确保"存对了",再谈"怎么用"。

## Schema 字段说明

```yaml
- id: <string>
  # 唯一标识符。格式: <模型架构缩写>-<核心现象关键词>-<日期YYYYMMDD>
  # 例: "llm-transpose-layout-20260706", "moe-scatter-backward-20260710"
  # 规则: 用现象/根因描述而非模型全名,确保跨项目可检索

  phenomenon:
    # 记录 agent 从 profiling 中观察到的所有相关信号,尽可能完整
    signals:
      - source: <string>  # 产出此信号的脚本名+参数,如 "parse_op_statistic" 或 "parse_kernel_details --filter Transpose"
        content: <string>  # 脚本输出的原文摘录(关键数值+判断),不做解读,只记事实
    raw_context: <string, optional>
      # 补充任何脚本没覆盖但 agent 观察到的信息
      # 如: trace_view 中肉眼看到的 pattern、源码中发现的结构特征

  analysis_path:
    # 记录从现象到根因的完整推理过程,按实际执行顺序
    steps:
      - action: <string>   # agent 做了什么(跑了什么脚本/读了什么代码/做了什么推理)
        observation: <string>  # 得到了什么结果/看到了什么
        reasoning: <string, optional>  # 为什么做这一步/这个结果说明什么

  root_cause:
    description: <string>  # 最终确认的根因
    bottleneck_type: <enum, optional>
      # Host-Bound / Compute-Bound / Memory-Bound / Allocator-Bound / Execution-Mode
      # 不明确或属于多种: 写 "mixed" 并在 description 中说明
    evidence: <string>  # 支撑此根因判断的关键证据

  optimization:
    attempts:
      - description: <string>  # 做了什么改动
        dimension: <string, optional>
          # eliminate_redundancy / reuse_and_precompute / hide_latency / equivalent_substitution
        implementation_detail: <string>  # 具体代码层面怎么改的(文件、函数、改法)
        equivalence_verification:
          method: <string>  # 怎么验证等价性的
          result: <string>  # 验证结果(通过/失败,具体数值)
        performance_result:
          metric: <string>  # 用什么指标衡量
          before: <string>
          after: <string>
          verdict: <string> # accepted / rejected / partial
        failure_reason: <string, optional>  # 如果 rejected,为什么失败

  final_state:
    adopted: <string>  # 最终采纳了哪个方案(或"无方案被采纳")
    end_to_end_before: <string>
    end_to_end_after: <string>
    remaining_bottleneck: <string, optional>  # 优化后暴露的新瓶颈
    is_terminal: <bool, optional>  # 是否判定为终局

  context:
    hardware: <string>   # 如 "Ascend 910B"
    cann_version: <string>  # 如 "CANN 8.0.0 / torch_npu 2.3.1"
    model_arch: <string>  # 架构类型而非具体模型名
    input_spec: <string>  # 测试输入规格
    profiling_level: <string>  # "L0" / "L1" / "L2"
    date: <string>  # YYYY-MM-DD
    notes: <string, optional>
```

## 填写原则

1. **完整优先**: 不确定某信息是否有用时,记下来。optional 字段能填就填。
2. **原文摘录**: phenomenon.signals.content 和 analysis_path.steps.observation 尽量贴脚本原始输出。
3. **失败必记**: optimization.attempts 中 rejected 的方案和 failure_reason 是最有价值的信息。
4. **一次优化阶段一个文件**: 每经过一轮完整的 Phase 2→4,写一个案例文件。
5. **不强求归类**: bottleneck_type 和 dimension 能判断就写,判断不了写"unclear"。

## 目录结构

```
<workspace>/
├── profiling/           # profiling 数据
├── evidence_db/         # 案例库(项目工作目录下)
│   ├── <id>.yaml        # 每个案例一个文件,扁平存放
│   └── ...
└── ...
```

文件名 = id 字段值 + `.yaml`
