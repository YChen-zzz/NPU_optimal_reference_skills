# Phase 2 双线分析模型：源码结构线 + Profiling 数据线

## 问题陈述

当前 Phase 2 只有一条分析线：**profiling → 源码定位**。即先从 profiling 数据发现"什么慢"，再用源码找到位置。这条线对"可见瓶颈"有效（如算子慢、host dispatch 慢、通信等待），但对"结构性冗余"完全失效。

**结构性冗余的本质**：每次调用本身看起来合理（profiling 数据无异常），但从代码结构出发能发现"多次调用其实可以合并"或"重复计算可以提前做一次"。

典型案例：
- **QKV 权重合并**：Q/K/V 各自的 Linear 每次调用都正常，但三个共享输入的 Linear 可以合并为一次 GEMM
- **KV cache 预分配**：每步 `torch.cat(old_kv, new_kv)` 分配+拷贝看起来正常，但预分配 buffer + index 写入可消除反复分配
- **Cross-attention K/V 预计算**：encoder 输出不变，但每个 decoder step 都重新计算 K/V projection

这些优化点的共同特征：
1. profiling 中看不到异常（每个算子表现正常）
2. 只有理解代码结构（多个调用共享输入 / 跨步复用可能）才能识别
3. 属于四维度中的"去重"和"复用"，但触发条件不在 profiling 而在源码

## 分析：两条线的本质差异

| | Profiling 数据线 | 源码结构线 |
|---|---|---|
| 起点 | profiling 数据中的异常 | 源码的计算结构/架构模式 |
| 能发现的 | 可见瓶颈（某算子慢/某段空闲/某处 wait） | 结构性冗余（可合并/可复用/可预计算） |
| 源码的角色 | 辅助——定位异常的源码位置 | 主导——从结构识别优化机会 |
| profiling 的角色 | 主导——发现异常 | 辅助——验证优化点是否值得做（量化收益） |
| 典型产出 | "Transpose 占 15% → 换布局" | "Q/K/V 三个 Linear 共享输入 → 合并 GEMM" |
| 四维度覆盖 | 替换、掩盖 为主 | 去重、复用 为主 |

两条线正交,不是互斥——同一个项目应该两条线都跑。

## 方案设计：Phase 2 拆为双线

```
Phase 2: 分析
├── Line A: 源码分析（结构性冗余识别）
│     通读源码(穿透框架) → 四维度逐层审视 → profiling 量化收益
│
└── Line B: Profiling 数据分析（可见瓶颈定位）
      采集 profiling → 脚本分析 → 五种分析模式推理 → 源码定位根因
```

两条线独立产出优化候选,最终合并为统一的优化方案清单交 ★A 用户确认。

### Line A: 源码结构分析（新增）

**目标**：从代码出发,用四维度思维框架主动发现优化机会——不依赖 profiling 异常信号,不依赖 pattern checklist。

**分析对象(三层)**:

1. **模型结构层**:模型由哪些组件构成、怎么组织
   - 注意力类型(MHA/GQA/MQA/cross-attention)、FFN 结构、Norm 选择
   - 重复结构(N 层循环、共享权重)、生成模式(autoregressive/diffusion/encoder-decoder)
   - 这一层回答"模型是什么"

2. **实现逻辑层**:具体代码怎么实现这些组件
   - 数据流路径:tensor 从输入到输出经过哪些变换,有没有绕远路
   - 生命周期管理:哪些中间结果被创建后立即丢弃、哪些跨步复用
   - 控制流:哪些分支在推理时永远走/永远不走、哪些条件判断是多余的
   - 这一层回答"代码怎么做的,有没有浪费"

3. **算法层**:当前的计算方法是否有更优的等价实现
   - 同一功能是否存在计算复杂度更低的算法
   - 是否有硬件更友好的等价表达(不改功能但改实现路径)
   - 这一层回答"有没有更好的做法"

**⚠ 穿透框架层:找到真实源码**

很多模型的推理路径被框架层层包裹(如 HuggingFace generate → GenerationMixin → model.forward → 各种 hook/wrapper),直接看入口代码看到的是框架调度逻辑而非模型计算。

agent 必须穿透框架到达"真实的模型实现代码":
- **识别框架 wrapper**:看到 `model.generate()`/`trainer.predict()`/`pipeline()` 时,不要停在这里——这是框架入口,不是模型逻辑
- **定位真实 forward**:沿调用链向下,跳过 `Module.__call__` / hook dispatch / mixin 方法,找到项目自己的 `modeling_xxx.py` 中的 `forward()` 实现
- **区分两类代码**:
  - 框架代码(site-packages 下):调度/hook/logging/兼容性处理——这里的开销属于 Line B(dispatch overhead)
  - 模型代码(项目目录下):真实的计算逻辑——这里的结构性冗余属于 Line A
- **常见框架穿透路径**:
  - HuggingFace: `generate()` → `_generate_*()` → `model()` → `modeling_xxx.py:XxxModel.forward()`
  - vLLM: `LLMEngine.step()` → `model_runner.execute_model()` → `model.forward()`
  - 自定义脚本: `main()` → `model(input)` → 找到 `nn.Module` 子类的 `forward()`

**分析方法:四维度驱动,不是 pattern 匹配**

agent 通读源码后,**用四维度作为思维透镜**,对每个计算逻辑块主动提问:

```
去重: 这段计算有没有多余的工作?
  - 有没有重复调用(同一输入的多次独立计算可合并)?
  - 有没有死逻辑(推理时不产生效果的分支)?
  - 有没有不必要的数据变换(绕了一圈又变回来)?

复用: 这段计算的结果/资源有没有被浪费?
  - 有没有每步都重新计算但结果不变的东西(可预计算)?
  - 有没有每步都分配释放同一块内存(可预分配复用)?
  - 有没有跨步/跨层可以共享的中间结果?

掩盖: 这段延迟有没有机会藏起来?
  - 有没有串行但互不依赖的计算(可并行/重叠)?
  - 有没有可以和通信重叠的计算段?

替换: 这段计算有没有硬件更友好的等价写法?
  - 有没有更高效的算法(如标准attention → FlashAttention)?
  - 有没有 NPU 融合算子可覆盖这组计算?
  - 当前 API 在 NPU 上是否有更优的等价表达?
```

**关键:这不是 checklist 匹配,是 AI 自主推理。** 四维度是思维工具(教 agent"从哪些角度看"),不是答案列表。agent 面对全新的代码结构,应该能自己发现"这三个 Linear 共享输入 → 去重:可合并",而不是查表看到"QKV 合并"才想到。

**profiling 在 Line A 中的角色:量化,不是发现**

Line A 发现的候选点需要 profiling 数据来量化"值不值得做":
- 识别到 Q/K/V 可合并 → 查 op_statistic:这三个 Linear 分别占多少时间? 合并的理论收益?
- 识别到 KV 每步 cat → 查 operator_memory:每次 cat 分配多大? 累计占总内存多少?
- 如果占比极低(如 <1%)→ 不值得改,跳过

这避免了"发现很多理论上的优化机会但实际收益微乎其微"的问题。

### Line B: Profiling 数据分析（现有）

保持不变——五种分析模式 + profiling_to_action 参考路径 + 各脚本 + profiling_to_source 的桥梁定位。

Line B 的逻辑链已完整:"profiling 说 X 慢 → 用五种模式定位 → 源码确认根因 → 选优化维度"。不需要额外改动或加 pattern。


### 需要新增的

1. **在 Phase 2 的流程中明确"双线并行"**——不是"先 profiling 再源码",而是两条线同时启动
2. **源码分析的方法论(四维度驱动)**——教 agent 怎么"用去重/复用/掩盖/替换审视源码",不是"从 profiling 找慢点"
3. **穿透框架层的指引**——教 agent 跳过框架 wrapper 找到真实模型代码

## 落地方案

### 改动 1: 根 SKILL.md Phase 2 描述

当前:
```
Phase 2  Profiling 分析
         └─ 阶段前采集 L1（信息不足以定位优化点时改采 L2）
```

改为:
```
Phase 2  分析（双线并行）
         ├─ Line A: 源码分析（通读源码 → 四维度逐层审视 → profiling 量化收益）
         └─ Line B: Profiling 分析（采集 L1 → 脚本分析 → 五种模式推理 → 源码定位根因）
```

### 改动 2: 扩展 `source_code_analysis.md`

当前定位:"从 Profiling 信号到源码位置"——只服务 Line B。

增加一个前置 section "主动源码分析(Line A)",包含:
- 三层分析对象(模型结构 / 实现逻辑 / 算法)的说明和分析要点
- 四维度驱动的提问框架(去重/复用/掩盖/替换对每个逻辑块怎么问)
- profiling 量化的方法(怎么用现有脚本数据验证候选点的收益)
- 与 Line B 的协作(Line A 发现的候选如何与 Line B 发现的瓶颈合并排优先级)

不写"pattern checklist"——四维度提问框架本身就是 agent 发现 pattern 的方法,不预设答案。

### 改动 3: 目录改名 `02_profiling_analysis` → `02_bottleneck_analysis`

原名只覆盖 profiling 分析(Line B),新名涵盖"瓶颈分析"的完整含义(Line A 源码分析 + Line B profiling 分析都是为了定位瓶颈)。

改名后该目录承载:
- `SKILL.md`: Phase 2 的完整分析 SKILL(双线)
- `references/source_code_analysis.md`: Line A 方法论(扩展后)
- `references/profiling_to_action.md`: Line B 的五种分析模式 + 参考路径
- `references/profiling_to_source.md`: Line B 的桥梁方法论
- `references/profiling_scripts_guide.md`: 脚本使用指南
- `scripts/`: profiling 解析脚本(服务 Line B + Line A 的量化)

需要同步更新的引用:
- 根 SKILL.md 中的子技能索引和 Phase 2 描述
- 01_preparation/SKILL.md 中"进入 Phase 2 分析"的引用
- profiling_collection.md 中引用 02 目录的路径

选项 B 确认:Line A 的源码分析方法论放在 `source_code_analysis.md`(扩展),不新建独立子技能目录。

### 改动 4: 流程中明确双线合并点

两条线产出独立的候选清单:
- Line A 产出:"源码结构候选"(QKV 合并、KV cache 等)
- Line B 产出:"profiling 瓶颈候选"(Transpose 多、dispatch 慢等)

在 ★A 用户确认前,合并为统一清单,按"预期收益"排序(收益来自 profiling 量化)。

### 不做的事

- 不写 pattern checklist(用四维度思维替代,agent 自己推理)
- 不修改 profiling 脚本(Line A 不依赖脚本检测)
- 不新建独立子技能目录(通过 source_code_analysis.md 扩展承载)
- 不取消 Line B(两条线互补)

## 开放问题

1. **四维度是否完备**:你提到"可能还有别的原语"。从 Line A 的视角看,四维度(去重/复用/掩盖/替换)是否覆盖了所有"从源码能发现的优化机会"? 如果 agent 在实际源码分析中发现了四维度问不出来的优化点,那可能需要补充第五维度。这个可以通过 opt_explore 的 explore_db 积累后验证。

2. **Line A 的深度控制**:通读源码 + 四维度逐层审视可能产出大量"理论上可以优化"的点,但很多收益极低。profiling 量化是筛选机制,但 agent 需要有"什么时候停止源码分析"的判断——建议:当 profiling 数据显示候选点占总时间 <1% 时停止追这个点。

3. **与 opt_explore 的关系**:model_opt 的 Line A 用四维度框架,是"有方法论指导的源码分析";opt_explore 是"完全自由的探索"。如果 agent 在 model_opt Line A 中用四维度问不出东西,可以切换到 opt_explore 做无框架约束的自由探索。
