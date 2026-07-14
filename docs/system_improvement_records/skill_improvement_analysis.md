# Skill 改进分析与落地方案

> 基于 skill_improvement_proposal.md 的缺口分析 + 当前 model_opt skills 内容 + 实际 eval session 暴露的问题，逐条回应 concerns 并给出可落地的方案。

## Concern 1: 等价变换候选生成方法论(缺口1)太薄弱/特例化

### 重新审视: 等价替换是独立于三原语的第四维度

我之前错误地把等价变换归为"去重的子集"。用具体例子反驳自己:

- `prod(dim)` → `sum(dim) == shape[dim]`:原始 prod 是**必要计算**,不是冗余,但 ReduceSum 在 NPU 上有高效 AI Core 实现而 ReduceProd 没有
- `F.one_hot(x).to(float)` → `zeros().scatter_()`:语义完全一样,后者 NPU 路径更短
- 一组 `Q*K^T → scale → mask → softmax` → `npu_fusion_attention`:逻辑等价,但融合算子在硬件上是完全不同的执行路径

这些都不是"去重"——没有任何"不必要的工作"被消除。两种实现做的**逻辑工作一样多**,差别纯粹在于物理执行路径对硬件的亲和度不同。

如果把这也叫"去重",则任何优化都是"去重"(都在减少某种 cost),三原语框架退化为无用。

### 四维度模型(三原语 + 等价替换)

| 维度 | 核心问题 | agent 的思维方向 |
|------|---------|-----------------|
| **去重** | 这个工作该存在吗? | 找能删掉/合并的 |
| **复用** | 这个结果已经算过了? | 找能缓存/预计算的 |
| **掩盖** | 这段延迟能藏起来? | 找能并行/重叠的 |
| **替换** | 同样的结果有没有硬件更便宜的写法? | 找数学等价但 NPU 更友好的实现 |

**替换与其他三者正交**: 可以对已经去重后的关键路径做替换(换更好的 NPU 算子);可以对被复用的计算做替换(缓存的东西换个更快的算法去算);可以对掩盖中的计算做替换。

### 等价替换的候选生成方法论

agent 面对一个算子/算子组想做"替换"时,搜索空间有三层:

```
层 1: NPU 融合算子替换(最优先)
  检查当前的算子组合是否有官方融合算子覆盖(npu_operator_reference.md)
  例: QKV + attention → npu_fusion_attention
      RMS + Norm → npu_rms_norm
      Gate + SiLU + Up → npu_swiglu / npu_ffn
  特点: 收益最大、风险最低(官方实现,精度有保证)

层 2: 换等价 API(同逻辑,不同 NPU kernel 映射)
  同一数学运算在 PyTorch 中有多种 API 表达,它们映射到不同的 NPU kernel:
  例: prod(dim) → sum(dim) == shape[dim]  (ReduceProd → ReduceSum + Equal)
      index_select → gather → embedding  (不同的 gather 实现)
      tensor[i,j] = v → view(-1).scatter_()  (逐元素 host 下发 → 单次 device scatter)
  方法: 列出当前算子的数学语义 → 搜同语义的其他 torch API → 对比两者在 kernel_details 中映射到什么 NPU 算子
  
层 3: 换算法(同结果,不同计算路径)
  例: per-token conv(unfold → pad+stack) 本质上两者都在做滑窗聚合,但 stack 路径的 backward 更 NPU 友好
      attention(标准) → FlashAttention(分块重计算)
      sort-based TopK → approximate TopK
  方法: 回到算法层面,问"这个功能在数学上还有什么别的实现方式",再对比各实现的 NPU kernel 链
```

**关键约束**: 每次替换后必须做等价性验证(见 Concern 3),因为替换是改实现不改语义——语义不变是铁约束。

### 落地方案

1. **在 `03_optimization/SKILL.md` 的三原语表中增加第四行"替换"**,并新建对应 reference
2. **新建 `03_optimization/references/equivalent_substitution.md`**: 包含上面三层搜索空间 + 每层的触发条件、搜索方法、验证要求
3. **从 `eliminate_redundancy.md` 中抽出"NPU 算子实现差异"小节**(prod→sum 那些例子),移到新文件,因为它们本质上是"替换"不是"去重"
4. 在主 SKILL.md 的"优化三原语"改为"优化四维度"(或保留"三原语"名称但明确标注等价替换为正交的第四维度)

**文件落地位置**: `03_optimization/references/equivalent_substitution.md`(新建)

---

## Concern 2: NPU 推理规则——是否需要补充大量规则?

### 问题本质

proposal 的"NPU 预判模型"列了 5 条规则,但你担心:
- 补很多规则维护成本高,且覆盖不全
- 当前"挨个尝试"虽然效率低,但泛化性和准确性有保证
- 规则可能过时(NPU 迭代快)

### 我的判断: 不需要独立的"规则库",规则应被消解到分析模式中

与其让 agent "背规则",不如让规则以两种方式自然出现:

**方式 1: 脚本直接检测并输出**(自动化的事实)
- compile 贯穿全程 → parse_trace_view §4 已自动判定 A/B 类并输出结论
- Accelerator Core = AI_CPU → parse_kernel_details 可自动检测并标记
- aic mte >> mac + shape 极小 → parse_kernel_details 的 Suspect Kernels 已覆盖

这些本质上是**脚本层的异常检测**(分析模式中的"异常定位"),agent 不需要知道规则,只需读脚本输出。

**方式 2: profiling_to_action 的信号组合路径**(agent 的推理参考)
当脚本输出多个信号、agent 需要组合判断时,文档给出推理路径作为参考。这些路径本身蕴含了规则,但 agent 是"按图索骥"而非"背规则":

```
脚本输出: compile 3066 个,分布贯穿全程(B 类)
  + step_trace: 利用率 29%
  + kernel_details: 硬件占比正常
→ 路径: 问题在执行模式(在线编译),不在算子层面
→ 行动: 关 jit_compile / 固定 shape / 图编译
```

Agent 不需要事先知道"NPU 上 aclop 路径会每步编译"这条规则——它只需要:
1. 跑脚本 → 看到 compile B 类 + 利用率低 + 硬件正常
2. 查 profiling_to_action → 匹配到这条路径
3. 执行对应行动

### 与"挨个尝试"的关系

这套架构**不取代试错**。它的价值是:
- 在做等价替换(Concern 1)时,通过**差异对比**模式(替换前后跑 profiling 看变化)快速判定某个候选是否有效
- 在候选太多时,通过脚本自动检测的信号做**初筛**(如发现某算子落 AI_CPU → 这个算子值得替换;发现 compile B 类 → 不是算子问题,不用在算子层面试)
- 试错本身被五种分析模式结构化了——不是盲试,而是"试→差异对比→纵向深入定位原因"

### 落地方案

- **不写独立的"NPU 规则库"文档**——避免规则膨胀和维护负担
- 在脚本中增加自动检测(部分已做: compile A/B; 待做: AI_CPU 标记)
- 在 `profiling_to_action.md` 中,用五种分析模式的框架组织现有和新增的信号组合路径
- 规则的"正确性保证"来自真实 profiling 数据验证,不来自人工维护——每条路径旁标注"来源: 经 XX 数据验证",随着案例库(Concern 6)积累,路径越来越可靠

---

## Concern 3: 等价性验证 + 与三原语的关系 + 是否还有别的原语

### 3.1 推理场景的等价性验证

proposal 给的协议偏训练(含 backward 梯度对比)。推理场景简化为:

```
推理等价性验证协议:
1. 构造代表性输入(覆盖: 最短/中位/最长序列, padding 边界, batch=1 和 batch>1)
2. Forward 对比: cosine >= 0.9999, max_abs_diff < 1e-4(fp32) / < 1e-2(fp16)
3. 关闭随机性: eval(), torch.manual_seed, dropout=0
4. 多次运行确认确定性: NPU 同输入应 bit-exact(jit_compile=False + allow_internal_format=False)
5. 边界: 空输入 / 单 token / 超长截断
```

**文件落地**: 新增 `04_accuracy_assurance/references/equivalence_verification.md`,与现有 `checklists.md` / `debugging_guide.md` 并列。

### 3.2 等价替换与原有三原语的关系

经 Concern 1 分析,等价替换**是独立于三原语的第四维度**,不是"去重"的子集。修正之前的判断:

```
原三原语: 去重 / 复用 / 掩盖  → 回答"工作量能否减少/复用/掩藏"
第四维度: 替换                → 回答"同样的工作有没有硬件上更便宜的等价写法"
```

**两者的交叉点**:一些手段兼具两个维度——如融合算子既是"替换"(换了实现)也有"去重"效果(消除 dispatch gap)。但维度本身正交:可以对已去重的关键路径做替换,也可以不去重直接替换。

**落地**: 在 `03_optimization/SKILL.md` 中将"优化三原语"升级为"优化四维度"(去重/复用/掩盖/**替换**),并新建 `equivalent_substitution.md` 作为第四维度的 reference。

### 3.3 还有没有别的原语?

在补入"替换"后,四维度的完备性检验:
- 去重: 减少工作量(同样结果、更少计算/搬运)
- 复用: 利用已有结果(空间换时间)
- 掩盖: 并行化延迟(让等待不可见)
- 替换: 换等价但硬件更友好的实现(同工作量、更低物理代价)

其他候选:
- "精度降级"(量化/混合精度): 跨维度——既是"替换"(换低精度算子)也有"去重"(减少 HBM 搬运量)
- "编译优化"(图编译/kernel 融合): 主要是"掩盖"(消除 dispatch gap)+"替换"(编译器重组 kernel)

**结论**: 四维度覆盖完备,无遗漏的正交方向。

---

## Concern 4: 训练场景暂时忽略

已确认。后续所有落地方案聚焦推理。proposal 中的"缺口 4"(训练推理线)暂不纳入当前迭代。

---

## Concern 5: 脚本诊断 pattern 是否正确 + 信号组合推理路径

### 问题本质

当前 profiling_to_action.md 给的是一堆"看到 X+Y → 结论 Z"的**具体实例**,但:
- 这些实例覆盖有限,遇到新场景 agent 不知道怎么推理
- 改 pattern(修正某条实例)可能方向不对——因为底层缺方法论
- 应该教 agent **怎么做分析**,而不是**给一堆分析结论让它匹配**

### 核心改进: 五种 Profiling 分析模式(方法论)

实际性能分析中,人使用 profiling 数据推理时,本质上只有五种思维模式。把它们显式化,agent 就能自己构造新的推理路径,不依赖预定义实例:

**模式 1: 横向关联(广度结合)**

同一个问题从多个文件/维度同时观察,交叉验证收敛到可靠结论。

```
目的: 消除单信号歧义(单信号有多种解释,多信号交叉才能确认)
方法: 针对同一怀疑点,分别从 step_trace / kernel_details / trace_view / operator_details 提取相关信息,看是否指向同一结论
何时用: 单一脚本给出模糊判断(如"Preparing 高")时,需要用其他文件确认或排除
```

**模式 2: 纵向深入(沿一个点逐层钻入)**

从高层信号出发,层层递进到更细粒度的数据,直到定位到源码行级的根因。

```
目的: 从"模糊的慢"收窄到"具体哪行代码、为什么慢"
方法: 每一步用更细粒度的文件回答上一步留下的"为什么"
何时用: 已确定一个可疑点(如某算子占比高),需要追到根因
典型路径: op_statistic(哪类) → kernel_details(为什么) → operator_details(谁触发) → 源码(为什么这样写)
```

**模式 3: 差异对比(两个状态的 delta)**

比较两份 profiling(优化前后 / L0 vs L1 / GPU vs NPU),变化量本身就是信息。

```
目的: 隔离变量,看什么变了什么没变
方法: 对比同一指标在两个状态下的差异,差异指向变化的原因
何时用: 判断优化是否生效、区分 profiler 开销与真实瓶颈、跨平台对比
关键: L0 vs L1 对比可分离 profiler 注入开销;优化前后对比用 diff_profiling
```

**模式 4: 异常定位(分布中找离群)**

看同类数据的分布,找打破 pattern 的异常点。

```
目的: 从大量正常数据中精确定位少数真正有问题的点
方法: 看分桶/分位数,找显著偏离均值的条目
何时用: 全局统计看起来"还行"但实际有隐藏瓶颈时;从千级 kernel 中圈定少数嫌疑对象
关键: 脚本已做了部分(Suspect Kernels、Top Stalls),但 agent 需理解"离群 = 线索"的逻辑
```

**模式 5: 时序因果(时间轴上的前后关系)**

利用事件的时间顺序推断因果——A 总是在 B 之前出现,说明 A 导致/阻塞了 B。

```
目的: 建立"谁导致了谁"的因果链
方法: trace_view 的 device 时间线 + host 下发 flow,观察前后关系
何时用: 知道"哪里慢"但不知道"为什么慢"时,从时序上看前因
关键: parse_trace_view 的 stall 聚合(kernel 对)就是时序因果的结构化输出
```

### 五种模式的协作流程

```
实际分析不是只用一种模式,而是组合:

1. 异常定位: 从全局数据中圈定嫌疑点
2. 纵向深入: 沿嫌疑点逐层钻到候选根因
3. 横向关联: 交叉验证候选根因(单来源不可信)
4. 时序因果: 在 trace 中确认事件间的因果关系
5. 差异对比: 优化后确认收益来自预期改动
```

### 与现有文档的关系

当前 profiling_to_action.md 里的信号组合(如"Host-Bound + empty_tensor → Allocator-Bound")本质是**横向关联**模式的具体实例。它们作为参考保留,但定位降级:

- **新定位**: 五种分析模式是方法论(教 agent 怎么想)
- **旧实例**: 降为"参考路径"(给 agent 已验证的组合作为快速匹配入口,但不是全部)
- agent 遇到实例没覆盖的新场景时,用五种模式自己构造推理

### 脚本 Suspect Signals 分级

配合分析模式,脚本输出也要分级:

- `DEFINITE`(确定性事实): 有充分数据判定的。如"compile 贯穿全程(B 类)"、"aic 列全 0(列名不匹配)"。agent 可直接采信。
- `SIGNAL`(信号/线索): 是异常点但原因不确定的。如"Preparing > Computing"、"某 kernel wait 异常大"。输出格式: "观察到 X,可能原因 A/B/C,建议用模式 N 交叉验证"。agent 需组合判断。

### 落地方案

1. **重写 `profiling_to_action.md`**: 开头加"五种分析模式"作为方法论框架,现有信号组合实例重组为"参考路径"附录
2. **各脚本 Suspect Signals 加分级标签**: DEFINITE / SIGNAL,后者附"建议交叉验证方式"
3. **新增若干路径实例**(从真实数据验证得来):
   - compile B 类 + 利用率低 + 硬件正常 → 执行模式问题
   - dispatch p90>>p50 + 算子极多 → 下发积压
   - Transpose/Cast >10% + mte 主导 → 布局不匹配
   - memory 高频抖动 + 同尺寸反复分配 + empty host 耗时高 → allocator 问题
   - 某算子 mac 高 + Block Dim 满 + cube_util 高 → 真 compute-bound(可能是终局)
4. **profiling_to_source.md 的定位漏斗对应"纵向深入"模式**,两文档互引但不重复

---

## Concern 6: 数据库组织 + 缺失的维度 + 与 skills 的关系

### 你期望的完整记录链

```
profiling 现象 → 分析路径 → 根因定位 → 优化方案 → 最终效果
```

### 当前阶段目标: 信息积累(完整+全面),暂不消费

现阶段案例库的唯一目的是**把优化过程中的信息尽可能完整地记录下来**,不丢失任何可能有价值的细节。消费端(检索、匹配、统计)是后续的事——先确保"存对了",再谈"怎么用"。

因此 schema 设计原则是:**宁可冗余不可遗漏**,字段含义要明确到 agent 看完说明就能正确填写。

### Schema 构造语法说明

以下是每个字段的**定义、填写规则和边界说明**,不是示例——agent 据此构造任意案例。

```yaml
- id: <string>
  # 唯一标识符。格式: <模型架构缩写>-<核心现象关键词>-<日期YYYYMMDD>
  # 例: "llm-transpose-layout-20260706", "moe-scatter-backward-20260710"
  # 规则: 用现象/根因描述而非模型全名,确保跨项目可检索

  phenomenon:
    # 记录 agent 从 profiling 中观察到的所有相关信号,尽可能完整
    # 每条 signal 是一个独立的观测事实
    signals:
      - source: <string>  # 产出此信号的脚本名+参数,如 "parse_op_statistic" 或 "parse_kernel_details --filter Transpose"
        content: <string>  # 脚本输出的原文摘录(关键数值+判断),不做解读,只记事实
        # 例: "Transpose: count=768, total=4.2ms, 占比15.3%, 累计占比78.2%"
    raw_context: <string, optional>
      # 补充任何脚本没覆盖但 agent 观察到的信息
      # 如: trace_view 中肉眼看到的 pattern、源码中发现的结构特征
      # 不限格式,自由文本,目的是不丢信息

  analysis_path:
    # 记录从现象到根因的完整推理过程,按实际执行顺序
    steps:
      - action: <string>   # agent 做了什么(跑了什么脚本/读了什么代码/做了什么推理)
        observation: <string>  # 得到了什么结果/看到了什么
        reasoning: <string, optional>  # 为什么做这一步/这个结果说明什么
    # 规则:
    # - 每个有意义的分析动作都记一步,包括走错方向又退回的
    # - 失败的分析尝试也要记(如"尝试用 operator_details 定位但文件不存在")
    # - reasoning 可省略(如果 action→observation 已经自明)

  root_cause:
    description: <string>  # 最终确认的根因,用一两句话说清楚
    bottleneck_type: <enum, optional>
      # 如果能归类: Host-Bound / Compute-Bound / Memory-Bound / Allocator-Bound / Execution-Mode(如在线编译)
      # 如果不能明确归类或属于多种: 写 "mixed" 并在 description 中说明
    evidence: <string>  # 支撑此根因判断的关键证据(哪个数据点最有说服力)

  optimization:
    # 记录所有尝试过的方案,无论成功失败
    attempts:
      - description: <string>  # 做了什么改动
        dimension: <string, optional>
          # 如果能归类到四维度: eliminate_redundancy / reuse_and_precompute / hide_latency / equivalent_substitution
          # 不确定可省略
        implementation_detail: <string>  # 具体代码层面怎么改的(文件、函数、改法)
        equivalence_verification:
          method: <string>  # 怎么验证等价性的(用了什么输入、什么指标、什么阈值)
          result: <string>  # 验证结果(通过/失败,具体数值)
        performance_result:
          metric: <string>  # 用什么指标衡量(L0 latency / step_trace computing / 特定算子耗时)
          before: <string>  # 优化前数值
          after: <string>   # 优化后数值
          verdict: <string> # accepted / rejected / partial,一句话结论
        failure_reason: <string, optional>  # 如果 rejected,为什么失败
    # 规则:
    # - 成功和失败的都记,失败的记 failure_reason
    # - 同一轮尝试了多个方案则 attempts 有多条
    # - implementation_detail 要具体到"改了哪个文件的哪个函数",不是泛泛说"换了实现"

  final_state:
    # 这轮优化结束后的最终状态
    adopted: <string>  # 最终采纳了哪个方案(或"无方案被采纳")
    end_to_end_before: <string>  # 整体性能(优化前)
    end_to_end_after: <string>   # 整体性能(优化后)
    remaining_bottleneck: <string, optional>  # 优化后暴露的新瓶颈(如果有)
    is_terminal: <bool, optional>  # 是否判定为当前硬件/架构下的终局(无进一步优化空间)

  context:
    # 环境和约束信息,确保案例可复现/可对比
    hardware: <string>   # 如 "Ascend 910B"
    cann_version: <string>  # 如 "CANN 8.0.0 / torch_npu 2.3.1"
    model_arch: <string>  # 架构类型而非具体模型名,如 "LLM decoder-only", "Encoder+MoE", "ViT"
    input_spec: <string>  # 测试输入规格,如 "seq_len=16, batch=1, fp16"
    profiling_level: <string>  # 用的什么采集级别 "L0" / "L1" / "L2"
    date: <string>  # YYYY-MM-DD
    notes: <string, optional>  # 任何其他对理解此案例有帮助的备注
```

### 填写原则(给 agent 的指导)

1. **完整优先**: 不确定某信息是否有用时,记下来。schema 里 optional 字段能填就填。
2. **原文摘录**: phenomenon.signals.content 和 analysis_path.steps.observation 尽量贴脚本原始输出,不要 agent 自己概括后丢失细节。
3. **失败必记**: optimization.attempts 中 rejected 的方案和 failure_reason 是最有价值的信息——它告诉未来"别走这条路"。
4. **一次优化阶段一个文件**: 每经过一轮完整的 Phase 2→4(分析+优化+验证),写一个案例文件。
5. **不做归类强求**: bottleneck_type 和 dimension 能判断就写,判断不了写"unclear"——不要为了填字段而乱归类。

### 落地方案

1. **目录**: `<workspace>/evidence_db/`(项目工作目录下,与 `profiling/` 同级)
   - `schema.md`: 上面的语法说明(给 agent 看)——此文件存放在 skill 目录 `model_opt/06_evidence_db/schema.md`,被 agent 读取后按规范在项目目录生成案例
   - `cases/`: 案例文件,扁平存放(不按类型分子目录——现阶段积累为主,分类是消费端的事)
   - 文件名 = id 字段值 + `.yaml`

2. **路径约定**: 与 profiling 目录规范一致——在主 SKILL.md 中统一说明,evidence_db 存在项目工作目录下,不存在 skill 目录中(skill 只存 schema 定义,项目存实际案例数据)

3. **写入时机**: 每个优化阶段(Phase 2→4)完成后,无论成功失败都写一个案例文件。

4. **当前不做的事**:
   - 不做检索工具(grep 足够)
   - 不做自动消费/匹配
   - 不按类型分目录(避免过早固化分类维度)
   - 不做 schema 校验(agent 按说明写即可,偶尔格式不完美没关系)

5. **入口**: 主 SKILL.md Phase 4→5 之间加一步:"记录本轮优化案例到 `<workspace>/evidence_db/`"---

## 汇总: 落地优先级与文件变动清单

| 优先级 | 改进项 | 落地位置 | 工作量 |
|--------|--------|---------|--------|
| P0 | 五种分析模式方法论 + 参考路径重组(concern 5) | 重写 `profiling_to_action.md` 前言 + 重组现有实例为附录 | 中 |
| P0 | 等价替换作为第四维度(concern 1+3) | 新建 `03_optimization/references/equivalent_substitution.md` + SKILL.md 升级"四维度" | 中 |
| P0 | 推理等价性验证协议(concern 3) | 新建 `04_accuracy_assurance/references/equivalence_verification.md` | 小 |
| P1 | 脚本 Suspect Signals 分级为 DEFINITE/SIGNAL(concern 5) | 各脚本诊断输出 + 附交叉验证建议 | 中 |
| P1 | 脚本自动检测增强(concern 2) | parse_kernel_details 加 AI_CPU 标记;parse_trace_view 已做 compile A/B | 小 |
| P1 | 从 eliminate_redundancy.md 抽出"NPU 算子实现差异"到 equivalent_substitution.md | 03_optimization/references/ | 小 |
| P2 | 案例库 schema + 初始目录(concern 6) | 新建 `06_evidence_db/` + schema 说明 + 首批案例 | 大 |
| P2 | 跨文件自动关联脚本(concern 5 长期) | 新脚本 `auto_correlate.py` | 大 |

### 不做的事(明确排除)

- 不写独立的"NPU 规则库"文档(concern 2: 规则消解到脚本检测 + 分析模式中)
- 不做训练场景相关改进(concern 4: 本轮聚焦推理)
- 不写学术化的"四轴理论"(concern 1: 给三层搜索空间,不给正交轴理论)
