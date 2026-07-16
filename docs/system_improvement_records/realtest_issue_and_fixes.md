# 实测发现的问题与改进方案

> 基于 skill 实测发现的四个问题及解决方案。

---

## 问题 1: agent 只读 SKILL.md 不深入 references

### 现象

agent 读了根 SKILL.md 和子技能 SKILL.md（了解 Phase 1→5 流程和大致方法），但**不深入读该 Phase 下的 references**。导致知道"要做四维度分析"但不知道四维度的具体手段（在 `eliminate_redundancy.md` 等文件中），知道"要跑脚本"但跳过了 memory 相关脚本（因为 `memory_profiling.md` 没被加载），知道"要查融合算子"但不系统（因为 `npu_operator_reference.md` 没被完整查阅）。

实测中遗漏的优化点全部对应未被加载的 reference：
- buffer 预分配 → `memory_profiling.md` 明确写了方案，未加载
- flat forward → `eliminate_redundancy.md` §框架调度层消除，未加载
- `npu_add_rms_norm` → `npu_operator_reference.md` 列了该算子，未系统查阅
- CANN 环境变量 → `graph_compile_and_cann.md`，未加载

### 根因

渐进式披露的设计是"SKILL.md 始终加载，references 按需加载"。但"按需"的判断由 agent 做——agent 读了 SKILL.md 后认为自己已经理解了方法，不会主动去读 references。问题不是"不知道在哪个 Phase"，而是"知道了 Phase 但不深入读该 Phase 的详细参考"。

具体机制：
1. SKILL.md 用一句话描述 reference 的内容（如"显存峰值分析"），agent 觉得自己不需要
2. references 中的具体方案（如"高频抖动 → 预分配 buffer + out= 写入"）agent 从未看到
3. 即使 SKILL.md 说"按需加载"，agent 的"需"判断本身就需要 references 的知识——形成鸡生蛋问题

### 方案

**1. 脚本→reference 强制绑定（已在 02_bottleneck_analysis/SKILL.md 落地）**

运行某个脚本后，**自动加载**其绑定的 reference，而非依赖 agent 判断"是否需要"。例如运行 `parse_operator_memory.py` 后必须加载 `memory_profiling.md`。这样 agent 不需要预判"我需不需要 memory 分析"——跑了脚本就自然会读到对应方案。

**2. 启动协议（确保 agent 知道自己在哪个 Phase）**

根 SKILL.md 开头加：
```
## ⚠ 启动协议

无论从哪个子技能进入,执行前必须:
1. 确认当前在哪个 Phase(参见下方「全流程」)
2. 确认上一个 Phase 的产出已完成
3. 按全流程顺序执行,不跳步
```

各子技能 SKILL.md 的 description 里标注所属 Phase。

**3. 审计表/门禁中的交叉引用（已在 proactive_source_analysis.md 和 npu_operator_reference.md 落地）**

审计表和门禁表格中显式引用具体 reference 文件名（如"对照 `npu_operator_reference.md` 检查"），让 agent 在填表时被迫打开 reference。

### 落地

- 根 SKILL.md: 核心原则前加"启动协议"段
- 各子技能 SKILL.md: description 末尾加"执行前参见根 SKILL.md 全流程"
- 02_bottleneck_analysis/SKILL.md: 强制脚本检查清单中绑定必读 reference（已落地）
- proactive_source_analysis.md: 审计表中引用 npu_operator_reference.md（已落地）
- 不改渐进式披露机制本身——通过脚本绑定和交叉引用让 reference 被自然加载

---

## 问题 2: evidence_db 构建被忽略

### 现象

evidence_db 的案例记录在实践中被 agent 跳过。

### 根因

1. 记录步骤在 Phase 5(工程化提交)中,但 05_engineering/SKILL.md 没有强调
2. "记录案例"没有强制检查点(不像精度验证有"必须通过"的约束)
3. 根 SKILL.md Phase 5 描述里只有一句"提交前按 schema 记录",不够醒目

### 方案

**把 evidence_db 记录提升为确认节点 B 的前置条件**:

当前确认节点 B(提交前审核)的流程:
```
1. 总结优化点及效果
2. 列出未采纳方案
3. ask_user_question 确认提交
```

改为:
```
1. 总结优化点及效果
2. 列出未采纳方案
3. 确认 evidence_db 案例已按 schema 记录(展示案例文件路径)
4. ask_user_question 确认提交
```

案例未记录 = 不允许提交。和"精度未通过 = 不允许提交"同等约束力。

### 落地

- 根 SKILL.md: 确认节点 B 加第 3 步(evidence_db 确认)
- 根 SKILL.md: Phase 5 描述中"记录案例"从建议改为强制("必须先记录案例再提交")
- 05_engineering/SKILL.md: 提交流程中加 evidence_db 检查步骤

---

## 问题 3: Phase 5 后缺少"继续优化?"询问

### 现象

跑完 Phase 5(提交)后直接结束,没有问用户要不要继续优化下一轮。

### 根因

流程图里 Phase 5 后有"瓶颈转移?→ 回到 Phase 2",但这个判断是 agent 内部做的,没有和用户确认。用户可能想继续,也可能觉得够了。

### 方案

**Phase 5 后加确认节点 C**:

```
Phase 5  工程化提交
   ↓
 ★ C  用户确认是否继续(展示本轮总结 + 剩余瓶颈 → 继续/停止)
   ↓
 ├─ 继续 → 回到 Phase 2 开启下一轮
 └─ 停止 → 结束
```

确认节点 C 的内容:
1. 展示本轮优化总结(性能提升 + 精度状态)
2. 展示当前剩余瓶颈(profiling 最新数据)
3. ask_user_question: 是否继续下一轮优化?

### 落地

- 根 SKILL.md: 全流程图 Phase 5 后加 ★C
- 根 SKILL.md: 加"确认节点 C"描述
- 根 SKILL.md: 迭代退出条件从"agent 判断"改为"用户确认"

---

## 问题 4: profiling 采集脚本应统一接口

### 现象

agent 每次优化会写一个新的采集脚本,把采集代码嵌入业务代码。优化后业务代码变了,采集脚本也要重写。

### 根因

当前 `profiling_collection.md` 给的是模板代码(直接 `with torch_npu.profiler.profile(...) as prof: model(input)`),agent 会复制模板到业务脚本里。没有强调"采集代码和业务代码分离"。

### 方案

**采集与业务分离:统一采集接口**

核心思想:业务推理逻辑用函数封装(如 `run_inference(model, input)`),采集脚本只调用这个函数。无论业务代码怎么优化,采集脚本不变。

设计指引:
```python
# business.py — 业务代码,被优化的是这里
def run_inference(model, input_data):
    """推理入口,优化改这里,采集脚本不改"""
    return model(input_data)

# profile.py — 采集脚本,只调用 run_inference,不改
def profile_run(profiling_dir, model, input_data):
    with torch_npu.profiler.profile(
        activities=[...],
        schedule=...,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir)
    ) as prof:
        run_inference(model, input_data)
        prof.step()
```

**原则**:
- 采集脚本只负责"包围"业务调用,不嵌入业务逻辑
- 业务代码的优化(改 forward、换算子、预分配等)都在 `run_inference` 内部,采集脚本无感知
- 多次优化对比时,用同一个采集脚本跑 before/after,确保采集条件一致
