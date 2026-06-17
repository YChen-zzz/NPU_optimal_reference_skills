# Host-Bound 深度诊断模式

## enqueue gap 分析

从 trace_view.json 提取 enqueue 事件，计算相邻 enqueue 的时间 gap：

1. 解析 trace_view.json，提取所有 `cat='enqueue'` 事件
2. 计算相邻 enqueue 的时间 gap
3. 按 gap 排序找 top 20
4. 对最大 gap，提取 gap 时段内的所有 host 事件（cpu_op + python_function）
5. 按 kernel 类型聚合 > 100μs 的 gap

## 逐层 kernel 序列分析

按 Step Id 过滤单步数据，然后按层分组统计：

```bash
awk -F',' 'NR>1 {print $6, $11, $12}' kernel_details.csv | \
awk '{dur+=$2; wait+=$3; n++; if(n%KERNELS_PER_LAYER==0){
  printf "Layer %d: compute=%dus, wait=%dus, ratio=%.0f%%\n",
  n/KERNELS_PER_LAYER, dur, wait, wait/(dur+wait)*100; dur=0;wait=0}}' | head -20
```

### TASK_QUEUE ramp-up 检测

- Layer 1-2: queue 还空 → 大 gap（ramp-up 阶段）
- Layer 3+: queue 已满 → gap ≈ 0（稳态）
- 如果 Layer 3+ 仍有大 gap → TASK_QUEUE 未生效或有其他 stall 源

## Host-Bound 优化策略

### 策略总体思路

```
图编译优先尝试（收益上限最高）
  ├─ 成功 → 直接享受图模式红利
  └─ 失败（算子不兼容 / 图太大内存爆炸 / 无法解决的 bug）
        → 回退 eager 模式，按 Level 0→4 逐级优化
```

### 图编译（优先尝试）

| 步骤 | 操作 |
|------|------|
| 1 | 用最简子模块（如单层 Linear）验证图编译基础能力 |
| 2 | 逐步扩大编译范围（+ Norm → + Attention → 完整模型） |
| 3 | 确认精度和性能 |

**放弃条件**（满足任一即回退 eager）：
- 算子不兼容且无替代方案
- 图太大导致编译期内存爆炸（OOM）
- 触发框架层 bug 且无法绕过

### Eager 模式分层优化（图编译不可用时）

按以下层级依次实施：

| Level | 手段 | 消除目标 | 说明 |
|-------|------|---------|------|
| 0 | CANN 环境调优 | 基础调度效率 | `TASK_QUEUE_ENABLE=2` 开启 Host-Device 异步流水；`CPU_AFFINITY_CONF=1` CPU 绑核减少调度抖动 |
| 1 | Python 框架开销消除 | `Module.__call__` 调度栈 | 扁平化 forward、monkey-patch 去除推理时冗余逻辑、移除多余全局 hook |
| 2 | 内存分配消除 | 运行时 tensor 分配 | 预分配输出 buffer + `out=` 写入；原地操作（`add_` / `mul_`） |
| 3 | 权重与数据 Prefetch | 运行时数据准备开销 | 权重预转置；中间结果缓存复用；跨层 prefetch（当前层计算时预加载下一层权重） |
| 4 | 数据布局优化 | 格式转换 / transpose 开销 | 选择 NPU 友好 layout；消除 `format_cast`；统一维度约定 |

**实施原则**：
- 从 Level 0 开始逐级实施，每级完成后重新 profiling 确认瓶颈是否转移
- Level 0–2 为低风险操作；Level 3–4 需微基准验证
- 不要跳层——如 Level 1 未做就做 Level 4，效果不明显且难以归因

## 核间同步分析

AI_CORE vs AI_VECTOR_CORE 切换是否导致额外 wait：

```python
core_seq = [(r['Accelerator Core'], float(r['Wait Time(us)'])) for r in step_n]
switches = sum(1 for i in range(1, len(core_seq)) if core_seq[i][0] != core_seq[i-1][0])
switch_wait = [core_seq[i][1] for i in range(1,len(core_seq)) if core_seq[i][0]!=core_seq[i-1][0]]
noswitch_wait = [core_seq[i][1] for i in range(1,len(core_seq)) if core_seq[i][0]==core_seq[i-1][0]]
print(f"切换率: {100*switches/(len(core_seq)-1):.0f}%")
print(f"切换 avg wait: {sum(switch_wait)/len(switch_wait):.0f}us")
print(f"不切换 avg wait: {sum(noswitch_wait)/len(noswitch_wait):.0f}us")
```

如果差异 < 10μs → 核切换不是主因，主因是 Python dispatch 固有延迟。

## 自回归 Decode 的 Host-Bound 本质

自回归 decode 的根本瓶颈分析：
1. 计算每层每步的 kernel 数和 compute 总量
2. 计算 per-kernel 的 dispatch overhead（Level0 的 avg wait）
3. 计算 dispatch/compute ratio
4. 如果 ratio > 1 → host-bound，eager 模式下无法解决 → 需要 fp16（启用融合算子减少 kernel 数）或图编译
