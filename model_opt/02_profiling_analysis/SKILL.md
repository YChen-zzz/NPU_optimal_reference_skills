---
name: NPU Profiling 分析
description: 分析昇腾 NPU profiling 数据以定位性能瓶颈并提出优化方向。当用户要求分析 profiling、定位瓶颈、查看性能数据时触发。
---

# NPU Profiling 分析

## 分析流程

按以下顺序逐步深入：

1. **step_trace_time.csv** — 设备利用率全貌
   - `Computing / (Computing + Free)` = 设备利用率
   - 利用率 < 20% → 严重 host-bound
2. **op_statistic.csv** — kernel 数量和类型分布
   - Transpose 多 → 布局问题
   - Pows+ReduceMean+Rsqrt → RmsNorm 未融合
   - 调用次数 / (steps × layers) 反推每层调用数
3. **kernel_details.csv** — wait time 分布
   - stall_ratio = total_wait / total_compute
   - avg wait/kernel
4. **operator_details.csv** — host 侧每个操作耗时分解
   - Host Self Duration = 纯 Python/C++ 调用开销
   - Device Self Duration = 实际 device kernel 时间
5. **trace_view.json** — enqueue gap + host 操作时间线

## 关键指标与告警

| 指标 | 含义 | 告警阈值 |
|------|------|----------|
| 设备利用率 | Computing/(Computing+Free) | < 20% 严重 |
| stall_ratio | Wait/Compute 比 | > 500% 严重 |
| avg wait/kernel | 每 kernel 平均等待 | > 100μs 严重 |
| empty_tensor 次数/步 | 每个算子需分配输出 buffer | > 500 次/步 严重 |
| 纯 host op 占比 | 无 device kernel 的 Python op 耗时比 | > 70% 严重 |

## 瓶颈分类

- **Host-Bound**: 设备利用率低，Free >> Computing → 问题在 host dispatch、Python 开销
- **Compute-Bound**: 利用率高，kernel 本身耗时大 → 优化算子或降精度
- **Memory-Bound**: 利用率高但 kernel 吞吐远低于理论峰值 → HBM bandwidth 竞争
  - 同时持有冗余权重（原始 + 预转置两份）导致全局带宽退化，表现为所有 kernel 均匀变慢
  - 大 shape 算子的 data movement 时间 >> 计算时间（访存密集型算子）
- **Allocator-Bound**: 表现类似 Host-Bound（设备空闲），但根因是 NPU allocator 触发同步
  - allocator 频繁同步：`empty_tensor` 次数异常多、MemSet 算子大量出现 → allocator 在每次分配时阻塞等待 device 完成
  - H2D/D2H 同步打断 pipeline：trace 中 NPU 计算流出现空泡，前后有 `.item()` / `.to(device)` / `.numpy()` 等操作
  - 碎片化严重：`reserved >> allocated` → 池中有大量空闲碎片但不满足新请求的连续大小要求 → 考虑 `expandable_segments` 或战略性 `empty_cache`

> **注意**: Allocator-Bound 容易与 Host-Bound 混淆——两者在 profiling 中都表现为"设备空闲"。区分方法：如果 operator_details 中 `empty_tensor` / `aten::empty` 的 Host Self Duration 占比显著，则为 Allocator-Bound；如果 Python dispatch wrapper 占比高，则为 Host-Bound。

## Host-Bound 根因定位

**关键原则**: 用 operator_details 而非 kernel_details 定位 host 时间花在哪。

1. 从 operator_details 按 Host Self Duration 排序 top 30
2. 按类别聚合：
   - tensor metadata ops（empty_tensor / view / as_strided）
   - Python dispatch wrapper（aten::matmul / aten::dropout）
   - ACL kernel launch（aclnnMm / aclnnRmsNorm）
   - format_cast + event sync
3. 计算各类别占比 → 找占比最大的类别
4. 如果 tensor metadata ops > 40% → 问题是 Python 框架开销，不是算子启动

## Pipeline Bubble 全局分析

**不要只看最大的单个 gap**，先做全局统计：

1. 用 awk/Python 统计 kernel_details 的 wait time 分布
2. 按算子类型聚合 wait → 找占比最大的类别
3. 计算 total_wait / total_duration = bubble ratio

### Level0 vs Level1 区分真实 bubble

CANN profiler Level1 在每个 kernel 前后插入 barrier → 破坏 TASK_QUEUE 流水 → Level1 看到的 bubble 大部分是 profiler 注入的。

验证方法：
- Level0 重新采集（去掉 aic_metrics）
- 对比 Level0 vs Level1 的 Computing 和 Free
- wall-clock 延迟交叉验证

## 跨版本对比

按 op name 聚合两版 operator_details 的 count 和 host_self → 计算 delta：
- 负值 = 改善
- 正值 = 退化
- 关注完全消除的 op、单次耗时变化、新增 op

## 跨平台公平对比（NPU vs GPU）

- NPU Level1 vs GPU torch.profiler 的 overhead 不同 → bubble ratio 不可比
- 统一到最低 overhead 配置 + 相同序列 + 相同 schedule
- 可比指标：kernel_sum / device_span / wall-clock

详见 [analysis_scripts.md](references/analysis_scripts.md) 获取分析脚本模板。
详见 [host_bound_patterns.md](references/host_bound_patterns.md) 获取 Host-Bound 深度诊断模式。
详见 [memory_profiling.md](references/memory_profiling.md) 获取显存峰值分析方法和常见显存陷阱。
详见 [profiling_to_action.md](references/profiling_to_action.md) 获取"profiling 特征 → 具体行动"的快速映射表。
