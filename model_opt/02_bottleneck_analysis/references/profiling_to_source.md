# 从 Profiling 现象定位到源码

Profiling 只能告诉你"哪个算子慢"，要动手优化必须先把它跨接到"源码里哪一行/哪个模块"。这一跨接依赖 profiling 文件里几个特定字段作为"桥"。本文只列经昇腾官方 profiler 数据说明确认存在的桥，以及每座桥的采集前提和断桥时的降级做法。

> 前提：这些桥全部依赖采集参数正确。若采集缺失对应开关，字段不会生成，桥即断裂——此时不能假装能精确定位，只能按降级路径做有限推断。采集参数见 01_preparation/references/profiling_collection.md §1。

## 五座桥（官方字段确认）

| 桥 | 字段 / 文件 | 采集前提 | 作用 |
|----|------------|---------|------|
| **Call Stack** | `operator_details.csv` 的 `Call Stack` 列 | `activities` 含 CPU + `with_stack=True` | 唯一能把一次算子调用映射到 Python 源码函数/行号的字段 |
| **Input Shapes** | `kernel_details.csv` / `operator_details.csv` 的 `Input Shapes` 列 | `record_shapes=True` | 区分同一算子类型的不同调用点（如 attention 的 QK^T vs FFN 的 up_proj） |
| **AI Core 指标** | `kernel_details.csv` 的 `aic_*` / `aiv_*` 占比列 | `aic_metrics=PipeUtilization` | 判断算子瓶颈性质（compute / memory / 搬运），决定优化方向 |
| **Accelerator Core** | `kernel_details.csv` 的 `Accelerator Core` 列；`data_preprocess.csv` | 基础采集即有；`data_preprocess.csv` 需 Level2 | 值为 AI CPU 的算子表示未落 AI Core，指向"换实现 / 改 dtype"的源码修改点 |
| **下发时序** | `trace_view.json` 的 HostToDevice flow、`AscendCL@opCompile` 事件（`parse_trace_view.py`） | NPU 采集即有（不依赖 with_stack） | host→device 下发链与在线编译停顿（A 预热 / B 每步），定位 host-bound 的空隙成因；开 with_stack 时 `cpu_op` 还带 Call stack |

## 组合使用：定位漏斗

单座桥都不足以直接定位，按下面顺序逐层收窄：

```
Input Shapes（哪次调用）
      ↓ 缩到同类算子里的具体调用点
Call Stack（哪行源码）
      ↓ 精确到函数/行号
AI Core 指标 / Accelerator Core（怎么改）
      ↓ 决定手段：融合 / 换布局 / 换实现 / 改 dtype
```

例：`op_statistic` 发现 Transpose 占 15% → `kernel_details --filter Transpose` 用 **Input Shapes** 确认是哪种 shape → `operator_details --filter Transpose` 用 **Call Stack** 定位到源码行 → **AI Core 指标** 判断是布局不匹配还是搬运开销 → 决定改法。

> host-bound 场景走另一条路：`parse_trace_view.py` 的**下发时序**桥先区分空隙成因（下发延迟 / 在线编译 / 同步），再对 H2D 拷贝、反复分配等**不换算子**的操作用 Call stack 定位到源码，做预取 / 预分配 / buffer 复用。

## 断桥时的降级路径

| 缺失字段 | 现象 | 降级做法（有限，不精确） |
|---------|------|------------------------|
| 无 `operator_details.csv`（未开 CPU/with_stack） | 无 Call Stack，无法定位源码行 | 先用 `parse_trace_view.py` 看 HostToDevice 下发链/opCompile 停顿定位 host 侧成因；再按算子类型 + shape 结合 forward 静态路径**人工推断**触发位置，结论标注为推断 |
| `aic_*` 占比列全为 0 | 硬件占比无效 | 检查是否误用了非 `PipeUtilization` 的 aic_metrics 组（列名不匹配导致读出 0）；重采或改用对应列名 |
| `Accelerator Core` 无 AI CPU 值 | 本次无 fallback | 无需处理；如怀疑 fallback 需 Level2 采 `data_preprocess.csv` 确认 |

> **注意（非桥）**：`kernel_details.csv` 的 `Name` 是算子顺序编号（如 `MatMul45`），**不含** module scope 路径，不能用作定位。`trace_view.json` 的下发时序 flow 不依赖 with_stack（见上表「下发时序」桥），但其 **Python 调用栈/源码映射** 仅在 `with_stack` 开启时才有——未开时它只能给时序，不能给源码行。
