# torch.compile NPU Compile Guide

## 1. 必须使用 `backend='npu'`

默认 `backend='inductor'` 在 NPU 上通常失败或无增益。`backend='npu'` 使用昇腾专用编译路径，对 elementwise chain 有显著 fusion 效果。

```python
@torch.compile(backend='npu', dynamic=False)  # 不是默认 inductor!
def fn(x):
    ...
```

## 2. PATH 排错：bishengir-compile 找不到

如果报 `npuc` / `bishengir` / `Invalid bishengir path` 错误，需要手动定位并添加路径：

```bash
find /usr/local/Ascend -name "bishengir-compile"
export PATH="/usr/local/Ascend/<version>/bisheng_toolkit/bishengir/bin:$PATH"
```

## 3. Compile Scope 策略

不要只 compile 最小子表达式。逐步扩大 scope，更大 scope 通常有更好的 fusion 效果：

1. 只包 activation
2. 包 activation + linear
3. 包 norm + linear + activation + linear

每种 scope 作为独立候选方案在 Lab 中对比。

## 4. 适用判断

| 场景 | 是否适合 | 说明 |
|------|---------|------|
| 多个 elementwise/pointwise 链（sigmoid+mul, relu+mul, div+sigmoid+mul） | ✅ 适合 | fusion 收益明显 |
| 已是单个大算子（mm, attention, norm API） | ❌ 不适合 | 无额外 fusion 空间 |
| tensor 很小 | ❌ 不适合 | dispatch overhead > fusion gain |

## 5. `dynamic=False` 在多 Regime 训练中的行为

`dynamic=False` 时，每个 shape 首次出现会触发一次编译，编译结果随后被缓存。在多 regime 训练（不同阶段 shape 不同）中，每个新 shape 都会经历一次编译开销。

## 6. 多卡注意事项：kernel_meta 冲突

首次多卡 compile 可能崩溃，报错 `unable to open output file kernel_meta/...`。原因是多个 rank 同时写同一 cache 目录。解决方法——每 rank 独立 cache 路径：

```bash
export TORCH_NPU_COMPILE_CACHE_DIR="/tmp/npu_compile_cache_${RANK:-0}"
```

## 7. 首次编译耗时与 Steady-State 分开记录

首次编译耗时（cold-start）与稳态执行时间（steady-state）必须分开记录。不得仅因 cold-start 较慢而否决稳态有收益的候选。
