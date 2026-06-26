# 优化日志条目模板

每批优化全量验证通过、git commit 完成后，必须更新一次优化日志。

## 日志文件位置

```
docs/optimization_log.md   ← 主日志文件
```

## 条目格式

```markdown
## 批次 N（<日期>）

### 本批优化点
1. 「优化点名称」— 简述和预期收益
2. ...

### 修改内容
- 文件: `xxx.py` — 具体改动描述
- 文件: `yyy.py` — ...

### 性能数据
| 指标 | 优化前 | 优化后 | 改善 | 数据来源 |
|------|------|------|------|------|
| 推理延迟 avg (ms) | XXX | YYY | -Z% | profiling/<timestamp> |
| 吞吐量 (samples/s) | XXX | YYY | +Z% | profiling/<timestamp> |

### 精度验证
- 连续输出 cosine: 0.9999+（min），全量测试集
- 生成文本：与 baseline 全量匹配
- 聚合分数相对误差: < 0.5%

### 未采纳方案
- [方案描述]：[实际效果]，[未采纳原因]

### Profiling 时间戳索引
- 优化前：`profiling/<timestamp_before>/`
- 优化后：`profiling/<timestamp_after>/`
```

## 示例

```markdown
## 批次 2（YYYY-MM-DD）

### 本批优化点
1. Flat Forward — 绕过 Module.__call__ 调用栈，预期减少 ~20% 延迟
2. 预分配 Buffer（out= 模式）— 消除运行时 empty_tensor 分配，预期 -15%
3. 权重预转置 — 消除每次 forward 的 aten::t 开销，预期 -5%

### 修改内容
- `model/flat_encoder.py` — 新增扁平化 encoder 实现
- `model/flat_decoder.py` — 新增扁平化 decoder 实现
- `inference/run_infer.py` — 切换为扁平化模型

### 性能数据
| 指标 | 优化前 | 优化后 | 改善 | 数据来源 |
|------|------|------|------|------|
| encoder 延迟 avg | 17.8 ms | 9.7 ms | -46% | profiling/YYYYMMDD_HHMMSS |
| 端到端吞吐量 | 56 seq/s | 103 seq/s | +84% | profiling/YYYYMMDD_HHMMSS |

### 精度验证
- encoder 输出 cosine: 0.9999+（min），全量测试集
- 生成文本：与 baseline 全量匹配
- 聚合分数相对误差: < 0.5%

### 未采纳方案
- StaticCache：在 NPU 上展开为更多子 kernel，比 DynamicCache 慢约 20%，已回退

### Profiling 时间戳索引
- 优化前：`profiling/YYYYMMDD_HHMMSS_before/`
- 优化后：`profiling/YYYYMMDD_HHMMSS_after/`
```
