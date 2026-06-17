# 算子融合思路

## 问题本质

NPU 上多个小算子的主要开销不在计算本身，而在 **kernel 间的 dispatch gap**——每个算子需要经历 host 调度、tiling 计算、ACL 下发等固定开销。当计算量很小但算子数量很多时，dispatch gap 总和可能远超计算时间。

## 思考路径

```
1. 从 profiling 识别“连续小算子组”
   → op_statistic.csv 中找出调用次数异常多且单次耗时很小的算子
   → kernel_details.csv 中 wait time >> duration 的算子

2. 查找可用的融合算子
   → dir(torch_npu) 搜索关键词（rms_norm, fusion_attention, swiglu, ffn 等）
   → 参考 CANN 算子文档确认 dtype/shape 约束

3. 运行时 Probe 验证可用性
   → try + synchronize，不可静态假设（CANN 版本、dtype 都可能影响）
   → probe 必须加 synchronize（NPU 异步执行，不 sync 无法捕获错误）

4. 替换后验证
   → 精度：输出与原始实现的 diff
   → 性能：重新 profiling，确认 kernel 数下降且端到端耗时下降
```

## 关键判断原则

### 融合不一定更快

融合算子有固定的初始化开销（tiling 计算等）。当原始算子组的总计算量很小时（如 decode 阶段 seq_len=1），融合的固定开销可能超过节省的 dispatch gap。

**判断方法**：在目标场景（相同 batch、seq_len、dtype）下 profiling 对比，而不是凭理论分析。

### dtype 兼容性

NPU 融合算子的 dtype 支持不统一，部分仅支持 fp16/bf16，部分支持 fp32。入参 dtype 不一致时会触发隐式转换，引入额外 kernel。通过 probe 检测而不是查文档。

### Fallback 设计

融合算子不可用时必须能回退到原始实现。代码中用 flag 切换（`if use_fused: ... else: ...`），不要在 forward 热循环里用 try/except。

## 常见融合模式参考

| 原始模式 | 融合思路 | 典型 NPU 算子 |
|---------|---------|-------------|
| 多步 Norm（Pow+Mean+Add+Rsqrt+Mul） | 合并为单个 Norm kernel | `npu_rms_norm` |
| Attention（QK^T+Scale+Mask+Softmax+AV） | FlashAttention 融合 | `npu_fusion_attention` |
| Gated FFN（分片+激活+分片+乘法） | 合并为单个 FFN kernel | `npu_swiglu` / `npu_ffn` |
| 残差+Norm（Add+RmsNorm） | 合并为单个 kernel，同时返回残差和 | `npu_add_rms_norm` |

## 注意事项

- NPU 融合算子的参数命名可能有拼写错误（如 `pre_tockens` 而非 `pre_tokens`）——以实际 API 为准
- 融合算子的返回值可能不止一个（如 `npu_fusion_attention` 返回 7 个值）——先用小输入试跑确认格式
- 融合算子与 `torch.compile` 图模式可能不兼容——需要时查找图模式兼容的替代算子
