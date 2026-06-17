# NPU 融合算子速查

## 通用注意事项

- **发现可用算子**：`[x for x in dir(torch_npu) if 'npu_' in x.lower()]`
- **查看签名**：故意传错参数触发报错，报错信息会打印完整 schema
- **Probe 必须加 synchronize**：NPU 异步执行，`try/except` 内不加 `torch.npu.synchronize()` 无法捕获错误
- **返回值数量不固定**：多数融合算子返回多个值，用 `result, *_ =` 解包更安全

## npu_rms_norm

```python
normed, rstd = torch_npu.npu_rms_norm(input, weight, epsilon=1e-6)
# normed: 归一化输出；rstd: 推理时可忽略
```
- dtype: fp16 / bf16 / fp32
- weight 必须与 input 同 dtype

## npu_fusion_attention

```python
attn_output, *_ = torch_npu.npu_fusion_attention(
    query, key, value,
    head_num=num_heads,
    input_layout="BSND",          # 可选: BSND / BNSD / BSH
    scale=1.0 / math.sqrt(d),
    pre_tockens=2147483647,       # 注意：Cann 官方拼写错误，不是 pre_tokens
    next_tockens=0,               # 同上；causal 时设 0
    pse=position_bias,            # 可选；dtype 必须与 query 一致
    atten_mask=mask,              # 可选
    keep_prob=1.0,                # 推理固定 1.0
)
# 返回 7 个值，推理只用第 1 个，其余用 *_ 丢弃
```
- dtype: fp16 / bf16 / fp32（eager）；GE 图模式下仅 fp16/bf16
- 与 `torch.compile` 图模式不兼容（multi-stream），需用 FIA 替代
- Fallback 场景：KV cache / cross-attention / output_attentions / head pruning

## npu_fused_infer_attention_score (FIA)

```python
attn_output, softmax_lse = torch_npu.npu_fused_infer_attention_score(
    query, key, value,
    num_heads=num_heads,          # 注意：参数名与 FA 不同（num_heads 而非 head_num）
    input_layout="BNSD",
    scale=...,
    pre_tokens=65536,             # 注意：拼写正确（与 FA 的 pre_tockens 不同）
    next_tokens=0,
)
# 返回 2 个值
```
- dtype: 仅 fp16 / bf16
- 图编译兼容（替代 npu_fusion_attention 的图模式方案）

## npu_add_rms_norm

```python
normed, rstd, residual = torch_npu.npu_add_rms_norm(
    x1, x2, weight, epsilon=1e-6
)
# normed:   rms_norm(x1 + x2) 的输出
# rstd:     推理可忽略
# residual: x1 + x2 的和（可直接作为下一层的残差输入）
```
- dtype: fp16 / bf16 / fp32
- **注意返回值顺序**：第 1 个是 normed，第 3 个是 residual，容易搞反

## npu_swiglu

```python
output = torch_npu.npu_swiglu(input, dim=-1)
```
- dtype: fp16 / bf16 / fp32
- dim 对应的维度大小必须能被 2 整除

## npu_ffn

```python
output = torch_npu.npu_ffn(input, weight1, weight2, activation="swish")
```
- dtype: **仅 fp16 / bf16**（fp32 不支持）
- activation 需为受支持类型

## npu_linear

```python
output = torch_npu.npu_linear(input_2d, weight)
# 内部自动做 weight 转置，不需要手动 .t()
```
- **仅支持 2D 输入**：3D+ 需先 `view(-1, dim)` 再调用，再 reshape 回去
- 不支持 bias

## 典型陷阱汇总

| 陷阱 | 说明 |
|------|------|
| `pre_tockens` 拼写 | CANN 官方 typo，写成 `pre_tokens` 会报 Unknown keyword |
| FA 返回值数量 | 用 `a, b, c = ...` 解包会报错，必须用 `a, *_ = ...` |
| pse dtype 不匹配 | pse 与 query dtype 不一致时静默产生错误结果，不报错 |
| FA vs FIA 参数名 | `head_num` vs `num_heads`，`pre_tockens` vs `pre_tokens` |
| npu_linear 3D | 传 3D tensor 直接报错，需手动 reshape |
| npu_add_rms_norm 顺序 | 第 1 个返回值是 normed不是 residual，搞反会导致精度错误 |
