# 推理适配已知问题

实际适配中遇到的、agent 无法从模型文档推导出的 NPU 推理问题。

## 精度类

| 现象 | 根因 | 解决 |
|------|------|------|
| Diffusion 生成图像面部模糊/伪影 | VAE decoder 在 NPU fp16 下精度不足 | VAE upcast 到 fp32，latents decode 前转 float32 |
| 分类置信度比官方低几个百分点 | 预处理 interpolation 不匹配 (如用 bilinear 替代 bicubic) | 严格从 config.json 读取 interpolation 字段 |
| 生成图像整体细节缺失 | 推理步数不足 (如 SD 默认 50 步被设为 20 步) | 使用模型默认步数，不为"快速验证"减步 |
| attention 层输出异常 | flash_attention 在当前 CANN 版本不兼容 | 加 `attn_implementation="eager"` |

## 精度敏感模块通用识别方法

不要硬记"哪个模块需要 upcast"，用以下通用流程定位：

1. 先整体 fp16 跑一遍
2. 检查输出 (冒烟 + 按输出性质验证)
3. 如果异常 → 逐模块 upcast 到 fp32，每次只改一个
4. 定位到问题模块后，仅对该模块做 upcast，其余保持 fp16

## 配置类

| 现象 | 根因 | 解决 |
|------|------|------|
| 模型加载时出 UNEXPECTED keys 警告 | 预训练模型包含 task head，用 AutoModel 加载只取 base | 正常，UNEXPECTED keys 是 pretraining head，不影响 |
| dtype 警告 `torch_dtype is deprecated` | transformers 新版用 `dtype` 参数 | 改用 `dtype=torch.bfloat16` |
| 国内环境下载样本数据 SSL 失败 | HTTPS 证书拦截/CDN 问题 | `ssl._create_unverified_context()` 或换 ModelScope 源 |
