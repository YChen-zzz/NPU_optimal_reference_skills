# .gitignore 模板

profiling trace 文件大、临时文件多，不应纳入 git。

```gitignore
# Profiling 输出目录
profiling/
*.pb
*.json.gz
ASCEND_PROFILER_OUTPUT/
PROF_*/

# 运行时临时文件
*.log
*.npy
*.pt
__pycache__/
*.pyc
output/

# 模型权重（如有单独存储方案）
weights/
*.safetensors
*.bin
```

**应当纳入 git** 的：源代码、脚本、文档、对比结果的摘要（非原始大文件）。
