---
name: CANN 环境配置参考
description: CANN 环境详细诊断命令、版本确认、环境变量清单与常见报错解决方案。作为 NPU 适配前期准备的次级参考文件。
---

# CANN 环境配置参考

## 1. 环境诊断命令

```bash
# 确认芯片型号、驱动版本、卡数、运行状态
npu-smi info

# 查看具体卡的健康状态
npu-smi info -t health -i 0

# 确认 CANN 工具链版本
cat $ASCEND_HOME_PATH/version.cfg
# 或：
cat ~/Ascend/ascend-toolkit/latest/version.cfg

# 确认 OPP 算子包
ls $ASCEND_HOME_PATH/opp/built-in/op_impl/

# 激活工具链（每次新 shell 后需执行）
source ~/Ascend/ascend-toolkit/latest/set_env.sh
```

## 2. torch_npu 验证代码片段

```python
import torch
import torch_npu

# 检查设备可见性
print("NPU 卡数:", torch.npu.device_count())
print("torch_npu 版本:", torch_npu.__version__)

# 简单功能验证
tensor = torch.ones(3, 3).npu()
print("设备:", tensor.device)   # 应输出 npu:0
print("加法测试:", (tensor + tensor).sum().item())  # 应输出 18.0

# torchair 版本
import torchair
print("torchair 版本:", torchair.__version__)
```

## 3. 环境变量配置清单

| 变量名 | 作用 | 示例值 |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | 控制可见 NPU 卡 | `0,1,2,3` |
| `ASCEND_GLOBAL_LOG_LEVEL` | 日志级别 (0=Debug,1=Info,3=Error) | `3` |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 日志输出到控制台 | `0` |
| `ASCEND_HOME_PATH` | CANN 工具链安装目录 | `~/Ascend/ascend-toolkit/latest` |
| `LD_LIBRARY_PATH` | 动态库搜索路径（set_env.sh 自动设置） | -- |

```bash
# 推荐写入 ~/.bashrc 或 activate 脚本
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
source ~/Ascend/ascend-toolkit/latest/set_env.sh
```

## 4. 版本配套确认方法

```bash
# PyTorch 版本
python3 -c "import torch; print(torch.__version__)"

# torch_npu 应与 PyTorch 主版本匹配
# 例：PyTorch 2.1.0 -> torch_npu 2.1.0.postX
python3 -c "import torch_npu; print(torch_npu.__version__)"

# 查询官方配套表
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha001/softwareinstall/instg/instg_0019.html
```

## 5. 常见报错及解决方案

### libhccl.so 缺失
```
OSError: libhccl.so: cannot open shared object file
```
**解决**：`set_env.sh` 未执行，或 CANN 工具链未安装。
```bash
source ~/Ascend/ascend-toolkit/latest/set_env.sh
ldconfig -p | grep hccl  # 验证库可见
```

### 错误码 EI0006 / 561103
```
[ERROR] RUNTIME(561103) 设备初始化失败
```
**解决**：多见于 `ASCEND_RT_VISIBLE_DEVICES` 设置与实际卡号不匹配，或驱动版本与 CANN 不配套。
```bash
npu-smi info  # 确认实际卡 ID
export ASCEND_RT_VISIBLE_DEVICES=0  # 从逻辑 0 开始
```

### torch_npu 导入后 npu 设备不可见
```python
import torch_npu
print(torch.npu.is_available())  # 输出 False
```
**解决**：检查 `npu-smi info` 是否正常，确认 `ASCEND_RT_VISIBLE_DEVICES` 已设置，确认 `set_env.sh` 已执行。

### 算子不支持 (OpNotSupported)
```
RuntimeError: Op [XxxOp] is not supported on NPU
```
**解决**：查订当前 CANN 版本的算子支持列表，优先升级 CANN。若无法升级，需手动替换算子实现（进入 Phase 3 优化阶段）。
