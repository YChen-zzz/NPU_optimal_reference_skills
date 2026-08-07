# 版本配套与环境问题

## torch / torch_npu / CANN 版本配套表

| CANN 版本 | PyTorch | torch_npu |
|-----------|---------|-----------|
| 8.3.RC1   | 2.8.0 / 2.7.1 / 2.6.0 | 2.8.0 / 2.7.1 / 2.6.0.post3+ |
| 8.2.RC1   | 2.6.0 / 2.5.1 | 2.6.0 / 2.5.1.post1 |
| 8.1.RC1   | 2.5.1 / 2.4.0 | 2.5.1 / 2.4.0.post4 |
| 8.0.0     | 2.4.0 | 2.4.0.post2 |

获取最新配套：`pip index versions torch_npu`，或查阅 torch_npu GitHub README。

## 环境层面已知问题

| 问题 | 表现 | 解决 |
|------|------|------|
| set_env.sh 未 source | `libhccl.so: cannot open shared object file` | 脚本开头 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| TBE 编译器缺依赖 | `ModuleNotFoundError: No module named 'decorator'` | `pip install decorator attrs` |
| 首次推理慢 | 新 shape 算子触发在线编译，耗时数十秒 | 正常现象，第二次恢复正常速度 |
