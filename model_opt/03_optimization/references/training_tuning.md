# 训练场景调优详解

## 流水优化（TASK_QUEUE_ENABLE）

### 原理

一级流水优化将部分算子适配任务从 Host 侧一级流水迁移至 Device 侧二级流水，
使两级流水负载更均衡，减少 dequeue 唤醒时间。

训练时反向传播 + 梯度更新会产生大量密集的算子下发，Host 端容易成为瓶颈（host-bound），
此时流水优化收益最为显著。推理场景通常为单向 forward，算子数量少，Host 很少成为瓶颈，
收益有限。

### 使能方法

```bash
export TASK_QUEUE_ENABLE=2
```

在启动训练脚本**之前**设置。

### 注意事项

- `ASCEND_LAUNCH_BLOCKING=1` 时 task_queue 关闭，`TASK_QUEUE_ENABLE` 不生效。
- `TASK_QUEUE_ENABLE=2` 时可能导致 **NPU 内存峰值上升**，遇 OOM 可减小 batch_size 或回退到 `=1`。
- 推理场景下建议先单独测试，确认有正向收益后再保留。

---

## CPU 绑核优化（CPU_AFFINITY_CONF）

### 原理

通过设置处理器亲和性，将任务绑定到 NPU 对应 NUMA 节点的 CPU 核心，
避免跨 NUMA 内存访问，减少调度开销。

训练时多卡数据并行，每张卡绑定对应 NUMA 节点收益明显。
推理场景如果使用 vLLM、TGI 等自带多进程/多线程调度的框架，强制绑核会限制
框架自身的线程并行度和 worker 调度灵活性，可能产生负优化。

### 使能方法

```bash
# 粗粒度绑核
export CPU_AFFINITY_CONF=1

# 细粒度绑核
export CPU_AFFINITY_CONF=2

# 自定义多卡绑核范围
export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:2-5,npu3:6-6
```

### 查看 NUMA 拓扑

```bash
lscpu
```

### 注意事项

- Docker 等虚拟化环境中 NUMA 拓扑可能与物理机不一致，建议根据实际映射自定义绑核范围。
- 绑核特性触发时机较后，一般会覆盖外界的绑核设置（如 `taskset`）。
- 对 CPU 瓶颈的模型有较大提升，对 NPU 瓶颈的模型能保证性能持平。

---

## 高性能内存库替换（tcmalloc）

### 原理

tcmalloc 为每个线程维护本地缓存（thread-local cache），小对象直接从本地缓存分配，
不走全局锁，减少多线程场景下的 malloc 锁竞争。

训练时 DataLoader 多 worker + 频繁 Tensor 创建/销毁，malloc 竞争严重，tcmalloc 收益明显。
推理时 batch 进 batch 出，内存分配模式相对稳定，收益较小。

### 安装

```bash
# openEuler
yum install gperftools

# Ubuntu
sudo apt install libgoogle-perftools4 libgoogle-perftools-dev
```

### 使能方法

```bash
# 确认库路径
find /usr -name "libtcmalloc.so*"

# 全局生效
export LD_PRELOAD="$LD_PRELOAD:<实际路径>/libtcmalloc.so"

# 仅对单个程序生效
LD_PRELOAD="<实际路径>/libtcmalloc.so" python train.py
```

### 验证是否生效

通过运行时检查进程的内存映射确认：

```bash
# 启动训练后，在另一个终端查看
cat /proc/<pid>/maps | grep tcmalloc
```

### 注意事项

- 库路径因安装方式和系统不同而异，务必先用 `find` 确认实际路径。
- `ldd $(which python)` 无法验证 `LD_PRELOAD` 注入的库，需用 `/proc/<pid>/maps` 确认。
- 类似替代品有 jemalloc，思路相同。

---

## 渐进式调优建议

建议按"流水优化 → 绑核优化 → tcmalloc 替换"顺序逐项开启，
每次只变更一个配置，观察性能变化后再叠加下一项。

如果某项优化叠加后性能不升反降，应将其去掉，用剩余项的组合重新测试，
找到最优子集。

### 推荐验证指标

| 指标 | 说明 |
|------|------|
| 单步耗时 | 对比调优前后的平均 step time |
| 吞吐量 | samples/sec 或 tokens/sec |
| NPU 内存峰值 | `npu-smi info` 观察，流水优化可能提升峰值 |
| CPU 利用率 | `top` / `htop` 观察绑核后 CPU 使用分布 |
