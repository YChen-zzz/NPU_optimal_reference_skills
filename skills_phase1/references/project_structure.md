## 项目目录结构

所有派生文件必须放入对应子目录，禁止在项目根目录平铺。

```
项目根目录/
├── train_gpt.py                    # 唯一的主训练脚本（持续就地修改）
├── run.sh                          # 唯一的主启动脚本
├── benchmarks/
│   └── supernodes/
│       ├── progress.md             # 优化进度追踪
│       └── sn_<name>.py            # 各 SN 的 Lab 脚本
├── ablations/                      # 多卡 ablation 用的短跑变体
│   ├── train_gpt_short_baseline.py
│   ├── train_gpt_short_<opt>.py
│   ├── run_short_baseline.sh
│   └── run_short_<opt>.sh
├── probes/                         # 单算子探测脚本
│   ├── probe_<name>.py
│   └── run_probe_<name>.sh
├── logs/                           # 所有运行日志
│   ├── baseline_short.log
│   ├── sn_<name>_L<N>.log
│   └── final_full_run.log
├── profiling/                      # profiling 输出（.gitignore）
└── custom_op/                      # 自定义算子（如有）
```

核心规则：
- `train_gpt.py` 是唯一的主脚本——ablation 通过的优化合入这里，不创建 `train_gpt_full_*.py` 变体
- 短跑变体和 run 脚本放 `ablations/`，probe 脚本放 `probes/`
- profiling 相关的 run 脚本也放 `ablations/`

## Git 分支策略

```
main（稳定版本，不直接修改）
  └── optimize/stage1（主工作分支）
        ├── 逐 SN 实施优化，每个 SN ablation 通过后 commit
        └── 所有 SN 完成后合入 main
```

1. 优化开始前: 从 main 创建 `optimize/stage1` 分支
2. 每个 SN ablation 通过并合入 train_gpt.py 后: `git commit -am "SN-<name>: <winner> (step_avg Xms→Yms)"`
3. 需要回滚: `git checkout -- train_gpt.py`
4. Stage 1 完成后: 用户确认后合入 main

不要在 git 操作上花超过 1 分钟。

## 日志管理

所有训练 run 的日志保存到 `logs/` 目录，文件名编码实验内容：

```
logs/
├── baseline.log
├── baseline_short.log
├── sn_loss_L4c_compile_sig.log
├── sn_attn_L0a_pre_tockens.log
├── v5_all_combined.log
└── final_full_run.log
```

统一格式：
```bash
torchrun ... ablations/train_gpt_short_X.py 2>&1 | tee logs/<descriptive_name>.log
```

快速对比：
```bash
grep "step:.*val_loss" logs/*.log
```
