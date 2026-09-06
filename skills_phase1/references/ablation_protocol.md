# 多卡 Ablation 详细流程

> 从主 SKILL.md 拆分的 reference 文件，覆盖短跑脚本构造、执行规则、验证与回退策略。

---

## 1. 短跑脚本构造方法

首次需创建，后续复用。从完整训练脚本派生一个 ~60 秒的短跑版本：

- 等比压缩 total steps（如 2090 → 210 步，约 1/10）
- 按相同比例映射所有 step-dependent schedule（lr、batch size、window size 的切换点）
- 保持相同 seed、数据、初始状态
- 输出格式不变（`step_avg` 可直接对比）

```python
num_scheduled_iterations = FULL_STEPS // 10
num_extension_iterations = FULL_EXT // 10
val_loss_every = num_scheduled_iterations + num_extension_iterations
```

---

## 2. 短跑与 Full Training 的关系（单向派生，禁止反推）

短跑脚本是从原始 `train_gpt.py` 的训练参数**单向派生**的产物。Full training 的参数**永远以 baseline git commit 中的原始 `train_gpt.py` 为唯一真相来源**，禁止从短跑脚本反向推算 full training 参数。

- **短跑创建时**，记录派生来源：在短跑脚本头部注释中标注 `# Derived from: train_gpt.py @ commit <hash>` 和原始完整参数值。
- **Full training 恢复时**：直接 `git show <baseline_commit>:train_gpt.py` 获取原始训练参数。
- **禁止**从短跑参数 × 压缩比来还原 full training 参数。

---

## 3. Ablation 执行规则

1. 先跑一次 baseline 短跑 → 记录 `step_avg` → 存入 `logs/baseline_short.log`
2. 每个有增益的方案放入 `ablations/` 子目录
3. 提交多卡短跑 → 结果写入 `logs/sn_<name>_L<N>.log`
4. 对比 baseline 的 `step_avg`
5. **只有 step_avg 下降才接受**

> 短跑的 `val_loss` 不用于判断正确性（步数太少未收敛），短跑只验证多卡环境下的真实速度增益。

---

## 4. 条件通过方案的短跑验证

对于"条件通过"的方案（高增益 + 误差在 dtype 精度可解释范围内），短跑同时承担速度验证 + 精度观察：

1. 记录 `step_avg` 和 `val_loss`
2. `val_loss` 在自然波动内 → 暂时 accept
3. `val_loss` 明显异常 → 降级为拒绝
4. 条件通过方案在 `progress.md` 中标记为 `✅cond`

---

## 5. 回退策略（val_loss 超标时）

1. **按嫌疑度排序**：Lab 阶段误差越大的方案嫌疑越高
2. **逐个回退验证**：从最高嫌疑开始
3. **定位元凶后决策**
