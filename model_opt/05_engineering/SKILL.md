---
name: npu-engineering-practice
description: 工程化实践：目录规划、版本管理、文档维护。当用户需要组织代码目录、管理 git 分支与提交、或维护优化文档时触发。
---

# NPU 适配工程化实践

## 工程目录规划

### 原则
- 目录命名按功能语义（scripts/、profiling/、accuracy/、evidence_db/ 等），具体结构由项目决定
- profiling 输出带时间戳 + `latest` 软链接，防覆盖且方便引用
- `.gitignore` 覆盖大文件（模型权重、profiling trace），详见 [templates/gitignore_template.md](templates/gitignore_template.md)
- 应当纳入 git 的：源代码、脚本、文档、对比结果的摘要（非原始大文件）

## Git 版本管理

### 分支策略

```
main 分支（稳定版本，不直接修改）
  └── optimize/<任务名>（主工作分支）
        ├── 逐批实施优化，每批 commit
        ├── explore/<方案A>（子分支，探索有冲突的优化路径）
        └── explore/<方案B>
```

**工作分支规则**：
- 所有优化工作均在 `optimize/` 主工作分支进行，不在 main 上直接修改
- 仅当用户确认优化效果后，才将工作分支合入 main

**子分支探索**（`explore/` 分支）：对互相冲突但都可能带来收益的优化方案，从工作分支开子分支并行评估，选定方案后合并回工作分支。

### 提交纪律

- **提交前必须完成以下步骤**（缺一不可）：
  1. 全量精度验证通过（Level 2，与原始 baseline 对比）
  2. Profiling 确认有收益（重新采集，对比优化前后数据）
  3. evidence_db 案例已按 [schema](../06_evidence_db/schema.md) 记录到项目工作目录的 `evidence_db/` 下
  4. 用户确认提交（展示总结 + evidence_db 路径，等待用户同意）
- 每批优化一个 commit，不要多批合并
- 实验性尝试前确认有干净的回退点（以当前工作分支 HEAD 为基点）

### 性能数据校验

- 每个性能数字都要能追溯到具体的 profiling 输出文件
- profiling 目录可能是 symlink -> 检查实际内容（`readlink -f`）
- baseline 不一定是最早的目录 -> 按 commit 或标注确认
- 对比时注明 warmup 策略（首次运行通常偏慢）

### 提交前用户确认流程

git commit 前，**必须**向用户展示本批总结并使用 `ask_user_question` 等待确认：

1. **总结内容**（必须包含以下信息）：
   - 本批实施的优化点列表及简述
   - 性能数据对比（优化前 vs 优化后，含数据来源）
   - 精度验证结果（指标 + 通过/未通过）
   - 修改的文件列表
   - 未采纳的方案及原因（如有）
2. **询问用户**：
   - 是否确认提交本批优化
   - 是否需要回退某些改动再提交
3. **用户确认后**才执行 git commit
4. 如用户要求调整，完成调整后重新验证精度和 profiling，再次走确认流程

commit message 格式：
```bash
# 格式: [类型] 内容: 简要数据
git commit -m "[perf] 融合 RmsNorm: encoder 延迟 -30%, cos=0.9999"
git commit -m "[perf] Flat Forward: 绕过 Module.__call__, throughput +45%"
git commit -m "[fix] 修复 causal mask 方向: 生成结果与 baseline 全量一致"
git commit -m "[revert] 回退 StaticCache: 在 NPU 上比 DynamicCache 更慢"
```

### 回退原则
回退要干净：删除相关代码（不留 `if use_xxx` 条件分支或注释块），回退后重新运行快速验证。

## Baseline 脚本

### 原则
- 逐一禁用所有优化点（不设置环境变量，不调用 patch，用原始模型）
- 保持其他条件完全一致（数据集、profiler 配置、warmup 步数、推理参数）
- 脚本自包含（不 import 有 side effect 的模块）

### 验证 baseline 有效性
确认 baseline 脚本没有意外引入优化模块（如 NPU 适配 patch、自定义算子等）。具体检查方式取决于项目的优化方式——检查 sys.modules、检查模型类型、或对比首次输出与官方示例。

## 文档维护

### 优化报告结构

```
1. 环境信息（硬件、固件、框架版本）
2. 总体对比（优化前 vs 优化后关键指标）
3. 分项优化技术（每项含 what + why + 效果）
4. 精度验证结果
5. 未采纳方案（及原因）
6. 后续优化方向
```

### 维护纪律
- 每个优化点描述必须包含 what（做了什么）+ why（为什么有效）
- 性能数字必须标注来源（哪次 profiling 的结果）
- 代码迁移/重命名后同步更新文档中的路径引用
