---
name: npu-engineering-practice
description: 自动管理 NPU 优化试验的 Git 分支、commit、回退点、证据文件和可复现命令；在精度与性能门禁通过后提交 accepted trial，隔离失败 trial，并保护用户未提交修改。用于 model_opt Phase 5 或独立的 NPU 优化工程化管理。
---

# Phase 5：证据与 Git

## 1. 工作区安全

- 优先在专用 optimize 分支或 worktree 中工作。
- 发现用户未提交修改时，不覆盖、不清理；使用独立 worktree/分支或绕开冲突文件。
- 每个 trial 记录 repository、branch、parent commit、tree 状态、diff hash 和 environment ID。
- 禁止 reset --hard、force push、复用 tag 或其他破坏历史的回退。

源码尚未使用 Git 时，只在工作副本初始化，不把 raw profiling、权重和编译大文件纳入版本库。

## 2. Trial 隔离

每个 trial 从 current iterative baseline 派生。

accepted trial：

- 对应唯一代码 diff；
- Phase 4 门禁通过；
- 自动提交到优化分支；
- 更新 current iterative baseline；
- 保存复现命令和 evidence refs。

rejected/inconclusive trial：

- 不进入 best branch；
- 保留 patch、独立 commit 或 source.diff；
- 保存准确 failure predicate 和重新打开条件。

## 3. 自动提交门禁

提交前自动检查：

1. 正确性与适用的训练门禁通过；
2. 性能收益超过计时噪声；
3. 无不允许的 regime/rank/memory 回退；
4. evidence_db trial 记录完整；
5. 修改文件与 Candidate 声明一致；
6. 当前 tree 不包含无关用户修改。

通过后无需逐次询问用户即可 commit。稳定主分支不自动合并，除非当前任务已明确授权。

## 4. Commit 粒度

- 一个可独立消融的 Action 一个 commit；
- bundle 可有汇总 commit，但必须保留各 Action 的独立历史或 patch；
- commit message 包含机制和关键结果，不声称未经 full run 的最终收益。

示例：

~~~text
[perf] attention work-domain: weighted short run -8.4%, accuracy pass
[perf] cache invariant mask: allocation count -120/step
[revert] fused loss: gradient mismatch in regime_large
~~~

## 5. 数据与目录

进入 Git：

- source/config；
- benchmark、profiling 和验证脚本；
- comparison summary；
- evidence_db 的结构化记录；
- 小型报告和 manifest。

不进入 Git：

- 模型权重；
- raw profiling/trace；
- compiler cache；
- 大型生成文件。

raw artifact 使用路径、identity/checksum 和不可变版本引用。

## 6. 交付

最终交付记录：

- best/release commit；
- 原始 baseline 与最终 full-run；
- 精度/训练结果；
- regime/rank coverage；
- accepted/rejected trial；
- 剩余 gap 与停止理由；
- 一键复现命令。

如果任务只授权优化和本地提交，不自动 merge、push 或发布。
