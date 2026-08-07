# GPU Teacher 证据与读取合同

## 推荐证据

| 证据 | 用途 |
|---|---|
| source、config、command、environment | 数学、精度、API、regime 和版本 |
| compiled graph inventory | distinct graph 与 signature coverage |
| readable/transformed graph | compile 前后的图结构 |
| pre/post-fusion IR | 删除、融合、materialization、layout、reuse |
| generated code/kernel metadata | 真实 dtype、work-domain、load/store、saved tensor |
| compiled profile/timeline | kernel、collective、overlap、rank 和 exposed interval |
| run log/manifest | compile、warmup、active window 和完整性 |
| eager GPU profile（可选） | 解释 compile delta，不作为 Teacher 最终状态 |

其他 compiler stack 可以使用等价 artifact，不要求特定文件名。

## Regime 与 Rank

- 先从 source/schedule 识别会改变执行结构的 regime。
- 每个 regime 保存多个 warm 后 steady step。
- 分布式负载优先保留 all-rank light trace；deep trace 选择代表、最慢和异常 rank。
- 记录 regime 在完整任务中的出现次数，供收益加权。
- transition、validation、checkpoint 和 steady window 分开。

## 读取账本

每个 artifact 记录：

artifact_id, backend, kind, path, identity, read_status, read_scope, read_method, extracted_facts, claim_ids, limitations

read_status 使用：

- fully_read：完整阅读适合人工消费的文本；
- targeted_read：读取已记录的行、graph、rank/regime 或 event window；
- parsed：确定性解析完整文件或声明范围；
- index_only：只读索引，不能证明 transformation/timing；
- inventory_only：只知道存在，不能支持结论；
- unread / unavailable。

高优先级 claim 至少引用 source contract、GPU pre/post chain、GPU compiled runtime、NPU source/runtime 中适用的证据。仅有 summary、PPT、文件名或相似 kernel 名不得给 A 级。

## Provenance

尽量建立：

~~~text
source/module
→ pre-compile node
→ transformed node
→ fusion group
→ generated kernel
→ runtime interval
~~~

缺失环节明确标记，不用名称相似性填补。

## Pack 缺失时

标准模式不跨机器自动采 GPU。生成 capture request，列出：

- 缺失的 source revision/config；
- 缺失的 regime、rank 和 step window；
- 缺失的 graph/IR/code/runtime family；
- 推荐的 capture 范围；
- 每项缺失阻塞哪些 claim。

只有 GPU source/compiler/regime 改变，或关键 coverage/provenance 缺失时才要求补采 GPU；NPU 优化迭代通常复用原 pack。
