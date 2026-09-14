# Streaming Multiquery 实施报告

## 当前状态

- 协议：`streaming_multiquery_v1`
- 实施范围：S0～S4 的本地代码、CPU 数据构建、结构/契约测试
- 状态：**S0～S4 CPU 工程闭环通过；S5 模型诊断未开始**
- 结论边界：当前结果只证明数据、预算、分支、奖励重放和训练输入契约可执行；没有运行模型、optimizer 或 GPU，**不构成学习有效证据**。
- 旧三阶段协议、冻结 E1/E3 YAML、checkpoint、318 runtime-gate scope 均未修改或重跑。

## S0 接管审计

开始实施时的只读状态：

- 仓库：`D:\Project\Age-Mem\AgeMem`
- 分支：`fix/training-protocol`，跟踪 `origin/fix/training-protocol`
- 基线提交：`58d3fcc`（`feat(agemem): ground oracle rewards with fact provenance`）
- 原工作树仅有用户提供、未跟踪的 `AGEMEM_STREAMING_MULTIQUERY_CHANGE_SPEC.md`；实施过程中未覆盖该文件。
- 仓库及父目录未发现 `AGENTS.md`，因此没有可应用的仓库级代理说明。
- 已完整读取 `STATUS.md`、`PROJECT_HANDOFF.md` 和根规范。交接文档中“下一步运行旧 E3 GPU”等历史指令已被本次明确的新协议请求覆盖；历史正文保留。

实际接口核对：

- `AgeMem_code_agentscope/memory_store.py` 提供 rollout-scoped、版本化、soft-delete store，但旧 budget wrapper 只计 `content`，不满足新 B。
- `trinity/common/experience.py` 和 `action_event_contract.py` 支持 token/action join 与 policy version，但基本单位仍是 step Experience，不能直接表达 K 份共享阅读各带 m 个只评分分支。
- `trinity/buffer/reader/queue_reader.py` 的 `consume_put_batch` 可消费整批，但新协议仍需先在发布边界验证完整 bundle。
- `AgeMem_code_agentscope/hotpotqa_benchmark/adapter.py` 可离线读取本地 DatasetDict；新 builder 复用数据布局，不复用旧三阶段 observation。
- 本机 HotpotQA fullwiki 为 11 个文件、645,926,725 bytes。

旧 provenance 的最新人工对齐结果是 400 动作中 TP/FP/FN/TN=`62/53/41/238`，precision=`0.539130`、recall=`0.601942`、F1=`0.568807`。它没有达到新规范建议的 `precision>=0.95, recall>=0.80`，因此当前语义奖励门禁为 **blocked**。来源 ID 正确不能替代正文事实语义。

## 配置与数据

CPU debug 配置：`configs/stream_mq/debug.yaml`。

- C=`4096`，B=`2048`
- ingest/answer max new tokens=`512/256`
- chunk target/max=`640/768`
- answer tail=`512`，retrieval payload cap=`2048`
- K=`4`，m=`4`，`std_ddof=0`
- 首轮 `train_scope=ingest_only`
- debug 历史目标 `alpha=2.0±10%`
- tokenizer：`agemem-debug-unicode-lexical-v1`（fixture-v1）；仅用于 CPU 结构构建，明确禁止模型运行

本机 `.venv` 没有 `transformers`，所以缓存中的 Hugging Face tokenizer 未被加载。生产数据必须在 S5 前用冻结 Qwen tokenizer/revision 重新构建到新的 output root，不能沿用 debug token 数。

真实 HotpotQA CPU 构建结果（`runs/`，按现有规则不进入 Git）：

| 指标 | 实测 |
|---|---:|
| histories | 24（train/dev/test = 16/4/4） |
| evaluated queries | 96（64/16/16） |
| queries per snapshot | 4 |
| chunks | 350 |
| source-registry sentences | 6,879 |
| history tokens min/mean/max | 7,412 / 7,981.96 / 8,775 |
| alpha min/max | 1.8096 / 2.1423 |
| chunk tokens min/max | 74 / 640 |
| QA / exact paragraph / normalized paragraph / SimHash-near cross-split overlap | 0 / 0 / 0 / 0 |
| question↔gold joins | 96/96 |
| gold support↔source registry joins | 全部 |

预定义坏样本/冲突处理计数：无法解析 supporting pointer 247、重复 question ID 1、跨 split exact paragraph 候选 4、无效 context 1、跨 split SimHash-near 候选 2；这些候选在拼 episode 前被拒绝，未从已构建 test 的低分结果中事后删除。

数据先分配 QA 和文档身份到 split，再在 split 内拼 episode。公开、问题、gold、source registry 和 manifest 分文件保存；policy serializer 使用字段白名单。答案文字自然出现在阅读来源中 92 次，这是 HotpotQA 事实正文，不是 gold 字段泄漏；阅读阶段不存在 question、answer、support label 或 source question ID 字段。

## 变更文件

新增实现：

- `AgeMem_code_agentscope/streaming_memory/{schema,token_budget,data_builder,environment,query_runner,reward_replay}.py`
- `trinity/common/streaming_multiquery_contract.py`
- `trinity/common/workflows/memory_context/train_streaming_multiquery.py`
- `scripts/agemem_stream_mq.py`
- `configs/stream_mq/debug.yaml`
- `tests/common/stream_mq_*_test.py`
- 本报告与规范入口

没有修改旧 memory store、旧三阶段 workflow、旧实验 YAML、checkpoint、artifact 或 runtime-gate 计数。

## 实际运行命令与结果

```powershell
.\.venv\python.exe -m unittest discover -s tests/common -p 'stream_mq_*_test.py'
# 20 tests, OK

.\.venv\python.exe scripts\agemem_stream_mq.py build-data --config configs\stream_mq\debug.yaml
# PASS: 24 histories / 96 queries / 350 chunks

.\.venv\python.exe scripts\agemem_stream_mq.py validate-data --manifest runs\stream_mq\data_v1\manifest.json
# PASS: 6 files; 96 exact question/gold joins; 6,879 source sentences

.\.venv\python.exe scripts\agemem_stream_mq.py env-smoke --config configs\stream_mq\debug.yaml --policy scripted
# PASS: first history 15/15 chunks observed; 2 active memories; B=1930/2048;
# 23 FIFO evictions; four independent branches share one immutable parent snapshot

.\.venv\python.exe scripts\agemem_stream_mq.py replay --run-dir runs\stream_mq\cpu_debug --profiles terminal,flat_state,dfa
# PASS: terminal=0, semantic mean=0.291667, FS=D=0.072917,
# max |FS-D|=0, duplicate_dfa_gpu_experiment_required=false

.\.venv\python.exe scripts\agemem_stream_mq.py gate --scope cpu --config configs\stream_mq\debug.yaml
# PASS

.\.venv\python.exe -m unittest tests.common.memory_store_test tests.common.m8_action_event_contract_test tests.common.hotpotqa_oracle_benchmark_test
# 39 tests OK, 3 environment-dependent skips
```

`compileall` 与 `git diff --check` 也通过。

## 验证

S1：

- 多 QA context 按 title+body digest 去重；相同标题但正文不同会保留为不同文档。
- 先 split 后拼 episode；validation 的 dev/test 互斥，跨 split exact/normalized/SimHash-near 文档门禁 fail closed。
- 按 tokenizer 计数、整句/整段拼 chunk，实际 debug history 落在 `2C±10%`。
- public/questions/gold/source registry 分离，manifest 锁定文件 digest 和行数；相同 digest 可幂等复用，身份冲突拒绝覆盖。

S2：

- 每次实际 chat-template render 后验证 `prompt_tokens + max_new_tokens <= C`。
- B 计算 canonical active payload，包含 content/title/tags/source_refs/custom；opaque environment ID 不计入。
- ADD/UPDATE 超预算或无效来源原子失败；来源必须已经观察。
- 每个 chunk 至少渲染给 policy 一次，FIFO 是 `environment_event`，不是 policy CLEAR。
- 阅读结束前不可 snapshot；m 分支克隆同一不可变 snapshot，不回写。
- 固定 retriever 只索引 active memory content，不解析 source ref 原文。

S3：

- terminal 先对 m 题 F1 求均值。
- extractive control 同时要求有效 source pointer 与真实存储正文包含来源句；“正确 ID + 无关总结”不得分。
- 删除/改坏最后有效表示会从 retained/exposed 回退到 absent；仍有等价表示时不会误清零。
- 静态固定检索、回答分支不写记忆时，FS 与 D 逐 rollout 完全相等。**因此当前配置明确跳过重复 D GPU 实验**；只有升级依赖语义和 reward version 后才能恢复 Flat-vs-D GPU 对照。

S4：

- `MemoryRolloutGroupBundle` 只有 K 份完整阅读都包含同一 m-query set 时才可发布。
- m 个分支先聚合为一个 rollout reward，再在 K 份记忆间按总体标准差计算 GRPO advantage。
- K=2,m=2 数值 fixture 得到约 `+1/-1`；复制评分向量不放大奖励。
- actor batch 只包含唯一 ingest read samples；reader token 的 actor-loss count 必须为 0。
- bundle 拒绝漏分支、重复 action、混合 policy version、非有限 logprob 和未结算失败。
- receipt 报告 K、m、共享前缀、actor samples、reader mask、reward std 和版本。

## 研究结果

- 尚无新协议自然模型轨迹、冻结 reader F1、group reward 方差、KL、梯度或 checkpoint 指标。
- 尚无 optimizer update，更无 dev/test 学习提升。
- CPU scripted smoke 和 mock/fixture PASS 只证明工程契约，不证明策略会学会保存知识。
- 旧 provenance grounder 未通过语义门禁；新 extractive control 仅是结构/可达性基线，不代表自由摘要语义准确率。

## 剩余阻塞与下一步

1. 用冻结 Qwen tokenizer/chat template/revision 重建 production split，并覆盖 2C 诊断和 5～10C 主 histories；当前 debug 只验证约 2C。
2. 冻结实际 policy、reader、optimizer/LoRA/KL/clipping/精度配置；当前 debug config 对这些字段保持 null/blocked。
3. 在新协议自然 ADD/UPDATE/DELETE/检索截断样本上做去重语义审计。旧 grounder 的 0.539/0.602 不允许开启 FS/D 训练。
4. 实现 S5 模型 runtime producer，把真实 token/logprob/action spans 转为已验证 bundle；当前 S4 只实现并验证发布边界，没有声称已接入一次真实 trainer step。
5. 先做冻结模型可答性、last-window、question-only、固定记忆与自然 group-variance 诊断；满足门禁后才预算 S6 Terminal pilot。
6. 当前 FS=D，S7 不安排重复 D GPU 臂；如后续加入动态版本/主动多步检索，必须用新 reward version 重新证明是否不等价。
