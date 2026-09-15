# streaming_dynamic_multiquery_v2：P0 当前实现审计

审计时间：2026-09-15。本文记录开始本轮增量实施时的实际工作区，不把旧交接文本中的计划当成当前事实。

## 仓库与约束

- 根目录：`D:\Project\Age-Mem\AgeMem`
- 基线提交：`cf80142d72b3009635234d5693fde4acf4a9c839`
- 分支：`fix/training-protocol`，跟踪 `origin/fix/training-protocol`
- 开始时唯一未提交项：用户提供、未跟踪的 `AGEMEM_DYNAMIC_LOGIC_INCREMENTAL_SPEC_V2.md`
- 仓库及父目录均未发现 `AGENTS.md`；因此没有额外仓库级代理规则。
- v1 已提交并推送；旧三阶段实现、冻结 YAML、checkpoint 和 318 scope 未在本轮审计前被修改。

## v1 实际范围

| 方面 | 已实现 | 已有验证 | 尚未验证／缺失 |
|---|---|---|---|
| 流式数据 | HotpotQA 先划分后拼接、public/questions/gold/registry 分离 | 24 histories、96 queries；跨 split overlap=0 | 生产 Qwen tokenizer 下 5–10C 尚无本地结果 |
| 双预算 | chat-template 后检查 C；canonical active payload 检查 B | debug C=4096、B=2048；环境 fixture 通过 | 模型 tokenizer/chat template 的生产计数未锁 |
| 多问题 | 读完后从不可变 snapshot 分出 m 个分支 | m=4 隔离测试通过 | 无真实 reader/model rollout |
| 训练契约 | K 完整组、m 先平均、共享 ingest token 只发布一次、reader mask=0 | K=2/m=2 手算契约测试通过 | `train_streaming_multiquery.py` 仅是发布边界；没有 v1 新协议真实 trainer step |
| v1 奖励 | terminal；extractive Flat-state；等价 DFA | CPU fixture FS=D，最大差值 0 | 自由摘要 grounder 未准入；无新协议学习结果 |
| 动态语义 | 无 | 无 | 没有时间 Oracle、版本 grounding、END/LIFE 或动态 monitor |

`streaming_multiquery_v1` 的代码位于 `AgeMem_code_agentscope/streaming_memory/`，发布契约位于 `trinity/common/streaming_multiquery_contract.py`。本轮在其下新增 namespaced `dynamic/`，不修改 v1 schema/reward 身份。

## v1 数据、模型与最近可见产物

最近可见 v1 manifest：`runs/stream_mq/data_v1/manifest.json`（runs 被 Git 忽略）。

- build ID：`stream-mq-e67353cd8abf`
- 产物记录的代码：`58d3fcc9e293f1adb388ece01606aad1b10e491b`，并带 dirty digest；它不是当前提交 `cf80142d...` 的干净重建
- tokenizer：`debug-lexical` / `fixture-v1`，不是 Qwen tokenizer
- 数据：24 histories（16/4/4）、96 queries、350 chunks、6,879 registry sentences
- alpha：1.8096–2.1423C；不是 v2 主压力档 5–10C
- v1 replay：terminal=0、semantic mean=0.291667、Flat=DFA=0.072917；仅为 scripted CPU fixture

旧 HotpotQA provenance 自由文本审计 precision/recall/F1=`0.539130/0.601942/0.568807`，低于 v2 建议准入线 `0.95/0.80`。该结果不能直接迁移为动态自由文本 grounder 的可信度。

本地工作站未发现已锁定的 Qwen policy/reader 路径、初始化 checkpoint、optimizer lock 或 GPU runtime receipt。之前用户在远端运行的 A6000/Qwen 实验只保留为历史叙述；本轮无法从本地文件验证其当前可用性、显存峰值或吞吐。

## 不变量核对

- C：v1 使用实际 chat template 计 `prompt + max_new_tokens`，可复用 `TokenAccounting`。
- B：v1 canonical payload 计 content/title/tags/source_refs/custom；v2 还必须把 claims 计入，不能直接沿用 v1 serializer。
- 旧版本正文：v1 环境内部保存 `memory_versions`；policy observation 没有读取 API。v2 需进一步把 audit ledger 与 policy store 分型。
- K/m：v1 契约按同一 history 的 K 份 memory 分组，m 题先平均；分支不回写。
- 训练：reader token 不入 actor loss，shared prefix 每 read rollout 一次；这在契约测试中成立，真实 runtime 未验证。
- 固定 reader/retriever：词法 retriever 有 CPU 实现；production revision 与真实 reader 均未锁。

## P0 结论

可直接复用 v1 的 tokenizer 预算、split-first 原则、snapshot 分支、词法检索和 K×m 发布契约思想；不能把静态 `support sentence` 覆盖器当成动态时间 grounder，也不能把 S4 发布函数称为真实 trainer 集成。v2 必须新增动态 public/private schema、独立 Oracle 与第二验证路径、版本隔离、source/content/time 分判、END/LIFE 以及 direct/compiled 差分验证。
