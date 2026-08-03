# Project Status

## Current milestone

M6：Extracted Triple/AP、显式状态与 Oracle 对照

状态：完成

## Completed

### M0

- [x] 完成仓库、分支、远程、环境和用户修改的接管检查
- [x] 用户完成 standalone demo 真实运行并确认
- [x] 六工具基础 smoke test 通过
- [x] 复现记录保存在 `docs/reproduction.md`

### M1

- [x] 实现严格 JSONL `TrajectoryStep`、AgentScope recorder hook 和 schema validation
- [x] 保存 observation、action、ToolResponse、memory before/after 和 env reward
- [x] 实现不调用 LLM/embedding 的确定性 `TrajectoryReplay`
- [x] 支持按 `task_id / rollout_id / timestep` 查询
- [x] M1 commit：`f5ceffd feat(agemem): add deterministic trajectory replay`

### M2

- [x] 定义 `MemoryStore` protocol 和 AgentScope adapter
- [x] 实现 rollout-scoped `InMemoryStore` 与 `RolloutMemoryStoreRegistry`
- [x] 实现 add/retrieve/update/delete/snapshot/restore/reset
- [x] update 保留版本历史，research-mode delete 使用 soft delete
- [x] M2 commit：`02536bb feat(agemem): isolate versioned memory stores by rollout`

### M3

- [x] 实现严格的 `ToyFact`、`ToyMemoryTask`、`StageInput`、`ToyAction` 和 episode schema
- [x] 新增 30 条人工两跳事实任务：20 train / 5 dev / 5 test
- [x] 每条任务标注 supporting、distractor、stale、duplicate fact IDs 和 answer
- [x] test split 使用 train 未出现过的实体组合
- [x] Agent 可见 `StageInput` 不包含 answer、fact IDs 或 Oracle labels
- [x] 实现与现有 AgeMem workflow 一致的三阶段协议
- [x] Stage 1→2 清空 STM 并保留 LTM；Stage 2→3 保留干扰上下文并追加问题
- [x] 复用 M2 manager、registry、snapshot/restore/reset 和版本化 update/soft delete
- [x] 使用确定性本地 embedding 和 timestep clock，不构造 OpenAI client
- [x] 实现 `GoldMemoryPolicy` 和五种显式 `ErrorMemoryPolicy`
- [x] gold policy 完成 30/30 任务
- [x] 覆盖干扰、重复 ADD、事实更新、过期检索和关键记忆误删
- [x] episode success 同时要求答案正确、当前 supporting memory 完整和 retrieval coverage 完整
- [x] 复用 M1 `TrajectoryRecorder` 输出完整 memory before/after JSONL
- [x] seed 和严格 M3 Oracle labels 保存于 ToolResult metadata
- [x] 相同 task/rollout/seed 生成字节级一致 JSONL 和相同 replay digest
- [x] 未下载真实 HotpotQA，未调用真实 LLM，未实现 AP/DFA/reward/Critic/GRPO

### M4

- [x] 从 M3 `ToolResultSnapshot.metadata["oracle_labels"]` 生成 9 种语义 AP
- [x] AP grounder 校验 task/rollout/stage/seed 和 fact-ID 语义集合，异常时 fail closed
- [x] AP 映射不读取工具名或裸 ADD/RETRIEVE 调用
- [x] 实现严格 `AutomatonSpec`、确定性校验和手工正向 DFA
- [x] 实现 q0→q4 progress chain、并行 update edge、reject 和 timeout 状态
- [x] 同一步 coverage/retrieval AP 使用固定优先级闭包，结果完全确定
- [x] 所有 progress edge 按 `edge_id` once-only，重复成功 UPDATE 也不能重复获奖
- [x] irrelevant store/retrieve 记录为 violation，M4 配置暂不启用负奖励
- [x] supporting memory delete 进入 rejecting state；未完成长循环进入 timeout state
- [x] 实现 M1 JSONL → Oracle AP → DFA → `RewardBreakdown` 的离线 replay
- [x] 每步分别保存 env、milestone、violation、trend、format 和 total reward
- [x] 提供 `terminal_only` 与 `terminal_dfa` 两个外部 JSON 配置 profile
- [x] Trend Shaping 固定为 0，未实现 Critic、自然语言抽取、负自动机或训练接入
- [x] 30/30 gold traces 接受，四类预定义 failure traces 全部拒绝
- [x] 重复 ADD/RETRIEVE/UPDATE、循环和 reward farming 测试通过
- [x] reward JSONL 和 replay digest 重复运行完全一致
- [x] 全链路不调用真实 LLM、在线/model embedding 或网络

### M5

- [x] 使用 `datasets.load_from_disk` 读取本地 HotpotQA fullwiki `DatasetDict`，运行期不下载数据
- [x] 校验 source train / validation / official test 规模分别为 90,447 / 7,405 / 7,405
- [x] 对官方 test 全量 7,405 条执行 label-blind 校验：answer 为 `None` 且 supporting labels 为空
- [x] 建立固定 6 train / 2 dev / 2 held-out test smoke split；train 来自 source train，dev/test 来自 source validation 且任务互斥
- [x] manifest 保存 source fingerprints、完整 smoke config digest、source index、Hotpot ID、type、level 和 supporting-fact 数量
- [x] supporting facts 只按精确 `(title, sent_id)` 解析，不使用标题或句子的字符串包含匹配
- [x] 从精确 source pointer 与句子正文生成稳定 SHA-256 fact IDs，并保留可审计 pointer
- [x] 适配真实任务中 2～4 条 supporting facts；不为真实句子伪造 M6 subject/relation/object 三元组
- [x] 复用 M3 三阶段环境、M2 rollout store、M1 `TrajectoryRecorder`/`TrajectoryReplay` 和 M4 Oracle AP/DFA
- [x] public `StageInput` 不含 answer、supporting IDs 或 Oracle labels；改变私有 answer 不改变三阶段 observation
- [x] 收集并重放 30 条轨迹：10 个真实任务 × gold / wrong-answer / missing-support 三种确定性策略
- [x] 10/10 gold episode 成功且 DFA 接受；20/20 失败对照 episode 失败且 DFA 拒绝
- [x] 计算 Answer EM/F1（含 HotpotQA yes/no/noanswer 规则）、support coverage、memory precision、Oracle cumulative retrieval recall@k、context-token estimate 和工具调用次数
- [x] 每条失败审计保存 exact source pointers、最终 memory 版本历史和逐步 AP/DFA state/edge trace，不复制完整上下文或答案
- [x] 完整 JSONL 只写入 gitignored `runs/`；提交固定 manifest、紧凑 JSON report、failure JSONL 和 Markdown 报告
- [x] 同一真实 smoke benchmark 重跑后 report、trajectory 和 reward 文件 SHA-256 完全一致
- [x] 全链路真实 LLM 调用数为 0；未开始自然语言 AP 抽取、Critic、GRPO 或模型训练

### M6

- [x] 先完成 M5 历史轨迹 schema 审计：30 条规范 rollout、224 个 action 与 224 条 reward 记录完整连接
- [x] 使用 namespaced v2 `ActionEvent`、`TrajectoryStepV2`、`RewardBreakdownV2` 和 `ActionCreditRecord`，不覆盖 M5 原文件
- [x] `action_id` 沿用原 tool-call ID；M5 规则/oracle/error-injector 轨迹的 token span、token IDs、old logprobs 和 policy version 保持 `None`，未伪造训练元数据
- [x] 20 个同一步双 edge 动作使用有序 `transition_ids` 保真；迁移 manifest digest 为 `3615ce1041b47ea30513e81f5ef812da4060df9fb854b843162c171443ac5452`
- [x] 实现严格 Triple/AP schema、精确 evidence span/digest、有限 confidence、unknown subject/category 与坏 evidence quarantine
- [x] 实现确定性 mock extractor 和 injected-client LLM adapter；LLM adapter 只使用 fake client 测试，M6 benchmark 真实 LLM 调用数为 0
- [x] Group cache 只缓存 action-independent candidates，并按 task/split/group/stage、observation、约束及 extractor/model/prompt 版本隔离；materialize 时重新绑定原始 `action_id`
- [x] 实现 rollout-scoped `StateTracker`：single-valued category 使用半开区间版本覆盖，multi-valued category 保留多值，并支持 reinforcement、quarantine、snapshot/restore/reset
- [x] 从公共 tool result、memory before/after delta、validated Triple 和 StateFact 生成 AP；不读取 `oracle_labels` 或 private role metadata，也不奖励裸 ADD/RETRIEVE
- [x] 每个派生 AP 经 `action_id` 和 Triple/State/Memory evidence ID 追溯到原始动作；离线奖励复用 M4 手工 DFA 与 once-only milestone 规则
- [x] 建立 10 个任务、34 个句子记录、37 个三元组的人工标注集，其中 24 条为 official supporting relevant facts、10 条为 irrelevant sample
- [x] 本地校验 exact source pointer、Hotpot ID、正文 SHA-256 与 stable fact ID；annotation corpus digest 为 `fa74d5098e8dd4040d66ca99ecd76346d4cc799a59ec3f3a4133ba9bab98edd0`
- [x] 在 M5 的 30 条规范 rollout / 224 个 action 上完成 human-backed mock 与 controlled-error 对照，两个 profile 均为 cache hit/miss 94/130、extractor calls 164、AP provenance 100%
- [x] human-backed mock：Triple F1 `1.0000000000`；AP F1 `0.9760765550`（FP=0、FN=10）；FA `0/20`、FR `0/10`
- [x] human-backed mock：action reward total MAE/RMSE/bias/max_abs 均为 `0`，trajectory signed/absolute error 均为 `0`；10 accepted / 20 rejected；10 条 first-divergence audit
- [x] human mock 是 Triple-extraction 上界，不是完整模型上界；10 个 AP FN 来自 fail-closed answer correctness，未造成 action 或 trajectory reward 误差
- [x] controlled error：Triple F1 `0.8695652174`（TP=30、FP=2、FN=7）；AP F1 `0.8369565217`（TP=154、FP=0、FN=60）
- [x] controlled error：FA `0/20`、FR `5/10 = 0.5`；action reward MAE/RMSE/bias/max_abs 为 `0.056919642857 / 0.140748953307 / -0.032366071429 / 0.5`
- [x] controlled error：trajectory signed error `-7.25`、absolute error `7.25`；5 accepted / 25 rejected；20 条 first-divergence audit
- [x] M6 紧凑报告不保存原始句子；最终 report digest 为 `e803f7752dc9e7357284887cf7716273bbd5396f62db1fc438d7cad95a2f9f92`
- [x] 未实现 Group Critic、GRPO 或模型训练

## Environment

- OS: Windows NT 10.0.26200.0
- PowerShell: 5.1.26100.8875
- Python: 3.10.20 (`.venv`)
- AgentScope: 1.0.21
- Pydantic: 2.13.4
- Datasets: 4.8.5
- PyArrow: 25.0.0
- PyTorch: 未安装（M6 离线 benchmark 不需要）
- Ruff: 0.15.9；pytest/Flake8 未安装；测试使用标准库 `unittest`

## Git state

- Current branch: `feat/m6-extracted-ap-state-tracker`
- M5 base: `6367569 feat(agemem): add HotpotQA Oracle benchmark`
- M6 schema audit commit: `d1d45ab feat(agemem): audit and migrate M5 action schema`
- M6 implementation commit: `1c8e5c1 feat(agemem): add extracted AP state benchmark`
- `PROJECT_HANDOFF.md` 是用户维护的未提交文件，Codex 未修改、未暂存
- M6 未推送远程

## Files changed in M6

- `AgeMem_code_agentscope/action_schema/`（新增）
- `AgeMem_code_agentscope/memory_extraction/`（新增）
- `configs/m6_extraction_benchmark.json`（新增）
- `data/annotations/m6_hotpotqa_manual_triples.json`（新增）
- `data/annotations/m6_hotpotqa_semantic_targets.json`（新增）
- `artifacts/m6_extraction_benchmark/`（新增）
- `docs/schema_audit_m6.md`（新增）
- `docs/m6_extraction_benchmark.md`（新增）
- `tests/common/m6_schema_migration_test.py`（新增）
- `tests/common/m6_extractor_test.py`（新增）
- `tests/common/m6_state_tracker_test.py`（新增）
- `tests/common/m6_grounding_reward_test.py`（新增）
- `tests/common/m6_extraction_metrics_test.py`（新增）
- `tests/common/m6_extraction_benchmark_test.py`（新增）
- `AgeMem_code_agentscope/README.md`
- `STATUS.md`

## Verification

- M5 adapter/benchmark/local-data tests：10/10 PASS
- M1～M5 core regression：60/60 PASS
- Existing tool-trace regression：28/28 PASS
- Combined scoped unittest suite：88/88 PASS
- M5 gold success + DFA acceptance：10/10
- M5 wrong-answer/missing-support failure + DFA rejection：20/20
- Official test label-blind validation：7,405/7,405
- Real smoke report/trajectory/reward repeated-run SHA-256 stability：PASS
- Oracle report schema/digest validation：PASS
- Python compile/import smoke：PASS
- M5 scoped `git diff --check`：PASS
- Oracle report digest：`c18b21b59506733b133ac3510b9c9136c780b79e14af2c96d74b81b6b8d8eef0`
- M6 schema/extractor/state/grounding/reward/metrics/benchmark tests：43/43 PASS
- M1～M6 加 existing tool-trace scoped regression：131/131 PASS
- M6 schema migration source preservation与重复运行确定性：PASS
- M6 annotation exact-pointer/hash/stable-ID validation：34/34 PASS
- M6 canonical ActionEvent↔ActionCredit join：224/224 PASS
- M6 benchmark canonical rollout/action coverage：30/224
- M6 real LLM calls：0
- M6 report schema/digest validation：PASS
- M6 report digest：`e803f7752dc9e7357284887cf7716273bbd5396f62db1fc438d7cad95a2f9f92`

## Known constraints

- M3 使用人工任务和规则 gold/error policy，不代表真实 HotpotQA 模型表现
- M3 Oracle labels 是结构化监督事件，不是 M4 AP，也没有产生逻辑奖励
- DELETE 不出现在公共 M3 动作空间；仅由错误策略调用现有 M2 soft delete 模拟关键记忆误删
- Stage 3 使用确定性 fact-ID metadata filter，这是 gold/error policy 的 Oracle 行为
- 完整 JSONL 含事实正文和 embedding，必须按敏感数据处理
- M4 使用 Oracle AP 上界，不代表后续自然语言 Extracted AP 的准确率
- 当前正向 DFA 是手工有限状态基线，没有 LTLf 编译或自动 Critic
- `violation_weight=0.0`：M4 记录无关存储/检索，但不启用负奖励或 Negative Automata
- `format=0.0`：输入已经通过 M1 严格 schema；M4 不额外设计格式奖励
- M5 的 `gold` 是 Oracle 上界；`wrong_answer` / `missing_support` 是确定性失败对照，不代表真实 base model 表现
- M5 smoke dev/test 都从 labeled source validation 派生；official test 因无标签只用于泄漏检查，不报告 Oracle 分数
- M5 retrieval 使用 fact-ID metadata 精确过滤；报告的 Recall@k 是 Oracle-directed 多次 top-1 检索的累计诊断，不是标准单查询模型 Recall@k
- context tokens 是对每个 timestep observation 的 tokenizer-independent 累计估算，会包含重复上下文
- M5 完整 trajectory/reward JSONL 含原始事实正文与 embedding，只保存在 gitignored `runs/` 并按敏感数据处理
- M6 human-backed mock 是人工标注驱动的 Triple-extraction 上界，不是真实 LLM 或完整 AP pipeline 的模型表现
- M6 relevance 和 required coverage slots 使用独立的人工 Oracle semantic target；它们不进入 candidate cache，但该 benchmark 不是端到端 label-free AP 系统
- Triple F1 只在 34 个完整人工标注句子、37 个 gold triples 上计算；controlled drop/corrupt 是合成错误，不代表真实 LLM 错误分布
- FA/FR 是 30 个 rollout 的终局指标；reward error 则在 224 个 `action_id` 精确连接上计算，并汇总到 30 条 trajectory
- M6 规则/oracle/error-injector 轨迹没有 token IDs、token logprobs 或 policy version；迁移器保持 `None`
- M6 没有运行真实 LLM，没有实现 Group Critic、GRPO 或训练接入
- 全局 `git diff --check` 仍只报告用户更新的 `PROJECT_HANDOFF.md` 两处 Markdown 行尾空格；M6 范围文件检查通过

## Failures and blockers

- 无未解决的 M6 测试或 benchmark 失败
- 本地 HotpotQA fullwiki 已可用；无需模型、GPU、API 或网络

## Next recommended action

等待用户验收 M6；未经明确授权，不进入 Group Critic、GRPO 或模型训练。
