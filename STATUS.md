# Project Status

## Current milestone

E1：1.5B terminal-only 正式扩大（24 条 / 8 step，无 nudge，不进入 E3）

状态：目标模型限定为 1.5B 与 4B（暂不考虑 7B）。1.5B M8b GPU smoke 已通过。E1 GRPO 基线三 seed（同一 6 条、1 step、无 nudge）train reward 与 held-out F1 全 0。Stage 3 格式 probe（nudge）mean F1 ≈ 0.21、max 1.0，该干预不并入基线。下一步是 1.5B 正式扩大：24 条 train、8 trainer step、seed 7、无 `<answer>` nudge；4B 排在这次之后。部署目标为组内远程 GPU 服务器，持久根目录冻结为 `/data/hjx/Age_mem`

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

### M6 False Reject 收尾

- [x] 逐条审计 controlled-error 的 5 条 False Reject；全部由 5 个相关事实的 `drop_relevant_fact` 注入一一解释
- [x] 每条均记录 task/rollout、Oracle/Extracted AP、首个差异 `action_id`、缺失 Triple/StateFact、grounding 与 DFA 差异
- [x] v2 audit 对 M5/M6/manifest 的 digest 与 byte SHA、逐文件 hash/行数、完整动作坐标、StateFact/AP evidence 进行 fail-closed 交叉校验
- [x] Oracle 与 controlled-error 两条 DFA/数值奖励流共重放检查 74 个动作；全部 gate 字段与错误计数由证据派生
- [x] StateTracker、AP grounding、action 对齐及 DFA 实现错误计数均为 0；human-backed FA/FR 仍为 0/0
- [x] 保存 `artifacts/m6_extraction_benchmark/false_reject_audit.json/.md` 与 `docs/m6_false_reject_audit.md`
- [x] M6 收尾门禁通过；audit digest 为 `59a582d31396b548c0aa2c9dfc78cb5c93f6d6347a8e073d1ce0d5f291648032`

### M7

- [x] 保留 M4 手工 DFA 为主基线；实现严格 Group Critic 输入/输出、mock critic 与 injected-client LLM adapter
- [x] Critic 输入升级为每个 group 仅绑定当前一条 `critic_only_privileged` HotpotQA fullwiki 完整记录，包含 question、answer、完整 context 与官方 `supporting_facts`；该私有 Oracle 记录进入 prompt/cache digest，但不进入 policy observation
- [x] validator 校验完整 action 坐标、AP evidence、命题定义域、依赖 DAG、可达接受状态、非接受初态及 state cap
- [x] 将合法 milestone DAG 确定性编译为正向 DFA；bad behavior 只审计，不实现负自动机
- [x] 无效/不可用 Critic 输出显式回退到手工 DFA 或 terminal-only；25 个 cyclic-invalid + 5 个 unavailable，共 30 次显式回退、静默采用 0 次
- [x] 全失败组只保留 `reward_eligible=false` 的反事实建议，不编译为训练奖励
- [x] 新 replay adapter 仅消费 `TrajectoryStepV2 + ActionCreditRecord`，不重跑 extractor/StateTracker/grounder，不调用 LLM
- [x] 90/90 条 hand-DFA profile/rollout replay 与 M6 action reward 完全一致，覆盖 30 rollouts / 224 actions
- [x] Oracle 与 human-backed FA/FR 均为 `0/20`、`0/10`；controlled-error 保留已解释的 `0/20`、`5/10`
- [x] Critic + 显式回退管线与手工 DFA 的 90/90 个 profile/rollout 终局结果、3 x 224 个逐动作奖励观测一致；25 组采用 Critic DFA，5 组因缺受控 AP 显式回退
- [x] milestone evidence 451/451 有效；150 次重复与 180 次 K=3 顺序排列检查 100% 稳定
- [x] hand-DFA 上 10 个重复 ADD + 10 个两步检索循环场景无 reward farming，once-only 和 progress cap 均通过；不外推为 Critic-DFA farming 结论
- [x] 调用成本只报告 mock 输入/输出与启发式 token：cold/cache hit/miss 为 360/30/30；provider token/cost 为 `None`，真实 LLM 调用为 0
- [x] 按 HotpotQA question type、精确 action count 和唯一真实干扰配置（Stage 1=6、Stage 2=3）报告结果
- [x] 保存 `artifacts/m7_group_critic/` 与 `docs/m7_group_critic_offline_validation.md`；加入当前任务完整私有参考后的 report digest 为 `87e41a0ccee6dc0dc24dfa18c898853b6530a42efdde5badbd3d0013accbac75`
- [x] 未实现 GRPO、模型训练、负自动机、LTLf 或真实 LLM 评测

### M8a

> 状态说明：M8a 的本地静态/离线门禁实现已完成。组内远程 M8b GPU smoke 已在 `e82bf54` / `checkpoints-attempt-002` 通过。在线 `ActionCreditRecord` 自动生成仍未实现，属于 E3/E4，不能在 terminal-only 基线中伪记为完成。

- [x] `TaskFileReader` 支持本地 Hugging Face `save_to_disk` Dataset/DatasetDict，并对 split、subset 与 row index fail closed
- [x] E1 dry-run 固定复用 M5 manifest 的 6 条 source-train 样本；运行时校验 train fingerprint、source index 顺序与 Hotpot ID，真实 fullwiki 90,447→6 已核对
- [x] 新增 2-GPU、K=2、单 trainer step 的 `agemem_e1_dry_run.yaml`；完整 K 组保持在同一 WorkflowRunner policy-freeze 窗口
- [x] E1 使用确定性 HotpotQA answer F1，记录 EM/P/R/F1；reward breakdown 只含 terminal 与 total，DFA milestone 关闭
- [x] E2 原始 `0.5/0.2/0.15/0.15` heuristic dense reward 作为显式独立对照保留
- [x] E1 运行时强制 `multi_step_grpo + step_wise_grpo + K>=2`，错误 algorithm/advantage/group scheduling fail closed
- [x] Stage 2 固定干扰不调用 provider；E1 禁止 provider distractor source
- [x] Trinity memory workflow 复用 M2 rollout-scoped、版本化、soft-delete MemoryStore，并保留 snapshot/restore/history
- [x] 在线 LLM 工具动作生成稳定 `action_id`，保存完整 response token IDs、old logprobs、token/character span、tool trace result 与 policy version
- [x] ActionEvent 与 Experience task/rollout/stage/timestep/EID 精确对齐；最终 buffer 边界重算 character→token span 并重查 ToolTrace join；同一 task 混合 policy version 时拒绝
- [x] 规则/oracle/random/error-injector 轨迹禁止进入 on-policy buffer；AgeMem ExperiencePipeline 在 operator 前后均强制契约存在，删除或篡改契约时 fail closed
- [x] `AgeMem_code_agentscope*` 已纳入 Trinity wheel，Pydantic 作为显式依赖；非 repo cwd 的 wheel 内 ActionEvent/M2 store import smoke 通过
- [x] M8a 初始范围为 46 项测试：43 PASS、3 SKIP；当前已由 M8b 锁扩展为 142 项 `m8a` scope（本地 139 PASS、3 SKIP）
- [x] M1～M7 相关回归 145/145、tool-trace 30/30 均通过
- [x] 本阶段未调用真实 LLM/embedding/网络，未运行模型、GPU、优化器或 checkpoint
- [x] **原未关闭项 1：** 组内远程 Linux/GPU 服务器上的 runtime tests、E1 单次更新和 checkpoint 新进程重载已在 `e82bf54` / `checkpoints-attempt-002` 通过；这仍不是正式 E1 统计
- [ ] **未关闭项：** 在线 `ActionCreditRecord` 自动生成器尚未实现；当前只有严格 schema、join 和 buffer validation（对应 E3/E4）

### M8b 上卡前准备

- [x] `configs/m8b_autodl_preflight.json` 锁定三份 YAML 的规范 LF SHA-256、M5 manifest、37 项 E1 契约、`m8a=142/all=318` 精确测试数和 `experience_buffer.path=null`
- [x] 跨平台 preflight 核对完整 40 位 commit、dirty state、递归 `.env`/ignored credential、非空 key、持久路径、空 job、依赖版本与 Trinity Config schema
- [x] 模型门禁固定 `Qwen/Qwen2.5-1.5B-Instruct`、完整 revision、Qwen2.5-1.5B 结构、单文件权重、最小 3 GB 权重体积及逐文件 SHA-256 manifest，不伪造模型来源
- [x] 数据门禁核对 fullwiki 三个 split 的规模/fingerprint，以及 6 条 train + 2 条 held-out 的 Hotpot ID 和规范内容 hash
- [x] 远程 GPU 门禁支持四卡宿主机通过 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 与 `CUDA_VISIBLE_DEVICES` 唯一选择 2 张 RTX A6000；保留完整物理清单，仅对选中卡要求总显存至少 48,000 MiB、空闲显存至少 47,000 MiB，并交叉核对 `nvidia-smi` 与 PyTorch device UUID
- [x] 严格 runtime gate 对 suite 发现数和执行数同时比对 lock；任意 `FAIL/ERROR/SKIP/unexpected success` 均失败，不能把本地 3 个 runtime SKIP 当通过
- [x] 冻结 DashScope endpoint、`text-embedding-v4` 256 维和 `qwen-max`；provider SDK 禁用隐式重试，成功/失败/eval 调用立即写入独立 fsync 元数据 JSONL
- [x] provider 记录包含 task/rollout/execution/call index、延迟、错误类型与真实 token usage，不保存 prompt/response/header/key；异常文本脱敏，usage 持久化失败时 fail closed，未报告金额保持 `None`
- [x] launcher、trainer、explorer 与 WorkflowRunner 不再吞掉运行失败；有限步训练提前耗尽、NCCL 同步失败、rollout/eval 失败均向上传播且不生成伪成功凭据
- [x] 训练与 benchmark receipt 记录进程 execution ID；训练凭据包含有限 loss/KL/reward 和 actor-update sentinel，E0 固定 model version 0，checkpoint eval 固定 model version 1
- [x] postflight 校验 `trainer_meta.json`、`latest_checkpointed_iteration.txt==1`、完整非空 model/optimizer/extra shards、真实 LoRA 与 `dummy_lora` 不同，以及 checkpoint eval 来自不同进程
- [x] 新增固定 2 条 held-out 样本的 E0 base eval 与 E1 checkpoint eval 配置；三份 YAML 都让 Trinity 在独立 job 内持久化 audit buffer
- [x] 将 M8a `save_to_disk` 测试改为只读 marker fixture + 内存 DatasetDict，消除 Windows 沙箱临时目录 ACL 假失败
- [x] 新增 model-manifest、preflight、runtime-gate、postflight CLI，以及分阶段 fail-closed 的 `scripts/autodl_m8b_smoke.sh`
- [x] `pyproject.toml` 新增 `m8b` extra（AgentScope + Datasets）；完整安装、运行顺序、停止条件与 provider 对账口径记录于 `docs/m8b_autodl_preflight.md`
- [x] 组内远程服务器上的 E0、E1 单 optimizer update、`global_step_1`、新进程 checkpoint eval 已在 `e82bf54` / `checkpoints-attempt-002` 通过（postflight pass=10）

### E1 terminal-only 多 seed 重复

- [x] 新增独立 train/eval YAML：同一 6 条 M5 train ID、K=2、1 trainer step、`terminal_only`、milestone 关闭；job 名与 seed 由环境变量注入
- [x] seeds 固定为 `7/17/27`，job 名为 `agemem-e1-terminal-only-repeat-s{seed}`，不复用 smoke seed `20260802` 或 `agemem-e1-terminal-only-dry-run`
- [x] 锁 `configs/e1_repeat.json` 与启动器 `scripts/agemem_e1_repeat.sh`：拒绝含 smoke job 的 checkpoint 根、要求 `flash-attn==2.8.1`、每个 seed 训练后新进程 held-out eval
- [x] 契约测试不计入冻结 318 项 runtime gate；冻结 dry-run YAML digest 保持不变
- [x] 远程已在 `/data/hjx/Age_mem/checkpoints-e1-repeat` 完成三 seed；train reward 与 held-out F1 全 0
- [x] seed 7 轨迹确认 Stage 3 已执行；7 条 Stage 3 记录均无 `<answer>`，多数继续 `Retrieve_memory` 或闲聊

### Stage 3 `<answer>` probe

- [x] `stage3_require_final_answer` 默认关闭，不改冻结 dry-run YAML
- [x] 启用时只在 Stage 3 最后一轮追加“不要调用工具、必须写 `<answer>`”，不含 gold 答案
- [x] 探针配置 `agemem_e1_stage3_answer_probe.yaml`：同一 6 条 train、T=0、bench、不训练
- [x] 远程已在 `checkpoints-e1-answer-probe-003` / commit `f971af7` 跑完；最后一轮 12/12 写出 `<answer>`，repair 未触发；`task_score` mean ≈ 0.208、max 1.0。该开关不并入 GRPO 基线

### 1.5B terminal-only 正式扩大（无 nudge）

- [x] 协议锁 `configs/e1_scale.json`：24 条 train（保留 M5 的 6 条前缀 + 18 条新样本）、8 trainer step、seed 7、K=2、`terminal_only`
- [x] 明确关闭 `stage3_require_final_answer` / `stage3_repair_untagged_answer`；job 名 `agemem-e1-terminal-only-scale`；不改冻结 dry-run YAML
- [x] 远端已用冻结 HotpotQA 选出 18 条额外 train 行；`selection_status=frozen`，24 条 ID/index/hash 已写入 lock 与 YAML

### Stage 1/2 反捷径 benchmark

- [x] 新增复用 M2 `MemoryStore` 的 `TokenBudgetMemoryStore`：按 active-content token 而非条目数计费，单条整段写入不能绕过预算；ADD/UPDATE/restore 超预算 fail closed，UPDATE 按 token 差额计费，soft delete、snapshot/restore 与版本历史保持可审计
- [x] Stage 1 普通策略只接收 `PublicStage1Input`（不透明 handle + 已观察正文），不能读取 question/answer、内部 fact ID 或 supporting/distractor labels；仅内置 Oracle 类型走专用私有路径
- [x] Stage 1 在固定 `toy-train-005`、seed 7 和 15 lexical-token LTM budget 下比较 Store-All、Store-None 与 Oracle-Safe-Store
- [x] Store-All 的 supporting recall / memory precision 为 `0.5/0.5` 且有 1 次预算拒绝；Oracle-Safe-Store 为 `1.0/1.0`
- [x] Stage 2 使用 6 条固定 dev/test case，覆盖 hard negative、partial relevance 与 delayed relevance；公开输入不含 `task_id`、split、原始消息/segment ID、`future_query` / `future_answer` 字段、scenario 或 Oracle role；每个 seed 使用与角色无关的不透明句柄；supporting message 可包含未来答案事实，但没有查询可用于判断相关性
- [x] Stage 2 schema 明确 budget scope 为 `retained_segment_text_only`（句柄、格式和控制提示不计入 payload budget），并要求 future answer 出现在 supporting text 且不出现在 distractor text
- [x] Always-Keep 的 support/budget/safe 为 `1.0/0.0/0.0`；Always-Clear 为 `0.0/1.0/0.0`；`opaque_id_control` 为 `0.667/1.0/0.667`；Oracle-Safe-Compress 为 `1.0/1.0/1.0`
- [x] 统一报告 7/7 gate PASS；schema `agemem.anti_shortcut_benchmark.v2`，Stage 1/2 输入 digest 已绑定，checksum `b5ced8e688194d3d9e7cb3a6b4bd8d256d7cc38610fcb56a1d8c37987a7b952c`
- [x] E2 context preservation 显式使用真实 `target_question`，避免把 Stage 2 第一条干扰消息误当目标问题；保留旧调用兼容回退
- [x] 新增 26 项测试与 2 项 E2 对齐回归通过；未调用真实 LLM、外部 embedding 服务、网络、GPU 或训练
- [x] E1 `terminal_only` 配置和 M3～M7 既有轨迹/报告未改写；两个 Oracle 策略只作为私有标签离线上界，不是可部署策略
- [x] 保留 v2 7/7 CI canary 不变，另建 `agemem.anti_shortcut_stress.v1`：Stage 1 覆盖 16 个含噪任务 × 50 seeds × 3 个全局预算，每个策略 2400 arms，并加入顺序、长短、opaque-ID、random-hash 与 entity-chain 基线
- [x] Stage 2 新增 6 个 dev/test 反事实 pair（12 个 future variants）× 50 seeds；同一 pair 的两个 future 复用同一公开输入和同一 query-blind decision，互斥 support 各自可放入预算但并集不可放入
- [x] Stage 2 query-blind Oracle safe success 精确为 `0.5`，query-aware Oracle 为 `1.0`；公开策略最高 `0.372`，random-hash 为 `0.350`，未超过理论上界
- [x] stress 11/11 integrity gates PASS，lexical-token checksum `ae88dc9b64cfab0b1b705ef365e9fccbaa400924a3309346f40b1ff492e1b214`；9/9 新增测试 PASS，CLI 支持冻结本地 Qwen tokenizer/revision 重跑

## Environment

- OS: Windows NT 10.0.26200.0
- PowerShell: 5.1.26100.8875
- Python: 3.10.20 (`.venv`)
- AgentScope: 1.0.21
- Pydantic: 2.13.4
- Datasets: 4.8.5
- PyArrow: 25.0.0
- PyTorch / Ray / vLLM: 当前 `.venv` 未安装；锁定 318 项 suite 中 3 个 runtime 接线测试因此 SKIP
- Ruff: 0.15.9；pytest/Flake8 未安装；测试使用标准库 `unittest`

## Git state

- Current branch: `feat/m6-extracted-ap-state-tracker`
- M5 base: `6367569 feat(agemem): add HotpotQA Oracle benchmark`
- M6 schema audit commit: `d1d45ab feat(agemem): audit and migrate M5 action schema`
- M6 implementation commit: `1c8e5c1 feat(agemem): add extracted AP state benchmark`
- M7 commit：`dd31d4d feat(agemem): add M7 group critic offline validation`
- M8a commit：`a94c301 feat(agemem): add M8a terminal-only training gates`
- M8a handoff commit：`c1fb0d4 docs(agemem): hand off M8a AutoDL smoke`
- M8b implementation commit：`4389fd5 feat(agemem): add M8b AutoDL smoke gates`
- M8b 上卡前执行包已完成本地验证与本地提交
- Stage 1/2 反捷径 benchmark commit：`7d3c45d feat(agemem): add stage 1/2 anti-shortcut gates`
- Git scratch-directory cleanup commit：`c26ecb8 chore(git): ignore local verification scratch directories`
- 上一已推送基线：`e82bf54ba48cd6f5a101510b33fe9db498890f49`（M8b 远程 GPU smoke 通过）。本提交为 E1 terminal-only 多 seed 重复协议；远端用本提交完整 40 位哈希作为 `AGEMEM_EXPECTED_COMMIT`

## Files changed in M6/M7/M8a

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
- `AgeMem_code_agentscope/group_critic/`（M7 新增）
- `configs/m7_group_critic.json`（M7 新增）
- `artifacts/m7_group_critic/`（M7 新增）
- `docs/m6_false_reject_audit.md`（M6 收尾新增）
- `docs/m7_group_critic_offline_validation.md`（M7 新增）
- `tests/common/m6_false_reject_audit_test.py`（M6 收尾新增）
- `tests/common/m7_group_critic_*_test.py`（M7 新增）
- `AgeMem_code_agentscope/README.md`
- `examples/agemem_hotpotqa/agemem_e1_dry_run.yaml`（M8a 新增）
- `trinity/common/action_event_contract.py`（M8a 新增）
- `trinity/common/hf_task_dataset.py`（M8a 新增）
- `trinity/common/workflows/memory_reward/reward_profiles.py`（M8a 新增）
- `trinity/common/workflows/memory_context/distractors.py`（M8a 新增）
- `trinity/common/workflows/memory_context/memory_store.py`、`train_hotpotQA.py`（M8a 接线）
- `trinity/common/models/vllm_model.py`、`trinity/explorer/workflow_runner.py`、`trinity/buffer/pipelines/experience_pipeline.py`（M8a 接线）
- `trinity/common/config.py`、`trinity/algorithm/algorithm.py`、`trinity/buffer/reader/file_reader.py`（M8a 门禁）
- `tests/buffer/task_file_reader_dataset_dict_test.py` 与 `tests/common/m8*_test.py`（M8a 测试）
- `docs/m8a_terminal_only_preflight.md`、`examples/agemem_hotpotqa/README.md`、`PROJECT_HANDOFF.md`（M8a 文档）
- `STATUS.md`

## Files changed in M8b-prep

- 锁与依赖：`.gitattributes`、`configs/m8b_autodl_preflight.json`、`pyproject.toml`
- 三份执行配置：`agemem_e0_frozen_eval.yaml`、`agemem_e1_dry_run.yaml`、`agemem_e1_checkpoint_eval.yaml`
- 核心门禁：`trinity/common/m8b_model_manifest.py`、`m8b_preflight.py`、`m8b_postflight.py`、`runtime_receipt.py`
- provider/rollout 接线：`trinity/common/auxiliary_provider.py`、`memory_store.py`、`train_hotpotQA.py`、`workflow_runner.py`
- fail-closed runtime：`trinity/cli/launcher.py`、`trinity/explorer/explorer.py`、`trinity/trainer/trainer.py`、`verl_trainer.py`
- CLI/脚本：`scripts/agemem_m8b_{model_manifest,preflight,runtime_gate,postflight}.py`、`autodl_m8b_preflight.sh`、`autodl_m8b_smoke.sh`
- M8b 测试：`m8b_provider_usage_test.py`、`m8b_preflight_test.py`、`m8b_model_manifest_test.py`、`m8b_runtime_gate_test.py`、`m8b_runtime_fail_closed_test.py`、`m8b_postflight_test.py`
- fixture/既有测试：`tests/buffer/task_file_reader_dataset_dict_test.py`、`tests/fixtures/m8a_saved_dataset_dict/`
- 文档：`docs/m8b_autodl_preflight.md`、`docs/m8a_terminal_only_preflight.md`、HotpotQA `README.md`、`PROJECT_HANDOFF.md`、`STATUS.md`

## Files changed in E1 terminal-only repeats

- `examples/agemem_hotpotqa/agemem_e1_repeat.yaml`
- `examples/agemem_hotpotqa/agemem_e1_repeat_eval.yaml`
- `configs/e1_repeat.json`
- `scripts/agemem_e1_repeat.sh`
- `tests/common/e1_repeat_contract_test.py`
- `examples/agemem_hotpotqa/README.md`、`docs/m8b_autodl_preflight.md`、`PROJECT_HANDOFF.md`、`STATUS.md`

## Files changed in Stage 3 answer probe

- `trinity/common/workflows/memory_context/utils.py`
- `trinity/common/workflows/memory_context/workflow_prompt.py`
- `trinity/common/workflows/memory_context/train_hotpotQA.py`
- `examples/agemem_hotpotqa/agemem_e1_stage3_answer_probe.yaml`
- `configs/e1_stage3_answer_probe.json`
- `scripts/agemem_e1_stage3_answer_probe.sh`
- `tests/common/e1_stage3_answer_probe_test.py`

## Files changed in 1.5B E1 scale

- `configs/e1_scale.json`
- `trinity/common/e1_scale.py`
- `scripts/agemem_e1_scale_select.py`
- `scripts/agemem_e1_scale.sh`
- `tests/common/e1_scale_contract_test.py`
- `examples/agemem_hotpotqa/agemem_e1_scale.yaml`
- `examples/agemem_hotpotqa/agemem_e1_scale_eval.yaml`
- `examples/agemem_hotpotqa/README.md`、`docs/m8b_autodl_preflight.md`、`STATUS.md`

## Files changed in Stage 1/2 anti-shortcut extension

- Stage 1：`AgeMem_code_agentscope/toy_hotpotqa/storage_baselines.py`、`tests/common/stage1_storage_baseline_test.py`
- Stage 2：`AgeMem_code_agentscope/toy_hotpotqa/stage2_challenge.py`、源码 fixture `data/toy/stage2_context_challenges.json`、wheel package-data 副本 `AgeMem_code_agentscope/toy_hotpotqa/data/`、`tests/common/stage2_context_challenge_test.py`
- 统一报告：`shortcut_benchmark.py`、`anti_shortcut_benchmark_test.py`、`artifacts/anti_shortcut_benchmark/`、`docs/anti_shortcut_benchmark.md`
- E2 对齐：`my_reward.py`、train/eval workflow 与 `tool_trace_test.py`
- 冻结门禁：`scripts/agemem_m8b_runtime_gate.py`、`configs/m8b_autodl_preflight.json`、`m8b_runtime_gate_test.py`
- stress 扩展：`shortcut_stress.py`、`scripts/agemem_anti_shortcut_stress.py`、`configs/anti_shortcut_stress.json`、成对反事实 fixture、`anti_shortcut_stress_test.py`、`artifacts/anti_shortcut_stress/` 与 `docs/anti_shortcut_stress.md`

## Presentation artifact

- `docs/project_presentation_materials.md`：15 页 PPT 素材、页级结论、图表建议、证据来源和后续 PPT 生成提示词；不包含 `.pptx`，不代表真实训练结果

## Verification

- M5 adapter/benchmark/local-data tests：10/10 PASS
- M1～M5 core regression：60/60 PASS
- Existing tool-trace regression（含 2 项 E2 显式问题对齐回归）：30/30 PASS
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
- M6 False Reject audit + 原 M6 scoped regression：47/47 PASS
- M6 False Reject audit gate：5/5 可解释，74 个 DFA 动作重放，实现错误 0，PASS
- M6 closeout + M7 combined scoped regression：85/85 PASS；M1～M7 相关回归：145/145 PASS
- M7 schema/validator/compiler/replay/benchmark tests：38/38 PASS
- M7 hand-DFA deterministic replay：90/90 exact，224 actions
- M7 Critic fallback：25 cyclic-invalid + 5 unavailable = 30/30 explicit；silent adoption 0
- M7 evidence integrity：451/451；repeat/permutation stability：150/150、180/180
- M7 hand-DFA reward farming：20/20 PASS（10 duplicate ADD + 10 two-step RETRIEVE loops）
- M7 report schema/digest/source-immutability validation：PASS
- M7 real LLM calls：0；provider token/cost：`None`
- M8a 初始 tests：46 discovered，43 PASS，3 SKIP（历史口径；已被下述 M8b lock scope 取代）
- M8a data/reward/distractor/memory/action/packaging 可执行项：43/43 PASS
- M8a 真实 fullwiki fixed split：90,447→6，顺序与 M5 Hotpot IDs 一致
- M8a wheel build：PASS；非 repo cwd 的 ActionEvent/M2 store import：PASS
- M1～M7 + tool-trace：145/145 + 30/30 = 175/175 PASS
- M8a real LLM / embedding / network / GPU / optimizer calls：0
- M8a Ruff check、py_compile、YAML parse、setuptools package discovery：PASS
- Stage 1 budget/store baselines：13/13 PASS；Stage 2 challenge：9/9 PASS；统一报告：4/4 PASS；合计 26/26 PASS
- E2 显式 target-question 对齐回归：2/2 PASS
- 反捷径统一报告：7/7 gates PASS，real LLM / external embedding service / network calls 0，schema v2 checksum `b5ced8e688194d3d9e7cb3a6b4bd8d256d7cc38610fcb56a1d8c37987a7b952c`
- 反捷径 stress：9/9 tests、11/11 integrity gates PASS；Stage 1 每策略 2400 arms，Stage 2 每策略 600 arms，checksum `ae88dc9b64cfab0b1b705ef365e9fccbaa400924a3309346f40b1ff492e1b214`
- M8b lock 精确测试数：`m8a=142`、`all=318`；suite discovery 与 lock 完全一致
- 本次未重跑完整 318 项 suite；新增 Qwen token-count snapshot 已 PASS，因此本地 all scope 推断为 318 RUN、315 PASS、3 SKIP，0 FAIL/ERROR/unexpected success
- M8b 当前 m8a scope 组成：原 M8a 46 项 + M8b 新增 61 项 + 反捷径 canary 26 项 + stress 9 项 = 142；本地 139 PASS、3 SKIP
- M8b 新增 61 项：provider 16、preflight 16、model manifest 2、runtime gate 5、runtime fail-closed 12、postflight 10，全部 PASS
- M8b strict runtime gate 本地结果必须为 FAIL：仅有的 3 个 SKIP 都因缺 Ray，且门禁按设计不放行 SKIP
- 1.5B 锁干净提交后的本地 preflight（`--no-write`）：18 PASS、0 FAIL、2 WARN、11 SKIP；两个 WARN 为未注入远程 key 与本地 ignored 凭据文件，11 个 SKIP 包含尚未配置本地 1.5B 模型路径以及只能在远程 Linux/GPU 环境验证的 GPU/runtime 项
- M8b 本地真实 fullwiki：90,447/7,405/7,405、三个 fingerprint、6 条 train + 2 条 held-out 的 ID/内容 hash 精确一致
- M8b 三份 1.5B YAML 规范 LF digest、37 项 E1 assertion、`experience_buffer.path=null` 与 lock 一致
- M8b real LLM / embedding / network / GPU / optimizer/checkpoint calls：0
- M8b Ruff、compileall、YAML/JSON parse 与 scoped diff check：PASS
- M7 report digest：`87e41a0ccee6dc0dc24dfa18c898853b6530a42efdde5badbd3d0013accbac75`

## Known constraints

- M3 使用人工任务和规则 gold/error policy，不代表真实 HotpotQA 模型表现
- M3 Oracle labels 是结构化监督事件，不是 M4 AP，也没有产生逻辑奖励
- DELETE 不出现在公共 M3 动作空间；仅由错误策略调用现有 M2 soft delete 模拟关键记忆误删
- Stage 3 使用确定性 fact-ID metadata filter，这是 gold/error policy 的 Oracle 行为
- 完整 JSONL 含事实正文和 embedding，必须按敏感数据处理
- M4 使用 Oracle AP 上界，不代表后续自然语言 Extracted AP 的准确率
- 手工正向 DFA 仍是主基线；M7 自动 Critic 仅为 deterministic mock 离线 shadow，尚无真实 LLM 评测或 LTLf 编译
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
- M6 没有运行真实 LLM；M7 新增 Group Critic 离线层，但没有运行真实 LLM，也没有实现 GRPO 或训练接入
- M7 Critic 结果来自固定 K=3 的 Oracle/error-policy smoke 轨迹，不代表真实模型 rollout 上的 Critic 表现
- M7 controlled-error 的 5 条 FR 是预期合成鲁棒性结果，已定位但没有被“修成 0”
- M7 只测量一个真实干扰配置（Stage 1=6、Stage 2=3），没有据此声称跨干扰强度泛化
- M8b 只完成本地静态/离线门禁和模拟 artifact 的 postflight 单测；真实 1.5B GPU smoke 已在远程 `e82bf54` / `checkpoints-attempt-002` 通过，但仍只是单次更新链路，不是正式 E1 统计
- E1 terminal reward 与固定 distractor 不调用辅助 LLM，但 memory embedding 仍访问已冻结的 DashScope，SUMMARY/CLEAR 仍可能调用已冻结的 `qwen-max`
- provider usage 记录不含请求/响应正文；OpenAI-compatible API 不报告货币金额时为 `None`，必须另与 DashScope 账单对账
- 锁定 suite 的 3 个 WorkflowRunner/ExperiencePipeline runtime tests 已在远程 smoke 变为 PASS；Windows 本地仍因缺 Ray 而 SKIP
- `agemem_e1_dry_run.yaml` 仅含固定 6 条数据、K=2 和 1 个 trainer step，不代表 E1 统计复现或正式结果
- E1 重复协议仍是同一 6 条样本、1 step、terminal-only；三个 seed 不能被说成 DFA、Extracted AP 或 E3
- v2 CI canary 的 Stage 1 结果仍只来自 1 个固定 toy task/seed；stress 已扩展到 16 个含噪任务、50 seeds 和三个全局预算，但当前规范 artifact 仍使用 `unicode-lexical-v1`，上卡后必须用同一冻结 Qwen tokenizer/revision 重跑
- v2 Stage 2 仍是 6 条独立合成 case；stress 另有 6 个成对反事实 context/12 future variants，并严格给出 query-blind `0.5` 上界，但这些仍是合成模板，不代表真实 HotpotQA 分布
- 当前 canary/stress 证明固定公开基线受预算和反事实边界约束，不证明已训练模型学会未来相关性、真实 HotpotQA 泛化、DFA 优于 terminal-only，或端到端 Answer EM/F1 已提升

## Failures and blockers

- 无未解决的本地可执行测试失败；318 项中仍有 3 个只能在完整 Linux runtime 关闭的 SKIP，因此 Windows 上严格 runtime gate 按设计为 FAIL
- M8b 远程 smoke 已通过；E1 三 seed 基线 terminal F1 为 0。格式 probe 已证明 nudge 后可评分，但不并入基线。24 条 E1 scale 行已冻结。当前阻塞是在空 checkpoint 根目录 `/data/hjx/Age_mem/checkpoints-e1-scale` 跑 8 step 训练。运行前重核远程空闲 GPU。不要进 E3，也不要跳到 4B
- DashScope provider 已冻结；货币成本仍须与 provider 账单对账，不得把 E1 声称为端到端无外部模型
- 当前 Windows 环境不能验证完整 Config/Ray/vLLM/veRL 或真实 GPU 重复运行
- 在线 `ActionCreditRecord` 当前只有 schema、精确 join 和 buffer validation；E3/E4 的 AP/DFA reward operator 尚未实现
- E5 的 DFA-state bucket、RTG、action-token mask 与动作级 loss 尚未实现

## Next recommended action

E1 三 seed 基线已在 `/data/hjx/Age_mem/checkpoints-e1-repeat` 完成（无 `<answer>`，reward 全 0）。格式 probe 已完成，nudge 不并入基线。24 条 train 已冻结。不要改冻结 dry-run YAML，不要进 E3，不要跳到 4B。下一步：空目录 `/data/hjx/Age_mem/checkpoints-e1-scale` 跑 `bash scripts/agemem_e1_scale.sh`。运行前重核 `nvidia-smi`。
