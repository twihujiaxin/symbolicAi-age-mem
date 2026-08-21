# AgeMem 神经符号记忆项目：PPT 素材文档

> 用途：供后续会话生成项目汇报、开题答辩或阶段性进展 PPT。<br>
> 建议版本：15 分钟、15 页；可压缩为 10 页。<br>
> 更新时间：2026-08-19<br>
> 当前代码分支：`feat/m6-extracted-ap-state-tracker`<br>
> 当前阶段：M8b-prep；真实 AutoDL E0/E1/checkpoint 尚未执行。

## 1. 汇报定位

### 建议标题

**Logic-Guided Agent Memory：面向长程记忆操作的可重放轨迹与 DFA 里程碑奖励**

### 副标题

从轨迹记录、rollout 隔离到 Oracle AP/DFA 离线验证：AgeMem 风格统一智能体的阶段性实现

### 一句话主线

本项目把 Agent 的记忆操作转化为可审计的动作、状态和自动机转移，先用确定性离线链路验证奖励正确性，再进入真实模型训练；当前已完成 M0-M7 与 M8b 上卡前准备，但尚未宣称 GPU 训练结果。

### 目标听众

- 研究导师、强化学习/Agent 方向评审：关注研究问题、对照设计和证据边界；
- 工程评审：关注 `action_id` 对齐、轨迹重放、MemoryStore 隔离、fail-closed 门禁；
- 后续接手开发者：关注文件入口、测试、未完成项和执行顺序。

### 汇报口径

- “已完成”指代码、数据适配、离线报告或门禁已经完成，不等于真实 GPU 训练成功；
- M3-M7 主要是 Toy/Oracle/mock/controlled-error 验证，不代表真实 LLM 的最终效果；
- M8a 的 terminal-only baseline 保持 DFA milestone 关闭，避免把后续 E3/E4 功能混入首轮训练对照；
- 所有真实 LLM、embedding、GPU、optimizer、checkpoint 调用数量，当前均为 0。

## 2. 核心叙事

```text
记忆动作具有延迟、稀疏、长程信用分配问题
                 ↓
记录每个动作及 memory before/after，保证可重放
                 ↓
从事实/状态生成 AP，使用手工 DFA 产生 once-only milestone reward
                 ↓
在 Toy 与真实 HotpotQA smoke 上验证数据、奖励和错误传播
                 ↓
用 mock Group Critic 做离线 shadow 与显式回退验证
                 ↓
完成 AutoDL 上卡前门禁，再比较 terminal-only 与 terminal+DFA 训练
```

### 研究问题

> 在相同统一 Agent、记忆工具、数据和终局任务奖励下，DFA 产生的里程碑奖励，能否比纯终局奖励更容易训练出有效的 `ADD / RETRIEVE / UPDATE` 行为？

### 研究假设

1. 终局奖励无法区分轨迹中哪些记忆动作带来真实进展；
2. 显式 AP 和 DFA 状态转移能提供更稳定、可解释、可重放的过程监督；
3. once-only milestone reward 可以降低重复 ADD、无效检索和循环动作带来的 reward farming；
4. 在进入训练前，必须先证明 AP grounding、`action_id` 对齐和 DFA replay 没有实现错误。

## 3. 推荐页级结构

### Slide 1｜标题页：Logic-Guided Agent Memory

**页面结论**
这是一个把 AgeMem 记忆工具与 GLARE 风格逻辑奖励结合起来的阶段性研究工程。

**页面内容**

- 主标题：Logic-Guided Agent Memory；
- 副标题：可重放轨迹、Oracle AP 与 DFA 里程碑奖励；
- 页脚：M0-M7 已完成，M8b-prep 已完成，真实 AutoDL smoke 待执行；
- 作者、日期、实验分支。

**建议图示**

用一条横向流程线展示：`Memory Tools → Trajectory → AP → DFA Reward → GRPO`。当前进度用高亮到 `Preflight`，不要把训练阶段画成已完成。

**讲解要点**

本项目的重点不是增加一个奖励项，而是先建立一条可解释、可审计、可确定性重放的奖励链路。

**来源**

`PROJECT_HANDOFF.md`、`STATUS.md`。

---

### Slide 2｜为什么需要记忆操作级奖励？

**页面结论**
记忆操作的收益通常在多个阶段之后才体现，纯终局奖励会把信用平均或错误地传播到整条轨迹。

**页面内容**

```text
Stage 1：候选事实 → ADD / UPDATE
Stage 2：查询尚未公开 → SUMMARY / CLEAR（或离线 KEEP / CLEAR / COMPRESS）
Stage 3：问题公开 → RETRIEVE / ANSWER
                              ↑
                      终局奖励在这里才出现
```

- 早期 ADD 是否保存了真正的 supporting fact？
- Stage 2 是否在预算内保留未来有用信息？
- Stage 3 RETRIEVE 是否覆盖了正确证据？
- UPDATE 是否替换了过期版本，而不是制造重复记忆？
- 重复调用和循环动作是否能“刷奖励”？

**建议图示**

左侧画三阶段时间轴，右侧对比两条奖励曲线：`Terminal-only` 只有最后一步有信号；`Milestone` 在存储、检索覆盖和正确回答处产生一次性进展信号。

**讲解要点**

这里的研究对象是信用分配，不是单纯提高工具调用次数。任何过程奖励都必须能追溯到动作和状态证据。

**来源**

`PROJECT_HANDOFF.md` 研究问题；M3/M4 设计。

---

### Slide 3｜研究范围与阶段边界

**页面结论**
项目采用“先验证奖励链路，再上卡训练”的分阶段策略，避免多个不确定因素同时进入实验。

**MVP 保留**

- 单个统一智能体；
- `ADD / RETRIEVE / UPDATE` 记忆工具；
- 三阶段 Toy Memory Environment 与 HotpotQA smoke；
- 正向手工 DFA、Terminal Reward、Milestone Reward；
- JSONL 轨迹、离线 replay、Oracle AP、严格测试。

**暂不纳入首轮主实验**

- `SUMMARY / CLEAR / DELETE` 的在线 DFA 奖励；
- Negative Automata、反事实推理、动态图数据库；
- 真实 Group Critic、GRPO 全量训练；
- 开放式 LTLf 自动生成。

Stage 1/2 sidecar 中的 KEEP/CLEAR/COMPRESS 是固定离线诊断策略；在线
SUMMARY/CLEAR 已存在于 E1 工具与 Experience 路径，可能调用冻结的 `qwen-max`，
但其 AP/DFA 奖励尚未接入主实验。

**建议图示**

用“已完成 / 当前准备 / 后续”三列路线图。红色或灰色标注明确未做功能，不要把它们画成现有结果。

**讲解要点**

范围收缩是实验控制：首轮只回答一个可证伪问题，减少抽取器、Critic、数据库和训练规模的混杂。

**来源**

`PROJECT_HANDOFF.md` 第 3 节、`STATUS.md` M8a/M8b。

---

### Slide 4｜系统架构：统一 Agent 到逻辑奖励

**页面结论**
系统把自然语言 Agent 行为拆成六个可审计层，每层都有独立契约和失败边界。

**架构层次**

```text
统一 Agent / AgentScope
        ↓ tool call + response token metadata
TrajectoryRecorder / ActionEvent
        ↓ memory before/after
Rollout-scoped MemoryStore
        ↓ validated Triple / StateFact / evidence
StateTracker + AP Grounding
        ↓ propositions AP_t
Hand-authored positive DFA
        ↓ once-only edge transitions
RewardBreakdown / ActionCreditRecord
        ↓ offline replay or future GRPO buffer
```

**关键工程约束**

- `action_id` 是动作主键，不能用易变的裸 timestep 代替；
- 每个 rollout 拥有独立 MemoryStore；
- 规则/oracle/error-injector 轨迹禁止进入 on-policy buffer；
- 契约缺失、join 不唯一、policy version 混合或写盘失败时 fail closed。

**建议图示**

画一张从左到右的数据流图，给每层标注输入、输出和验证器；用锁形图标表示 fail-closed 边界。

**来源**

`trinity/common/action_event_contract.py`、`trinity/common/workflows/memory_context/memory_store.py`、`docs/schema_audit_m6.md`。

---

### Slide 5｜M1-M2：可重放轨迹与 rollout 隔离

**页面结论**
在讨论奖励之前，先确保每次 Agent 行为可持久化、可查询、可确定性重放，且并行 rollout 不共享状态。

**M1 轨迹记录**

- `TrajectoryStep` / JSONL schema validation；
- 保存 observation、action、tool result、environment reward；
- 保存 memory before/after；
- 支持 `task_id / rollout_id / timestep` 查询；
- `TrajectoryReplay` 不调用 LLM 或 embedding。

**M2 MemoryStore**

- `MemoryStore` protocol 与 AgentScope adapter；
- `InMemoryStore` 与 rollout registry；
- `add / retrieve / update / delete / snapshot / restore / reset`；
- update 保留版本历史；research mode delete 使用 soft delete。

**建议图示**

上半部画一条 JSONL 轨迹样例：`observation → action → tool_result → memory_before/after → reward`；下半部画两个隔离 rollout 的 store，证明状态不相互可见。

**讲解要点**

如果轨迹不能重放，后续的 AP、DFA 和 reward 差异无法定位；如果 rollout 共享记忆，训练信号会被污染。

**来源**

`docs/reproduction.md`、M1/M2 测试与对应 commit：`f5ceffd`、`02536bb`。

---

### Slide 6｜M3：三阶段 Toy Memory Environment

**页面结论**
先用可控环境覆盖“正确、失败、重复、循环、过期事实”等行为，再接入真实 HotpotQA 数据。

**环境设计**

- 30 条人工两跳事实任务：20 train / 5 dev / 5 test；
- Agent 可见 observation 不包含 answer、fact ID 或 Oracle labels；
- Stage 1：清空 STM，保留 LTM，处理初始事实；
- Stage 2：加入固定干扰上下文；
- Stage 3：追加问题，检索 supporting memory 并回答；
- `GoldMemoryPolicy` 与 5 类显式 `ErrorMemoryPolicy`。

**覆盖场景**

- distractor；
- duplicate ADD；
- fact UPDATE；
- stale/expired retrieval；
- supporting memory 被误删；
- 循环动作与 reward farming。

**结果**

- gold policy：30/30 episode 成功；
- 相同 task/rollout/seed：JSONL 字节级一致，replay digest 一致；
- 未调用真实 LLM、embedding 或网络。

**建议图示**

用三栏 Stage 1/2/3 卡片，下面放一条 memory 版本线：`v1 → v2(update) → active/stale`。

**来源**

`STATUS.md` M3、Toy environment tests。

---

### Slide 7｜Stage 1/2 反捷径压力测试

**页面结论**

固定容量与查询延迟挑战能排除 Store-All、Always-Keep、Always-Clear 和当前 min-ID control，但只证明 benchmark 对这些固定策略有区分力，不代表模型已学会泛化相关性。

**Stage 1：LTM 存储预算**

固定 `toy-train-005`、seed 7、15 个 `unicode-lexical-v1` active-content tokens：

| Policy | Support recall | Memory precision | Budget rejects |
|---|---:|---:|---:|
| Store-All | 0.500 | 0.500 | 1 |
| Store-None | 0.000 | 0.000 | 0 |
| Oracle-Safe-Store | 1.000 | 1.000 | 0 |

**Stage 2：query-delayed context challenge**

6 条 dev/test case 覆盖 hard negative、partial relevance、delayed relevance；公开输入不含 `task_id`、split、原始消息/segment ID、`future_query` / `future_answer` 字段、scenario 或 Oracle role，每个 seed 使用与角色无关的不透明句柄。Supporting message 可以包含未来答案事实，但当时没有查询可用于判断相关性。

| Policy | Support | Budget | Safe success |
|---|---:|---:|---:|
| Always-Keep | 1.000 | 0.000 | 0.000 |
| Always-Clear | 0.000 | 1.000 | 0.000 |
| Opaque-ID control | 0.667 | 1.000 | 0.667 |
| Oracle-Safe-Compress | 1.000 | 1.000 | 1.000 |

**证据边界**

- 两个 Oracle 策略使用私有 labels，只是不可部署的离线上界；
- 7/7 gates PASS，真实 LLM / external embedding service / network calls 为 0；报告 schema v2 checksum 为 `b5ced8e688194d3d9e7cb3a6b4bd8d256d7cc38610fcb56a1d8c37987a7b952c`；
- Stage 2 budget scope 为 `retained_segment_text_only`，句柄、格式和控制提示不计入 payload budget；
- E1 `terminal_only` 配置与 M3-M7 artifacts 未改写；
- 当前结果不证明真实 LLM、真实 HotpotQA 泛化或 DFA 优于 terminal-only。

**建议图示**

左右放两张策略对照图：左图显示 Store-All 在固定 LTM budget 下丢 support，右图显示 Always-Keep、Always-Clear 与 min-ID control 均未达到 Oracle。Oracle 柱使用虚线并标“privileged upper bound”。

**来源**

`docs/anti_shortcut_benchmark.md`，checksum：`b5ced8e688194d3d9e7cb3a6b4bd8d256d7cc38610fcb56a1d8c37987a7b952c`。

---

### Slide 8｜M4：Oracle AP 与手工正向 DFA

**页面结论**
M4 证明了“轨迹 → AP → DFA → 逐步 reward”这条离线链路能够确定性重放，并能抵抗重复和循环刷奖励。

**逻辑链路**

```text
M3 oracle_labels
      ↓ semantic AP（不读取裸工具名）
AP grounder
      ↓ task / rollout / stage / seed / fact-ID 校验
Hand DFA: q0 → q1 → q2/q3 → q4
      ↓ edge_id once-only
RewardBreakdown
```

**关键机制**

- 9 类语义 AP；
- q0→q4 正向 progress chain，并行 update edge、reject、timeout；
- milestone edge 以 `edge_id` 去重；
- irrelevant store/retrieve 记为 violation，但 M4 暂不启用负奖励；
- supporting memory 删除进入 rejecting state。

**结果**

- 30/30 gold traces 接受；
- 四类预定义 failure traces 全部拒绝；
- 重复 ADD/RETRIEVE/UPDATE、循环和 reward farming 测试通过；
- reward JSONL 与 replay digest 重复运行完全一致。

**建议图示**

画 DFA 状态图，突出 `q0→q1→q3→q4` 的进展边、`q_reject` 和 `q_timeout`；旁边用小表展示重复 edge 不再奖励。

**来源**

M4 实现、`STATUS.md` M4、`docs/m5_hotpotqa_oracle_benchmark.md`。

---

### Slide 9｜M5：真实 HotpotQA 数据适配与 Oracle Benchmark

**页面结论**
在不调用 LLM 的条件下，数据适配、supporting facts、答案不可见性和 Oracle reward 链路已经在真实 fullwiki 上闭环。

**数据证据**

| 项目 | 数值 |
|---|---:|
| source train | 90,447 |
| source validation | 7,405 |
| official test | 7,405 |
| smoke train / dev / held-out | 6 / 2 / 2 |
| 规范轨迹 | 30（10 个任务 × 3 策略） |
| 真实 LLM 调用 | 0 |

**数据安全与正确性**

- supporting fact 按精确 `(title, sent_id)` 解析；
- 句子正文生成稳定 SHA-256 fact ID，并保留 source pointer；
- official test 7,405 条通过 label-blind 检查；
- public StageInput 不包含 answer、supporting IDs 或 Oracle labels；
- gold / wrong-answer / missing-support 三种确定性策略。

**结果**

- gold：10/10 成功、DFA 接受；
- failure controls：20/20 失败、DFA 拒绝；
- 报告包含 Answer EM/F1、support coverage、memory precision、retrieval recall@k、context tokens、tool calls。

**建议图示**

左侧画数据 split 与 smoke 子集抽取，右侧画三种策略的 10/10、0/10、0/10 success 对比柱状图。注明这是 Oracle/确定性策略，不是 base model 成绩。

**来源**

`docs/m5_hotpotqa_oracle_benchmark.md`、`data/splits/hotpotqa_smoke_manifest.json`。

---

### Slide 10｜M6：从 Triple 到 StateTracker 与 Extracted AP

**页面结论**
M6 把 Oracle AP 之外的抽取链路显式化，并保证每个派生 AP 都能追溯到原始 `action_id`。

**新增组件**

- 严格 Triple/AP schema；
- evidence span/digest 与坏 evidence quarantine；
- deterministic mock extractor 与 fake-client adapter；
- rollout-scoped StateTracker：single-valued 半开区间覆盖，multi-valued 保留多值；
- AP grounding 只使用公共 tool result、memory delta、validated Triple、StateFact；
- group cache 只缓存 action-independent candidates，materialize 时重新绑定 `action_id`。

**规模与指标**

| Profile | Triple F1 | AP F1 | False Accept | False Reject | Reward MAE |
|---|---:|---:|---:|---:|---:|
| human-backed mock | 1.000 | 0.976 | 0/20 | 0/10 | 0.0000 |
| controlled error | 0.870 | 0.837 | 0/20 | 5/10 | 0.0569 |

**重要解释**

- human-backed mock 是人工标注驱动的抽取上界，不是真实 LLM 表现；
- controlled-error 的 5 条 FR 是人为 `drop_relevant_fact` 注入，不应被“修成 0”；
- M6 benchmark 真实 LLM 调用数为 0。

**建议图示**

画 `sentence → Triple → StateFact → AP → DFA` 的证据链，所有箭头汇聚到 `action_id`；右侧放两行指标表，不要使用“模型准确率”字样。

**来源**

`docs/m6_extraction_benchmark.md`、`docs/schema_audit_m6.md`。

---

### Slide 11｜M6 收尾：错误如何传播？

**页面结论**
5 条 False Reject 全部可解释为预期抽取器漏抽，未发现 StateTracker、AP grounding、action alignment 或 DFA 实现错误。

**统一因果链**

```text
drop_relevant_fact（人工注入）
        ↓
相关 Triple 缺失
        ↓
对应 StateFact / supporting AP 缺失
        ↓
coverage fail-closed
        ↓
DFA 未从 q1 进入 q3/q4
        ↓
终局正确答案仍被拒绝，产生可解释 reward error
```

**审计证据**

- 5/5 FR 有 task、rollout、Oracle/Extracted AP、首次差异 action、缺失 Triple、StateTracker、grounding、DFA 转移和注入类型；
- DFA/reward action checks：74；
- 实现错误计数：0；
- human-backed FA/FR：0/20、0/10；
- M7 entry gate：PASS。

**建议图示**

做一张“注入错误 → AP 丢失 → DFA 转移 → reward 差异”的 Sankey/因果链图；右下角放绿色 `implementation error = 0`，橙色 `expected extractor omission = 5`。

**来源**

`docs/m6_false_reject_audit.md`，audit digest：`59a582d31396b548c0aa2c9dfc78cb5c93f6d6347a8e073d1ce0d5f291648032`。

---

### Slide 12｜M7：Group Critic 与手工 DFA 的离线验证

**页面结论**
在不调用真实 LLM 的前提下，Critic 只作为 deterministic mock shadow；无效输出显式回退到手工 DFA，且 replay 结果保持一致。

**验证结果**

| 检查项 | 结果 |
|---|---:|
| hand-DFA profile/rollout replay | 90/90 exact |
| milestone evidence integrity | 451/451 |
| repeat stability | 150/150 |
| K=3 permutation stability | 180/180 |
| invalid/unavailable critic explicit fallback | 30/30 |
| silent critic adoption | 0 |
| reward-farming scenarios | 20/20 |
| real LLM calls | 0 |

**错误归因**

- Oracle、human-backed mock：FA/FR 均为 0；
- controlled-error：保留已解释的 5/10 FR；
- Critic+fallback 与 hand-DFA 逐动作奖励 agreement：1.000。

**建议图示**

画三路决策：`valid critic → compile DFA`、`invalid/unavailable → hand DFA fallback`、`unsupported counterfactual → reward_eligible=false`。旁边放 90/90 和 30/30 指标。

**讲解要点**

M7 验证的是离线管线的稳定性与回退安全，不是证明真实 LLM Critic 已经有效。

**来源**

`docs/m7_group_critic_offline_validation.md`。

---

### Slide 13｜M8a：Terminal-only 训练基线契约

**页面结论**
M8a 将首轮 GPU smoke 收缩为可控的 terminal-only baseline，确保奖励对照不被 DFA/AP 接线污染。

**固定配置**

- 单统一 Agent；
- 2 GPU：1 rollout + 1 trainer；
- 6 条固定 train 样本；
- K=2 grouped rollouts；
- `multi_step_grpo + step_wise_grpo`；
- 1 个 trainer step；
- reward breakdown 仅有 terminal 与 total；
- Stage 2 使用固定 distractor，不调用 provider distractor；这不代表 SUMMARY/CLEAR 或 memory embedding 没有外部调用；
- milestone/DFA reward 关闭。

**本地证据**

- M8a 原始历史 subset：46 项，43 PASS、3 SKIP；
- M8b 锁定后 m8a scope：133 项，130 PASS、3 个缺 Ray 的环境性 SKIP；
- M1-M7 相关回归：145/145；tool-trace：30/30；
- 本阶段真实 LLM/embedding/network/GPU/optimizer/checkpoint：0。

**两项未关闭**

1. AutoDL Linux runtime 与 E1/checkpoint 真实执行；
2. 在线 `ActionCreditRecord` 生成器（留到 E3/E4）。

**建议图示**

用“baseline contract”卡片展示固定样本、K、step、reward components；旁边用两个未完成标签明确区分“外部执行”和“后续功能”。

**来源**

`docs/m8a_terminal_only_preflight.md`、`STATUS.md`。

---

### Slide 14｜M8b-prep：上卡前证据门禁

**页面结论**
M8b-prep 已把租卡前风险转化为版本锁、数据/模型 provenance、运行时 fail-closed 和 postflight 证据，但尚未运行真实 GPU。

**门禁内容**

- 三份 YAML canonical-LF SHA-256 与 M5 manifest digest；
- 固定完整 Git commit、模型 40 位 revision、逐文件模型 SHA-256 manifest；
- fullwiki 三 split fingerprint、6 train + 2 held-out 行内容 hash；
- 依赖版本与 Trinity structured Config；
- 恰好 2 张 GPU、每卡总显存 ≥76,000 MiB、空闲显存 ≥74,000 MiB、UUID 对齐；
- provider metadata-only JSONL：`task_id / rollout_id / execution_id / call_index`，立即写入并 `fsync`；
- receipt：model version、process execution ID、有限 loss/KL/reward、actor update sentinel；
- postflight：E0、E1 step 1、checkpoint shards、LoRA 差异、新进程 eval。

**本地结果**

- 定向 M8b tests：61/61 PASS；反捷径 tests：26/26 PASS；另有 2 项 E2 target-question 对齐回归；
- 全量锁定 suite：308 discovered/executed，305 PASS、3 SKIP、0 FAIL、0 ERROR；
- 本地 preflight：18 PASS、0 FAIL、2 WARN、11 SKIP；
- 严格 runtime gate 对 SKIP fail closed；
- 真实 GPU/LLM/optimizer/checkpoint：0。

**建议图示**

画“上传代码/数据/模型 → preflight → runtime gate → E0 → E1 → Ray restart → checkpoint eval → postflight”的闸门式流程。所有尚未执行的阶段用虚线框。

**来源**

`docs/m8b_autodl_preflight.md`、`configs/m8b_autodl_preflight.json`。

---

### Slide 15｜结论、未完成项与下一步

**页面结论**
项目已经完成奖励链路的离线可解释性验证，下一步不是直接扩大训练，而是先完成严格 GPU smoke，再实现在线 credit 与正式对照。

**已得到的证据**

- 轨迹可记录、可查询、可确定性 replay；
- rollout memory 隔离与版本历史可审计；
- Oracle AP → 手工 DFA → once-only reward 链路稳定；
- 真实 HotpotQA smoke 数据适配与 label-blind 检查通过；
- 5 条 controlled-error FR 全部可解释；
- Group Critic 无效输出显式回退，不静默污染奖励。

**当前不能声称**

- 不能声称 DFA 已优于 terminal-only 训练：尚无真实 E1/E3/E4 训练结果；
- 不能声称真实 LLM AP/Group Critic 性能：当前使用 mock/fake client；
- 不能声称真实模型已摆脱 Store-All 或主题偏移捷径：当前只有确定性 sidecar；
- 不能声称 GPU/checkpoint smoke 通过：AutoDL 尚未执行；
- 不能声称在线 `ActionCreditRecord` 已接入：当前仍是 schema/join/validation。

**推荐下一步**

1. 推送当前已验证的本地提交，轮换本地凭据；
2. AutoDL 上执行 `bash -n`、严格 preflight 和 `308/308` runtime gate；
3. 依次完成 E0 frozen eval、E1 单次 update、checkpoint 保存、重启后 eval、postflight；
4. 只有 M8b smoke 通过后，设计 E3/E4 在线 `ActionCreditRecord` 生成与 terminal-only 对照；
5. 最后才进入多 seed、扩大数据和正式 DFA-vs-terminal 研究。

**建议收束句**

> 先证明奖励链路不会撒谎，再让模型学习它；先完成可审计的单步 smoke，再讨论规模化训练收益。

## 4. 汇报数据总表

| 阶段 | 核心产物 | 规模/结果 | 证据边界 |
|---|---|---|---|
| M1 | TrajectoryRecorder / Replay | JSONL 可重放、查询、确定性 digest | 无真实模型 |
| M2 | MemoryStore / rollout registry | 版本化、snapshot/restore、隔离 | CPU/内存实现 |
| M3 | 三阶段 Toy Environment | 30 tasks；gold 30/30 | 规则 policy |
| Anti-shortcut | Stage 1/2 固定策略压力测试 | 26/26 tests；7/7 gates | Toy/Oracle sidecar |
| M4 | Oracle AP + hand DFA | gold 30/30；failure 全拒绝；once-only | Oracle 上界 |
| M5 | HotpotQA smoke adapter | 90,447/7,405/7,405；30 trajectories | 确定性策略 |
| M6 | Extracted AP / StateTracker | mock AP F1 .976；controlled FR 5/10 可解释 | fake/mock extractor |
| M7 | Critic offline validation | 90/90 replay；30 fallback；20/20 farming | deterministic mock |
| M8a | Terminal-only contract | 133 scope；130 PASS、3 local SKIP | 尚未上卡 |
| M8b-prep | AutoDL evidence gates | 61/61 targeted；308 all scope | 尚未真实执行 |

## 5. 图表与素材清单

后续生成 PPT 时建议制作以下图表，避免堆砌代码截图：

1. **研究路线图**：M0 → M1 → M2 → M3/M4 → M5 → M6 → M7 → M8a/M8b；
2. **三阶段环境时间轴**：Stage 1 记忆写入、Stage 2 干扰、Stage 3 检索回答；
3. **反捷径双对照图**：Store-All/None/Oracle-Safe-Store 与 Always-Keep/Clear/Opaque-ID-Control/Oracle-Safe-Compress；
4. **系统数据流图**：ActionEvent → MemoryStore delta → Triple/StateFact → AP → DFA → Reward；
5. **DFA 状态图**：`q0/q1/q2/q3/q4/q_reject/q_timeout`，标注 once-only edges；
6. **M5 策略对比柱状图**：gold 10/10 success，wrong-answer 0/10，missing-support 0/10；
7. **M6 指标表/散点图**：human-backed 与 controlled-error 的 Triple/AP F1、FA/FR、reward MAE；
8. **M6 错误传播因果链**：注入漏抽 → AP 缺失 → DFA 转移失败；
9. **M7 fallback 流程图**：valid critic / invalid critic / unavailable critic；
10. **M8b 闸门流程图**：preflight → runtime gate → E0 → E1 → checkpoint eval → postflight；
11. **证据边界矩阵**：已验证、mock/Oracle、待 AutoDL、待 E3/E4。

## 6. 视觉与排版建议

- 整体风格：学术工程型、克制、清晰，避免营销式大标题和装饰性卡片；
- 主色：深墨色文字 + 青绿色（已通过）+ 橙色（受控错误/警告）+ 红色（未完成/停止条件）；
- 每页只保留一个主结论，指标不超过 5 个；
- 用实线表示已验证链路，用虚线表示尚未执行链路；
- 代码只展示短接口名或 JSON 字段，不贴大段源代码；
- 所有指标标注数据类型：`Oracle`、`mock`、`controlled-error`、`local preflight` 或 `AutoDL pending`；
- 不展示 API key、完整轨迹正文、完整上下文、模型目录内容或 ignored `config`；
- 结论页必须保留“当前没有真实 GPU/LLM 训练结果”的免责声明。

## 7. 给后续 PPT 生成会话的直接提示词

```text
请阅读 docs/project_presentation_materials.md、STATUS.md、PROJECT_HANDOFF.md，
生成一份 15 页中文学术工程汇报 PPT，主题为
“Logic-Guided Agent Memory：面向长程记忆操作的可重放轨迹与 DFA 里程碑奖励”。

严格遵守素材文档中的阶段边界：M0-M7 与 M8b-prep 只按已验证证据呈现；
不要把 mock、Oracle 或 controlled-error 指标写成真实 LLM/训练结果；
不要声称 AutoDL、GPU、optimizer、checkpoint 或在线 ActionCreditRecord 已完成。

优先制作：研究问题、系统数据流、三阶段环境、反捷径双对照、DFA 状态图、M5/M6/M7 结果图、
M8b 闸门流程和未完成项。每页一个结论，图表优先于代码截图，保留数据来源和免责声明。
```

## 8. 证据来源索引

| 用途 | 文件 |
|---|---|
| 总体状态与测试数字 | `STATUS.md` |
| 研究范围、路线与交接约束 | `PROJECT_HANDOFF.md` |
| M3/M4/M5 基础与 Oracle benchmark | `docs/m5_hotpotqa_oracle_benchmark.md` |
| M6 抽取指标 | `docs/m6_extraction_benchmark.md` |
| M6 五条 False Reject 审计 | `docs/m6_false_reject_audit.md` |
| M7 Critic 离线验证 | `docs/m7_group_critic_offline_validation.md` |
| Stage 1/2 反捷径报告 | `docs/anti_shortcut_benchmark.md`、`artifacts/anti_shortcut_benchmark/anti_shortcut_benchmark.json` |
| M8a terminal-only 契约 | `docs/m8a_terminal_only_preflight.md` |
| M8b AutoDL 门禁与 smoke 顺序 | `docs/m8b_autodl_preflight.md` |
| 固定数据清单 | `data/splits/hotpotqa_smoke_manifest.json` |
| M8b 版本锁 | `configs/m8b_autodl_preflight.json` |
