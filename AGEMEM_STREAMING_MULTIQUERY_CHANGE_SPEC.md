# AgeMem 长信息流与多问题记忆训练：更改技术规范

> 版本：v1.0  
> 日期：2026-09-14  
> 接收方：在现有 Age-Mem 仓库中工作的 Codex（GPT-5.6 Sol）  
> 文档性质：待实施的技术设计、迁移方案与验收契约；不代表已经修改代码或完成实验。  
> 核心改动：将固定的“记忆构建—纯干扰—单问题回答”改为“受预算约束的连续信息流—同一记忆快照上的多个独立问题”。

**阅读导航：** 执行前读 0～3 节；数据实现读 4～7 节；环境与奖励读 8～11 节；训练接线读 12～14 节；配置、实施和验收读 15～22 节；第 23 节可直接作为 Codex 启动指令。第 24 节为延后研究，第 25 节为来源。

## 0. 执行摘要与使用方法

### 0.1 本次目标

实现一个可重放、可训练、可审计的新协议 `streaming_multiquery_v1`：

1. 同一个模型逐批接收长度超过其单次上下文上限的信息流。
2. 主压力档累计正文长度为上下文上限的 5～10 倍，另保留 1 倍以内和约 2 倍的诊断档。
3. 阅读时不公开未来问题；模型自行管理有限上下文和长期记忆。
4. 阅读结束后冻结状态，从同一快照独立回答多个问题。
5. 用多个问题的平均表现评价前面的记忆构建行为。
6. 分开检验环境是否可学习、语义奖励是否准确、逻辑结构是否提供额外信息、信用分配是否有效。

### 0.2 本文的执行边界

- 当前用户请求是生成实施文档。本文件未修改项目仓库、数据、远端环境或训练配置。
- 执行者收到用户“按本文实施”的指令后，默认连续推进本地代码、CPU 数据构建、重放、必要测试和文档更新；不要每完成一个小步骤就重复请求确认。
- 用户仅把文件放进仓库而没有要求执行时，不推定已获执行授权。
- 不从本文推定已有 GPU、外部付费 API、推送远端或发布权限。需要这些动作时，先完成可审查配置和本地验证；结合当时会话已有授权判断是否还需确认。
- 缺模型、缺数据、缺远端权限时，继续完成独立的 CPU 工作，明确记录阻塞项；不伪造运行结果。
- 本文是新协议的目标规范，不是绕过仓库 `AGENTS.md`、权限控制或既有保护机制的指令。

### 0.3 第一轮交付重点

第一轮先交付：数据构建器、双预算环境、多问题快照分支、独立测试、冻结诊断、正确的共享前缀训练接线。动态事实更新、真实 LLM Critic 和完整动作级新算法在后续阶段实现。

**不要把“新协议跑通”和“DFA 已被证明有效”写成同一个结论。** 固定检索下 Flat 与 DFA 可能等价；这是需要报告的实验结果。

### 0.4 建议放置位置

执行者把本文保存为仓库的 `docs/streaming_multiquery_change_spec.md`，并在 `STATUS.md` 顶部新增指向它的当前状态。若该路径已有用户文件，保留其内容并合并或使用新版本名。下文新增路径均为建议，须先核对实际仓库。

## 1. 依据、已知状态与结论边界

### 1.1 本文依据

- 用户上传的 `Agentic Memory(1).pdf`，重点为 §3.3～3.5、附录 A.3 和 C.4。
- 用户上传的 `GLARE(1).pdf`，重点为 §4.2～4.5 和附录 Algorithm 1。
- 用户上传的 `PROJECT_HANDOFF(1).md` 与 `STATUS.md`，以顶部 2026-09-14 更新为准。
- 本轮讨论确定的长信息流、隐藏未来问题、多问题独立评估方案。
- HotpotQA、MuSiQue 和 LongMemEval 官方资料，链接见第 25 节。

本文作者只核对了论文、交接和状态文档，没有读取本地实际仓库、400 条动作明细或远端训练目录。下列状态为文档记录，不是本轮重新验证的事实。

### 1.2 必须保留的历史记录

| 项目 | 文档记录 | 新方案如何处理 |
|---|---|---|
| 36-step E1 pilot | LoRA 参数变化，32-dev F1 约 0.246，无提升 | 保留为原协议负结果，不覆盖、不自动重跑 |
| fact-memory 冻结诊断 | 24 tasks、96 rollouts、319 Experiences、400 唯一动作 | 可复用审计方法，不冒充新协议轨迹 |
| 终局组内差异 | 只有 2/24 题标准差非零 | 新协议继续监测，不默认多问题必然解决 |
| 旧文本 grounder | 103 个确认正例仅命中 23，recall 0.223 | 禁止把旧 exact-text 低召回直接当语义遗忘率 |
| provenance grounding | 文档称已加入，CPU replay 待验收 | 先核对实际实现，再复用其可信部分 |
| M7 | mock/injected-client 离线 Critic；真实 LLM 调用 0 | 不写成真实 Critic 已验证 |
| 旧 E5 | 动作级 advantage、RTG、token mask 未实现 | 本文新阶段另命名，禁止伪记为已完成 |
| 历史 runtime gate | 冻结计数 318 | 保持旧 scope；新协议另设 suite 与 manifest |

旧文件后部关于“直接运行无 QR E3”“step 30 OOM”等历史说明，不得覆盖顶部最新状态。新文档更新要解决当前入口歧义，但不要删除历史实验记录。

### 1.3 两篇论文与本项目的区别

- AgeMem 的三阶段是同一 episode 的三个片段；其末端奖励包含任务、上下文、记忆分量，不等于纯答案 F1。
- 原论文 terminal advantage 广播至各步骤，不代表已准确区分每个动作的贡献。
- GLARE 提供符号状态与逻辑奖励思路，但不能补足 policy 在决策时不可观察的未来信息。
- GLARE 上传版本正文 §4.5 与附录算法在归一化范围上存在差异。执行者须明确本实现公式，不以“照论文”代替精确定义。
- 本方案是新任务协议，不宣称严格复现任一论文。旧 E0/E1/E3 分数不能与新协议分数直接相减归因。

## 2. 研究问题与可证伪假设

### 2.1 首要研究问题

在有限单次上下文、有限持久记忆和未知未来查询的条件下，能否通过训练提高同一份记忆对多个后续问题的平均效用？

### 2.2 需要分开回答的问题

| 编号 | 问题 | 必要对照 |
|---|---|---|
| RQ-S1 | 长输入是否确实形成记忆压力？ | question-only、last-window、固定记忆基线 |
| RQ-S2 | 学习写入是否有用？ | 固定阅读器与检索器下的冻结策略、训练策略 |
| RQ-S3 | 多问题是否改善记忆评价？ | 单问题/多问题，报告相同计算预算下结果 |
| RQ-S4 | 额外语义监督是否有用？ | Terminal vs Flat，同任务与初始化 |
| RQ-S5 | 时序逻辑本身是否额外有用？ | Flat vs DFA，匹配 AP、奖励尺度和信用粒度 |
| RQ-S6 | 动作级信用是否额外有用？ | 同奖励下 trajectory vs action-return |
| RQ-S7 | 更长信息流是否泛化？ | 训练长度分布与独立 10C 长度测试 |

### 2.3 不得预设的结论

- 输入达到 10C，不等于独立有效信息也达到 10C。
- 增加问题数量可能降低评价噪声，也可能降低组内奖励方差。
- 隐藏未来问题的策略不一定能达到知道未来问题的 Oracle。
- 静态百科 QA 不能充分证明动态事实更新或生产会话记忆能力。
- 多条问题来自同一历史，统计上相关；不能当完全独立样本计算置信区间。

## 3. 新协议与旧协议的迁移

| 维度 | 旧协议 | 新协议 |
|---|---|---|
| 基本样本 | 单问题、三阶段轨迹 | 一个历史流与一个候选问题池 |
| 输入方式 | Stage 1 事实，Stage 2 固定干扰 | 有效事实和背景分布于持续输入 |
| 压力来源 | 阶段切换、干扰、局部预算 | 累计长度与真实双预算 |
| 未来问题 | Stage 3 单题公开 | 阅读后按冻结抽样规则逐题公开 |
| 评价对象 | 一条问答轨迹 | 一份阅读结束状态上的多个独立分支 |
| 首轮检索 | 既有多种模式 | 固定、本地、只索引实际保留记忆 |
| 首轮训练 | 多能力联合探索 | 优先只训练记忆构建，回答器固定 |
| 原文访问 | 需审计 QR 等旁路 | 默认没有可读取的完整历史索引 |

不强行把旧 `stage_id=1/2/3` 改含义。新 schema 使用 `phase=ingest/query`，如旧接口要求数值 stage，可映射为 1/3，但必须同时记录 `protocol_version` 和 `phase`，禁止凭 stage 推断旧语义。

## 4. 术语与预算口径

### 4.1 变量定义

| 符号 | 含义 |
|---|---|
| H | 一条 episode 的完整输入历史流 |
| C | 单次调用的输入加预留输出 token 上限 |
| L | 去重后历史正文 token 数，不含未来问题和答案标签 |
| alpha=L/C | 输入长度比 |
| B | 所有可恢复持久记忆的有效 payload token 上限 |
| Q(H) | 数据构建时保留的候选问题池 |
| m | 每轮独立评价的问题数，起步为 4 |
| K | 同一历史的候选记忆构建 rollout 数，pilot 为 4 |
| S_k=(M_k,C_tail,k) | 第 k 次阅读结束的记忆和预算内上下文快照 |
| F_kj | 第 k 份快照对第 j 个问题的官方答案 F1 |

### 4.2 C 的精确定义

每次调用都满足：

`tokens(rendered_chat_prompt) + max_new_tokens <= C`

必须通过冻结 tokenizer 和实际 chat template 计数，包含 system、工具 schema、消息角色、标题、句子号、返回结果和模型写入的摘要。L 单独对输入正文计数，另报告序列化后的完整流 token 数。比较不同模型时记录 tokenizer 差异，不声称不同 tokenizer 的相同 token 值等价。

### 4.3 B 的精确定义

- 计入 active memory 的正文、可检索标题、模型自定义字段、自由文本 tags/keys 和来源引用序列。
- 采用规范化序列化后的整体 token 数计费，不仅数条目、不只数正文。
- 系统生成且严格定长/定结构的内部控制字段可不计费，但不得承载模型自由文本。
- 内部版本历史和 tombstone 可以保留作审计，但 policy 不能读取旧正文或免费 restore。
- 若开放 restore，恢复后的内容重新计入 B；任何模型可恢复的旧内容都不能成为绕过 B 的免费存储。
- 不允许在 ID、tool error、文件名、超长 source refs 或备注字段中建立第二份记忆。

### 4.4 首轮工程默认值

以下是待诊断的起步配置，不是已经验证的最优超参数。

| 配置 | 起步值 |
|---|---:|
| C | 4096 |
| ingest / answer 最大生成 token | 512 / 256 |
| 文本块目标正文 token | 512～768，按句子边界切分 |
| B | 2048 |
| alpha 诊断档 | <=1、约 2、约 5、约 10 |
| 首轮训练主档 | 约 2 和约 5；通过后加入 5～10 连续长度 |
| 每块最大决策次数 | 2；NEXT 消耗一次决策并立即推进 |
| 实际问题数 m | 4 |
| 候选问题池目标 | 8～12；按长度和数据可用性调整，最少 m |
| K | 4 |
| 回答阶段尾部上下文 cap | 512 |
| 回答阶段检索 payload cap | 2048 |

如果 system/tool schema 加固定预算已经无法装入 C，应使预检失败，调节新配置，而不是静默截断工具说明。不得通过直接更改模型架构位置编码把“运行预算”误实现成模型能力修改。

## 5. 数据源与划分规则

### 5.1 主数据

复用 HotpotQA，优先读取已有本地数据。train 使用带标签训练集；新 dev/test 从带标签 validation/distractor 来源派生。官方 fullwiki test 无答案和 supporting labels，不用于本地有监督训练和评分。

官方 fullwiki dev 的检索上下文可能缺 gold 段落。适配器逐个解析 `(title,sentence_index)`；缺证据时明确报告并采用合法的 distractor 来源重建，不能将缺证据样本默认为可回答。下载/来源更换需记录新 fingerprint，不能静默覆盖已冻结数据。

### 5.2 先划分，再拼 episode

1. 建立原始 QA ID 与段落规范化 hash 清单。
2. 保留官方 train/validation 的基础边界。
3. 从未用于反复调参的带标签 validation 部分建立新 dev/test。
4. 如要求文档不重叠，按共享文档/段落连接关系分组或排除跨集合冲突；记录保留率和被排除项。
5. 每个 split 只使用自己划分中的 QA 与背景段落池。
6. 检查原问题、精确段落和近重复文档重叠；背景文本同样参与检查。

不能只对 QA ID 去重就声称文档完全隔离；也不能声称解决了基座模型预训练污染。旧 32-dev 只作历史回归，不作为新方案的最终盲测。

### 5.3 数据量规划

| 阶段 | 独立历史数量 | 用途 |
|---|---:|---|
| debug | 20～50 | 人工抽查、确定性验证、CPU gate |
| pilot train | 200 | 初步学习信号 |
| pilot dev | 50 | 配置比较与奖励系数选择 |
| pilot test | 100 | 冻结后最后评测 |
| 扩展 train | 约 1000 起 | 仅在 pilot 有可解释信号后 |

这些数字不是论文级统计充分性保证。若原始可用数据不足，输出实际数量和原因，不重复相同历史凑数。同一历史的多个长度、顺序、问题子集变体共享 `history_family_id`，不能跨 train/dev/test。

## 6. Episode 构建算法

### 6.1 两级问题集合

`Q_pool` 是完整候选问题池；`Q_eval` 是某次 rollout 组实际回答的 m 道题。阅读 policy 不可见两者。一个组的 K 条轨迹必须共享 H、排列、Q_eval、预算、retriever 和 reader 版本；只允许 policy 采样随机性不同。

Q_eval 的抽样独立于模型行为。允许阅读结束后逻辑上揭示，但其 seed 可在采样前由环境固定。不得根据模型保留内容挑选可答题。训练可以轮换 Q_eval；dev/test 固定问题集合和顺序。

### 6.2 推荐实现步骤

1. 从当前 split 选择若干 QA，获取完整候选上下文段落。
2. 以标题、正文 hash 和源版本构建文档身份；同名不同文本不可粗暴合并。
3. 合并、去重段落，保留每一道题的原始证据指针。
4. 以整篇来源段落为基本选择单元，不能只抽 gold 句子再随机塞噪声。
5. 在固定目标长度下选择文档子集；被保留的问题必须具有全部标注证据。
6. 如果不足长度，增加同 split 的完整 QA 上下文或背景文档；背景选择不能依赖模型表现。
7. 如果超过长度，先撤销整块背景；若需要撤销某个问题的必要文档，同时移除该候选问题。禁止直接截断并保留失效标签。
8. 问题池去重，检查有效题数 >=m；否则继续采样或明确构建失败。
9. 按预先定义的顺序策略排列文档，按完整句子切块。
10. 记录证据在新流中的 chunk、句子和 token 位置，计算 digest。

给构建器设最大重采样次数，失败报告必须区分“长度不可达”“有效问题不足”“跨 split 冲突”“源指针失效”。不要无限循环。

### 6.3 顺序与难度控制

- 第一版保留单段落内部句序，只打乱文档块顺序。
- 后续可加入同文档分块交错，但保留标题/来源，不能破坏语言可理解性。
- 报告问题证据位于早、中、晚的位置和两跳间距离。
- 位置平衡在构建器中使用 gold 作离线分层可以接受，但不能给 policy 暴露标签或固定“第几块总重要”的模式。
- 长度控制组可固定核心历史、向各位置加入不同背景；这些变体必须留在同一 split 并作为成对实验。
- 自然规模组应同时增加独立事实和候选问题，不把所有 10C 都做成同一个 2C 核心加填充。
- 不人工修改事实以制造冲突。动态更新数据采用后续独立 schema。

### 6.4 第一版不需要 LLM 出题

直接复用原始 question、answer、supporting_facts。LLM 改写、问题生成、摘要数据增强都会引入额外语义校验工作，暂不加入。未来新增生成题必须有可追溯证据和人工/独立模型审计，不进入既有冻结 test。

## 7. 数据契约与公开/私有隔离

### 7.1 文件组织

建议使用三类文件，结构身份通过 ID 对齐：

| 文件 | 内容 | 可访问组件 |
|---|---|---|
| `episodes.public.jsonl` | chunks、公开来源、预算 | ingest environment |
| `episodes.questions.jsonl` | episode 对应候选问题文本 | query scheduler，阅读后才公开单题 |
| `episodes.gold.jsonl` | answers、support refs、分层标签 | evaluator / reward processor |
| `manifest.json` | split、来源/配置/tokenizer digest、统计、版本 | 实验基础设施 |

文件分开不是全部安全保证。policy observation 必须使用显式白名单 serializer；不能把环境对象 `__dict__`、完整 dataclass 或 debug repr 送入提示。

### 7.2 主要对象

```python
class StreamEpisodePublic:
    schema_version: str       # agemem.stream_mq.episode.v1
    episode_id: str           # 内部连接，不必展示给 policy
    history_family_id: str
    chunks: list[PublicChunk]
    context_budget_tokens: int
    memory_budget_tokens: int

class PublicChunk:
    chunk_id: str
    text: str
    sources: list[PublicSourceSpan]  # 无 supporting/not_support 标签
    content_token_count: int

class QueryRecord:
    query_id: str
    episode_id: str
    question: str

class QueryGold:
    query_id: str
    answer: str
    support_refs: list[SourceSentenceRef]
    source_question_id: str

class SourceSentenceRef:
    source_dataset: str
    source_revision: str
    document_key: str
    title: str
    sentence_index: int
    sentence_sha256: str
```

这是 schema 意图，执行者应使用仓库已有 dataclass/Pydantic 方式，避免引入平行技术栈。

### 7.3 公开来源的限制

- 文档标题、句子号可公开，便于真实来源声明；原始 QA ID、support labels、difficulty、future query ID 不公开。
- 模型只能引用已观察的来源；引用未来 chunk 或不存在的句子必须拒绝。
- 引用原文不等于正文保存了该事实，grounder 必须分别检查。
- 私有文件缺失时，训练评分失败并报告，不能退化为按工具调用次数奖励。

### 7.4 Manifest 必备内容

包括构建代码 commit、dirty patch digest、原始数据 fingerprint、split 规则、每个来源 ID、构建/排列/选题 seeds、候选/实际题数、tokenizer 名称及 revision、chat-template digest、C/B/L、chunk 分布、证据覆盖、重复率、各类丢弃数量、全部文件 SHA-256。

正文、金标和逐动作原文沿用项目已有 gitignored runs/data 策略，不提交到公共远端；小型原创测试 fixture 可以提交。历史特权审计文件继续遵循既有权限约定，不复制进入新提示或公开报告。

## 8. 阅读环境与上下文生命周期

### 8.1 阅读流程

```python
for chunk in episode.chunks:
    context.admit(chunk)             # 精确计数，必要时按公开规则腾出空间
    for decision in range(max_decisions_per_chunk):
        observation = render_public_ingest(context, memory_view, budgets)
        action = policy.sample(observation)
        result = tools.execute_one(action)
        recorder.append(action, result, before_state, after_state)
        if action.type == "NEXT":
            break
snapshot = freeze(memory, retained_context, tokenizer_and_protocol_versions)
```

模型必须确实看到所有成功接收的 chunk。溢出处理可以移除旧上下文，不能静默丢弃尚未向 policy 展示的新 chunk。`observed_chunk_ids` 和 source visibility ledger 由环境记录，用于奖励验证，不作为可读原文档案。

### 8.2 工具语义

首轮每个模型回复最多一个完整工具动作，减少多动作 token 归因歧义。

| 动作 | 作用 | 校验 |
|---|---|---|
| ADD | 写入模型提供的记忆正文和已观察来源 | schema、来源可见性、全局 B |
| UPDATE | 原子替换指定 active memory 的正文/来源 | 版本、语义来源、预算差额；失败不改变状态 |
| DELETE | 删除 active memory | 不允许通过免费 restore 读取旧正文 |
| SUMMARY | 模型提供 `replacement_text`，替换指定可见上下文片段 | 片段必须当前可见；不调用辅助模型生成摘要 |
| CLEAR | 移除指定可见上下文片段 | system/tool schema/当前决策必要指令不可删 |
| NEXT | 接收下一块，无持久写入 | 不直接奖励 |

模型输出截断、非法 JSON、多动作、超预算或无效来源时，返回短的确定性错误；不得返回 gold、隐藏原文或纠正后的正确答案。动作失败也消耗决策次数。到达块内决策上限后自动推进，记录 `environment_event`，不能伪装为 policy 动作。

### 8.3 SUMMARY 和 STM 的能力边界

SUMMARY 由当前 policy 自行生成摘要；原项目可能调用 qwen-max 的实现不得未经标注直接沿用。如果复用外部摘要器，必须另设 `assisted_summary` 实验臂并记录模型、费用与调用量。

SUMMARY 默认只改变 STM；不会自动写入 LTM。要跨更长历史保留，模型需显式 ADD/UPDATE。阅读时 STM 可帮助跨批次融合，回答时只保留有限尾部，因此首轮结果主要支持受限记忆构建，不自动证明全部 STM 技能均已学会。

### 8.4 自动溢出处理

默认使用可审计的 FIFO 移除最旧可删除消息组，为当前 chunk、下一次响应和工具回执预留空间。每次移除记录 IDs、token 数、触发原因和状态 hash。

- 该 FIFO 是环境兜底，不记为 policy 的 CLEAR，不给予策略奖励。
- 必须给模型至少一次看到新 chunk 并处理它的机会。
- 不从 loss 中删除这些早期生成动作；它们的 prompt 和 old logprobs 仍保留在训练轨迹。
- 不允许框架 tokenizer 在环境外再次偷偷截断；用最终 prompt hash 检查实际送入模型的内容。
- 摘要、标题、工具返回和 error message 都参与 C 计数。

### 8.5 LTM 访问

环境不把全部 M 每轮拼进提示；这可能绕过或耗尽 C。通过预算内的简短 memory handle 列表和受限检索接口访问。如首轮允许 ingest 检索，只能检索当前 M，参数与调用成本固定并记入上下文；默认关闭以简化变量。

所有全量 history、原始 QA context、reward-side source registry、旧版本原文和另一 rollout 的 memory，均不可被 policy 工具访问。

## 9. 阅读结束快照与多个独立问题

### 9.1 快照定义

保存 `S_k=(M_k,C_tail,k)`：

- M 为实际 active memory 的完整冻结副本。
- C_tail 为按确定性规则选取的、最多 512 token 的阅读末尾可保留上下文；规则和 payload 在全部实验臂一致。
- 不默认额外清空全部 STM。若研究只靠 LTM 的情况，增加 `answer_tail_tokens=0` 消融，不能与主结果混合。
- 保存 memory digest、tail digest、snapshot ID、policy version 和已观察来源账本。

### 9.2 分支协议

每道题从同一个不可变 S_k 克隆得到独立分支：

1. 加入该题公开 question。
2. 使用固定 retriever 从 M_k 获取预算内记忆正文。
3. 构建回答 prompt：固定指令 + question + C_tail + retrieved payload。
4. 检查输入加输出预算 <=C，生成答案。
5. 分支结束；其回答、检索缓存和新增上下文不返回父快照。

首轮回答分支不允许写记忆、不调用工具循环，一次固定检索后一次回答。后续 `interactive_query` 才开放模型主动 RETRIEVE；必须新 protocol/config version。

### 9.3 回答器设置

第一轮采用 `train_scope=ingest_only`：

- 记忆构建 policy 是待训练模型。
- reader 使用同系列、同规模的冻结初始 checkpoint；不同奖励臂复用同一 reader。
- reader prompt、tokenizer、temperature（默认 0）、答案提取规则和检索方法完全一致。
- reader 是环境评分链中的固定组件；其生成 token 不参加 policy gradient，不伪造 old logprobs。
- 固定 reader 可作为相同底座模型的独立逻辑角色实现；资源如何部署由现有 provider/runtime 支持决定，不宣称必须同时在显存常驻两份完整模型。
- 主政策的冻结/训练对比都用这个 reader，另做自身模型回答能力的诊断，不混同两个口径。

这一步测“写入策略带来的收益”。后续联合学习统一 policy 的阅读与回答，要使用第 13 节的分支 loss 契约，并重新设 matched baseline。

### 9.4 固定检索器

默认复用或实现本地确定性词法检索（例如已锁定的 BM25），冻结分词、大小写、stopwords、排序和 tie-break 规则。只索引 M 中实际保留的文本，不能索引 source refs 对应的完整原句。

按排序累加序列化 payload 直到 token cap；超长条目按固定规则跳过或明确截断。若截断，reward 只能检查真正进入回答 prompt 的内容，不能把被截去的 supporting fact 算作已暴露。

固定检索可能限制最终上界，应单独报告全量保留索引诊断。该诊断如果不遵守 B，必须标为资源更宽松的参考，而非同预算对照。

## 10. 语义 grounding、来源与奖励基础设施

### 10.1 先验证旧 provenance，再验证新协议

若旧 96-rollout/400-action 数据在本地可用，先运行已实现的 provenance CPU replay，记录结果与版本。旧文件不可用时，不阻塞新协议的数据/环境开发，但不得声称旧 grounder 已验收；必须在新协议独立样本上补做语义审计后再开语义奖励训练。

### 10.2 三个不同概念

| 概念 | 判定 |
|---|---|
| 有效来源 | 指针存在、已经观察、版本/正文 hash 正确 |
| 事实保留 | 实际存储正文保留了该来源中的目标语义 |
| 事实暴露 | 实际回答 prompt 中包含保留下来的相应语义 |

仅“指向 supporting sentence”的空壳记忆不能得到事实保留分。空正文、无关摘要和伪造 source refs 必须被拒绝或判为不支持。UPDATE 改正文后重新判定，不无条件继承原 support；失败 UPDATE 不修改状态。

### 10.3 首轮可实现的语义策略

- CPU 结构 gate 可使用原创 fixture 的确定性语义规则，不用于宣称真实自然语言准确率。
- 自然轨迹可采用已校验 source pointer + 受控 extractive 模式，作为独立标记的 `extractive_control`，验证可达性。
- 自由摘要模式沿用经人工审计的 grounder，或者增加独立、冻结的语义判定器；任何辅助模型都记录版本、调用与失败。
- 没有足够语义验证能力时，可以继续 Terminal-only 和离线诊断，不启用凭指针直接给分的伪 Oracle 奖励。
- “已确认支持”“确认不支持”“不明确”三类分开报告；不明确项不强行当负例。

### 10.4 审计报告

新样本覆盖 ADD、UPDATE、DELETE 后重存、摘要、检索截断和无效指针。按去重语义样本统计 precision/recall/F1，再报告动作展开口径；不把重复同一句的几十个动作当几十个独立审计样本。

结构正确率要求 100%。语义训练准入建议以预先冻结的阈值管理，例如 precision>=0.95、recall>=0.80，同时报告样本量、各类分布和区间。这些是工程目标而非统计显著性标准；样本过少或关键类别缺失时，即使点估计达标也不自动认为可靠。阈值需要在自然结果揭示前冻结，不能看结果后降低。

## 11. 奖励定义与实验臂

### 11.1 共同任务奖励

对第 k 份快照、m 个问题：

$$R^{task}_k=\frac{1}{m}\sum_{j=1}^{m}F_{kj}.$$

答案使用现有、已经验证的 HotpotQA 官方 EM/F1 规则，含 yes/no/noanswer 特殊处理。格式提示各臂一致；解析失败 F1=0，并单独统计，不把格式奖励当内容正确。

### 11.2 成本与约束

主 pilot 先用硬 C/B 和决策次数上限约束成本，`cost_weight=0`，避免同时引入一套调参变量。记录生成 token、保留 token、工具次数、自动移除、失败调用和延迟。

后续成本臂采用归一化成本且预先设定权重，所有奖励臂共用；不能把成本惩罚只加给某一方法。预算失败返回确定性错误并消耗动作额度；是否另有违规罚分必须写入 reward version。

### 11.3 静态语义参考分

对题 j，令 S_j 为去重后的官方支持事实集合。定义阅读结束时语义保留覆盖 c^M_kj，和实际回答 prompt 的暴露覆盖 c^E_kj，均在 [0,1]。

此处 c^E 专指从 M_k 经检索实际暴露的证据；C_tail 自带的证据另记 `tail_support_coverage`，两者并集另记 `total_prompt_support_coverage`。这样可以区分长期记忆与最近上下文，且保证本节的静态状态与自动机分数使用同一口径。最终 F1 可正常利用全部合法 prompt。

$$U_{kj}=\tfrac12 c^M_{kj}+\tfrac12 c^E_{kj}.$$

覆盖只是一种辅助监督，不假设官方支持句是唯一合法解法。保留改写和等价证据的审计口径，避免把覆盖分直接等同最终任务成功。

### 11.4 奖励对照矩阵

| 臂 | 定义 | 用途 |
|---|---|---|
| T | R_task | 主 terminal 基线 |
| FS | R_task + lambda*mean_j(U_kj) | 强静态状态监督，不使用自动机 |
| FE | R_task + lambda*归一化 once-only 有效事件分 | 可选，检验事件计数与最终状态差别 |
| D | R_task + lambda*归一化逻辑状态分 | 检验状态/依赖，必须与 FS/FE 解释清楚 |

默认 `lambda=0.25` 仅为起步值；零和少量正值在 dev 上比较，选定后冻结。保证语义总分归一化到 [0,1]，不随 m、事实数、动作数或历史长度线性增长。

FS 是必要强对照。如果 D 只是用自动机计算 U，那么 D 与 FS 的逐轨迹奖励应完全相同。此时不运行两套重复 GPU 训练，不宣称自动机本身带来收益。

### 11.5 自动机首轮语义

对每个 query、每个 support 建立状态：`absent`、`retained`、`exposed_valid`。允许：

- 已观察且经语义验证的写入：absent -> retained。
- 最后一份有效表示被删除或改坏：retained -> absent。
- 对应有效表示进入回答 prompt：retained -> exposed_valid。
- 重复保存、重复检索不会增加覆盖。
- 等价表示仍存在时，删除冗余副本不会错误清零。
- 自动检索事件标记为 `actor=environment`；只能间接影响早期写入奖励，不能当作模型主动 RETRIEVE 的训练证据。

事实集合之间不强加任意“F1 必须先于 F2”的顺序，也不要求存齐全部证据才承认部分进展。ANSWER 由任务 F1 评分，不再在语义项重复加一份答对奖励。

读取结束前删除事实应使最终保留分下降，不能通过“先存再删”保留永久积分。若采用增量状态势分，必须记录回退负增量并验证其和等于终态减初态。

**有限静态任务中的这个自动机，主要是可解释实现与错误检测机制。它很可能与强静态状态分等价。** 要研究额外时序收益，后续需要真实的旧/新版本、有效期、主动多步检索等依赖；不能为制造差距而给 Flat 较差的语义标签。

首轮 D 的终态数值固定为：absent=0、retained=0.5、exposed_valid=1；先在单题事实集合上平均，再在 m 道题上平均。因此，在固定检索、回答分支不修改记忆的主协议中，D 应与 FS 等价。S3 的目标是验证这个等价和回退正确性；S7 不为该等价实例另跑 D 训练。后续改变依赖时升级 `reward_version` 并重新给出公式。

如实现可选 FE，单事实分数为 `0.5*ever_valid_store + 0.5*ever_valid_exposure`，同样先按事实、再按题平均；删除不会撤回历史 store 项。FE 的用途是检验错误的历史积分是否偏好存后遗忘，不将其作为唯一 Flat 基线。

### 11.6 不宣称未证明的 shaping 性质

将非零终态语义覆盖加到任务奖励会改变优化目标。不因写成势函数差就声称保持原始任务的最优策略。若后续采用理论上的 potential-based shaping，必须另行定义折扣、初末状态处理和证明前提。

## 12. 多问题共享前缀的训练拓扑

### 12.1 一条样本实际是一棵小树

一条记忆构建轨迹 tau_read,k 后接 m 个独立 query branches。它不是 m 条互不相关的完整 episode，也不是先后回答 m 题的线性轨迹。

必须记录：

| 字段 | 含义 |
|---|---|
| `group_id` | 相同 H/Q_eval/预算的 K 份记忆构建组 |
| `read_rollout_id` | 某一次记忆构建 |
| `snapshot_id` | 阅读结束状态 |
| `query_branch_id` | 某快照下的一道问题分支 |
| `query_id` | 对应原始问题身份 |
| `shared_prefix_id` | 共享阅读前缀，只存一份逻辑身份 |
| `policy_version` / `reader_version` | 生成策略与固定 reader 版本 |

同一环境事件和 policy 动作分别编号；`ActionCreditRecord` 若按题拆分，连接键是 `(action_id,query_id,reward_version)`，之后生成明确的 m 题平均聚合记录，不能把同一 action_id 的多个评分误当重复损坏。

### 12.2 第一轮 ingest-only GRPO

每个历史组采样 K 份记忆，得到 R_k。只在 K 份独立记忆间计算：

$$A_k=\frac{R_k-\operatorname{mean}_{l=1..K}R_l}{\operatorname{std}_{l=1..K}R_l+\epsilon}.$$

首轮将 A_k 广播给该份记忆的所有可训练阅读生成 token。这个方法称为 `read_trajectory_advantage`，不称为动作级归因。

- buffer 中每个阅读动作只出现一次逻辑样本。
- 回答分支只提供评测/奖励，不进入 actor loss。
- K 不能替换成 K*m；四道题并不是四份不同记忆。
- 组内方差为零时，任务优势为零。记录 KL/其他项，不用跨题混组或人为扰动奖励制造信号。
- 不丢弃全失败组后假装保持原数据分布；若采样重加权，需单独版本和对照。

### 12.3 损失归一化

在不改变底层 PPO/GRPO clipping 与 KL 契约的前提下，明确采用按历史、按 rollout 等权，再在其有效训练 token 内取平均：

$$L=\frac1{N_H}\sum_h\frac1K\sum_k
\frac1{n^{read}_{hk}}\sum_{t\in read(h,k)}\ell_{hkt}.$$

这定义的是显式长度归一化的 surrogate，不能声称等于未归一化的完整序列 REINFORCE。若沿用框架原 token reduction，必须记录真实公式，并让所有方法和消融一致。

特别检查框架 `advantage` 是否在进入 trainer 后又被全 batch 二次归一化。不要让不同历史、不同 phase 或 query 分支被隐式重新混组。

### 12.4 最小数值验收例

使用专用 fixture（K=2、m=2，不改变 pilot 默认 K/m）：第一份记忆的两个 F1 为 [1,0]，第二份为 [0,0]。任务回报分别为 0.5 和 0，组均值 0.25；采用总体标准差时 std=0.25，优势接近 +1 和 -1。若框架使用样本标准差，应按其实际公式得到相应值，并在配置中明确 `std_ddof`，不能两种口径混用。

不论每份记忆分成多少个问题分支，阅读 ActionEvent 的逻辑数量都不变。分支评分先聚合，不对共享前缀复制两次。为只验证平均聚合，可在测试中复制相同评分向量确认回报不膨胀；真实数据构建仍禁止重复问题凑 m。

## 13. 后续联合训练与动作信用分配

### 13.1 联合训练的目标分解

当阅读与回答均由同一待训练 policy 产生时，平均任务回报目标的 score-function 结构为：

$$\nabla J=\mathbb E\left[
\bar R_k\nabla\log p_\theta(\tau^{read}_k)
+\frac1m\sum_jR_{kj}\nabla\log p_\theta(\tau^{answer}_{kj}\mid S_k,q_j)
\right].$$

该式说明共享前缀承接平均分支回报，各回答分支承接自己的回报并带 1/m 权重。它是设计依据，不意味着 GRPO 的组归一化 surrogate 与原始梯度完全相同。

- 阅读部分仍只训练一次，不把它复制 m 次然后与回答 token 直接拼平。
- 回答优势可对同一历史、同一道题跨 K 份记忆计算；不能在不同难度题之间直接互减。
- phase/token reduction 以及阅读/回答相对权重必须显式记录。
- 共享权重引起的 reader 能力变化属于联合训练结果，不能与 ingest-only 固定 reader 结果混称。

### 13.2 后续动作 RTG

只有在动作级增量奖励语义可靠、trajectory 基线有效后再启用。

设 r^read_ku 是每个阅读动作即时奖励，B_kj 是第 j 分支在阅读完成之后的剩余回报，则阅读动作 t 的无折扣 return 为：

$$G^{read}_{kt}=\sum_{u=t}^{T_{read}}r^{read}_{ku}+\frac1m\sum_jB_{kj}.$$

每个回答分支自身计算局部未来回报。不得把问题分支 j 的后面错误接成 j+1 的“未来”；它们是同父快照的兄弟分支。

若使用 gamma<1，必须定义环境步骤/模型动作的折扣单位和分支距离；首轮动作版用 gamma=1，避免引入额外时间偏好。

### 13.3 必须避免的实现错误

- 任务最终回报在即时 reward、分支 return 和共享前缀 return 中重复计数。
- 把各动作 return-to-go 再相加，作为轨迹总奖励。
- 同一个 assistant turn 的多个动作重叠使用同一 token 范围。
- 把 SUMMARY 输出、JSON 参数或 NEXT 的真实生成 token 从损失中漏掉。
- 把 system、user、tool observation 和固定 reader 输出加入 actor loss。
- 使用新模型重算的 logprobs 冒充 rollout 时的 old_logprobs。
- 用 `(stage,dfa_state)` 相同就断言两个 observation 是相同 Markov 状态。

### 13.4 bucket 的限制

DFA 状态会丢失记忆正文、剩余预算、证据身份和输入进度。相同 DFA 状态不保证动作可比。若研究 bucket baseline，至少加入同一 history/query、phase、输入位置、预算状态等条件并审计样本量；稀疏时按预先规则回退，记录回退率。

动作级方法是单独实验变量。不得同时更换数据、grounder、reader、奖励系数和 advantage，然后把提升全部归因于信用分配。

## 14. Buffer、日志与运行时契约

### 14.1 原子 rollout bundle

新 buffer 单位建议为 `MemoryRolloutGroupBundle`：一个历史的 K 份阅读轨迹，加各自 m 个评分分支、共享版本与完整奖励。仅当预期分支全部完成或按显式失败策略结算后，才能发布 bundle。

优先复用已有 `consume_put_batch` 的整组消费思想，但必须检查其真实单位；`train_batch_size=8` 不保证消费了 8 条完整 rollout。禁止通过固定 Experience 数量切断组。

### 14.2 可变长度与失败策略

- 不按裸 timestep 对齐不同轨迹。
- 网络/运行时失败与模型答错分开；基础设施失败不应无条件给 0 并进入训练。
- 允许确定性重试固定 reader 的基础设施错误，记录次数；不能为了获得更高分反复采样答案。
- 组中缺分支且无法恢复时，整个组标记 infrastructure_failed，不把残缺组默默进入 GRPO。
- 若 policy 生成了非法动作，属于有效模型行为结果，保留轨迹并按公开环境规则处理。

### 14.3 Receipt 最低字段

每个训练步记录 history_count、K、m、完整组数、read rollout 数、branch 数、read ActionEvent 数、environment event 数、trainable tokens、零方差组比例、reward 各分量均值/标准差、非零 advantage 比例、grad_norm、KL、loss、policy/reader/retriever/grounder 版本、checkpoint digest 和数据 manifest digest。

Receipt 必须能区分“optimizer 调用成功”“参数发生变化”“产生非零任务梯度”“dev 表现提升”。四者不相互替代。

## 15. 基线、消融与统计分析

### 15.1 环境和记忆基线

| 基线 | 定义 | 解释边界 |
|---|---|---|
| Question-only | 固定 reader 只看问题 | 参数知识参考 |
| Last-window | 原始流最后预算内部分，不调用记忆策略 | 最近窗口能否答题 |
| FIFO-memory | 在 B 内按顺序保留完整条目 | 有限记忆公开基线 |
| Fixed-summary | 固定模型/规则摘要，在相同 B 内保留 | 记录额外调用和成本 |
| Frozen-policy | 当前未训练 policy，使用同样工具和 reader | 训练收益的主要对照 |
| Learned-policy | 训练后的同构 policy | 与 Frozen 比较 |
| Gold-support | 每题将完整支持证据提供给 reader | 信息特权诊断，不是可部署同预算策略 |
| Full-store RAG | 全文保留并检索 | 若超出 B，标为资源放宽的系统参考 |

不把 Store-All 称为天然作弊。若资源不限，它可能合理；本实验通过明确 B 来研究选择性保留。

Gold-support 也须满足单次 C。若支持证据本身超出 C，标记 oracle_context_infeasible；不可截断后继续把失败当推理能力不足。

### 15.2 方法对照

优先顺序：Frozen -> T -> FS -> 有实质区别的 D -> 可选 action-return。FE 用于定位“事件计数/终态保留”的区别，不替代强 FS 对照。

单问题与多问题比较至少报告两种预算口径：相同独立历史/阅读 rollout 数，以及相近总生成 token/reader 调用成本。不能把多花 m 倍回答计算的提升全部归因于监督设计。

### 15.3 最低报告指标

1. **任务**：按历史先平均问题 F1，再跨历史平均；EM；格式失败率；问题类型分层。
2. **记忆**：保留/检索/实际暴露的语义覆盖、token 占用、条目数、压缩比、冗余。
3. **压力**：alpha、chunk 数、C/B、证据年龄、跨 chunk 距离、正文去重率。
4. **策略行为**：各工具次数、非法/超预算率、NEXT、自动 FIFO 次数、SUMMARY token 变化。
5. **奖励**：分量分布、组内 std、零方差率、Flat/DFA 差异率、排序一致性、grounder 审计结果。
6. **训练**：可训练 token、参数变化、非零梯度比例、KL、训练/评测 token 与耗时。
7. **分支完整性**：快照隔离、m/K 完整性、action-credit join、实际 policy prompt 一致性。

“暴露的证据”不等于“模型因果上使用了证据”。如需声称使用，增加去除/替换检索证据的成对干预评测。

### 15.4 统计原则

- 以独立 history/history_family 为重采样单位计算成对差值区间，不能把 m 道相关问题当 m 个独立历史。
- 同一评测集尽量不重复文档；若历史间复用大量文档，报告重叠并按共享来源簇分析敏感性。
- 固定同一 test 输入、问题、reader 和检索预算比较方法。
- pilot 单 seed 用于诊断，不写“稳定显著提升”。有明确效应后再扩展 seeds/样本。
- 失败和低分题不能因影响平均数被从 test 删除；数据坏样本按预定义规则处理并报告全部分母。

## 16. 配置草案

以下 YAML 是目标接口示例，执行者应先实现和校验这些字段，再生成真正可运行的配置。它不是可以直接塞入当前 Trinity parser 的现成配置。所有相对路径以仓库根为准。

```yaml
schema_version: agemem.stream_mq.config.v1
protocol: streaming_multiquery_v1
experiment_family: stream_mq_v1

data:
  source: local_hotpotqa
  source_path: null
  source_revision: null
  split_manifest: configs/stream_mq/split_manifest.json
  output_root: runs/stream_mq/data_v1
  build_seed: 20260914
  alpha_targets: [2.0, 5.0, 10.0]
  alpha_tolerance: 0.10
  candidate_query_target: 12
  min_candidate_queries: 4
  queries_per_snapshot: 4
  preserve_sentence_boundaries: true
  deduplicate_paragraphs: true
  max_build_attempts_per_episode: 100
  question_selection_depends_on_policy: false

model:
  policy_path: null
  policy_revision: null
  tokenizer_path: null
  tokenizer_revision: null
  chat_template_sha256: null
  reader_path: null
  reader_revision: null

environment:
  context_total_tokens: 4096
  persistent_memory_tokens: 2048
  chunk_target_tokens: 640
  chunk_max_tokens: 768
  ingest_max_new_tokens: 512
  answer_max_new_tokens: 256
  decisions_per_chunk: 2
  one_action_per_response: true
  auto_eviction: fifo_oldest_removable_message_group
  answer_tail_tokens: 512
  original_history_lookup: false
  source_pointer_dereference_by_policy: false
  old_memory_version_readable: false
  external_summary_model: false

query:
  mode: independent_snapshot_branches
  retriever: frozen_local_lexical
  retrieval_payload_tokens: 2048
  retrieval_top_k: 8
  reader_temperature: 0.0
  branch_writeback: false
  query_stage_policy_tools: false
  required_answer_format: answer_tags

reward:
  profile: terminal
  grounder_version: null
  task_metric: hotpotqa_official_f1
  semantic_lambda: 0.0
  semantic_component_cap: 1.0
  cost_weight: 0.0
  format_bonus: 0.0
  negative_trend_reward: false
  positive_trend_reward: false

training:
  train_scope: ingest_only
  rollouts_per_history: 4
  ingest_temperature: 0.6
  advantage_mode: read_trajectory
  std_ddof: 0
  group_key: history_queryset_budget_policyversion
  publish_complete_group_only: true
  shared_prefix_storage: once
  reader_tokens_in_actor_loss: false
  loss_reduction: history_rollout_token_mean
  history_groups_per_step: 2
  gamma: 1.0
  optimizer_config_source: existing_verified_small_model_profile

runtime:
  run_root: runs/stream_mq/terminal_seed7
  checkpoint_root: null
  seed: 7
  resume: false
  require_resolved_model_and_data_versions: true
  emit_receipt: true
```

### 16.1 配置不变量

- 所有 null 的数据/模型/revision 必须在模型运行前解析并冻结，不能用“latest”顶替。
- 若独立 answer reader 与 policy 不同 tokenizer，分别计数并绑定两个模板。
- T 臂 semantic_lambda 必须为 0；FS/D 使用单独配置，不能运行时改同一 lock。
- 回答 prompt 的分配顺序固定为：预留输出、保留完整指令与问题、装入最多 512 token 的尾部、剩余空间装入最多 2048 token 的检索 payload。每一步以实际模板重新计数，保留完整句子/消息边界。固定指令与问题本身放不下时明确失败，不能截断问题。记录各部分实际值，所有臂使用同一算法。
- alpha_tolerance 是相对于 alpha_target 的允许误差。标记为主 5～10C 的样本同时满足严格闭区间 [5,10]：允许范围是目标容差区间与 [5,10] 的交集。约 2C 的诊断不受该区间约束。1C 以内诊断单独配置，不混入该示例的训练列表。
- optimizer、学习率、LoRA rank/target modules、KL 系数、clipping、梯度累积、mini-batch 和模型精度等从实际已验证的小模型配置解析并写入新 lock。禁止仅留下 `existing_verified_small_model_profile` 字符串就宣称配置完整；若旧实现无法提供某项，由执行者在运行前选择并记录，不以本文虚构值代替。
- 新 checkpoint 根必须独立；只有核对模型、数据、配置、reader 和 optimizer state 完整匹配后才允许显式 resume。

## 17. 模块改造与建议文件清单

### 17.1 已知可复用入口（必须先核验）

| 现有路径 | 可复用能力 | 注意 |
|---|---|---|
| `AgeMem_code_agentscope/action_schema/` | 动作身份、轨迹 schema | 新增 protocol/branch 字段，兼容旧记录 |
| `AgeMem_code_agentscope/toy_hotpotqa/` | 内存隔离、预算、规则 fixture | 不把 Oracle 策略放入训练 buffer |
| `AgeMem_code_agentscope/memory_extraction/` | 来源/三元组/状态接口 | 核对自然语义能力，不把 mock 当模型 |
| `trinity/common/workflows/memory_context/memory_store.py` | 现有 memory store | 增加预算与快照，不改旧默认行为 |
| `trinity/common/workflows/memory_context/train_hotpotQA.py` | 原工作流参考 | 优先新增 workflow，避免继续堆叠开关 |
| `trinity/common/workflows/memory_reward/reward_profiles.py` | 奖励配置 | 新 namespaced profiles |
| `trinity/common/action_event_contract.py` | action/Experience join | 扩展共享前缀与分支校验 |
| `trinity/buffer/reader/queue_reader.py` 等 | 整组消费 | 核验实际 bundle 单位 |
| `trinity/common/runtime_receipt.py` | 运行证据 | 新字段兼容追加 |

### 17.2 建议新增模块

```text
AgeMem_code_agentscope/streaming_memory/
  schema.py
  data_builder.py
  split_audit.py
  source_registry.py
  token_budget.py
  environment.py
  tools.py
  snapshot.py
  query_runner.py
  reward_replay.py
  metrics.py

trinity/common/workflows/memory_context/train_streaming_multiquery.py
trinity/common/streaming_multiquery_contract.py

configs/stream_mq/
examples/agemem_streaming_multiquery/
scripts/agemem_stream_mq.py
tests/common/stream_mq_*_test.py
docs/streaming_multiquery_change_spec.md
docs/streaming_multiquery_implementation_report.md
```

如现有代码已有相同能力，应做适配而非再写一套。新增工具不能自动调用旧 QR 的 `stage3_index_observed_context`；完整原文索引旁路默认禁止。

### 17.3 不应改动

- 原 1.5B/4B E1 dry-run YAML 和其 digest。
- 已关闭实验的 checkpoint、lock、报告和 M3～M7 原始轨迹。
- 冻结 318 scope 的预期计数；新 suite 单独发现并冻结。
- 用户未提交的无关修改、模型 provider 凭据、原始数据内容。
- 已有默认工作流行为；新增配置默认不开启新协议。

## 18. CLI 契约与执行顺序

下面命令是执行者需要交付的接口，不是假定仓库中已经存在。先实现 `--help` 和解析验证，再在实施报告中写入实测命令。

```bash
# 1. 构建和静态检查（CPU，本地数据）
python scripts/agemem_stream_mq.py build-data --config configs/stream_mq/debug.yaml
python scripts/agemem_stream_mq.py validate-data --manifest runs/stream_mq/data_v1/manifest.json

# 2. 规则 fixture 环境与快照测试（CPU）
python scripts/agemem_stream_mq.py env-smoke --config configs/stream_mq/debug.yaml --policy scripted

# 3. 对已有真实轨迹重放（CPU，绝不重新调用 policy）
python scripts/agemem_stream_mq.py replay --run-dir runs/stream_mq/frozen_diag --profiles terminal,flat_state,dfa

# 4. 新协议自己的 gate
python scripts/agemem_stream_mq.py gate --scope cpu --config configs/stream_mq/debug.yaml

# 5. 以下需要模型环境；先生成配置、预检，按会话已有授权执行
python scripts/agemem_stream_mq.py preflight --config configs/stream_mq/frozen_diag.yaml
python scripts/agemem_stream_mq.py diagnose --config configs/stream_mq/frozen_diag.yaml
python scripts/agemem_stream_mq.py train --config configs/stream_mq/terminal_pilot.yaml
python scripts/agemem_stream_mq.py evaluate --config configs/stream_mq/terminal_eval.yaml
python scripts/agemem_stream_mq.py report --run-dir runs/stream_mq/terminal_seed7
```

`replay` 的输入必须来自对应新协议；不能把旧 96 条三阶段轨迹直接套成多问题数据。旧 provenance replay 使用仓库真实已有入口，先搜索确认名称，不在本文虚构。

每个 mutating CLI 应拒绝覆盖非空、身份冲突的输出根；允许幂等读取已完成同 digest 产物，明确报告 reused。脚本退出码区分配置、数据、基础设施和模型行为失败。`--help` 不加载 GPU、模型或发出外部请求。

## 19. 分阶段实施计划

新里程碑使用 S0～S8，与旧 M/E 编号分离。

| 阶段 | 交付 | 通过后继续 |
|---|---|---|
| S0 接管审计 | 仓库/版本/未提交修改、现有接口地图、旧 provenance 状态 | S1 |
| S1 数据 | builder、split audit、manifest、20～50 条 debug、坏样本报告 | S2 |
| S2 环境 | C/B、工具、FIFO、可见性与快照分支、CPU fixture | S3 |
| S3 奖励重放 | grounding 审计入口、T/FS/D、状态回退与多题聚合 | S4 接线可并行推进依赖较少部分 |
| S4 训练接线 | 完整 group bundle、共享前缀一次、reader mask、receipt | S5 |
| S5 冻结模型诊断 | 可答性、last-window、策略差异、自然奖励信号 | S6 条件满足后 |
| S6 T 小规模训练 | 真实非零更新与独立 dev 评测 | S7 |
| S7 奖励对照 | FS、与 FS 不等价时的 D；固定其他变量 | S8 按证据选择 |
| S8 后续研究 | 主动检索、统一 policy、动作 RTG、动态事实 | 独立配置与实验报告 |

### 19.1 S0 的具体动作

在实际仓库读 `AGENTS.md`、`STATUS.md`、现有交接和本文。用 `git status --short`、`git branch --show-current`、`git rev-parse HEAD` 和 `rg` 定位接口。不要重新 git init 或向项目内部重复 clone。

文档中的 `D:\Project\Age-Mem\AgeMem`、`/data/hjx/Age_mem` 仅是历史路径，需现场确认。已有未提交工作保留；必要时使用隔离分支/工作目录，不强制清理。

### 19.2 S1～S4 可连续推进

在用户授权实施后，执行者应完成所有可独立进行的本地工作。旧审计不可访问、Ray 不在本机或模型尚未就位，只阻塞依赖它们的验证，不阻塞 schema、builder、预算环境和组装接口。

### 19.3 S5 的继续/调整条件

先选 20～50 个冻结 debug/dev 历史，不用 test 调参。以下是决策规则，不是随意降低门槛的理由：

- Gold-support 也普遍答不出：优先缩短问题/证据难度、检查格式和 prompt，不启动长流 RL 试图补基础阅读能力。
- Last-window 与记忆方法基本相同：检查证据位置、尾部重复和原文旁路；自然结果如实报告。
- 自然轨迹全部得到相同 R：检查策略差异、grounder 和读写瓶颈；可以降低到 2C 诊断，不能往 reward 加随机噪声。
- FS/D 相同：保留等价报告，不运行重复臂；T vs FS 仍有独立研究价值。
- 语义 grounder 未通过：Terminal 可在环境/训练门禁通过后研究；FS/D 保持离线。

### 19.4 S6 的算力规模说明

先用小配置验证一次真实更新，再开展受预算约束的 pilot。若每步 2 个历史组、200 条历史，则一轮完整覆盖约 100 个训练步；12-step smoke 只覆盖至多 24 个历史实例，不能写成 200 历史训练完成。

可先在预先冻结的 24-history 子集跑 12-step smoke，并在 0/6/12 评测小 dev。参数变化且没有提升属于有效结果。随后是否训练完整 200-history pilot，依据 S5 信号与已有执行授权，不因“有 GPU”自动重复已关闭实验。

### 19.5 文档更新

每个阶段更新统一 implementation report：变更文件、真实命令、验证结果、失败项、未验证项、下一依赖。`STATUS.md` 顶部写当前 S 阶段并链接报告；历史段标为历史，避免多个“唯一下一步”竞争。

## 20. 必要测试与验收矩阵

只编写真正保护关键行为的测试。不要为了增加测试数测试文件存在、复制配置值或重复断言实现细节。

| 编号 | 必须验证的行为 | 验收 |
|---|---|---|
| D1 | 同种子同输入构建稳定 | JSONL/manifest 内容 digest 一致；时间戳独立处理 |
| D2 | 支持证据全部在输入 | 每个 query 的 ref 可解析且出现于 stream |
| D3 | fullwiki 缺证据 | 明确失败/重建，不静默可回答 |
| D4 | 切块与去重 | 句子来源不丢失，同名异文不错误合并 |
| D5 | split 污染 | QA/正文跨集合冲突被报告并按规则处理 |
| D6 | gold 隔离 | 改 answer/support labels 不改变阅读 observation；改 query 文本不改变阅读 observation |
| E1 | C 预算 | 实际 rendered prompt+输出预留从未越限 |
| E2 | B 预算 | 大正文、metadata、来源列表、UPDATE/restore 均无法绕过 |
| E3 | 原子 mutation | 无效 ADD/UPDATE/DELETE 不改变状态 |
| E4 | 输入可见性 | 每个 chunk 至少一次进入模型 observation；未来来源不可引用 |
| E5 | FIFO | 确定性移除、审计完整、不产生假 policy action |
| E6 | 快照分支 | 一分支的回答/记忆改动不能影响兄弟分支或父快照 |
| E7 | 原文旁路 | source registry、旧版本、全历史索引对 policy 不可读 |
| R1 | 语义与指针分离 | 正确指针+无关正文不得获得 support |
| R2 | 状态回退 | 存后删会丢失保留分；保留等价副本则不丢分 |
| R3 | once-only 与上限 | 重复 ADD/RETRIEVE/循环不增加归一化终态分 |
| R4 | 暴露口径 | 检索截断掉的证据不能记为暴露 |
| R5 | 多题聚合 | 改变问题顺序不改变平均奖励；m 变化不产生线性奖励膨胀 |
| R6 | Flat/DFA 等价 | 在等价 fixture 上完全一致，在定义的回退差异上可解释 |
| T1 | 组单位 | K=4,m=4 是 4 份记忆、16 分支，不是 16 份记忆 |
| T2 | 前缀唯一 | 同一阅读 action 在 actor batch 只出现一次逻辑训练样本 |
| T3 | mask/logprobs | 仅生成阅读 token 可训练；old logprobs 对齐且 policy version 一致 |
| T4 | 残缺组 | query 缺失的 bundle 不进入更新 |
| T5 | 零方差 | 优势为零，无跨组/跨题伪差异 |
| T6 | 奖励账本 | 不重复累计 branch 回报或各动作 RTG |
| T7 | 真实更新 | 非零信号 fixture/小模型产生非零任务梯度；纯零信号行为符合预期 |
| C1 | 兼容性 | 旧默认路径与冻结配置不变，新 scope 独立 |

### 20.1 真实验证要求

CPU tests 不冒充 GPU runtime PASS；SKIP 写明依赖。mock reader/规则轨迹只验证结构，真实自然轨迹和真实优化器结果分别报告。

新 gate 的测试清单先通过 discovery 建立 manifest，不预先编造“应有多少项”。旧冻结 318 gate 依其原契约执行；新测试不塞进旧计数以伪造通过。

### 20.2 轻量语义反例集合

至少包含：存入同义改写、正确来源却空正文、跨两句合并、引用未见来源、删掉唯一表示、删掉冗余副本、UPDATE 改成无关内容、检索只命中标题、检索结果被 token cap 截断、重复 ADD 后删除原记录。

这些反例用于找真实奖励错误，不要求自动机接受所有“答对但无证据”的轨迹。任务正确性与证据监督应分栏显示。

## 21. 实施风险与处理方式

| 风险 | 识别信号 | 处理 |
|---|---|---|
| 预算低于可用信息下的合理可达范围 | 特权压缩可行、公开策略长期无区分 | 报告信息条件差异，放宽 B/降低难度做诊断 |
| 基础工具协议没学会 | 非法/截断动作占多数 | 可新增独立工具 SFT 初始化；全部对照使用相同初始化 |
| reader 瓶颈 | 完整证据仍低 F1 | 先改诊断任务或回答协议，别归因记忆 |
| 全文旁路 | 不写入也能高检索覆盖 | 关闭实际 policy 全文入口，保留作资源放宽基线 |
| 来源奖励虚高 | ID 正确但内容错误 | 内容一致性审计，禁用未验收语义奖励 |
| 多题平均无差异 | 多种 M 但组内 R 接近 | 检查题目覆盖/可答性，报告 m 的真实作用 |
| 固定检索让逻辑冗余 | FS/D 完全同分同序 | 不重复训练；后续引入真正交互依赖 |
| token mask 错位 | loss 不变或前缀重复 | 精确 action/token join，真实梯度检查 |
| 文档历史状态冲突 | 多个“当前下一步” | 统一 STATUS 顶部、新报告单入口 |
| 长 rollout 成本失控 | 每步 token/reader 调用激增 | 先报告预算再运行；降低 debug 数量，不篡改评价分母 |

工具 SFT 如采用，示范可来自规则/教师，但必须作为独立 SFT 数据集；不能把规则动作伪装为当前 policy 的 on-policy 样本。

## 22. 完成定义与交付清单

### 22.1 第一轮工程完成

- [ ] 当前仓库状态和接口审计已记录。
- [ ] 新数据 schema、builder、split 检查和 manifest 可运行。
- [ ] 新环境对 C/B、来源可见性和快照隔离有关键测试。
- [ ] 单次阅读、多个独立问题的真实分支语义正确。
- [ ] 固定 reader 与 retriever 版本可冻结，不可访问全历史旁路。
- [ ] T/FS/D 重放入口可用，语义能力状态明确。
- [ ] 完整 K 组与共享前缀一次的训练接线已实现。
- [ ] 新配置、CLI、receipt 和报告可复现。
- [ ] 原实验配置、轨迹和 checkpoint 不受覆盖。
- [ ] CPU 与模型运行验证结果分开，未验证项不写 PASS。

### 22.2 研究结论完成

工程完成不要求一定得到正结果。研究结论需进一步提供：固定对照上的评测、组内信号分析、误差归因、预算和成本、样本/seed 限制，以及到底验证了存储学习、过程监督、逻辑依赖还是信用分配中的哪一项。

### 22.3 实施报告模板

```markdown
# Streaming Multiquery 实施报告

## 当前状态
- 当前 S 阶段：
- 实际 commit / dirty patch digest：
- 已完成：
- 未完成与阻塞：

## 配置与数据
- protocol / reward / grounder / reader / retriever 版本：
- 独立 history / history_family / QA / rollout / branch 数：
- C / B / alpha 分布：
- manifest 与配置 digest：

## 变更文件
| 路径 | 改动目的 | 兼容影响 |
|---|---|---|

## 实际运行命令与结果
| 命令 | 环境 | 退出码 | 产物 | 结论 |
|---|---|---|---|---|

## 验证
- CPU：
- 真实模型冻结诊断：
- 真实训练与参数变化：
- 未验证 / SKIP：

## 研究结果
- T/FS/D 是否产生有效差异：
- 方差、F1、证据覆盖与成本：
- 结论及不能支持的结论：

## 下一步
- 可继续执行的具体任务：
- 需要的模型/数据/运行权限及现有授权状态：
```

## 23. 提供给 Codex（GPT-5.6 Sol）的启动指令

以下内容供用户在实际仓库中明确要求实施时使用。不要把文件中任何代码块当成已经执行过的日志。

```text
请在当前 Age-Mem 仓库中实施 docs/streaming_multiquery_change_spec.md。

先读取 AGENTS.md、STATUS.md、现有交接文档和这份规范，核对当前 Git 状态与实际接口。
按新规范实现 streaming_multiquery_v1，保留旧三阶段协议和全部冻结实验产物。

优先连续完成 S0～S4 的本地代码、CPU 数据构建、关键测试和实施报告。
合理的模块划分和命名由你根据现有代码决定，不要为每个可逆的小改动反复请求确认。
缺少模型或旧审计轨迹时，继续完成不依赖它们的工作，并明确记录阻塞项。

重点落实：
1. HotpotQA 多样本重组为长信息流，按 tokenizer 控制长度，并先划分再拼接。
2. 问题在阅读结束前不可见，gold 永不进入 policy observation。
3. 每次调用严格满足 C，所有可恢复的长期记忆 payload 满足 B。
4. 每份阅读快照独立回答 m 道题；分支之间不共享回答或修改。
5. 首轮 ingest-only，固定 reader 和 retriever，只索引实际保留的记忆。
6. K 份记忆构建是 GRPO 的组；m 个问题不是 m 份独立阅读轨迹。
7. 共享阅读前缀在 actor batch 只训练一次，固定 reader token 不进 loss。
8. 核验 provenance grounding 的内容语义；来源 ID 正确不等于保存了事实。
9. 如果 Flat-state 与 DFA 等价，报告等价并跳过重复 GPU 实验。
10. 更新 STATUS 顶部和 implementation report，列出实测与未验证项。

不要自动重跑旧 E1/E3，不要覆盖旧 YAML、checkpoint 或冻结 318 scope。
模型运行、外部付费服务和远端写入按照本次会话已有授权处理；如仍需授权，
先给出已验证的配置、命令和预算，使确认成为最终执行前的具体步骤。

最后交付：变更摘要、文件列表、真实验证结果、关键指标、明确的剩余阻塞。
不要把代码存在、mock PASS 或 optimizer 被调用写成学习有效。
```

## 24. 延后功能及进入条件

| 功能 | 进入条件 | 新增实验要求 |
|---|---|---|
| 主动 RETRIEVE | 固定检索下已确认写入策略可评价 | 限定查询次数、标注 policy 动作、同预算对照 |
| 统一 policy 读写答 | ingest-only 训练与分支 loss 正确 | 独立联合训练基线、避免共享前缀重复 |
| 动作级 return/advantage | 自然增量奖励可靠、trajectory 基线明确 | 同奖励只改信用粒度 |
| 动态事实更新 | 时间/版本来源可验证 | 定义有效时间、冲突解决、旧事实仍可能历史有效 |
| 真实 Group Critic | 人工逻辑和 grounding 基线可信 | 独立 schema/质量审计、失败回退、真实调用成本 |
| MuSiQue 泛化 | HotpotQA 主流程冻结 | 原始 split 和证据适配，新数据不调旧 test |
| LongMemEval 会话评测 | 能处理 timestamp/session 与较长历史 | 遵循官方评测；改短历史则标为自定义版本 |

静态 HotpotQA 中暂时不强制每条轨迹包含 UPDATE/DELETE；“没有必要更新”可以是正确策略。也不强迫所有工具出现以满足统计表。

## 25. 来源与引用说明

### 25.1 用户提供的项目和论文材料

1. `Agentic Memory(1).pdf`：三阶段 episode、终局复合奖励与 step-wise GRPO。重点核对正文 §3.3～3.5、附录 A.3/C.4。
2. `GLARE(1).pdf`：符号状态、逻辑自动机与步骤奖励。正文 §4.5 与附录 Algorithm 1 的归一化差异已在本文说明。
3. `PROJECT_HANDOFF(1).md`：项目架构、数据契约、M/E 阶段和冻结规则。
4. `STATUS.md`：以 2026-09-14 顶部当前状态为准，后部历史结果保留但不覆盖当前说明。

### 25.2 官方外部资料

- [HotpotQA 官方主页](https://hotpotqa.github.io/)：数据下载、distractor/fullwiki 设置和官方评测。
- [HotpotQA 数据下载与格式](https://github.com/hotpotqa/hotpot#data-download-and-preprocessing)：fullwiki dev 可能缺证据、test 无标签、question/answer/context/supporting_facts 字段。
- [HotpotQA 官方评测脚本](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)：答案与支持事实评测逻辑。
- [MuSiQue 官方仓库](https://github.com/stonybrooknlp/musique)：多跳 QA 数据和评测入口。
- [LongMemEval 官方仓库](https://github.com/xiaowu0162/LongMemEval)：长期会话记忆能力、数据结构和自定义历史构建。

外部资料在 2026-09-14 的本轮讨论中核对。执行者下载或更新依赖时仍需记录实际 revision。本文的 C/B、数据量、m/K、阶段划分、lambda 与训练拓扑是本项目拟议设计，不是上述论文已验证的标准配置。
