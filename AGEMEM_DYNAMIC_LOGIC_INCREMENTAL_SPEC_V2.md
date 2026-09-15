# AgeMem 动态记忆与逻辑监督：v2 增量实施技术文档

> 版本：v2.0
> 日期：2026-09-15
> 接收方：在用户现有 Age-Mem 仓库工作的 Codex（GPT-5.6 Sol）
> 前置条件：用户已执行上一版 `AGEMEM_STREAMING_MULTIQUERY_CHANGE_SPEC.md`。执行深度、当前代码和训练结果须由接手者核验。
> 文档性质：下一轮增量实施规范，不是已完成的实验报告。
> 建议仓库路径：`docs/dynamic_memory_logic_incremental_spec_v2.md`。

## 0. 执行摘要

### 0.1 这次做什么

复用 v1 已实现的流式输入、双预算、多问题快照、训练 buffer 和日志。在其上增加**具有时间与版本语义的动态记忆任务、可信语义判定、可回退的状态监督，以及公平的奖励对照**。

首要研究问题：

> 在未来问题未知、单次上下文和持久记忆容量受限的条件下，对事实来源、时间有效性和记忆保留过程的显式监督，是否比终局奖励及同等信息量的终态监督更有效地训练记忆管理？

本轮先研究固定回答器下的记忆构建。主动检索、联合训练、自动 Logic Critic 和新动作级优化器不是首轮前置条件；若 v1 已实现，则保留为独立可选配置。

### 0.2 关键决定

1. 不重建 v1，不重新跑已关闭实验，不把旧上传的 `STATUS.md` 当成用户执行后的现状。
2. 新协议命名为 `streaming_dynamic_multiquery_v2`；旧 `streaming_multiquery_v1` 保留。
3. 主任务先覆盖“当前状态、历史状态、多次更新、跨实体时间连接”；更复杂的追溯纠错单列后续扩展。
4. 自然语言内容与程序生成的事实时间线成对生成；隐藏答案由独立 Oracle 求解，不能流入策略观察。
5. 先固定语义判定和奖励，再比较训练；不要同时更换 grounder、reader、数据与优化算法。
6. **强规则实现与 DFA 实现若语义相同，奖励必须相同。只训练其中一个，不为了制造创新而削弱规则基线。**
7. 先验证过程监督是否有价值，再验证规则生成、形式化编译或信用分配是否提供额外贡献。
8. 所有“更准确、泛化、更高效”的表述，都须对应实际指标和对照。运行成功不能代替方法有效。

### 0.3 执行权限与持续推进

用户要求按本文实施后，接手者应持续完成已授权的代码、数据、CPU 检查、配置和报告，不在每个阶段重复询问确认。用户会话已有的 GPU、远端与外部服务授权继续适用。本文不替代实际权限控制；缺少某项资源时完成其余独立工作，并记录具体阻塞。

当前请求仅生成本规范；本文没有修改用户项目或启动训练。

## 1. 从 v1 迁移：先核验，再增量改造

### 1.1 P0 当前实现盘点

先读取当前工作区 `AGENTS.md`、`STATUS.md`、v1 实施报告、实际配置与最近运行 receipt；检查 Git 状态。使用 `rg` 查找，不假定历史路径仍存在。记录：

| 核验项 | 必须留下的证据 |
|---|---|
| 当前仓库 | 根路径、commit、未提交修改概述、当前分支 |
| v1 完成范围 | 已实现／已验证／未验证／缺失，逐项区分 |
| 数据与模型 | manifest、tokenizer/model revision、checkpoint 身份 |
| 双预算 | 实际 C/B、模板计数口径、旧版本正文是否可恢复 |
| 多问题分支 | K/m、共享快照隔离、问题顺序无污染 |
| 训练接线 | 整组消费、前缀只训练一次、reader 是否冻结 |
| 奖励 | 实际 T/FS/D 公式、grounder、归一化和 reward version |
| 已有结果 | 原始结果位置、样本量、seed、失败率、是否可重放 |
| 当前成本 | 已知显存峰值、吞吐、GPU/模型是否可用 |

输出 `docs/v2_current_implementation_audit.md`。没有某项结果就写未验证，不要求用户重新上传才能开始本地盘点。不把旧 v1 的“待实现”当成当前缺失，也不把用户“已执行”自动解释为所有实验通过。

### 1.2 变更映射

| v1 模块 | v2 处理 |
|---|---|
| 环境、chunking、C/B 限制 | 复用；增加动态事件语义和版本字段 |
| 快照、K×m 问答分支 | 复用；加强时间查询与跨问题隔离 |
| 静态 HotpotQA 数据 | 保留为回归与静态对照，不重写标签 |
| 来源 grounding | 复用接口；增加正文蕴含、有效期和版本校验 |
| v1 T/FS/D | 原封保留；新增 namespaced v2 reward profiles |
| 固定检索器、固定 reader | 主实验继续使用 |
| shared-prefix loss 和 GRPO | 主实验继续使用；不顺带重构优化器 |
| v1 action-RTG／联合训练 | 若已完成则单独记录；首轮主比较不启用 |
| M7 Mock Critic | 保留工程验证身份，不视为真实 Critic 结果 |
| 历史实验与 checkpoint | 归档引用；不覆盖、不静默 resume |

本文取代 v1 中“动态事实更新延后”的排期；不修改 v1 的数学定义和历史结果。新配置具有新的 protocol/schema/reward 版本。若发现 v1 泄漏或训练接线错误，先修复并标记受影响结果，不沿用其有效性结论。

## 2. 相关工作与贡献边界

| 工作 | 已有贡献中与本项目重合的部分 | v2 必须比较或澄清的部分 |
|---|---|---|
| [Mem-α](https://arxiv.org/html/2509.25911v1) | 分块构建记忆，以后续问答收益训练写策略，固定检索和回答模型 | 本项目的固定 reader 训练范式已有先例；差异应落在监督语义 |
| [UMA](https://arxiv.org/html/2602.18493v2) | 隐藏未来问题、多问题复用、记忆与问答分组优化 | 多问题平均奖励和联合训练不作为新算法；其原文检索与本文 memory-only 条件不同 |
| [Mem-T](https://arxiv.org/html/2601.23014v2) | 树引导的记忆检索奖励与构建信用分配 | 不声称首次提供细粒度记忆奖励或过程归因 |
| [Memory-T1](https://arxiv.org/html/2512.20092v1) | 检索中的证据 grounding 和时间一致性奖励 | 不把时间奖励本身作为首创 |
| [Memory-R2](https://arxiv.org/abs/2605.21768) | 相同中间状态上的局部重采样与全局优化 | 后续若研究局部反事实信用，须作为直接对照 |
| GLARE（用户上传） | 组级逻辑分析、符号状态与自动机奖励 | 复用基础方法；新增贡献必须解释记忆场景的具体变化 |

上述工作链接见第 24 节。这里不是穷尽性新颖性证明。

### 2.1 本轮可以验证的候选贡献

- 记忆专用的状态监督：对当前／历史有效性、信息丢失与合法替代表示给出一致判定。
- 可靠的奖励落地：避免正确来源指针配错误正文、改写漏判、失效版本误奖。
- 在相同监督信息和预算下，证明检查整个保留过程的奖励是否优于只检查终态。
- 若之后实现自动规则生成：证明形式化校验与编译是否降低错误奖励，并迁移到新任务结构。

### 2.2 不得提前写入论文结论

- “首次小上下文流式多问题记忆训练”。
- “DFA 比任意程序状态跟踪更有表达能力”。
- “出现 RETRIEVE 表明模型因果上使用了证据”。
- “加入状态势分必然保持原任务最优策略”。
- “比 terminal 好即可证明逻辑结构有用”。
- “新增动态任务就证明方法新颖或达到 SOTA”。

## 3. 实验因素与实施范围

### 3.1 必须解耦的因素

| 因素 | 首轮固定值 | 后续可变值 |
|---|---|---|
| 信息流 | 动态 v2，同一数据 manifest | 静态迁移、长度外推 |
| Policy | 同一 1.5B 或 4B 初始化 | 另一规模复验 |
| Reader | 固定版本，固定解码 | 共享策略联合训练 |
| Retriever | 冻结词法检索，仅索引保留记忆 | 主动检索／混合检索 |
| Grounder | 经审计并冻结 | 泛化到自由摘要 |
| 训练优势 | v1 已验证的轨迹级优势 | 显式动作回报算法 |
| 奖励 | T、终态监督、过程监督 | 自动规则、局部信用 |
| 环境预算 | 相同 C/B/调用额度 | 预算敏感性 |

### 3.2 首轮交付与后续项

**必须交付：** P0 盘点、动态数据生成器与独立 Oracle、语义 schema、版本与来源校验、强终态规则评分器、过程评分器及等价 monitor、CPU 重放、训练 profile、验收和报告接口。

**有条件推进：** 真实轨迹诊断与有授权资源下的 T/END/LIFE pilot。

**单独排期：** 自动 LLM Critic、追溯纠错、学习主动检索、动作级优势新算法、全量外部基线复现。不能用这些较大项目阻塞首轮基础交付。

## 4. 协议不变量

1. 阅读策略只看到公开信息流、当前合法记忆和预算内近期上下文；不知道具体未来问题。
2. 主压力档按冻结 tokenizer 的正文 token 计算 `alpha=L/C∈[5,10]`；1C/2C 仅作诊断档。
3. 每次调用满足 `tokens(rendered_prompt)+max_new_tokens<=C`；不能只限制输入正文。
4. 可恢复记忆的正文、自由字段、来源引用和自定义键均计入 B。
5. 被删除或覆盖的旧正文只在审计侧保留；策略不能通过旧版本 API、日志、source_id、缓存或错误信息取回。
6. 读取结束冻结 `(M, C_tail)`；m 道题各自从相同快照开始，不共享前一题答案、检索结果、KV cache 或 scratchpad。
7. 同一个 history 的 K 份记忆使用同一组问题及同一预算。共享阅读前缀只进入一次 actor loss。
8. Gold、事件真值图、奖励 AP、Oracle 支持集和审计标签都只在数据／奖励侧。
9. 可以在事后用隐藏问题和答案计算训练奖励；这些信息不得成为阅读期反馈或可见工具返回。
10. C_tail 仍然合法；分别报告来自记忆和来自尾部上下文的证据。增加“全部关键证据已退出尾部”的预定义测试子集。

## 5. 动态事实的时间语义

### 5.1 先采用简单、明确的公开世界规则

首轮数据是受控、单一可信来源的事件流。相同实体关系的状态变更有明确生效时间。用户知道任务涉及事实变更；策略可以知道通用规则，但不能得到 Oracle 生成的状态表。

时间统一采用离散整数时间，公开文本可渲染为日期。有效区间使用左闭右开 `[valid_from,valid_to)`；空终点表示持续有效，只有“当前值已知”时才可作此解释。

区分两个时钟：

- `observed_at`：事件在流中被读到的时刻／chunk 位置。
- `effective_at`：事件在任务世界中开始生效的时刻。

首轮单调配置要求生效时间随观察顺序不下降，且不包含撤回历史事实的追溯纠错。未来生效、乱序到达和追溯纠错分别使用新 profile，不能默认套用“最后收到的值总是正确”。

### 5.2 状态更新不等于旧事实为假

“周一改为李四”不会使“周一之前由张三负责”变成错误。世界版本更新后，旧版本仍可能对历史问题有价值。

关系声明必须区分 `single_valued` 与 `set_valued`。首轮只要求前者；集合添加／移除作为独立扩展。`unknown` 表示无法从可见证据确定，不等同于明确的 `none`。

### 5.3 明确的最小示例

| 观察顺序 | 公开文本 | 隐藏世界含义 |
|---|---|---|
| 1 | 3 月 1 日起，北辰项目由张三负责。 | 张三从时间 1 有效 |
| 2 | 3 月 8 日起，北辰项目负责人改为李四。 | 张三区间关闭，李四开始 |
| 3 | 3 月 12 日起，北辰项目负责人改为王五。 | 李四区间关闭，王五开始 |

阅读结束可问当前负责人、3 月 5 日负责人、3 月 10 日负责人、李四何时开始负责。问题独立回答。

合法记忆可以是时间线、带有效期的条目、保留的事件序列或忠实摘要；无需使用 Oracle 的内部 schema。保存完整事件序列且在问答时推导区间同样正确，不能因为没有显式 `valid_to` 而扣分。

### 5.4 不强制工具脚本

可用 UPDATE 改写一条时间线，也可 ADD 一个新版本并保持事件顺序；两者均可有效。不得规定“必须 UPDATE 才有奖励”“必须先存事实 A 再存 B”“必须 RETRIEVE 两次”。

## 6. 数据集构造规范

### 6.1 三层数据

| 数据层 | 目的 | 允许的结论 |
|---|---|---|
| D0 规则 fixture | 精确检验语义、预算、monitor、奖励 | 代码与数学契约正确 |
| D1 受控动态自然语言 | 程序时间线＋文本渲染，用于训练 pilot | 在受控动态任务上的学习效果 |
| D2 外部自然数据 | 从合法训练／评测数据适配 | 独立数据分布上的泛化 |

D2 优先复用 v1 已有数据与官方开放的记忆数据格式。没有完整时间标注的自然样本只用于可支持的答案评测，不能补造 Oracle 标签后称为官方金标。

### 6.2 D1 生成顺序

1. 冻结任务族、实体词表、关系类型、事件生成规则、渲染器和随机种子。
2. 先按 `history_family_id` 划分 train/dev/test，再生成同族变体；同一时间线的改写、不同长度填充和配对错误版本始终同 split。
3. 生成事件时间线及关系类型，独立 Oracle 求解状态和历史查询。
4. 将事件渲染成自然语言。首轮用确定性模板；若引入 LLM 改写，保留映射、冻结生成模型，并对改写是否改变事实单独审计。
5. 添加同 split 的背景、重复信息、其他实体的有效更新。背景不能全是无意义填充；记录唯一事实数、更新数、实体数和证据跨度。
6. 按 tokenizer 和句子边界组块，将目标 L/C 放在指定区间；更长文本不能靠无限重复同一句凑数。
7. 构建候选问题池及答案、参考证据、合法替代证据表示；问题在策略读完后才揭示。
8. 用独立验证器重新计算答案；检查时间边界、未知答案、来源可见性与支持集。
9. 产生公开输入与私有标签的分离文件、manifest 和数据统计。

不要用被测试的 policy 挑选“它恰好会错”的 test 样本。难度控制由预先定义的事件结构和预算决定。

### 6.3 首轮任务族

| task_family | 必须构造的能力 | 样本特点 |
|---|---|---|
| `current_state` | 用最新生效值回答 | 至少两次值变化，旧值词频可更高 |
| `historical_state` | 保留历史值及时间定位 | 当前问题与历史问题共用历史 |
| `multi_update` | 区分多个中间版本 | 包含 A→B→A，不能只按 value 去重 |
| `temporal_join` | 同一查询时间连接两个关系 | 两跳关系更新时间错开 |
| `retention_under_budget` | 合并／删冗余同时保留有效语义 | 重复描述、有限 B、多实体竞争 |

每条历史候选池建议 8～12 题，至少满足 m=4。可重叠 task_family，但统计题型主标签。训练每次按固定规则抽题；测试问题池固定。题型抽样概率作为任务分布的一部分记录，不能因方法不同而变化。

对于 `temporal_join`，例如“某日期项目负责人所属团队”，必须按同一日期求出负责人和其团队，不把两个关系各自的最终值直接拼起来。

`retention_under_budget` 首轮检验记忆有用性，不要求某种删除动作。删除冗余副本且保留等价表示不应受罚。

### 6.4 时间可回答性与证据范围

每个问题保存 `answer_available_at`：该题最终参考答案及必要限定信息首次能由已观察前缀确定的 chunk。未来的变更可能改变某个问题答案时，不能过早将该问题标为可评分。

- “读取结束时当前负责人”需看到最后一次相关生效变更后，才开始对最终版本评分。
- 历史点查询必须有证据支持该时间点的状态；首轮单调规则下，后续事件越过查询点后可确定它。
- 结束时仍缺乏依据的样本明确标为未知／不可回答，不能把 Oracle 的未来世界真值当成策略应该知道的答案。

首轮 D1 主训练池仅纳入 `answerable=true` 且有非空合法证据义务的问题。未知／不可回答样本保留在独立诊断集中，先只评估答案与拒答行为；它们不能用空支持集算满覆盖。若后续要训练拒答的过程监督，另定义“证据不足”的验证器、奖励和数据 profile。

### 6.5 预算可行性

分别构造：

- `feasible_control`：存在一份对候选问题池通用的公开规则摘要，在 B 内保留全部必要语义；用于验证能否学习。
- `selection_pressure`：通用完整存储可能超过 B；研究有限预算下的取舍，不要求所有题满分。

可行性参考记忆由公开时间线规则构建，不用每次抽到的具体问题来选择条目。另行提供看过问题的 per-query gold-support，只作信息特权诊断。

报告每种参考的实际 token、是否符合 B/C。不存在可行表示时，不能把达不到 Oracle 分数归因于训练失败。

### 6.6 初始规模与数据比例

建议起步值，可根据 P0 已有能力调整并在采样前冻结：

| 集合 | 独立 history family 数 | 用途 |
|---|---:|---|
| D0 fixture | 40～80 | 确定性验收，不作统计学习结论 |
| D1 train pool | 400 | CPU 可先完整生成，pilot 先用其中 100 |
| D1 dev | 80 | 诊断、选 checkpoint、选 lambda |
| D1 test | 120 | 最终冻结比较；不边训练边查看 |
| 自然语言语义审计 | 约 200 个去重判定单元起步 | 来源／版本／改写等分层 |

训练主长度 5C；7.5C/10C 作为预先划分的外推或后续混合档。一个 family 的不同长度变体必须一起分配。首轮先只用动态训练池，另用 v1 静态 held-out 评估迁移；如需混合训练，再单列配置，所有方法采用相同比例。

## 7. 数据与日志 schema

### 7.1 公开流记录

```json
{
  "schema_version": "dynamic_stream_v2",
  "history_id": "h_demo",
  "history_family_id": "family_demo",
  "chunks": [
    {
      "chunk_id": "chunk_a",
      "observed_at": 0,
      "text": "3 月 1 日起，北辰项目由张三负责。",
      "source_refs": ["src_a"]
    }
  ]
}
```

示例只展示字段，不是长度达标的训练样本。来源 ID 为固定结构的不透明定位符，不编码答案、任务难度或 gold 身份。公开时间信息来自文本，不能由私有标签自动补到 observation。

### 7.2 私有事件记录

```json
{
  "event_id": "gold_event_a",
  "entity": "北辰项目",
  "relation": "负责人",
  "value": "张三",
  "operation": "set",
  "observed_at": 0,
  "effective_at": 1,
  "source_ref": "src_a",
  "source_text_hash": "computed_at_build_time"
}
```

私有文件可以用事件图推导有效区间；首轮单调状态由下一次相关变更关闭旧区间。策略无需输出这些内部字段。

### 7.3 查询与证据标签

```json
{
  "query_id": "q_demo",
  "question": "3 月 5 日北辰项目由谁负责？",
  "task_family": "historical_state",
  "query_time": 5,
  "answers": ["张三"],
  "answer_available_at": 2,
  "support_alternatives": [["gold_event_a", "gold_event_b"]],
  "answer_type": "entity",
  "answerable": true
}
```

只有 `question` 及不含答案的公开格式说明进入 query prompt。`answer_available_at` 和 support alternatives 由求解器计算，不手抄示例数值。证据集合是参考，不一定穷尽自然语言中的所有合法推导。

### 7.4 运行记录扩展

复用 v1 的 history/group/read_rollout/snapshot/query/action 身份；增加：

- `protocol_version`、`time_semantics_version`、`world_oracle_version`。
- `memory_revision_before/after`、`content_hash_before/after`。
- `mutation_type`、`mutation_success`、`replaced_revision_ids`。
- `source_validation`、`semantic_validation`、`temporal_validation`。
- `checkpoint_index`、`eligible_queries`、`coverage_by_query`、`conflict_by_query`。
- `reward_profile`、`monitor_version`、`logic_spec_hash`、`private_label_digest`。
- `actor=policy|environment|reader`；固定检索事件不能变成 policy 动作。

所有奖励侧字段保存在私有 sidecar，不拼回下一次模型观察。数据加载器采用明确白名单投影，不能用“去掉几个已知 gold 字段”的黑名单实现。

## 8. Memory Store、版本和工具语义

### 8.1 三类对象必须分开

| 对象 | 内容 | 策略是否可读 |
|---|---|---|
| World Oracle | 完整真值事件、推导区间、正确答案 | 否 |
| Agent Memory | 模型实际保存的正文与元数据 | 是，在 C/B 内 |
| Audit Ledger | 过去版本、动作结果、奖励判定 | 否 |

“可追踪来源”不等于“可以通过来源随时取回原文”。source registry 可用于奖励核验，但不得成为额外 RAG 索引。

### 8.2 工具操作

- ADD：生成新 memory ID，保存 policy 提交内容；来源声明须指向已观察内容。
- UPDATE：原子替换指定当前版本。新内容和来源声明重新验证，不能自动继承旧语义结论。
- DELETE：移除策略可读取表示；审计保留旧 revision 不代表它仍有保留分。
- SUMMARY／MERGE：若 v1 支持则复用；合并后重新验证正文，多个来源可合并但不得制造不受支持的关系。
- RETRIEVE：仅从当前活动记忆检索，返回具体 revision 和实际展示文本；超 C 截断后的未展示部分不计暴露。
- NEXT：前进到下一 chunk，不因跳过写入而强行处罚；是否损失必要语义由状态分体现。

原本可执行但语义错误的 ADD/UPDATE 不要由私有 Oracle 代替策略修正或拒绝。例如来源真实但值错误：环境可以接受该条目，奖励侧给出失败判定；工具不能返回“正确答案其实是李四”。纯公开结构错误、无效 ID、超预算可按既有公开规则拒绝。

### 8.3 版本与预算

允许在一个活动条目内保存多个历史版本，只要实际序列化内容计入 B。覆盖后旧内容默认不可读。恢复历史如果未来开放，须重新计入 B 且符合公开工具契约。

内部版本计数和定长 hash 可以作为控制字段；任意可读自由文本、来源列表和可检索 metadata 不能免费存储。索引不能缓存已删除正文；snapshot restore、retriever cache 和 multi-query cache 均按 revision/hash 隔离。

失败操作不改变记忆，也不推进 memory revision。批量操作有部分成功语义时必须逐项记录，不能用一个总 success 遮盖部分失败。

## 9. Grounding：来源正确、正文正确、时间正确分别判定

### 9.1 三重判定

1. **来源有效**：对应公开内容已被读取、定位和 hash 匹配。
2. **正文支持**：实际记忆内容表达了被声称保留的关系，允许忠实改写和组合推导。
3. **时间适用**：该表示足以支持查询的时间语义，且没有把历史值误写成当前值。

检索暴露另在实际最终 prompt 上运行语义判定，不使用完整存储正文替代。记忆中正确、展示时丢失了日期限定，暴露判定应反映损失。

### 9.2 判定接口

```python
class GroundingDecision:
    label: str             # supported | unsupported | unclear
    source_ok: bool
    content_ok: bool | None
    temporal_ok: bool | None
    evidence_revision_ids: list[str]
    reason_code: str
    validator_version: str
```

Oracle reference IDs 仅奖励侧使用。优先评估整个相关保留集合；单个条目不够，但多个条目合起来足够时应认定支持。不能要求每条摘要重复全部来源才能给分。

### 9.3 实现路径

| 模式 | 方法 | 使用边界 |
|---|---|---|
| `controlled_semantics` | 独立解析器与事件求解器检查受控文本或 policy 自写结构化声明 | D0/D1 起步；明确不代表自由语言鲁棒性 |
| `extractive_control` | 公开证据引用＋实际保存正文＋时间语义核验 | 可达性与 grounding 诊断 |
| `audited_free_text` | 冻结语义判定器＋去重人工审计 | 审计过关后才用于自由摘要训练 |

如果结构化写入模式由模型生成 claim 字段，可以训练，但应报告工具 schema 提供的额外结构。不能由环境读取 hidden world 后自动给模型生成正确 claim。

不要把同一段生成逻辑复用成“独立验证器”后声称交叉验证。至少使用不同实现路径求解，例如事件扫描生成标签、时间区间求交验证标签。

### 9.4 审计样本与判定阈值

复用 v1 审计流程，新增：旧／新版本混淆、历史有效、无日期摘要、正确指针错误正文、重复副本删除、A→B→A、两跳时间错配、检索截断。

统计 source/content/time 三部分，以及联合支持判定的 precision/recall/F1。按去重样本和 history family 报告；不把同一句重复 30 次当 30 个独立成功。

首轮建议预冻结目标：联合支持 precision≥0.95、recall≥0.80，unclear≤0.10；同时报告区间和关键类别样本量。这些是工程准入目标，不是已有实测结论。不能看结果后放宽阈值而继续沿用原实验名。

`unclear` 不能静默当负例，也不能通过丢弃困难 rollout 选择性训练。受控主实验应能确定评分；自由文本出现不确定时执行固定复核流程。仍未解决则整个组仅用于诊断，不将其残缺奖励送入语义训练；报告排除率和选择偏差。若不确定率高，暂停自由文本语义臂，继续受控模式和 terminal 基线。

## 10. 可验证状态与逻辑 monitor

### 10.1 使用语义状态，不使用工具名称充当里程碑

参考谓词：

| 谓词 | 意义 |
|---|---|
| `observed(source)` | 来源已在公开前缀出现 |
| `supported(rep, fact)` | 当前保留表示忠实支持某事实 |
| `time_applicable(rep, query)` | 表示适用于查询时间 |
| `recoverable(query, memory)` | 记忆支持足以回答该题的合法推导 |
| `ambiguous(query, memory)` | 同一查询时间存在未区分的冲突值 |
| `exposed(query, prompt)` | 实際问答 prompt 含适用证据 |

以上 query 可为奖励侧事后查询；不意味着阅读策略看见问题。命名和实际 schema 由执行者统一，不在不同模块用“valid”分别代表来源、正文和时间三种概念。

### 10.2 允许退回的状态

每个参考证据义务可使用 `absent / partial / supported / conflicting`，问答分支另记录 exposure。版本更新、覆盖、删除和压缩都能使状态上升或下降。

- 删除最后一份有效表示：支持状态丢失。
- 删除重复副本：如果剩余表示足够，状态保持。
- 保存新版本但丢失必要历史限定：当前题可能进展，历史题可能退步。
- 添加正确限定或保留另一种合法表示：可消除冲突。
- 无效工具调用：状态不变；不能拿重复调用刷进展。

不要把历史“曾达到 supported”作为永不失效的接受状态。保留 `ever_supported` 只能用于诊断，不能冒充当前有效性。

### 10.3 逻辑层的实现边界

采用“数据语义层＋有限控制状态”的组合。无限实体和日期由语义层处理；每个有界 episode 根据事件/查询实例化有限 monitor，不把所有实体、日期组合展开成指数级全局 DFA。

工具驱动的 transition 负责更新当前状态，chunk checkpoint 负责评分，query branch 负责 exposure。复杂历史义务必须列出所需状态变量和重放规则；不能只写一句 LTL 公式却没有可执行语义。

### 10.4 等价规则实现是必须的强对照

实现两个后端读取相同 GroundingDecision、相同 checkpoint 和相同 reward spec：

- `direct_state_evaluator`：直接规则求值。
- `compiled_monitor_evaluator`：通过 monitor 的转移与输出求值。

在首轮手工规则配置下，两者应逐事件状态、逐 checkpoint 分数、最终奖励完全一致。它们用于交叉校验，不是两个应当打出不同分数的学习方法。

若不同，默认按 bug 处理，先定位再训练。只有两份明确不同的监督规范，才可能产生可解释差异。手写程序理论上可以实现相同自动机；论文不得宣称自动机天然压过这种等价实现。

## 11. 奖励规范：终局、终态、保留过程分开

本章新增 `dynamic_reward_v2`，不更改 v1 的 FS/D 含义。首轮使用固定 reader，仅训练 ingest policy。

### 11.1 公共答案奖励

对同一历史的第 k 份记忆、问题 j：`F[k,j]∈[0,1]`。

- 实体／时间／状态查询主用 canonical exact match，日期和单位按预定义解析规则归一化。
- 集合答案使用预定义集合匹配；多跳实体答案仍用 EM。
- 自然语言数据按该数据官方指标，不将不同数据集的原始指标混在同一表头。
- D1 主分数使用 EM，token F1 可作辅助；解析失败按答错统计并单列。

$$R_k^{task}=\frac{1}{m}\sum_{j=1}^{m}F_{k,j}.$$

每个 history 先对问题平均，再跨 history 平均。工具次数与成本先用硬预算约束，额外 format/cost reward 默认关闭，避免干扰归因；如保留 v1 已验证项，所有臂必须完全相同并记录公式。

### 11.2 证据覆盖与歧义

令每题参考支持方案为若干合法证据组合。对每种方案计算实际保留语义的比例，再取最大值，得到 `coverage_M[k,j,t]∈[0,1]`；完整合法摘要或推导表示经 grounder 确认后可覆盖对应义务，不要求逐字复制来源。

令 `conflict_M[k,j,t]∈{0,1}` 表示存在会影响该题的、未被时间限定区分的矛盾声明。不同时间区间取不同值不算冲突。对于受控任务，时间未指明且仍可从事件顺序唯一恢复的表示，不应误判为冲突。

定义：

$$u^M_{k,j,t}=coverage^M_{k,j,t}\,(1-conflict^M_{k,j,t}).$$

从当前记忆实际进入 query prompt 的 payload 同理得到 `u_E[k,j]`。尾部原文覆盖单列，不混入 memory-only 的 u_E；任务 F 正常使用完整合法 prompt。

这是一种显式辅助目标，不是回答正确率的无偏估计。参考方案不完备的自然任务要通过语义补充或报告限制，不能把所有非参考解法判错。

### 11.3 终态分 END

$$U_k^{end}=\frac{1}{m}\sum_j\left(\frac12u^M_{k,j,T}+\frac12u^E_{k,j}\right).$$

END 使用全部相同的来源、版本、有效期与冲突信息，是**强时间状态基线**。不能将 END 做成只数 ADD 或完全不理解历史时间的弱基线。

### 11.4 保留过程分 LIFE

只在固定的外部 chunk 结束后评分，不在每次工具调用后累积正分。设 chunk checkpoint 为 t，定义 eligibility：

`eligible[j,t] = (t >= answer_available_at[j])`。

分母由历史和抽到的问题决定，和 policy 的动作数、是否成功无关。每个 checkpoint 中，策略已经没有机会再看该块原文之后的记忆状态参与评分。实际结算点为 chunk 提交并完成公开环境的边界处理之后；若边界触发合法 STM 清理，按清理后的持久记忆 M 评分。

$$n_j=\sum_t eligible_{j,t},\qquad V_k^{retention}=\frac1m\sum_j\frac{\sum_t eligible_{j,t}\,u^M_{k,j,t}}{n_j}.$$

先在每题的 eligible checkpoint 内平均，再对 m 题平均，避免较早可回答的问题仅因评分机会更多就获得更大权重。所有主训练问题必须满足 `n_j>=1`。某题分母为零时标记 `no_retention_opportunity`，不偷偷除以 1、不从 m 中移除该题；该 history 配置进入诊断并停止作为语义训练输入，而非给模型记一次失败。

定义过程分：

$$U_k^{life}=\frac12U_k^{end}+\frac12V_k^{retention}.$$

END 与 LIFE 均在 [0,1]。LIFE 明确偏好“在可确定后保持可恢复”，而 END 只看最终结果。这是可检验的额外目标，不声称对原始终局目标最优策略不变。

**边界：** 任意两个 checkpoint 之间的短暂删除又恢复，如果不影响边界状态，本 profile 不处罚。所有事件仍被 monitor 记录，但不虚称奖励捕捉了每个瞬时错误。若以后需要更细粒度，必须固定与动作数无关的采样规则并新增 profile。

### 11.5 主实验臂

| ID | 奖励 | 所检验内容 |
|---|---|---|
| `V2_T` | R_task | 终局学习基线 |
| `V2_END` | R_task + lambda·U_end | 时间／版本语义辅助监督是否有用 |
| `V2_LIFE` | R_task + lambda·U_life | 额外保留过程监督是否优于终态监督 |
| `V2_LIFE_MONITOR` | 与 V2_LIFE 完全相同，改用编译后端 | 只作离线等价验证，默认不重复 GPU 训练 |

起步 lambda=0.25，dev 可比较 0.1/0.25/0.5 中预先选定的少量值；给 END/LIFE 相同调参预算。只凭一个 lambda 失败，不得断言整个方法无效；也不能无限调参直到 test 有提升。

初始目标是验证 V2_END 对 V2_T、V2_LIFE 对 V2_END。主结果都按未加 shaping 的任务 EM/F1 判断，不能仅展示训练奖励升高。

### 11.6 具体数值例：终态相同，过程不同

专用 fixture：两道题、两个 eligible checkpoint，两条轨迹最终 memory 与 exposure 都完整，且最终两题均答对。

| 轨迹 | checkpoint 1 的两题 u_M | checkpoint 2 的两题 u_M | U_end | V_retention | U_life |
|---|---|---|---:|---:|---:|
| A | [1,1] | [1,1] | 1 | 1 | 1 |
| B | [0,1] | [1,1] | 1 | 0.75 | 0.875 |

B 必须有合法恢复路径，例如第二个 checkpoint 的公开块重述了缺失事实；不能从隐藏日志凭空恢复。lambda=0.25 时，T 均为 1，END 均为 1.25，LIFE 分别为 1.25 和 1.21875。

这个例子只证明两个辅助目标有差异。B 最终同样答对，所以差异不自动等于更优任务信用；必须用真实训练和中间不可见查询评测检验这种偏好是否有益。

### 11.7 不用重复进展奖励刷分

不可在每次 ADD/UPDATE/RETRIEVE 直接加固定正奖励。重复写入、删除后重建、循环检索不能使上述状态分超过其范围。

如记录状态增量，只作解释日志，必须包含退回的负增量，并验证累计等于末态减初态。不能把这些增量又叠加到已经计入的 END/LIFE 总分，造成双计。

## 12. 训练接线与算法范围

### 12.1 首轮沿用 v1 已验证的 ingest-only GRPO

同一历史生成 K 份独立记忆，各回答 m 个问题，先形成每份记忆的 R_k，再在 K 个 R_k 之间归一化：

$$A_k=\frac{R_k-mean(R)}{std(R)+\epsilon}.$$

使用 v1 锁定的总体／样本标准差口径。A_k 广播到该份阅读轨迹的 policy 生成 token。不是动作级信用分配。

保留 v1 的：整组消费、共享前缀只训练一次、reader 输出不进 actor loss、old_logprobs 身份、按 history/rollout 的长度归一化和 KL 规则。不要在 reward profile 改动时顺便更换这些因素。

### 12.2 chunk checkpoint 是奖励事件，不是额外策略动作

固定检索、Oracle 检查、monitor 转移不能产生可训练 token。每项评分按 action_id／checkpoint_id 关联，不按裸 timestep 猜测。

公开环境反馈保持所有奖励臂一致。训练时不能为了方便把“这个 chunk 得到 0.5 分”插入 policy prompt，这会泄漏未来问题及私有参考信息。

### 12.3 已有 v1 联合训练怎么办

如果 v1 已经训练 reader 或主动检索，不删除成果。P0 记录其初始化和结果，新增固定 reader 分支进行本轮方法隔离。可以从同一个已验证 checkpoint 初始化各主臂，但必须披露它已有的训练历史，不能称为原始 base model。

之后研究联合训练时，继续采用记忆回报对 m 题平均、回答按同一道题跨 K 份记忆归一化的设计。它与 UMA 直接相关，不能作为本项目首创。

### 12.4 动作级信用另立实验

本轮只记录原始事件分数、边界状态与终局任务收益，为后续算法准备数据。不要把 RTG 简单塞进现有 advantage 字段后宣称完成新算法。

后续需独立定义：即时奖励分配、共享前缀的平均分支收益、gamma 的时间单位、不同长度轨迹如何比较、baseline 信息范围、缺失动作如何处理。不得把不同语义状态的第 t 个动作直接组归一化。

对移除/替换某记忆后的固定 reader 重跑，仅称为局部证据依赖诊断。它不是完整策略级因果效应，因为后续检索和策略适应可能不同。

## 13. 评测与强基线

### 13.1 必须的系统参考

| 基线 | 资源条件 | 用途 |
|---|---|---|
| Question-only | 同 reader，无历史 | 参数知识和答案分布捷径 |
| Last-window | 同 C，保留尾部 | 判断关键证据是否真被移出 |
| FIFO-memory | 同 B、同 reader/retriever | 有限存储基础参考 |
| Frozen-policy | 同架构同初始化，不训练 | 训练收益 |
| Rule timeline | 从公开流解析并维护时间线，同 B/C | 受控数据可行性上界参考；明确有任务专用解析器 |
| Gold-support | 看过问题的特权支持证据，同 C | reader 能否解题；不是公平策略基线 |
| Full-store RAG | 若放宽 B，明确标识 | 系统参考，不与受限方法作公平成本结论 |

对于规则时间线基线，解析器不能读取 private event 文件；若实现的是 Oracle 直接填充，则改名 `privileged_world_oracle`，不能冒称公开规则策略。

### 13.2 外部方法的比较层级

首轮内部消融先回答机制问题，不要求一口气复现所有论文。但若准备论文，应依研究主张补充：

- 记忆构建与多问题训练：Mem-α、UMA。
- 稠密过程归因：Mem-T；若提出局部重采样再加 Memory-R2。
- 时间检索：Memory-T1。

建立 `baseline_fidelity.md`，记录每个基线的原始／适配实现、工具、原文可访问性、C/B、模型、训练量与评分。禁用原文检索后的 UMA 只能称 `UMA-adapted-memory-only`，不能冒称论文官方默认配置。不要跨数据／模型／预算直接引用论文分数相减。

### 13.3 配对反例与中间查询

除了普通 held-out 答案评测，增加固定配对子集：

1. 同最终值、不同历史路径：仅记最后值无法回答历史问题。
2. 同事件集合、不同有效时间：忽略时间无法正确关联。
3. A→B→A：按 value 去重会丢掉中间版本。
4. 删除单个副本 versus 删除最后有效表示：奖励应区分。
5. 正确来源＋错误正文 versus 忠实摘要：区分“指针正确”和“语义正确”。
6. 同终态、不同早期保留轨迹：验证 END/LIFE 的实际差异。

增加 `checkpoint_probe`：从若干预先固定的中间快照分叉，在不改变主轨迹的情况下提出当时可回答的问题。只作评测，不向主 ingest 分支反馈答案。这能检验 LIFE 的持续可恢复性是否真的提高，而非只提升它自己定义的代理分数。

checkpoint_probe 与最终问答是不同测试设置，分别报告；不能把 v2 主协议悄悄改成在线先问后记。

### 13.4 指标

- 主指标：每 history 平均答案 EM；D2 按官方 EM/F1。
- 分层：当前／历史／多次更新／时间连接／预算压力、5C/10C、证据年龄。
- 记忆：完整可恢复率、部分覆盖、历史版本保留率、时间歧义率、实际暴露覆盖。
- 监督：grounder precision/recall/F1、unclear、规则/monitor 等价率、END/LIFE 差异率与差异原因。
- 学习：非零优势组率、reward std、有效训练 tokens、参数变化、任务梯度、KL、dev 曲线。
- 代价：policy 与 reader 总 tokens、grounder/critic 调用、吞吐、峰值显存、总 GPU 时长。
- 协议：预算超限、旁路泄漏、分支污染、缺失组、基础设施失败。

监控压缩与语义质量的关系：不能仅凭记忆 token 更短就认定更好。静态 held-out 表现下降也要报告，避免动态任务提高来自灾难性偏置。

### 13.5 统计与公平预算

以 history_family 为成对重采样单位计算差值区间，不能把同一历史的 m 题当独立样本。先冻结样本、seed、超参选择规则和指标。

单 seed pilot 是诊断。出现明确效应后再按预算扩展到至少 3 个训练 seed；报告逐 seed 结果和 history 采样的不确定性，不把两种方差混为一谈。

END/LIFE 的评分成本不同，分别报告相同训练样本／更新数与总计算成本。无外部模型成本的 CPU monitor 也记录 CPU 时间。不能只匹配 policy tokens 而忽略额外语义判定模型开销。

## 14. 建议模块与接口

这些路径是建议，不是对当前仓库的事实描述。P0 先将现有模块映射到职责，优先扩展已存在的接口，避免新增一套平行框架。

```text
<existing_streaming_package>/dynamic/
  schema.py
  world_generator.py
  world_oracle.py
  independent_validator.py
  renderers.py
  query_builder.py
  split_audit.py
  temporal_grounder.py
  state_evaluator.py
  lifecycle_monitor.py
  reward_profiles.py
  checkpoint_probe.py
  replay.py
  metrics.py

configs/stream_dynamic_v2/
  data_debug.yaml
  replay.yaml
  frozen_diagnostic.yaml
  pilot_terminal.yaml
  pilot_end.yaml
  pilot_life.yaml

tests/stream_dynamic_v2/
docs/v2_current_implementation_audit.md
docs/v2_dynamic_memory_implementation_report.md
docs/v2_baseline_fidelity.md
```

### 14.1 数据与语义接口

```python
build_dynamic_dataset(config) -> DatasetManifest
validate_dataset(manifest) -> DatasetValidationReport
solve_world(events, question_spec) -> OracleAnswer
project_public_observation(history, chunk_index) -> PublicObservation
validate_memory_semantics(memory_snapshot, visible_sources, private_query) -> GroundingDecision
evaluate_checkpoint(memory_snapshot, checkpoint, private_queries) -> CheckpointScore
evaluate_exposure(rendered_query_prompt, private_query) -> ExposureScore
```

Oracle 与 private_queries 只能传入评估函数，不能成为公开 tool handler 的依赖。测试／进程边界都应验证此限制。接口实际拆分遵循现有依赖注入方式，不要求照抄类名。

### 14.2 奖励与训练接口

```python
compile_monitor(logic_spec, episode_schema) -> CompiledMonitor
replay_dynamic_rewards(rollout_bundle, profile, validators) -> RewardReplayReport
aggregate_memory_reward(branch_scores, checkpoint_scores, profile) -> MemoryReward
attach_reward_to_bundle(bundle, reward_report) -> TrainingBundle
```

返回对象包含各分量、分母、unclear/失败状态与版本 hash。不要只返回一个 float 而无法检查 reward 为什么变化。

monitor update 和 direct evaluator 共享输入契约，但不要简单互相调用后宣称两个独立后端验证通过。用独立状态求值路径进行差分验证。

## 15. 配置示例与冻结规则

### 15.1 待解析的起步配置

以下为规范示例。示例中的 `null` 必须由 P0 根据真实环境解析；未解析不能启动模型运行，也不能宣称已交付可运行配置。C/B 等数值优先继承 v1 已验证配置；下列值仅为缺省起点。

```yaml
experiment:
  name: dynamic_memory_v2_pilot
  protocol_version: streaming_dynamic_multiquery_v2
  schema_version: dynamic_stream_v2
  reward_version: dynamic_reward_v2
  seed: 7

inherit:
  v1_runtime_lock: null
  initialization_checkpoint: null

model:
  policy_revision: null
  tokenizer_revision: null
  reader_revision: null
  train_reader: false

budget:
  context_total_tokens: 4096
  persistent_memory_tokens: 2048
  ingest_max_new_tokens: 512
  answer_max_new_tokens: 256
  answer_tail_tokens: 512
  retrieved_payload_tokens: 2048
  max_decisions_per_chunk: 2

data:
  manifest: null
  generator_version: dynamic_world_v2_0
  time_semantics: monotonic_single_valued_v1
  train_length_ratio: 5.0
  evaluation_length_ratios: [5.0, 7.5, 10.0]
  query_pool_min: 8
  query_pool_max: 12
  questions_per_snapshot: 4
  memory_rollouts_per_group: 4
  split_unit: history_family_id
  text_render_mode: deterministic_templates
  main_feasibility_profile: feasible_control

memory:
  allow_raw_source_retrieval: false
  allow_audit_history_read: false
  accessible_metadata_counts_in_budget: true
  retriever_revision: null

grounding:
  mode: controlled_semantics
  validator_revision: null
  unclear_policy: review_then_diagnostic_only
  natural_audit_precision_target: 0.95
  natural_audit_recall_target: 0.80

reward:
  profile: V2_LIFE
  lambda_semantic: 0.25
  end_memory_weight: 0.5
  end_exposure_weight: 0.5
  life_end_weight: 0.5
  life_retention_weight: 0.5
  checkpoint_schedule: every_chunk_commit
  checkpoint_normalization: per_query_eligible_mean_then_query_mean
  cost_weight: 0.0
  format_weight: 0.0
  backend: direct_state_evaluator

training:
  advantage_mode: inherited_read_trajectory
  optimizer_lock: null
  shared_prefix_weighting: once_per_read_rollout
  answer_tokens_in_actor_loss: false
  automatic_cross_group_reward_normalization: false

evaluation:
  main_metric: history_macro_exact_match
  checkpoint_probe: true
  bootstrap_unit: history_family_id
  frozen_test_manifest: null
```

`max_decisions_per_chunk=2` 可能对动态更新工具不足。先检查一次调用是否允许多项动作、NEXT 是否占额度；如需增加，所有臂一起调整，并在真实采样前锁定，不能只放宽 LIFE。

检索 payload cap 不是保证能装下的承诺；以渲染后 C 检查为准，输出预留、system、query、tail 都要计入。若按 v1 算法缩减实际检索量，记录实际值；不得静默截断 system 或问题。

### 15.2 配置锁最少内容

保存 model/tokenizer/reader/retriever/grounder revision、chat template hash、初始化 checkpoint hash、manifest digest、实际 C/B、K/m、reward 公式与系数、optimizer 和 loss reduction、KL 与 std 口径、时间语义版本、代码 commit、环境版本。

如果当前代码中有全局 reward whitening、batch 内二次 advantage normalization，必须核对并显式处理，避免破坏 K 份记忆内部的分组定义。

新 run/checkpoint/cache 根使用 `dynamic_v2/...`；resume 需要完整匹配模型、数据、奖励、优化器和运行身份。动态 reward 不能继续写进静态 v1 checkpoint 的训练历史。

## 16. 必须的验收测试

测试集中于真实风险；不为每个数据类 getter 添加镜像测试，也不重复整个旧 runtime gate。旧 gate 只在受影响公共接口上运行必要回归，新 scope 独立记录。

### 16.1 数据与时间语义

| ID | 场景 | 必须结果 |
|---|---|---|
| D01 | 当前值与历史值不同 | 各自按查询时间得到正确答案 |
| D02 | `[from,to)` 边界 | to 时刻归下一有效版本 |
| D03 | A→B→A | 中间 B 可恢复，两个 A 不被错误合并 |
| D04 | 两跳更新时间错开 | 按同一 query_time 连接 |
| D05 | 不完整时间证据 | unknown 或未 eligible，不使用未来标签 |
| D06 | 同族改写／长度变体 | 不跨 train/dev/test |
| D07 | 数据超过 5C 但重复很多 | 统计揭示，不按字符数或重复 token 虚报压力 |
| D08 | 同公开前缀、不同隐藏未来 | 决策前 observation 字节一致；固定随机状态时策略行为一致 |
| D09 | 规则可行性参考超过 B/C | 标记不可行，不记作同预算满分上界 |

### 16.2 存储与 grounding

| ID | 场景 | 必须结果 |
|---|---|---|
| M01 | 正确 pointer，错误 value | source_ok=true、content_ok=false |
| M02 | 正确 value，但错误有效期 | temporal_ok=false |
| M03 | 无日期但可由事件序列恢复 | 允许合法推导，不机械判错 |
| M04 | 合并摘要保留必要语义 | 允许不同表达获得支持 |
| M05 | 删除冗余副本 | coverage 不变 |
| M06 | 删除最后有效表示 | coverage 下降 |
| M07 | UPDATE 正文变错 | 不继承旧 support 判定 |
| M08 | 失败 UPDATE／DELETE | 状态和 revision 不变 |
| M09 | 旧版本、检索缓存、source_id 取原文 | policy 无法免费读取 |
| M10 | 检索截掉时间限定 | 按实际 prompt 重算 exposure |
| M11 | 活动 metadata 藏大段原文 | 计入 B 或按 schema 拒绝 |

### 16.3 奖励与 monitor

| ID | 场景 | 必须结果 |
|---|---|---|
| R01 | 第 11.6 节 fixture | 数值与手算一致 |
| R02 | direct 与 compiled 后端 | 同语义逐状态、逐分量一致 |
| R03 | ADD/DELETE/RETRIEVE 循环 | 分数不因动作数无界增长 |
| R04 | 重复正确条目 | 按语义集合计覆盖，不重复计分 |
| R05 | 未 eligible 的最终版本 | 不提前奖励，也不提前惩罚缺失 |
| R06 | 某题缺失且其他题正常 | 分母不根据 policy 成功率缩小 |
| R07 | 不明确语义／基础设施失败 | 按显式策略处理，不静默给零 |
| R08 | v1 静态 reward regression | v1 已冻结输入与输出不变 |
| R09 | END/LIFE 真正相同的轨迹集合 | 如实报告零差异，不伪造信号 |
| R10 | 前缀 score_delta | 正负增量可重放；未重复加进总回报 |

### 16.4 训练与分支

- K=2、m=2 的手算 GRPO fixture，前缀一次，分支平均。
- m 改变不复制 read token，也不让语义分线性膨胀。
- policy/environment/reader 事件的 loss mask 明确。
- 同 snapshot 的问题调换顺序不改变其他分支初始输入和状态。
- 所有采样问题都完成或组被明确标记失败，不能残缺组入 buffer。
- old_logprobs 绑定原始 rollout policy；reward-only replay 不改变它们。
- 模型基础设施不可用只影响对应测试，不将 SKIP 记成 PASS。
- 如需 GPU smoke，检查一次真实 optimizer update、非零有效 policy 梯度、checkpoint 完整性和加载后评测身份；不把权重变化单独当成任务学习成功。

## 17. 实施阶段、交付物和停止条件

阶段顺序是实施依赖，不是重新设计一个三阶段训练 episode。

| 阶段 | 工作 | 必须产物 | 继续条件 |
|---|---|---|---|
| P0 | 核验已执行 v1 | 当前实现审计与路径映射 | 已知哪些可复用，未伪造当前状态 |
| P1 | 动态世界、数据、Oracle | D0/D1、manifest、独立验证报告 | 时间语义、split 和无泄漏通过 |
| P2 | 存储／版本／grounding | 接口实现与审计报告 | 受控语义可靠，自由语义按模式分开 |
| P3 | END/LIFE 与 monitor | 手算 fixture、差分重放报告 | 同语义后端一致，不同目标差异合理 |
| P4 | 冻结策略真实诊断 | K×m 轨迹、成本、失败归因 | 环境可达、reader 可答、奖励可信 |
| P5 | 训练 profile 接线 | T/END/LIFE 锁与必要 smoke | 共用训练接线正确，有有效信号 |
| P6 | 小规模方法比较 | pilot 曲线、dev、预算报告 | 按预定停止规则得到可解释结果 |
| P7 | 扩展验证 | 多 seed、动态/静态/OOD 分析 | 确有值得复验的效应 |
| P8 | 可选自动 Critic／联合训练 | 独立实验规范与结果 | 不把多个变量混在主结论里 |

### 17.1 P4 冻结诊断建议

先使用 dev 中固定的 24～40 个 histories，K=4、m=4；不是训练后挑最好看的样本。复用同一批真实轨迹离线评分 T/END/LIFE，不为每个奖励臂重新采样。

输出：grounding 不确定率、各 profile reward/std/排序、END-LIFE 差异、差异对应的实际状态、总 token 和峰值显存。

构造反例可以证明定义正确，但不能替代自然轨迹差异。如果真实轨迹全部失败、只会 NEXT、或 END/LIFE 全部相同，先诊断，不直接延长训练。

### 17.2 P6 pilot 冻结参数

根据 v1 可承受成本决定具体预算。建议起点为每臂最多 30 次 optimizer update、dev 每 10 步评测、单 seed。实际更新数必须在开始前写入计划；这是工程诊断上限，不是充分收敛或统计结论。

三个臂共享初始化和预先排定的 history/question 抽样计划；训练后因 policy 不同导致 rollout 不同是正常的，不要求逐 token 相同。

主 checkpoint 按相同 dev 主指标选，平分时选较早 checkpoint。另报告末步表现；不能各方法挑不同题型最优点。

终止条件包括：确定性泄漏／预算／loss mask 错误、持续基础设施失败、非有限损失、超过已授权成本上限。训练无增益可作为负结果关闭，不能为追求正结果无限加步数。

### 17.3 缺资源时的完成定义

接手者没有 GPU 时，完成 P0～P3、配置生成、必要 CPU 验证和可运行命令；P4 以后标为资源待满足。真实语义审计资料缺失时，继续受控模式，明确自由文本模式尚未准入。

不能用旧文件里“只允许旧 provenance replay”的历史约束阻塞用户已经授权的新 v2 实施。仍适用的仓库保护、权限和真实证据门禁继续遵守；若产生具体冲突，说明来源和受影响动作，完成其余工作。

## 18. 失败诊断与后续决定

| 发现 | 优先解释和检查 | 下一步 |
|---|---|---|
| Grounder 奖励高，答案仍低 | 错误语义、时间丢失、检索截断、reader 推理 | 先检查真实 payload 和 gold-support reader |
| 来源判定正确但自然改写低召回 | 判定器依赖表面形式 | 补语义审计；不把低分全归因遗忘 |
| T 全组零差异，END 有差异 | 终局信号稀疏或 reader 过弱 | 检查可达性，再开展受控奖励 pilot |
| END 与 LIFE 全部相同 | 信息只在末尾可判定、没有保留机会、策略未形成差异 | 查 eligibility 和轨迹；不伪造目标差距 |
| LIFE 分数高而 dev EM 不升 | 辅助目标与任务不一致、过度保留占预算 | 分析 lambda、查询分布、证据竞争 |
| LIFE 优于 END，但耗时大幅增加 | 更多语义计算或更长动作 | 按同总成本补充比较 |
| 编译 monitor 与 direct 不一致 | 转移、状态回退或 denominator bug | 停止语义训练，修复差分 |
| 当前题提高、历史题降低 | 历史版本被覆盖／摘要丢时间 | 检查表示与任务分布，不只加 DELETE 处罚 |
| 动态测试提高、静态回归下降 | 任务专化或灾难性遗忘 | 单列静态混合训练对照 |

“DFA 与强规则等价”是预期正确性结果。应跳过重复训练，并把研究重点放在监督目标、规则生成可靠性或可迁移性；不能因此人为移除强基线的状态信息。

## 19. 后续自动 Logic Critic 的技术约束

本节是候选扩展，不是 P0～P6 的阻塞条件。只有固定语义监督在真实轨迹上有作用，才值得研究自动化生成。

### 19.1 目标

从公开任务语义与训练轨迹中生成记忆义务和依赖，减少逐任务手写规则；通过确定性校验与编译控制不一致奖励。这比“把手工计数包装成 DFA”更能检验 GLARE 迁移价值，但仍需相关工作比较。

### 19.2 Critic 输出 schema

输出必须包含 `spec_version`、`predicate_schema`、`obligations`、`dependencies`、`allowed_alternatives`、`invalidation_conditions`、`evidence_refs`、`confidence`、`fallback_reason`。

规则描述语义结果，不规定唯一工具序列。新增时态关系必须在受支持的有限语法内。编译器校验未定义谓词、矛盾义务、不可达状态、循环奖励、未来信息依赖和不合法引用。

### 19.3 信息边界与 fallback

训练 Critic 可按明确配置看到训练私有标签或成功轨迹，但必须标明监督成本和信息特权；所有比较臂匹配可用标签。测试规则模板冻结，不能看 test gold 后生成“适配 test 的规则”并称为零样本泛化。

读入策略始终不知道未来问题；Critic 输出也不能作为阅读期提示。全失败组不得由 Critic 凭空断言某个记忆操作是成功必需条件。校验失败回退至固定 END/LIFE 或 terminal，按预先冻结策略执行并报告回退率。

### 19.4 自动化的有效对照

同一组自然轨迹比较：

- LLM 直接逐步打分。
- LLM 生成规则＋直接规则解释器。
- 相同规则＋校验＋编译 monitor。
- 人工固定语义规则参考。

分开测量规则语义错误与执行错误。编译不会修复源规则的事实误解；更稳定的分数也可能稳定地错误。最终仍要测任务学习和跨关系／跨模板迁移。

## 20. 研究结论与允许的表述

| 获得的证据 | 可以表述 | 仍不能表述 |
|---|---|---|
| CPU fixture 与差分通过 | 实现满足受控契约 | 模型学会动态记忆 |
| grounder 审计改善 | 在审计分布上减少奖励误判 | 全领域可靠、因果归因准确 |
| END 优于 T | 额外时间状态监督有帮助 | DFA 结构提供额外收益 |
| LIFE 优于 END | 保留过程辅助目标在该设置有效 | 逻辑形式天然优于等价程序 |
| 固定规则在新关系上迁移 | 规则／表示有一定复用能力 | 自动 Critic 已验证 |
| 自动 Critic 改善训练且审计可靠 | 自动生成并验证规则有经验价值 | 首次解决全部 memory credit 问题 |
| 证据替换使答案改变 | 该干预下答案依赖相关信息 | 完整策略级因果机制已识别 |

建议研究标题暂用“面向动态记忆维护的可验证状态过程监督”。在形成独特算法和完整证据前，不使用“首个”“因果最优”“通用记忆推理”等强宣称。

## 21. CLI 与运行产物契约

优先扩展 v1 CLI；若不存在统一入口，建议新增下述命令。以下为执行者需要实现并验证的接口示例，不保证当前仓库已有。

```bash
python scripts/agemem_dynamic_v2.py audit-current
python scripts/agemem_dynamic_v2.py build-data --config configs/stream_dynamic_v2/data_debug.yaml
python scripts/agemem_dynamic_v2.py validate-data --manifest runs/dynamic_v2/data/manifest.json
python scripts/agemem_dynamic_v2.py env-smoke --config configs/stream_dynamic_v2/data_debug.yaml
python scripts/agemem_dynamic_v2.py replay --run-dir runs/dynamic_v2/frozen_diag --profiles V2_T,V2_END,V2_LIFE
python scripts/agemem_dynamic_v2.py compare-backends --run-dir runs/dynamic_v2/frozen_diag
python scripts/agemem_dynamic_v2.py preflight --config configs/stream_dynamic_v2/pilot_terminal.yaml
python scripts/agemem_dynamic_v2.py diagnose --config configs/stream_dynamic_v2/frozen_diagnostic.yaml
python scripts/agemem_dynamic_v2.py train --config configs/stream_dynamic_v2/pilot_terminal.yaml
python scripts/agemem_dynamic_v2.py train --config configs/stream_dynamic_v2/pilot_end.yaml
python scripts/agemem_dynamic_v2.py train --config configs/stream_dynamic_v2/pilot_life.yaml
python scripts/agemem_dynamic_v2.py evaluate --run-dir runs/dynamic_v2/pilot_terminal
python scripts/agemem_dynamic_v2.py report --experiment-root runs/dynamic_v2
```

实际顺序：先用 env-smoke 产出的 fixture run-dir 做 replay/compare-backends；P4 真实 diagnose 完成后，再对真实 frozen_diag 重放。不要把示例路径不存在误认为算法失败。

CLI `--help` 必须解释输入输出和退出码。对不可恢复 schema/数据错误返回非零；对资源缺失返回清楚原因，不伪造空“成功”报告。最终实施报告写实际验证过的命令与路径。

### 21.1 每个实验必须留下

- config lock、数据与代码身份、初始化 checkpoint 身份。
- 分离的公开 rollout 与私有语义／奖励 sidecar。
- 按 history/query 的答案与指标，原始分母、失败原因。
- 按 checkpoint 的 coverage/conflict/eligibility、各 reward profile。
- monitor/direct 差分报告和代表性反例。
- token、GPU、CPU、reader 与 grounder 调用成本。
- 实际训练步、checkpoint 完整性、选择规则、dev/test 使用记录。

生成统一 `docs/v2_dynamic_memory_implementation_report.md`，包括“已实现”“已验证”“实测结果”“阻塞／待验证”“下一步”五部分。更新当前 `STATUS.md` 顶部链接和阶段；历史状态保留在历史区域，避免同时出现互相冲突的“唯一下一步”。

## 22. Definition of Done

### 22.1 代码和 CPU 交付完成

- [ ] 已核验实际 v1 完成范围，未重做可复用模块。
- [ ] 新协议和旧协议可分别运行，版本身份不混淆。
- [ ] 动态数据生成、独立 Oracle、split 与预算可行性检查已完成。
- [ ] 来源／正文／时间／实际暴露四种判定区分明确。
- [ ] 旧正文、隐藏标签与多问题分支不存在旁路。
- [ ] V2_T/END/LIFE 公式、分母、eligibility 与示例通过验证。
- [ ] 同语义 direct/monitor 差分一致；不重复安排等价训练臂。
- [ ] 未经审计自由语义被明确隔离，unclear 处理可追踪。
- [ ] 继承的共享前缀训练与 mask 契约未破坏。
- [ ] 运行配置无未解析必需字段，或明确列为资源阻塞。
- [ ] 相关回归通过；SKIP 与真实通过区分。
- [ ] 交付实际命令、报告和当前状态，未声称未运行的 GPU 结果。

### 22.2 研究验证完成

- [ ] 冻结自然轨迹上的语义评分可信，差异有具体状态解释。
- [ ] 在相同初始化、数据与预算下完成 T/END/LIFE 方法比较。
- [ ] 主指标按原始答案任务评分，未用辅助奖励代替。
- [ ] 当前／历史／多跳／长度子集及静态回归均报告。
- [ ] 对值得扩展的效应完成额外 seed／数据验证并报告成本。
- [ ] 结论没有越过数据范围和基线覆盖范围。

代码交付完成与研究效果成立是两件事。负结果也可以完成一轮研究验证；不得以“必须正提升”作为无限增加实验的理由。

## 23. 可直接交给 Codex 的启动指令

```text
请在当前 Age-Mem 仓库中，按《AGEMEM_DYNAMIC_LOGIC_INCREMENTAL_SPEC_V2.md》进行增量实施。

我已经执行过上一版流式多问题技术文档。请先读取实际 AGENTS.md、STATUS.md、上一版实施报告、代码和最近运行结果，输出 P0 当前实现审计；不要把旧上传文档中的待完成事项当成当前状态，也不要重复重建已完成模块。

这轮重点是：动态时间／版本数据、独立 Oracle、可信来源与语义判定、强终态监督 END、保留过程监督 LIFE，以及 direct evaluator 与 compiled monitor 的等价验证。复用现有双预算、多问题隔离、共享前缀和 GRPO 接线，保持旧协议可复现。

按 P0→P3 连续完成可执行的本地工作，再根据现有资源和本会话授权推进 P4→P6。缺 GPU、模型或自然语义审计资料时，继续其他独立实现，明确记录阻塞，不伪造验证。沿用我已经提供的授权，不要每完成一个小步骤就重复问是否继续。

不要强制固定 ADD/UPDATE/RETRIEVE 顺序。来源指针正确不代表正文正确；历史事实不因当前状态更新而自动失效。策略不能读取 private labels、审计旧正文或原文索引旁路。

主实验比较 V2_T、V2_END、V2_LIFE，保持训练算法和其他变量相同。相同逻辑的直接规则和自动机必须同分，差异先按 bug 处理；不为等价实现重复启动 GPU 训练。首轮不新加自动 LLM Critic、局部反事实优化器或联合 reader 训练。

每阶段更新 v2 实施报告，最终给出：改动文件、实际验证命令、通过/失败/未运行、数据/配置/checkpoint 身份、研究结果边界和下一步。修复当前接口的实际问题，但保留历史实验、用户无关修改和旧 reward/profile 身份。
```

## 24. 参考来源与可复用项目

以下条目在 2026-09-15 核对。各项方法细节以上文简述为限，执行复现时继续记录实际代码 commit 和数据 revision。

1. **v1 本项目规范**：`AGEMEM_STREAMING_MULTIQUERY_CHANGE_SPEC.md`。本文件为增量补充，不覆盖旧版本。
2. **用户上传项目资料**：`PROJECT_HANDOFF(1).md`、`STATUS.md`，记录的是之前状态；用户执行后应以实际仓库核验为准。
3. **AgeMem / GLARE**：用户上传的 `Agentic Memory(1).pdf`、`GLARE(1).pdf`。引用上传版本，不替换为未核对的新版本公式。
4. **Mem-α**：[论文](https://arxiv.org/html/2509.25911v1)、[官方代码](https://github.com/wangyu-ustc/Mem-alpha)、[数据](https://huggingface.co/datasets/YuWangX/Memalpha)。可参考分块和多问题数据组织；不能直接拿评测分区训练。
5. **UMA**：[论文 v2](https://arxiv.org/html/2602.18493v2)、[官方代码](https://github.com/ictnlp/unified-memory-agent)。与多问题平均收益和分组优化直接相关；原文检索权限需单列。
6. **Mem-T**：[论文 v2](https://arxiv.org/html/2601.23014v2)、[官方代码](https://github.com/yanweiyue/Mem-T)。涉及树引导奖励和事后记忆构建归因。
7. **Memory-T1**：[论文](https://arxiv.org/html/2512.20092v1)。涉及证据 grounding 和时间一致性检索奖励。
8. **Memory-R2**：[论文](https://arxiv.org/abs/2605.21768)。涉及相同中间状态的局部重采样与全局优化。
9. **MemAgent**：[论文](https://arxiv.org/html/2507.02259v1)、[官方代码](https://github.com/BytedTsinghua-SIA/MemAgent)。作为有限上下文分块训练的相关工作；读入时问题已知，与主协议不同。
10. **MemoryAgentBench**：[官方项目](https://github.com/HUST-AI-HYZ/MemoryAgentBench)。参考增量输入与多问题评测，不将其测试集混入训练。

本文的动态数据生成、eligibility、END/LIFE 权重、规模与实施顺序是本项目拟议设计，尚未获得实验验证，也不作为“已有论文已证明有效”的结论。

## 25. 冲突优先级与本轮最小目标

当前用户明确指令和实际权限约束优先；本文给出 v2 的目标规范。当前仓库真实证据优先于旧上传状态的完成描述。保留 v1 定义和历史结果，冲突通过新版本配置与明确迁移记录解决，不靠静默改写旧实验。

**本轮最小研究闭环：同一批动态信息流，可信地评估记忆的时间有效性；以同一训练接线比较终局、终态和保留过程监督；用独立问答结果判断额外监督是否有益。**
