# 神经符号 Agent Memory 项目交接文档

> 面向：VS Code 中的 Codex 插件  
> 项目方向：AgeMem 式可学习记忆管理 + GLARE 式 LTLf/DFA 逻辑奖励  
> 文档版本：v1.2  
> 更新时间：2026-07-30  
> 本地项目根目录：`D:\Project\Age-Mem\AgeMem`  
> 当前状态：用户本地已有项目；Codex 接管后必须先检查仓库，不能重新初始化或覆盖现有代码

---

## 1. 一页摘要

本项目计划构建一个能够自主执行长期记忆和短期上下文管理的 LLM Agent，并使用结构化时序逻辑奖励改善强化学习中的长程信用分配。

核心路线：

```text
AgeMem 的统一记忆策略
    +
GLARE 的三元组状态抽取、里程碑分析和自动机奖励
    =
Logic-Guided Agent Memory
```

Agent 的动作空间最终包括：

```text
环境动作：
    open / take / put / heat / clean / answer / ...

长期记忆操作：
    ADD / UPDATE / DELETE

短期上下文操作：
    RETRIEVE / SUMMARY / FILTER
```

训练流程：

```text
同一任务采样 K 条轨迹
        ↓
记录观察、动作、记忆变化和环境反馈
        ↓
观察抽取为三元组，动作映射为命题
        ↓
显式状态跟踪器生成 AP_t
        ↓
Logic Critic 分析轨迹组
        ↓
生成里程碑、依赖关系和坏行为
        ↓
编译为 LTLf/DFA
        ↓
自动机重放轨迹，产生逐步奖励
        ↓
Step-wise GRPO 更新统一策略
```

第一版只验证一个问题：

> 在相同 Agent、记忆工具和终局任务奖励下，DFA 里程碑奖励是否比纯终局奖励更容易学会有效的 `ADD / RETRIEVE / UPDATE` 行为？

---

## 2. 研究假设

### 2.1 核心假设

AgeMem 将终局奖励广播到整条轨迹，可以缓解记忆操作的延迟监督问题，但仍无法区分一条轨迹中哪些记忆动作真正产生了任务进展。

本项目假设：

1. 将开放式轨迹转化为显式符号状态，可以降低逐步语义打分的不稳定性；
2. 将记忆行为和任务行为组织成带依赖关系的里程碑，可以提供更准确的过程监督；
3. DFA 状态转移产生的奖励，比单次 LLM 打分更一致、可复现、可解释；
4. 正向自动机和负向自动机可以分别奖励有效记忆行为、惩罚过期检索和提前删除；
5. 更准确的逐步奖励最终能够提高长程任务成功率、记忆质量和训练样本效率。

### 2.2 需要用实验回答的问题

```text
RQ1：DFA 奖励是否提高任务成功率？
RQ2：DFA 奖励是否提高训练样本效率？
RQ3：提升来自自动机，还是仅仅来自增加了额外奖励？
RQ4：三元组抽取错误会造成多大性能损失？
RQ5：Logic Critic 生成的里程碑是否可靠？
RQ6：统一 Agent 是否优于独立 Memory Manager？
RQ7：逻辑奖励能否泛化到未见任务和更长轨迹？
```

---

## 3. 项目范围

### 3.1 MVP 必须包含

- 一个基于 AgentScope 的单智能体循环；
- `ADD / RETRIEVE / UPDATE` 三种记忆工具；
- 每个 rollout 独立的记忆状态；
- 一个三阶段 toy memory environment；
- 完整、可重放的 JSONL 轨迹；
- Oracle AP；
- 一个手工定义的正向 DFA；
- Milestone Reward；
- Terminal-only 和 Terminal+DFA 两种设置；
- 单元测试和离线奖励重放；
- 在 toy 环境上完成小规模 GRPO smoke test。

### 3.2 MVP 暂时不做

- `SUMMARY / FILTER / DELETE`；
- Negative Automata；
- Trend Shaping；
- 全失败组的反事实里程碑；
- 任意开放式 LTL 自动生成；
- Belief/Probabilistic Automata；
- PostgreSQL 作为训练 rollout 后端；
- ALFWorld、ScienceWorld 全量训练；
- 多智能体协同；
- GUI、网页服务和生产部署。

这些功能只有在 MVP 证明 DFA Reward 有效后再依次加入。

---

## 4. 当前工作区状态

用户本地项目位置：

```text
D:\Project\Age-Mem\AgeMem
```

本交接文档生成环境无法直接读取用户 Windows `D:` 盘，因此不能预先断言本地项目的 Git 状态、依赖状态或现有修改。VS Code 中的 Codex 打开该目录后，必须首先完成只读检查：

```text
当前是否为 Git 仓库
当前分支和未提交修改
AgeMem 代码版本和目录结构
AgeMem_code_agentscope 是否存在
是否已有 .venv/conda 环境
requirements/pyproject 的依赖约束
是否已有用户自己添加的代码
是否已有 STATUS.md、实验结果或配置文件
```

因此 Codex 不应：

- 再次 `git init`；
- 再次 clone AgeMem 到项目内部；
- 删除或覆盖未提交修改；
- 假设上游仓库结构与当前最新版完全一致；
- 直接开始修改训练代码。

第一阶段应当是：

```text
接管检查 → 记录现状 → 复现已有 standalone demo → 再开始增量开发
```

---

## 5. 推荐上游项目与技术栈

### 5.1 上游项目

| 组件 | 推荐项目 | 用途 |
|---|---|---|
| 直接基线 | [AgeMem](https://github.com/y1y5/AgeMem) | 记忆工具、AgentScope demo、Trinity-RFT 工作流 |
| Agent 层 | [AgentScope](https://github.com/agentscope-ai/agentscope) | Agent 循环、消息、工具调用 |
| RFT 层 | [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) | Explorer、Buffer、Trainer、GRPO |
| 主环境 | [ALFWorld](https://github.com/alfworld/alfworld) | 文本长程任务规划 |
| 扩展环境 | [ScienceWorld](https://github.com/allenai/ScienceWorld) | 复杂自然语言观察和科学任务 |
| LTLf 编译 | [LTLf2DFA](https://github.com/whitemech/ltlf2dfa) | LTLf 转最小 DFA |
| 最终数据库 | [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL 中的向量检索 |

### 5.2 推荐运行环境

```text
日常代码阅读/standalone demo：Windows 或 WSL2 均可
RL训练、ALFWorld、ScienceWorld、MONA/LTLf：优先 WSL2 Ubuntu 或原生 Linux
Python：优先遵循 AgeMem requirements；新模块目标为 Python 3.11
编辑器：VS Code + Codex 插件
训练：Linux + NVIDIA GPU + CUDA
快速调试：模型 API 或小型本地模型
测试：pytest
```

当前项目位于 Windows：

```text
D:\Project\Age-Mem\AgeMem
```

建议分两步处理：

1. 先直接用 VS Code 打开该目录，完成代码理解、standalone demo 和非 GPU 模块开发；
2. 进入 Trinity-RFT、ALFWorld、ScienceWorld 或 LTLf/MONA 阶段前，在 WSL Linux 文件系统中建立训练副本，例如：

```text
~/projects/Age-Mem
```

若暂时通过 WSL 访问原目录，对应路径通常是：

```text
/mnt/d/Project/Age-Mem/AgeMem
```

但大规模训练和大量小文件 IO 不建议长期运行在 `/mnt/d`，更适合把仓库 clone 到 WSL 自己的 Linux 文件系统，并通过 Git 同步代码。迁移前必须先提交或妥善保存 Windows 工作区的本地修改。

### 5.3 版本原则

1. 第一阶段严格使用 AgeMem 仓库声明的依赖版本；
2. 不要一开始强行升级到最新 AgentScope；
3. 首次成功运行后记录：
   - AgeMem commit；
   - AgentScope 版本；
   - Trinity-RFT commit；
   - Python 版本；
   - CUDA 和 PyTorch 版本；
4. 后续升级必须单独建分支并重新运行 smoke test；
5. 不允许为了修复单个报错而无记录地批量升级所有依赖。

---

## 6. 总体架构

```text
┌──────────────────────────────────────────────────────────┐
│                    AgentScope Agent                      │
│  observation + STM + retrieved memory + task → action   │
└──────────────────┬───────────────────────┬───────────────┘
                   │                       │
                   ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  Memory Tools   │     │ Environment Tool │
          │ ADD/UPDATE/...  │     │ open/take/...    │
          └────────┬────────┘     └────────┬─────────┘
                   │                       │
                   ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  MemoryStore    │     │ EnvAdapter      │
          └────────┬────────┘     └────────┬─────────┘
                   └───────────┬───────────┘
                               ▼
                       ┌───────────────┐
                       │ TraceRecorder │
                       └───────┬───────┘
                               ▼
                 task-level group of K trajectories
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     ┌──────────────────┐             ┌──────────────────┐
     │ Triple Extractor │             │ Action Mapper    │
     └────────┬─────────┘             └────────┬─────────┘
              ▼                                │
     ┌──────────────────┐                      │
     │ StateTracker h_t │◄─────────────────────┘
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ AP Grounder      │
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ Logic Critic     │
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ Validator        │
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ DFA + Reward     │
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ Trinity-RFT      │
     │ Buffer + Trainer │
     └──────────────────┘
```

---

## 7. 关键设计决策

### 7.1 统一 Agent

MVP 使用同一个模型同时决定：

- 环境动作；
- 是否保存信息；
- 是否检索信息；
- 是否更新信息。

独立 Memory Manager 作为后续基线，而不是 MVP 主架构。

### 7.2 有限轨迹使用 LTLf

Agent episode 是有限的，因此主语义采用：

```text
LTLf → DFA
```

而不是无限轨迹上的普通 LTL/Büchi automaton。

### 7.3 Critic 先输出结构化里程碑

第一版 Critic 不直接输出任意字符串形式的 LTL，而是输出：

```json
{
  "milestones": [],
  "dependencies": [],
  "bad_behaviors": [],
  "evidence_steps": [],
  "confidence": 0.0
}
```

由确定性代码将里程碑依赖图编译成 DFA。等结构化版本稳定后，再增加 LTLf 表达和 LTLf2DFA。

### 7.4 训练和持久化使用不同 MemoryStore 后端

```text
InMemoryStore：
    用于并行 rollout
    快速 snapshot/reset
    每个 rollout 完全隔离

PostgresMemoryStore：
    用于最终持久化实验
    支持版本、审计、metadata 和 pgvector
```

两者必须实现同一个 `MemoryStore` 协议。

### 7.5 自动机训练时使用，测试时默认 shadow execution

主实验中：

- 自动机用于训练奖励；
- 测试时策略独立执行；
- DFA 只在后台记录满足率；
- DFA 不阻止动作、不提示重规划。

若测试时自动机进行干预，必须单独命名为：

```text
Ours + Runtime Monitor
```

不能与主结果混合。

---

## 8. 推荐目录结构

项目根目录已经存在，因此不要重建或移动 AgeMem 原有目录。Codex 应先识别现有结构，再增量增加本项目文件。建议目标结构如下，其中“原有 AgeMem 文件”保持原位：

```text
D:\Project\Age-Mem\AgeMem\
├── <原有 AgeMem 文件与目录，保持原位>
├── PROJECT_HANDOFF.md
├── README.md
├── STATUS.md
├── .env.example
├── configs/
│   ├── agent.yaml
│   ├── memory.yaml
│   ├── reward.yaml
│   ├── toy.yaml
│   ├── alfworld.yaml
│   └── training.yaml
├── src/
│   └── logic_memory/
│       ├── __init__.py
│       ├── agent/
│       │   ├── agent.py
│       │   ├── prompts.py
│       │   └── tool_adapter.py
│       ├── memory/
│       │   ├── base.py
│       │   ├── models.py
│       │   ├── in_memory.py
│       │   ├── postgres.py
│       │   └── tools.py
│       ├── environments/
│       │   ├── base.py
│       │   ├── toy.py
│       │   ├── alfworld.py
│       │   └── scienceworld.py
│       ├── trajectory/
│       │   ├── models.py
│       │   ├── recorder.py
│       │   └── replay.py
│       ├── symbolic/
│       │   ├── triples.py
│       │   ├── extractor.py
│       │   ├── action_mapper.py
│       │   ├── state_tracker.py
│       │   ├── grounder.py
│       │   ├── critic.py
│       │   ├── validator.py
│       │   ├── compiler.py
│       │   └── automaton.py
│       ├── rewards/
│       │   ├── terminal.py
│       │   ├── logic.py
│       │   └── composite.py
│       ├── training/
│       │   ├── explorer.py
│       │   ├── reward_adapter.py
│       │   └── workflow.py
│       └── evaluation/
│           ├── metrics.py
│           └── evaluator.py
├── scripts/
│   ├── run_demo.py
│   ├── collect_rollouts.py
│   ├── annotate_triples.py
│   ├── build_automata.py
│   ├── replay_rewards.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/
│   ├── toy/
│   ├── annotations/
│   └── splits/
├── artifacts/
│   ├── trajectories/
│   ├── automata/
│   ├── metrics/
│   └── checkpoints/
└── docs/
    ├── reproduction.md
    ├── reward_design.md
    └── experiment_log.md
```

如果现有仓库没有 `src/` 布局，不要仅为了形式统一而立即搬动 AgeMem 原代码。可以先增加：

```text
logic_memory/
```

或者：

```text
extensions/logic_memory/
```

具体位置由 Codex 在检查现有 import、打包方式和测试结构后决定。原则是：

- 不破坏 AgeMem 已有入口；
- 新模块与上游代码尽量解耦；
- 通过 adapter 接入现有 memory tools 和 training workflow；
- 避免把研究代码散落到多个无关上游文件中。

---

## 9. 核心数据契约

所有模块先围绕稳定的数据模型开发，避免后期因字段不一致反复重写。

### 9.1 MemoryItem

```python
class MemoryItem:
    memory_id: str
    content: str
    metadata: dict
    embedding: list[float] | None
    version: int
    status: str
    created_at: str
    updated_at: str
    source_rollout_id: str | None
    source_step: int | None
```

`status` 至少支持：

```text
active
superseded
discarded
```

### 9.2 MemoryEvent

```python
class MemoryEvent:
    event_id: str
    operation: str
    target_memory_id: str | None
    arguments: dict
    result: dict
    rollout_id: str
    timestep: int
```

### 9.3 TrajectoryStep

```python
class TrajectoryStep:
    task_id: str
    rollout_id: str
    stage: int
    timestep: int
    observation: str
    action_text: str
    tool_calls: list[dict]
    tool_results: list[dict]
    memory_before: list[dict]
    memory_after: list[dict]
    env_reward: float
    done: bool
    old_logprob: float | None
```

### 9.4 Triple 与 StateFact

```python
class Triple:
    subject: str
    category: str
    value: str
    confidence: float
    evidence_text: str


class StateFact:
    subject: str
    category: str
    value: str
    confidence: float
    source_step: int
    valid_from: int
    valid_to: int | None
```

### 9.5 Critic 输出

```python
class Milestone:
    milestone_id: str
    proposition: str
    description: str
    evidence_steps: list[int]
    confidence: float


class CriticOutput:
    milestones: list[Milestone]
    dependencies: list[tuple[str, str]]
    bad_behaviors: list[dict]
    counterfactual_used: bool
    warnings: list[str]
```

### 9.6 AutomatonSpec

```python
class AutomatonSpec:
    states: list[str]
    initial_state: str
    accepting_states: list[str]
    rejecting_states: list[str]
    transitions: list[dict]
    source_milestones: list[str]
```

### 9.7 RewardBreakdown

```python
class RewardBreakdown:
    env: float
    milestone: float
    violation: float
    trend: float
    format: float
    total: float
    automaton_state_before: str
    automaton_state_after: str
```

---

## 10. 分阶段实施计划

## M0：已有仓库接管与上游复现

### 目标

接管 `D:\Project\Age-Mem\AgeMem` 的已有项目，在不修改核心算法的情况下运行 AgeMem standalone AgentScope demo。

### 任务

1. 检查当前目录和 Git 状态；
2. 列出未提交、未跟踪和被忽略文件；
3. 记录当前分支、remote 和 commit；
4. 检查用户已有改动，禁止覆盖；
5. 只有在工作区状态明确后，才建议是否创建开发分支；
6. 检查 Python、`.venv`、conda 和 requirements；
7. 阅读：
   - 根目录 README；
   - `AgeMem_code_agentscope/README.md`；
   - memory manager；
   - 六个 memory tools；
   - standalone main；
8. 优先复用兼容的现有虚拟环境；没有时才建立新环境；
9. 使用本地环境变量配置 API 密钥；
10. 运行 standalone demo；
11. 手工执行：
   - add；
   - retrieve；
   - update；
   - delete；
12. 把复现命令、输出和问题写入 `docs/reproduction.md`；
13. 创建或更新 `STATUS.md`。

### 验收标准

- demo 可启动；
- Agent 能成功调用至少三种 memory tools；
- `ADD → RETRIEVE → UPDATE → RETRIEVE` 结果正确；
- 密钥未写入仓库；
- 复现步骤可以在新终端重复执行。
- 用户原有修改保持完整；
- `STATUS.md` 记录本地仓库的真实状态和版本。

### 禁止事项

- 不修改 GRPO；
- 不加入 LTL；
- 不迁移数据库；
- 不升级全部依赖；
- 不开始 GPU 训练。

---

## M1：轨迹记录与可重放

### 目标

每一次 Agent 行为都能被持久记录和确定性重放。

### 任务

1. 实现 `TrajectoryStep`；
2. 为 AgentScope 工具调用增加 recorder hook；
3. 每步保存 memory before/after；
4. 保存 observation、action、tool result、env reward；
5. 输出 JSONL；
6. 实现 `TrajectoryReplay`；
7. 确保 replay 不调用 LLM；
8. 加入 schema validation；
9. 为损坏 JSONL、缺失字段和重复 timestep 编写测试。

### 验收标准

- 一次 demo 可以生成完整 JSONL；
- replay 后得到相同的 memory 状态序列；
- 同一文件重复 replay 结果完全一致；
- 轨迹可以按 `task_id / rollout_id / timestep` 查询；
- 单元测试通过。

---

## M2：MemoryStore 抽象与 rollout 隔离

### 目标

把 AgentScope 工具与具体数据库实现解耦。

### 任务

1. 定义 `MemoryStore` protocol；
2. 包装现有 AgeMem memory manager；
3. 实现 `InMemoryStore`；
4. 实现：
   - add；
   - retrieve；
   - update；
   - delete；
   - snapshot；
   - restore；
   - reset；
5. 为每个 `rollout_id` 建立独立 store；
6. 测试两个并行 rollout 不互相可见；
7. update 使用版本化语义，不直接销毁旧版本；
8. delete 在研究模式下使用 soft delete。

### 验收标准

- memory tool 不知道具体后端；
- snapshot/restore 后内容一致；
- 两条 rollout 无状态污染；
- 旧版本可审计；
- memory tests 全部通过。

---

## M3：三阶段 Toy Environment

### 目标

建立一个不依赖 ALFWorld 的最小、确定性、可验证环境。

### 示例任务

```text
Stage 1:
    apple 位于 drawer1
    microwave 位于 kitchen

Stage 2:
    清空 STM
    加入 banana、tomato、sink 等干扰事实

Stage 3:
    Heat apple and put it on dining table
```

### 任务

1. 定义环境状态；
2. 定义合法动作；
3. 定义状态转移；
4. 定义环境成功条件；
5. 输出自然语言 observation；
6. 同时输出仅供调试使用的 Oracle AP；
7. 创建不同难度：
   - 无干扰；
   - 有干扰；
   - 事实更新；
   - 过期事实；
   - 关键记忆被删除；
8. 划分 train/dev/test；
9. 保证 test 中存在未见对象组合。

### 验收标准

- 固定 seed 时轨迹可复现；
- gold action sequence 可以完成所有任务；
- 明确错误的 action sequence 会失败；
- Stage 之间 LTM 保留、STM 按协议重置；
- 测试集不会访问训练答案。

---

## M4：Oracle AP + 手工 DFA + 离线奖励

### 目标

在完全排除 LLM 抽取误差的情况下验证自动机奖励链路。

### 第一版 DFA

```text
q0 --useful_fact_stored--> q1
q1 --useful_fact_retrieved--> q2
q2 --target_object_acquired--> q3
q3 --task_goal_achieved--> q4(accept)
```

### 奖励

第一版只使用：

\[
r_t^{logic}
=
\lambda_{pos}
\cdot
\mathbb{I}[\text{首次发生有效进展转移}]
\]

\[
R_t
=
r_t^{env}
+
\beta r_t^{logic}
+
r_t^{fmt}
\]

### 任务

1. 实现 `AutomatonSpec`；
2. 实现 DFA runner；
3. 实现 progress transition；
4. 每条 progress edge 只奖励一次；
5. 实现 accepting/rejecting/timeout；
6. 实现 `RewardBreakdown`；
7. 对成功和失败 gold traces 离线 replay；
8. 加入循环和重复调用测试；
9. 检查 Agent 无法通过重复 `ADD` 刷奖励。

### 验收标准

- 所有 gold success trace 被接受；
- 预定义 failure trace 不被接受；
- 同一 progress edge 不会重复奖励；
- 轨迹重放奖励完全确定；
- 不依赖在线 LLM；
- 不使用 Trend Shaping。

---

## M5：自然语言三元组抽取与显式状态跟踪

### 目标

用自然语言 observation 替换 Oracle AP，同时保留 Oracle 管线作为上界。

### 任务

1. 定义严格 JSON schema；
2. 实现 `TripleExtractor` 接口；
3. 实现一个 mock extractor；
4. 实现一个 LLM extractor；
5. 对完全相同 observation 做 group batching/cache；
6. 实现 `StateTracker h_t`；
7. 保存 confidence、source step 和有效时间；
8. 实现 Markovian overwrite；
9. 将状态映射为 AP；
10. 人工标注一批 observation；
11. 计算 triple/AP Precision、Recall、F1；
12. 比较 Oracle AP 与 Extracted AP 的自动机结果。

### 验收标准

- 非法 JSON 不会直接进入状态；
- 未知 subject/category 有明确策略；
- 更新事实不会物理删除旧证据；
- 抽取错误可以定位到具体 step；
- 能报告 AP 对最终奖励的误差传播。

---

## M6：Group-Level Logic Critic

### 目标

根据同一任务的 K 条轨迹生成结构化里程碑和依赖关系。

### 输入

```text
task description
K trajectories
terminal outcomes
memory events
AP traces
```

### 输出

```text
milestones
dependencies
bad behavior tags
evidence step IDs
confidence
warnings
```

### 任务

1. 先实现 hand-authored critic；
2. 再实现 LLM critic；
3. 要求每个里程碑提供 evidence；
4. 验证 dependency graph 无环；
5. 验证所有 proposition 已定义；
6. 验证接受状态可达；
7. 验证初始状态不会直接接受；
8. 验证成功轨迹与公式基本一致；
9. 无法验证时回退到 terminal-only reward；
10. MVP 中全失败组不使用反事实奖励，只记录。

### 验收标准

- Critic 输出可被 schema parser 接受；
- 同一输入、低温设置结果足够稳定；
- 无效输出会被拒绝而不是静默使用；
- 每个自动机可追溯到轨迹证据；
- validator 有独立单元测试。

---

## M7：自动机离线验证

### 目标

在开始 RL 前确认自动机与环境成功条件具有合理一致性。

### 必须计算

\[
\text{False Accept Rate}
=
P(\text{DFA accepts}\land\text{environment fails})
\]

\[
\text{False Reject Rate}
=
P(\text{DFA rejects}\land\text{environment succeeds})
\]

此外报告：

- compilation/validation success rate；
- milestone precision/recall；
- violation detection accuracy；
- reward 与 terminal success 的相关性；
- 每条轨迹获得的 progress edge 数；
- 重复工具调用得到的奖励；
- Critic 调用成本和延迟。

### 验收标准

- 报告可以按任务类型拆分；
- 自动机错误可以追溯到 extractor、state tracker 或 critic；
- 不存在明显 reward farming；
- 结果足够稳定后才允许进入 M8。

---

## M8：接入 Trinity-RFT 和 Step-wise GRPO

### 目标

先复现 terminal-only 训练，再增加 DFA reward。

### 集成关系

```text
Explorer:
    为每个 task 采样 K 条完整轨迹

Logic Processor:
    构造 AP、Critic 输出和 DFA

Reward Processor:
    replay 轨迹并写入 step rewards

Buffer:
    保存 token、old logprob、reward、metadata

Trainer:
    计算 group-relative advantage
    更新 policy
```

### 训练顺序

```text
E0：Base model，无训练
E1：Terminal-only GRPO
E2：Terminal + 手工 milestone
E3：Terminal + LLM milestone，无 DFA
E4：Terminal + DFA reward
```

### 任务

1. 复现 AgeMem terminal-only smoke run；
2. 确认 reward adapter 不改变 token/action 对齐；
3. 接入手工 DFA；
4. 小 batch、小数据、短训练运行；
5. 检查 loss、KL、reward 和工具调用频率；
6. 再接入自动 Critic；
7. 保存每次训练的完整配置；
8. 固定 seed 并至少运行多个 seed；
9. 对 checkpoint 做冻结评测。

### 验收标准

- E1 可以稳定运行；
- E2/E4 奖励进入正确 timestep；
- 训练无 NaN、logprob 错位或 rollout 污染；
- checkpoint 在冻结测试中可以加载；
- 结果不是仅由工具调用次数增加造成。

---

## M9：正式 Benchmark 与完整功能

扩展顺序：

```text
1. Toy/PDDL：符号正确性
2. HotpotQA：复现 AgeMem 训练接口
3. ALFWorld：主要任务规划结果
4. ScienceWorld：复杂观察和泛化
5. LongMemEval/MemBench：长期记忆诊断
```

完整功能增加顺序：

```text
1. DELETE 与过期事实
2. Negative Automata
3. SUMMARY / FILTER
4. STM context pressure
5. bounded Trend Shaping
6. 全失败组反事实分析
7. Hard DFA → belief/soft state
8. PostgreSQL + pgvector
```

---

## 11. Baseline 与消融矩阵

### 11.1 外部基线

| ID | 方法 | 目的 |
|---|---|---|
| B0 | ReAct，无外部记忆 | 基础 Agent 能力 |
| B1 | Full History / Sliding Window | 仅依赖上下文 |
| B2 | Static RAG | 固定存储和 Top-k 检索 |
| B3 | A-MEM 或 Mem0 | 非 RL 结构化记忆 |
| B4 | Memory-R1 | 分离 Memory Manager |
| B5 | AgeMem | 统一记忆策略 |
| B6 | Ours | AgeMem + LTLf/DFA Reward |

### 11.2 奖励对照

| ID | 奖励 | 验证内容 |
|---|---|---|
| R0 | Terminal only | 稀疏奖励基线 |
| R1 | Heuristic dense reward | 人工过程奖励 |
| R2 | LLM per-step judge | 非结构化语义奖励 |
| R3 | Flat milestones | 里程碑但无状态机 |
| R4 | LTLf/DFA | 完整方法 |

### 11.3 核心消融

```text
A1：去掉 Logic Reward
A2：Oracle AP → Extracted AP
A3：去掉显式 h_t
A4：去掉 Negative Automata
A5：去掉 Trend Shaping
A6：去掉 counterfactual milestones
A7：统一 Agent → 分离 Memory Manager
A8：训练时 DFA → 训练+测试 runtime monitor
```

---

## 12. 评价指标

### 12.1 任务指标

- Task Success Rate；
- Normalized Environment Score；
- 平均完成步数；
- timeout rate；
- invalid action rate。

### 12.2 记忆指标

- Retrieval Recall@k；
- 关键事实保留率；
- 过期记忆率；
- 冲突记忆率；
- update accuracy；
- 平均 memory items；
- 平均 memory tool calls；
- retrieved context tokens。

### 12.3 符号指标

- Triple Precision / Recall / F1；
- AP Precision / Recall / F1；
- milestone Precision / Recall；
- automaton validation rate；
- LTLf compilation rate；
- DFA satisfaction rate；
- False Accept Rate；
- False Reject Rate。

### 12.4 训练与效率

- success-vs-rollout curve；
- 达到固定成功率所需样本；
- 多 seed 均值和标准差；
- 训练 token；
- 推理 token；
- Critic 调用次数；
- 训练时间和推理延迟；
- GPU 和 API 成本。

---

## 13. Reward 设计规则

### 13.1 不奖励裸工具调用

错误：

```text
调用 ADD 就奖励
调用 RETRIEVE 就奖励
```

否则模型会通过滥用工具刷奖励。

应奖励：

```text
stored_useful_fact
retrieved_current_relevant_fact
used_retrieved_fact_for_progress
updated_stale_fact
```

### 13.2 里程碑奖励只发一次

```python
if transition.is_progressive and transition.edge_id not in visited_edges:
    reward += lambda_pos
    visited_edges.add(transition.edge_id)
```

### 13.3 MVP 不使用无条件 Trend Reward

如果每个里程碑之间的所有 timestep 都获得正奖励，Agent 可能通过拖延完成任务来累计回报。

后续使用 Trend Shaping 时必须：

- 有总上限；
- 与实际状态势函数绑定；
- 不允许循环累积；
- 单独做无 Trend 消融。

### 13.4 逻辑奖励不能淹没环境奖励

配置中分别记录：

```yaml
reward:
  env_weight: 1.0
  logic_beta: null
  milestone_weight: null
  violation_weight: null
  trend_weight: 0.0
```

具体系数通过 dev set 调整，不在代码中硬编码。

---

## 14. 测试策略

### 14.1 单元测试

必须覆盖：

```text
memory add/update/delete/retrieve
snapshot/restore/reset
rollout isolation
trajectory serialization
triple schema validation
state overwrite/versioning
AP grounding
DFA transition
accept/reject
edge reward once-only
reward replay determinism
critic output validation
```

### 14.2 集成测试

```text
Agent → tool → memory → ToolResponse
ToyEnv → observation → extractor → state → AP
AP trace → DFA → step rewards
K rollouts → critic → automaton → replay
Explorer → reward adapter → Buffer
```

### 14.3 测试原则

- 单元测试不能调用真实 LLM；
- 使用固定 fixtures 和 mock responses；
- 真实模型测试使用 `integration` 标记；
- 网络/API 测试默认不运行；
- 每个已修复 bug 必须加入回归测试；
- 不允许只用打印输出代替断言。

---

## 15. Codex 工作规则

Codex 在执行本项目时必须遵循以下规则。

### 15.1 开始每个阶段前

1. 完整阅读本文件；
2. 阅读 `STATUS.md`；
3. 查看 `git status`；
4. 阅读本阶段涉及的上游 README；
5. 给出仅覆盖当前阶段的计划；
6. 确认不会覆盖用户已有修改。

### 15.2 实施过程中

- 一次只完成一个 milestone；
- 优先写最小可运行代码；
- 修改后运行相关测试；
- 不在无测试的情况下大规模重构；
- 不把 API key、token、数据库密码写入代码；
- 所有实验配置写入 YAML/JSON；
- 所有轨迹和结果带 `task_id / rollout_id / seed`；
- 记录外部仓库 commit；
- 不把 LLM 输出视为可信输入，必须 schema validate；
- 如果发现设计假设错误，先更新文档和状态，不静默绕过。

### 15.3 每个阶段结束时

Codex 必须更新 `STATUS.md`：

```text
完成了什么
修改了哪些文件
运行了哪些命令
哪些测试通过
有哪些失败
当前已知风险
下一步建议
```

### 15.4 阻塞时

如果遇到以下情况，应停止并向用户说明：

- 需要选择训练模型或 GPU 配置；
- 需要付费 API；
- 上游依赖存在重大版本冲突；
- 需要改变研究问题；
- 需要下载大型模型或数据集；
- 需要删除或覆盖大量用户文件；
- 自动机定义与 benchmark 目标不一致。

---

## 16. VS Code + Codex 使用方式

### 16.1 推荐打开方式

第一阶段直接打开已有项目：

```text
VS Code
    ↓
File → Open Folder
    ↓
D:\Project\Age-Mem\AgeMem
    ↓
在该窗口中打开 Codex 插件
```

进入 RL 和 Linux-only 依赖阶段后，推荐把仓库通过 Git 同步到 WSL 的：

```text
~/projects/Age-Mem
```

然后使用 VS Code 的 “Open Folder in WSL”。不要让 Codex直接复制一个包含未提交修改的工作区；先检查并保存 Git 状态。

### 16.2 第一次交给 Codex 的提示词

在 Codex 插件中粘贴：

```text
请完整阅读 PROJECT_HANDOFF.md。

当前项目根目录是 D:\Project\Age-Mem\AgeMem，并且已有代码。
只执行 M0：已有仓库接管检查与 standalone demo 复现。

先只读检查系统、Python、Git 状态、当前分支、remote、未提交修改、
目录结构、requirements、已有虚拟环境和 AgeMem_code_agentscope，
然后给出计划。不要 git init，不要重复 clone，不要覆盖或清理用户修改。

不要开始 LTL、自动机或强化学习实现。
不要写入任何 API 密钥。
完成后创建并更新 STATUS.md，运行能够安全执行的 smoke test，
最后汇报修改文件、测试结果、阻塞项和下一步。
```

### 16.3 M1 提示词

```text
请阅读 PROJECT_HANDOFF.md 和 STATUS.md。
只执行 M1：轨迹记录与可重放。
先定位 AgentScope 的消息循环、memory tool 调用点和 ToolResponse 返回点，
设计最小 TrajectoryStep schema，再实现 JSONL recorder 和无 LLM replay。
为序列化、重复 timestep、缺失字段和 replay determinism 编写测试。
不要开始 M2 或修改训练逻辑。
```

### 16.4 M2 提示词

```text
请阅读 PROJECT_HANDOFF.md 和 STATUS.md。
只执行 M2：MemoryStore 抽象与 rollout 隔离。
保持现有 AgentScope 工具接口兼容，先实现 InMemoryStore、
snapshot/restore/reset 和 rollout_id 隔离。
update 使用版本化语义，delete 使用 soft delete。
完成后运行 memory 和 integration tests 并更新 STATUS.md。
```

### 16.5 M3-M4 提示词

```text
请阅读 PROJECT_HANDOFF.md 和 STATUS.md。
实现 M3 和 M4，但不要调用真实 LLM：
先建立三阶段 Toy Environment 和 Oracle AP，
再实现手工 DFA、once-only milestone reward 和离线 trajectory replay。
必须包含成功、失败、重复 ADD、循环动作和过期事实测试。
如果 Oracle AP 管线未通过，不要进入自然语言抽取。
```

### 16.6 M5-M7 提示词

```text
请阅读 PROJECT_HANDOFF.md 和 STATUS.md。
按 M5、M6、M7 顺序工作：
先定义严格的 Triple/Critic JSON schema 和 mock，
再接入真实 LLM extractor/critic。
所有输出必须经过 validator。
计算 AP F1、False Accept、False Reject 和 reward determinism。
不要开始 GRPO，直到离线验证报告生成并通过验收标准。
```

### 16.7 M8 提示词

```text
请阅读 PROJECT_HANDOFF.md、STATUS.md、AgeMem 和 Trinity-RFT 文档。
只执行 M8。
先复现 terminal-only smoke training，再增加手工 DFA reward。
确认逐步奖励与 token/action timestep 对齐，检查 NaN、KL、logprob、
rollout memory isolation 和 checkpoint loading。
不要直接开始全量 benchmark。
```

---

## 17. 推荐接管与复现命令

项目已经存在，因此本节改为“接管检查命令”。以下命令是交接参考，不应盲目执行。Codex 必须先检查系统和上游 README。

### 17.1 Windows PowerShell 只读检查

```powershell
Set-Location 'D:\Project\Age-Mem\AgeMem'
Get-Location
Get-ChildItem -Force
git status --short --branch
git remote -v
git log -1 --oneline
py --version
Get-ChildItem -Force .venv, venv -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -File -Filter 'requirements*.txt'
```

如果 `git status` 失败，Codex 只记录该情况，不执行 `git init`，先向用户确认该目录是否只是源码副本。

### 17.2 建立环境

只有在确认没有可复用环境后，才根据本地 README 建立新环境。示例：

```powershell
Set-Location 'D:\Project\Age-Mem\AgeMem'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\AgeMem_code_agentscope\requirements.txt
```

如果 PowerShell 阻止激活脚本，不要未经用户同意修改全局执行策略；可以直接使用：

```powershell
.\.venv\Scripts\python.exe
```

### 17.3 运行 standalone demo

复现时优先采用本地仓库 README 中的实际命令。典型入口可能是：

```powershell
$env:DASHSCOPE_API_KEY = Read-Host 'Enter DASHSCOPE_API_KEY locally'
python -m AgeMem_code_agentscope.main
```

注意：

- 实际模块路径取决于本地仓库结构和版本；
- 不要把真实 key 写入 shell history、文档或仓库；
- 可以创建 `.env.example`，但只能写变量名；
- `.env` 必须加入 `.gitignore`。
- 用户应自行在本地终端设置密钥，不要在 Codex 对话中发送密钥；
- 如果项目当前只能在 WSL/Linux 运行，应记录具体原因，再制定迁移方案。

---

## 18. STATUS.md 模板

Codex 在 M0 创建：

```markdown
# Project Status

## Current milestone

M0

## Completed

- [ ] Existing repository inspected
- [ ] User changes recorded and protected
- [ ] Environment created
- [ ] Standalone demo runs
- [ ] Memory tool smoke test passes

## Environment

- OS:
- Python:
- PyTorch:
- CUDA:
- AgeMem commit:
- AgentScope version:
- Trinity-RFT commit:

## Commands run

## Tests

## Known issues

## User decisions needed

## Next recommended action
```

---

## 19. 风险清单

| 风险 | 表现 | 缓解方式 |
|---|---|---|
| AP 抽取错误 | DFA 错误转移 | Oracle AP 上界、confidence、版本证据 |
| Critic 幻觉 | 不存在的里程碑 | evidence step、schema、validator |
| 奖励作弊 | 重复 ADD/拖延 | edge once-only、工具成本、无 Trend MVP |
| 奖励泄漏 | Critic 使用测试答案 | Critic 仅训练使用，测试 shadow execution |
| 状态污染 | rollout 相互读写 | 独立 store、snapshot、隔离测试 |
| 依赖冲突 | AgentScope/Trinity 无法运行 | 固定上游 commit、先复现再升级 |
| 训练成本过高 | 无法完成实验 | toy→subset→full，先小模型/LoRA |
| 数据库瓶颈 | 并行 rollout 变慢 | rollout 使用 InMemoryStore |
| 逻辑奖励压过任务奖励 | 接受率高但任务失败 | False Accept、reward scale ablation |
| 全失败反事实不可靠 | 伪里程碑 | MVP 关闭，后续人工子集验证 |

---

## 20. 最终论文级交付物

代码：

- 可运行的 AgentScope Agent；
- MemoryStore 两种后端；
- 三阶段任务协议；
- TraceRecorder 与 Replay；
- Triple Extractor；
- StateTracker；
- Group-Level Critic；
- LTLf/DFA compiler 与 runner；
- Logic Reward；
- Trinity-RFT workflow；
- benchmark adapters；
- 单元和集成测试。

实验：

- external baseline；
- reward baseline；
- ablation；
- AP/critic/automaton 诊断；
- 多 seed；
- 成本和效率分析；
- 典型成功、失败和 reward-hacking case study。

文档：

- 环境复现；
- 数据格式；
- 奖励设计；
- benchmark 协议；
- 训练和评测命令；
- 已知限制；
- 可复现实验配置。

---

## 21. 项目完成定义

只有同时满足以下条件，才能认为项目核心完成：

1. 新机器按 README 可以复现环境；
2. 所有核心模块有测试；
3. 轨迹可以确定性重放；
4. Oracle AP 与 Extracted AP 结果分别报告；
5. Critic 和 DFA 错误可追溯；
6. Terminal-only baseline 已复现；
7. DFA reward 在至少一个交互 benchmark 上有稳定结果；
8. 提升不依赖测试时自动机干预；
9. False Accept/Reject 被明确报告；
10. 训练配置、seed、模型、commit 和数据 split 完整保存；
11. 没有凭据或敏感信息进入仓库；
12. 论文结论不超过实验实际支持范围。

---

## 22. 立即执行的第一步

Codex 当前只应执行：

```text
M0：已有仓库接管检查与上游复现
```

不要直接实现完整方案。正确顺序是：

```text
检查并保护用户已有项目
    ↓
复现 Agent 和记忆工具
    ↓
记录并重放轨迹
    ↓
隔离 MemoryStore
    ↓
Toy Environment + Oracle AP
    ↓
手工 DFA Reward
    ↓
自然语言抽取
    ↓
自动 Critic
    ↓
离线验证
    ↓
GRPO
    ↓
正式 Benchmark
```

这条顺序是本项目最重要的工程约束。
