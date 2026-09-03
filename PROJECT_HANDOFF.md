# 神经符号 Agent Memory 项目交接文档

> 面向：VS Code 中的 Codex 插件  
> 项目方向：AgeMem 式可学习记忆管理 + GLARE 式 LTLf/DFA 逻辑奖励  
> 文档版本：v2.1<br>
> 更新时间：2026-09-03<br>
> 本地项目根目录：`D:\Project\Age-Mem\AgeMem`  
> 当前状态：M0～M7 已完成；目标模型限定为 1.5B 与 4B、暂不考虑 7B；M8a、1.5B M8b-prep 与 Stage 1/2 反捷径 canary/stress 离线门禁已实现，4B 将使用独立模型/config lock。部署目标已从 AutoDL 改为组内远程 GPU 服务器，工作区根目录为 `/data/hjx/Age_mem`。1.5B 远程准备已推进到步骤十二“使用冻结 tokenizer 重跑反捷径 stress”；本地已将 Stage 2 统一预算改为 19、重生成 lexical stress 报告，并把 runtime gate 锁更新为 `m8a=142`、`all=318`。该修复包含在当前提交中；远程冻结 Qwen tokenizer 重跑、真实严格预检、E0、E1 单次更新和 checkpoint 新进程重载均尚未执行

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

主实验中 Agent 的动作空间包括：

```text
任务输出：
    ANSWER

长期记忆操作：
    ADD / UPDATE / DELETE

短期上下文操作：
    RETRIEVE / SUMMARY / CLEAR
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
- 一个 HotpotQA 风格的三阶段 toy memory environment；
- 完整、可重放的 JSONL 轨迹；
- Oracle AP；
- 一个手工定义的正向 DFA；
- Milestone Reward；
- Terminal-only 和 Terminal+DFA 两种设置；
- 单元测试和离线奖励重放；
- 在 HotpotQA 小数据子集上完成小规模 GRPO smoke test。

### 3.2 原始奖励 MVP 暂时不做

- `SUMMARY / CLEAR / DELETE` 的在线 DFA 奖励；
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

Stage 1/2 反捷径 sidecar 中的 KEEP/CLEAR/COMPRESS 是固定离线诊断策略，
没有改动 E1 `terminal_only` 配置。在线 SUMMARY/CLEAR 已存在于 E1 工具与
Experience 路径；当前尚未接入的是其 AP/DFA 奖励。

---

## 4. 当前工作区状态

用户本地项目位置：

```text
D:\Project\Age-Mem\AgeMem
```

截至 2026-09-02，Codex 已在 Windows `D:` 盘工作区完成本地检查、1.5B M8a/M8b-prep、A6000 双卡选择门禁与 anti-shortcut stress 实现。后续会话仍必须先读取 `STATUS.md` 并重新执行只读检查，不能把本文记录当作实时 Git 状态：

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

截至 2026-09-02，本地代码、报告和 scoped tests 核验后的阶段状态为：

```text
M0 已完成：已有仓库接管与上游复现
M1 已完成：轨迹记录与可重放
M2 已完成：MemoryStore 抽象与 rollout 隔离
M3 已完成：HotpotQA 风格三阶段 Toy Memory Environment
M4 已完成：Memory Oracle AP + 手工 DFA + 离线奖励
M5 已完成：真实 HotpotQA 数据适配与 Oracle Benchmark
M6 已完成：自然语言三元组抽取、显式状态跟踪与 False Reject 收尾
M7 已完成：Group Critic 与自动机离线验证；真实 LLM 调用为 0
M8a 本地门禁实现已完成，但仍有两项未关闭：组内远程 GPU 服务器 runtime/E1 smoke 尚未执行；在线 `ActionCreditRecord` 自动生成器尚未实现（后者属于 E3/E4）
M8b-prep 已完成：模型/数据/配置锁、严格预检、provider 遥测、运行时 receipt、E0/E1/checkpoint eval 与 fail-closed 一键脚本
Stage 1/2 反捷径 sidecar 已完成：保留固定 v2 CI canary，并新增 16-task/50-seed/3-budget Stage 1 与成对反事实 Stage 2 stress；两套报告均不改写 E1 或 M3～M7 artifact
```

“已完成”仍须以当前工作区、`STATUS.md`、报告 digest 和测试结果共同核验。M8a/M8b-prep 只表示上卡前契约、门禁和执行编排已建立，不表示远程服务器预检、E0、E1、optimizer update 或 checkpoint 重载已通过。若历史实现与当前数据契约不一致，优先做非破坏性兼容或迁移，不重写已完成阶段。

---

## 5. 推荐上游项目与技术栈

### 5.1 上游项目

| 组件 | 推荐项目 | 用途 |
|---|---|---|
| 直接基线 | [AgeMem](https://github.com/y1y5/AgeMem) | 记忆工具、AgentScope demo、Trinity-RFT 工作流 |
| Agent 层 | [AgentScope](https://github.com/agentscope-ai/agentscope) | Agent 循环、消息、工具调用 |
| RFT 层 | [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) | Explorer、Buffer、Trainer、GRPO |
| 主训练数据 | [HotpotQA](https://hotpotqa.github.io/) | 多跳问答、supporting facts、AgeMem 三阶段训练 |
| 跨域扩展 | 2WikiMultiHopQA / MuSiQue | 检验记忆策略跨数据集泛化 |
| 可选规划环境 | [ALFWorld](https://github.com/alfworld/alfworld) | 仅在需要证明任务规划泛化时使用 |
| LTLf 编译 | [LTLf2DFA](https://github.com/whitemech/ltlf2dfa) | LTLf 转最小 DFA |
| 最终数据库 | [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL 中的向量检索 |

### 5.2 推荐运行环境

```text
日常代码阅读/standalone demo：Windows 或 WSL2 均可
RL训练、HotpotQA 全量实验、MONA/LTLf：优先 WSL2 Ubuntu 或原生 Linux
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
2. 进入 Trinity-RFT、HotpotQA 全量训练或 LTLf/MONA 阶段前，在 WSL Linux 文件系统中建立训练副本，例如：

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
│   ├── hotpotqa.yaml
│   ├── alfworld.yaml              # M9 可选扩展
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
│       │   ├── hotpotqa.py
│       │   └── alfworld.py        # M9 可选扩展
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
    action_id: str
    operation: str
    target_memory_id: str | None
    arguments: dict
    result: dict
    rollout_id: str
    timestep: int
```

### 9.3 ActionEvent 与 TrajectoryStep

```python
class ActionEvent:
    # 跨 replay、AP grounding、DFA reward 和训练 buffer 的稳定连接键
    action_id: str
    task_id: str
    rollout_id: str
    stage_id: int
    timestep: int
    assistant_turn_id: int
    action_index_in_turn: int

    # oracle / random / error_injector / llm
    source: str
    action_type: str
    action_text: str
    arguments: dict
    result: dict

    # 仅 LLM rollout 必须存在；规则轨迹允许为 None
    response_token_ids: list[int] | None
    token_start: int | None
    token_end: int | None
    old_logprobs: list[float] | None
    policy_version: str | None


class TrajectoryStep:
    schema_version: str
    task_id: str
    rollout_id: str
    stage_id: int
    timestep: int
    observation: str
    actions: list[ActionEvent]
    memory_before: list[dict]
    memory_after: list[dict]
    env_reward: float
    done: bool
```

兼容原则：历史 `TrajectoryStep.stage` 可以在读取时映射为 `stage_id`；历史单值 `old_logprob` 不得伪装为 token-level `old_logprobs`。同一 assistant turn 产生多个工具调用时，每个调用必须有独立 `action_id` 和 `action_index_in_turn`。

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
    cost: float
    total: float
    automaton_state_before: str
    automaton_state_after: str
```

读取 v1.3 及更早版本的派生奖励记录时，缺失的 `cost` 按 `0.0` 处理；兼容读取不能覆盖原记录。

### 9.8 ActionCreditRecord

M3～M5 的原始动作轨迹保持不可变；M4 之后产生的 AP、DFA 状态和奖励以派生记录保存，并通过 `action_id` 关联：

```python
class ActionCreditRecord:
    schema_version: str
    action_id: str
    atomic_propositions: list[str]

    dfa_spec_id: str
    transition_id: str | None
    dfa_state_before: str
    dfa_state_after: str

    reward_breakdown: RewardBreakdown
    return_to_go: float | None
    advantage: float | None
    reward_version: str
```

这样可以在不重新采样轨迹的情况下重算 AP、DFA、奖励、Return-to-Go 和 Advantage。`return_to_go` 只作为对应动作的训练权重，不能再次求和作为轨迹总奖励。

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

## M3：HotpotQA 风格三阶段 Toy Memory Environment

### 目标

建立一个与 AgeMem 三阶段训练语义一致、但不依赖完整 HotpotQA 数据和在线 LLM 的最小确定性环境。M3 不再模拟 `open/take/put` 等 ALFWorld 物体操作。

### 示例任务

```text
Stage 1：记忆构建
    观察事实 A：The Eiffel Tower is located in Paris.
    观察事实 B：Paris is the capital of France.
    Agent 决定是否 ADD / UPDATE。

Stage 2：查询延迟的上下文干扰
    问题和答案尚未公开；重置或压缩 STM，加入若干干扰事实。
    M3 只验证关键事实仍保留在 LTM；SUMMARY/CLEAR 暂不作为必需动作。

Stage 3：检索问答
    问题：Which country contains the city where the Eiffel Tower is located?
    Agent 需要 RETRIEVE supporting facts 后回答 France。
```

### 任务

1. 定义 `MemoryEpisode`、`StageInput` 和阶段切换协议；
2. 使用 20～50 条人工合成的两跳事实任务；
3. 为每条任务标注：
   - `supporting_fact_ids`；
   - `distractor_fact_ids`；
   - `stale_fact_ids`；
   - `answer`；
4. 输出自然语言 observation 和仅供调试/监督使用的 Oracle labels；
5. 复用 M1 的轨迹记录器和 M2 的 rollout 隔离；
6. 覆盖无干扰、有干扰、事实更新、重复事实、过期事实和关键记忆误删；
7. 划分 train/dev/test，并让 test 包含未见实体组合；
8. 实现不调用 LLM 的 gold policy 和明显错误 policy。

### 验收标准

- 固定 seed 时 episode 和轨迹完全可复现；
- gold policy 能完成全部任务；
- 错误 policy 会在预期条件下失败；
- Stage 之间 LTM 保留，STM 按协议重置；
- 每个 rollout 使用独立 MemoryStore；
- 不读取真实 HotpotQA test 答案，不发生答案泄漏。

---

## M4：Memory Oracle AP + 手工 DFA + 离线奖励

### 目标

在排除自然语言抽取误差和 RL 不稳定性的情况下，验证“记忆语义事件 → DFA 转移 → 逐步奖励”链路。

### 第一版原子命题

```text
observed_supporting_fact
stored_supporting_fact
stored_irrelevant_fact
updated_stale_fact
deleted_supporting_fact
retrieved_supporting_fact
retrieved_irrelevant_fact
supporting_coverage_complete
answered_correctly
```

这些命题描述动作产生的语义结果，而不是裸工具调用。`ADD` 本身不能直接获得奖励。

### 第一版 DFA

```text
q0 --stored_supporting_fact--> q1
q1 --supporting_coverage_complete--> q2
q2 --retrieved_supporting_fact--> q3
q3 --answered_correctly--> q4(accept)
```

`updated_stale_fact` 可作为并行进展边；`stored_irrelevant_fact`、`deleted_supporting_fact` 和无关检索先记录为 violation，MVP 中再决定是否启用负奖励。

### 奖励

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
r_t^{task}
+
\beta r_t^{logic}
+
r_t^{fmt}
\]

### 任务

1. 实现 `AutomatonSpec`、DFA runner 和 `RewardBreakdown`；
2. 从 M3 Oracle labels 生成 AP，而不是调用 LLM；
3. 每条 progress edge 只奖励一次；
4. 实现 accepting/rejecting/timeout；
5. 对成功、失败、重复调用和循环轨迹离线 replay；
6. 验证重复 `ADD/RETRIEVE` 不能刷奖励；
7. 分别保存 task reward、logic reward 和 format reward；
8. 第一版不启用 Trend Shaping。

### 验收标准

- 所有 gold success traces 被接受；
- 预定义 failure traces 不被接受；
- 同一 progress edge 不重复奖励；
- 奖励重放完全确定；
- 自动机只奖励有效内容变化，不奖励裸工具调用；
- 整条链路不依赖在线 LLM。

---

## M5：真实 HotpotQA 数据适配与 Oracle Benchmark

### 目标

将 M3/M4 的确定性管线接到真实 HotpotQA，同时继续使用 `supporting_facts` 生成 Oracle AP。此阶段只做 rollout、离线奖励和评测，不训练模型。

### 三阶段构造

```text
Stage 1：按消息或段落提供候选事实，Agent 使用 ADD/UPDATE 构建 LTM
Stage 2：问题尚未公开，注入 distractor、施加 STM token budget，并进行 SUMMARY/CLEAR 或离线上下文控制
Stage 3：公开问题，Agent 使用 RETRIEVE 获取 supporting facts 并回答
```

### 任务

1. 适配 AgeMem 已有 HotpotQA fullwiki/distractor 数据读取逻辑；
2. 将 `supporting_facts` 映射为稳定的 fact IDs；
3. 明确 train/dev/test，禁止把答案文本写进 Agent observation；
4. 构造小规模 smoke split，再扩展到正式 split；
5. 收集 base model、规则策略和 gold policy 轨迹；
6. 用 Oracle AP 运行 M4 自动机；
7. 计算 Answer EM/F1、supporting-fact coverage、Memory Precision、Retrieval Recall@k、context tokens 和工具调用次数；
8. 保存每条失败轨迹对应的事实、记忆状态和自动机状态。

### 验收标准

- 小规模 HotpotQA episode 可确定性生成与重放；
- supporting fact 的匹配不依赖脆弱的纯字符串包含；
- Oracle AP 自动机可以解释成功和失败轨迹；
- 数据 split 和答案不可见性有测试；
- 在进入 M6 前生成一份 Oracle benchmark 报告。

---

## M6：自然语言三元组抽取与显式状态跟踪

### 目标

用自然语言 observation 和工具结果生成 Extracted AP，同时保留 M5 Oracle AP 作为监督上界。开始抽取器实现前，先完成一次历史轨迹 Schema 审计，确保 M3～M5 产物可以按动作关联后续 AP、DFA 奖励和动作级 GRPO。

### 任务

1. 阅读 `STATUS.md`、M5 Oracle benchmark 报告和 M3～M5 轨迹样本；
2. 审计并报告以下字段的真实存在性和可用性：
   - `action_id`；
   - `stage_id`；
   - `assistant_turn_id / action_index_in_turn`；
   - `token_start / token_end`；
   - `response_token_ids / old_logprobs / policy_version`；
   - `ActionCreditRecord / RewardBreakdown`；
3. 对规则/oracle/error-injector 轨迹允许 token 和 logprob 字段为 `None`；不得为通过校验而伪造数值；
4. 若历史轨迹缺少稳定动作键，编写确定性迁移器和迁移测试，保留原始文件并输出新 schema 版本；
5. 定义严格的 Triple/AP JSON schema；
6. 实现 `TripleExtractor` 接口、mock extractor 和 LLM extractor；
7. 规则映射确定性的 memory tool actions；
8. 对相同 observation 做 group batching/cache；
9. 实现 `StateTracker h_t`、版本信息、source step 和 confidence；
10. 对冲突事实执行 Markovian overwrite，但保留历史证据；
11. 将状态和 memory delta 映射为 AP；
12. 每个 AP 和派生奖励必须通过 `action_id` 追溯到原始动作；
13. 使用 HotpotQA supporting facts 和人工小样本评估抽取；
14. 计算 Triple/AP Precision、Recall、F1；
15. 比较 Oracle AP 与 Extracted AP 的奖励、接受状态和错误传播。

### 验收标准

- 非法输出不会进入状态；
- 未知 subject/category 有明确策略；
- 抽取错误可以定位到具体 timestep；
- 更新事实不会物理删除旧证据；
- 能报告 Oracle→Extracted 导致的 False Accept/Reject 增量；
- `ActionEvent ↔ ActionCreditRecord` 的 `action_id` 连接完整且唯一；
- 历史轨迹迁移可重复、可回滚，不覆盖 M3～M5 原始产物；
- 在进入 M7 前生成 `docs/schema_audit_m6.md` 和抽取评测报告。

---

## M7：Group Critic 与自动机离线验证

### 目标

先证明手工 memory DFA 可靠，再把 Group-Level Logic Critic 作为可替换的自动里程碑生成器。自动 Critic 不是进入 M8 的硬依赖。

### Critic 输入输出

```text
输入：当前任务唯一一条 `critic_only_privileged` HotpotQA fullwiki 完整记录（question、answer、context、official supporting_facts）以及 K trajectories、terminal outcomes、memory events、AP traces；私有 Oracle 记录只进入 Critic prompt/cache digest，不进入 policy observation
输出：milestones、dependencies、bad behavior tags、evidence step IDs、confidence、warnings
```

### 任务

1. 保留 M4 手工 DFA 作为主基线；
2. 实现结构化 Critic schema、mock critic 和 LLM critic；
3. 要求每个里程碑引用证据 step；
4. 验证依赖图无环、命题已定义、接受状态可达且初态不接受；
5. 无效 Critic 输出回退到手工 DFA 或 terminal-only reward；
6. MVP 中全失败组只记录反事实建议，不把它直接作为训练奖励；
7. 在真实 HotpotQA 离线轨迹上计算 False Accept Rate 和 False Reject Rate；
8. 检查 reward farming、循环奖励、重复工具奖励和 Critic 稳定性；
9. 输出按问题类型、轨迹长度和干扰强度拆分的离线报告。

### 验收标准

- 手工 DFA 在离线 HotpotQA 轨迹上达到可接受的一致性；
- 自动机错误可追溯到 extractor、state tracker、critic 或数据适配层；
- 无效 Critic 输出不会被静默采用；
- 不存在明显 reward farming；
- M8 可以先使用手工 DFA，不等待自动 Critic 完美。

---

## M8：HotpotQA 上接入 Trinity-RFT 和动作级 GRPO

### 目标

先复现 AgeMem 的 HotpotQA terminal-only 轨迹级 GRPO，再逐步增加 memory DFA reward 和动作级信用分配。奖励改进与信用分配改进必须作为两个独立变量进行对照。

### 集成关系

```text
Explorer：为同一个 HotpotQA task 采样 K 条完整三阶段轨迹
Logic Processor：构造 Oracle/Extracted AP，并选择手工 DFA 或 Critic DFA
Reward Processor：重放轨迹，通过 action_id 把 reward 写回对应动作
Buffer：保存 ActionEvent、ActionCreditRecord、token-level old_logprobs 和 policy_version
Trainer：先支持 trajectory advantage，再支持 stage/DFA-state-conditioned action advantage
```

### 训练顺序

```text
E0：Base model，无训练
E1：AgeMem Terminal-only GRPO
E2：Terminal + Heuristic Dense Reward
E3：Terminal + Oracle AP + Hand-authored DFA（监督上界）
E4：Terminal + Extracted AP + Hand-authored DFA + Trajectory Advantage
E5：Terminal + Extracted AP + Hand-authored DFA + Action-level Advantage（主方法）
E6：Terminal + Extracted AP + Group Critic DFA + Action-level Advantage（M7 稳定后可选）
```

### M8a：上卡前本地门禁（已完成）

M8a 不执行模型训练，只关闭 E1 在租卡前可以用 CPU/静态检查发现的问题：

- 支持读取本地 Hugging Face `save_to_disk` DatasetDict，按 M5 manifest 固定选择 6 条 source-train 样本，并在运行时核对 train fingerprint 与 Hotpot ID；
- 新增 `agemem_e1_dry_run.yaml`：2 GPU、K=2、固定干扰、`multi_step_grpo + step_wise_grpo`、1 个 trainer step；
- E1 只使用确定性 HotpotQA terminal answer F1，不加入工具、记忆、context、timeout 或 DFA reward；
- 复用 M2 rollout-scoped、版本化、soft-delete MemoryStore；
- 在线 LLM 工具动作保存稳定 `action_id`、完整 response token IDs、逐 token old logprobs、token span 和冻结 policy version；
- K 组必须在同一 WorkflowRunner 的 policy-freeze 窗口，组内混合 policy version 时 fail closed；
- ActionEvent 必须与 ToolTrace、Experience task/rollout/stage/timestep 及可选 ActionCreditRecord 精确连接；最终 buffer 边界重算 character→token span 并重查 ToolTrace join；
- 规则、Oracle、random 和 error-injector 轨迹禁止进入 on-policy buffer；AgeMem pipeline 在 operator 前后均强制契约存在，删除或篡改时 fail closed；
- Trinity wheel 同时包含 `trinity*` 与复用的 `AgeMem_code_agentscope*` 契约包。

原始 M8a scoped 结果已被 M8b 冻结 runtime gate 取代。当前锁定发现数为 `m8a=142`、`all=318`；少跑、漏跑、数量漂移、FAIL、ERROR、unexpected success 或任意 SKIP 都判失败。本地因缺 PyTorch/Ray/vLLM 仍有 3 个环境性 SKIP，只能作为诊断，必须在组内远程服务器完整 Linux 环境变为 PASS。

当前 E1 仍使用 DashScope embedding，SUMMARY/CLEAR 仍可能调用 `qwen-max`；只有 terminal reward 与固定 distractor 已去除辅助 LLM 调用。M8b-prep 已为首轮 smoke 冻结 endpoint、embedding/chat model，并记录无正文的调用、错误、延迟和 usage；provider 不返回货币金额时保持 `None` 并在实验后与账单对账。M8a 也尚未在线生成 DFA `ActionCreditRecord`，E3/E4/E5 不得提前宣称完成。

### M8b-prep：远程 GPU 上卡前执行包（已完成）

M8b-prep 只完成可执行的上卡前约束与证据链，不表示任何真实 GPU 阶段成功。冻结输入和门禁如下：

- `configs/m8b_autodl_preflight.json` 锁定 E1、E0、checkpoint eval 三份 YAML 的 canonical UTF-8/LF SHA-256、M5 manifest digest、依赖范围和精确测试发现数；
- E1 固定 M5 的 6 条 source-train 样本、K=2、1 个 trainer step；E0 与 checkpoint eval 固定 2 条 held-out validation 样本。预检同时核对 fullwiki 三个 split 的大小/fingerprint、8 个 Hotpot ID 和每条选中记录的 canonical JSON 内容 SHA-256，不能只凭 ID 或 fingerprint 放行；
- 模型固定为 `Qwen/Qwen2.5-1.5B-Instruct` 的小写 40 位 commit revision。下载完成且目录不再变化后，必须先设置 `TRINITY_MODEL_REVISION`，再生成一次离线逐文件清单：

```bash
export TRINITY_MODEL_REVISION=<Qwen模型的完整40位commit revision>
python scripts/agemem_m8b_model_manifest.py \
  --model-path "$TRINITY_MODEL_PATH" \
  --repository-id Qwen/Qwen2.5-1.5B-Instruct \
  --revision "$TRINITY_MODEL_REVISION"
```

生成的 `$TRINITY_MODEL_PATH/.agemem_model_manifest.json` 保存 repository/revision、完整文件集合、大小和 SHA-256。严格预检会重新计算所有物料文件，核对 Qwen2.5-1.5B 结构、tokenizer/chat template、单文件 `model.safetensors` 和不少于 3 GB 的权重；模型目录漂移时不得用 `--force` 掩盖，须重新确认来源和 revision。

- `scripts/agemem_m8b_preflight.py --mode autodl` 要求固定 40 位代码 commit、干净工作树、无嵌套 `.env`/远程 ignored 凭据、模型/数据/checkpoint 均位于当前冻结的持久根目录 `/data/hjx/Age_mem`、至少 80 GiB checkpoint 空间，以及空的固定 E0/E1 job 路径；内部 mode 名 `autodl` 仅作为严格 Linux/GPU 模式的兼容名称保留；
- GPU 门禁允许四卡宿主机通过 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 与 `CUDA_VISIBLE_DEVICES` 显式选择两张 RTX A6000；报告保留完整物理清单，仅对选中卡要求总显存至少 48,000 MiB、执行前空闲显存至少 47,000 MiB，并要求 PyTorch 重映射设备与 `nvidia-smi` 物理 UUID 一致；
- `scripts/agemem_m8b_runtime_gate.py --scope all` 锁定 `m8a=142`、`all=318`，任何测试数量漂移、FAIL、ERROR、unexpected success 或 SKIP 都 fail closed；本地仍有 3 个环境性 SKIP，必须在组内远程服务器变为 PASS；
- DashScope endpoint、`text-embedding-v4` 256 维和 `qwen-max` 已冻结。每次成功、失败、eval 或 malformed-response 调用立即写入独立、加锁、`fsync` 的 `<checkpoint_job_dir>/trajectories/auxiliary_provider_calls.jsonl`，键为 `task_id / rollout_id / execution_id / call_index`；只保存 provider/model/outcome/error type/latency/usage 等元数据，不保存 prompt、response、header 或 key。SDK retry 关闭，遥测无法持久化时调用本身失败；未返回货币成本时保持 `null`，不得估造；
- launcher、Explorer、Trainer 和 NCCL 同步路径已改为失败向上传播；外围异常日志只记类型。训练和评测仅在真实步骤完成后写严格 JSON receipt，非有限指标、失败 rollout/eval、提前耗尽或写盘错误均不得生成成功证据；
- E0 receipt 固定为 `bench_step_0_model_0.json`，E1 更新 receipt 固定为 `trainer_step_1.json`，checkpoint 新进程评测 receipt 固定为 `bench_step_1_model_1.json`。每份 receipt 记录 `process_id` 和每进程唯一的 `process_execution_id`；checkpoint eval 的执行 ID 必须与训练不同。

组内远程 GPU 环境、固定模型与凭据就绪后，唯一推荐的端到端 smoke 命令是：

```bash
bash scripts/autodl_m8b_smoke.sh
```

该脚本按顺序执行严格 preflight/runtime gate、E0、E1 单次更新、停止并重启 Ray、checkpoint eval 和只读 postflight；拒绝已有 Ray cluster、旧日志、旧 postflight 证据或被占用的固定 job 路径，任一 CLI/receipt/checkpoint/postflight 失败都会立即停止。`scripts/autodl_m8b_preflight.sh` 仍可单独运行只读门禁，并且绝不启动 Ray 或训练。

postflight 必须同时证明：E0 是 step/model version `0/0` 且 2 条 held-out 无失败；E1 step 1 有有限 loss/KL/reward 和 `training/actor_update_completed=1`；`trainer_meta.json` 与 `latest_checkpointed_iteration.txt` 均指向 1；model/optimizer/extra-state shard 完整非空且 rank 集合闭合；训练后 LoRA 存在且与 `dummy_lora` 的权重 SHA-256 不同；checkpoint eval 是 step/model version `1/1`、held-out taskset/数量/分数完整，并且其 `process_execution_id` 与训练 receipt 不同。

真实远程 GPU 验收状态如下，当前均未执行、不得勾选或宣称成功：

- [ ] 组内远程服务器严格 preflight 与 318 项 runtime gate 全通过且 0 SKIP；
- [ ] E0 model-version-0 held-out 冻结评测通过；
- [ ] E1 单次 optimizer update、step-1 receipt 和 `global_step_1` 通过；
- [ ] 停止/重启 Ray 后，model-version-1 checkpoint 新进程评测通过；
- [ ] postflight 全部检查通过并保存最终 JSON 报告。

详细上卡顺序、停止条件和安全要求见 `docs/m8b_autodl_preflight.md`。

#### Stage 1/2 反捷径 sidecar（已完成）

该 sidecar 是上卡前的确定性可证伪检查，不进入训练 buffer，也不修改现有 E1
`terminal_only` YAML、M3～M7 轨迹或报告：

- Stage 1 在 `toy-train-005` 的 seeded 观察流中同时放入 supporting facts 和 distractor，并以恰好容纳两条 supporting facts 的 15 lexical-token LTM budget 比较 `store-all`、`store-none` 与 `oracle-safe-store`；
- `TokenBudgetMemoryStore` 复用 M2 `MemoryStore`，按 active-content token 而不是条目数计费，因此单条整段写入不能绕过预算；ADD/UPDATE/restore 超预算 fail closed，UPDATE 按新旧内容差额计费，soft delete 与版本历史保持可审计；
- 固定 seed 7 时，Store-All 因先写入一条 distractor 而拒绝后续 supporting fact，support recall / memory precision 均为 `0.5`；Oracle-Safe-Store 两项均为 `1.0`；
- Stage 2 使用 6 条固定 dev/test challenge，覆盖 hard negative、partial relevance 和 delayed relevance；公开输入不含 `task_id`、split、原始消息/segment ID、`future_query` / `future_answer` 字段、场景标签或 Oracle role，每个 seed 使用与角色无关的不透明句柄；supporting message 正文可以包含未来答案事实，但当时没有查询可判断其相关性；
- Stage 2 schema 明确 budget scope 为 `retained_segment_text_only`，只对保留 segment 正文计 token，句柄、格式和控制提示不计入该 payload budget；并验证 future answer 必须落在 supporting text 且不落在 distractor text；
- `always_keep` 的 support recall 为 `1.0` 但 budget compliance / safe success 为 `0.0`；`always_clear` 的 budget compliance 为 `1.0` 但 support recall / safe success 为 `0.0`；`opaque_id_control`（仅保留字典序最小句柄）为 `0.667/1.0/0.667`；`oracle_safe_compress` 的 support recall、distractor removal、budget compliance 与 safe success 均为 `1.0`；
- Oracle 策略只证明同一预算下存在可行上界，不是可部署模型，也不证明未知未来查询下可以在线达到该上界；
- E2 context preservation 现在由 workflow 显式传入真实 `target_question`，不再把 Stage 2 第一条干扰消息误当问题；旧调用保留兼容回退；
- 26 项 canary 测试与 2 项显式问题对齐回归全部通过，历史锁由 280 增至 308；新增 9 项 stress 回归后，完整本地结果为 318 RUN、315 PASS、3 个环境性 SKIP、0 FAIL/ERROR；两份报告均可确定性重建与校验。

规范报告位于 `artifacts/anti_shortcut_benchmark/` 与
`docs/anti_shortcut_benchmark.md`，checksum 为
`b5ced8e688194d3d9e7cb3a6b4bd8d256d7cc38610fcb56a1d8c37987a7b952c`；该 SHA-256 仅用于确定性重复和输入绑定，不提供来源认证。

独立 stress 报告位于 `artifacts/anti_shortcut_stress/` 与
`docs/anti_shortcut_stress.md`，schema 为 `agemem.anti_shortcut_stress.v1`，
lexical-token checksum 为
`ae88dc9b64cfab0b1b705ef365e9fccbaa400924a3309346f40b1ff492e1b214`。其协议为：

- Stage 1 选取 train/dev/test 中 16 个含 distractor、duplicate 或 stale fact 的任务，覆盖 50 个 order seeds 和 12/20/28 三个全局预算；每个策略 2400 arms，所有三事实任务达到 6/6 permutation coverage；
- 非 Oracle Stage 1 输入只含 `budget_tokens` 与公开事实，隐藏 task ID、split、seed 和私有角色；加入 reverse-order、shortest/longest、opaque min/max、random-hash 和 entity-chain 强基线；
- Stage 2 使用 6 个 dev/test counterfactual pair、12 个 future variants 和 50 seeds；同一 pair 的两个 future 共享完全相同的公开输入，并复用一次 query-blind decision；
- 两 future support 互斥、各自可装入 19-token budget、并集不可装入，目标段 token/大写词差均为 0，因此 query-blind safe-success 上界严格为 `0.5`；pair-blind Oracle 正好 `0.5`，query-aware Oracle 为 `1.0`，公开策略最高 `0.372`；
- stress 的 11/11 gate 是协议完整性/可证伪性门禁，不是模型能力门禁。当前 artifact 仍使用 `unicode-lexical-v1`；远程服务器模型 manifest 就绪后，必须通过 CLI 注入冻结的 Qwen tokenizer、完整 40 位 revision 与 tokenizer assets digest 重跑。

### 任务

1. 复现 AgeMem HotpotQA terminal-only smoke run；
2. 固定 `π_old` 完成同一组 K 条 rollout 后再更新参数，并保存逐 token `old_logprobs`；
3. 确认 reward adapter 通过稳定 `action_id` 对齐动作，不依赖易变的裸 timestep；
4. 先接入 E3/E4 的轨迹级 Advantage，确认奖励链路正确；
5. 再实现 E5：按 `decision_key=(stage_id, dfa_state_before)` 对齐语义决策位置；
6. 为每个动作计算即时奖励、Return-to-Go 和动作级 Advantage；
7. 状态 bucket 样本不足时依次回退到 stage-level、trajectory-level Advantage；
8. 将同一动作 Advantage 赋给该动作 token span，system/user/tool-result token 必须 mask；
9. loss 先在动作内部按 token 平均，再对动作平均，避免长 JSON 参数获得额外权重；
10. 规则 Oracle 和 ErrorInjector 轨迹不得进入 on-policy GRPO buffer；
11. 使用小数据、小 batch、短训练检查 loss、KL、reward、工具调用频率和 context tokens；
12. 防止不同 rollout 共享记忆；
13. 保存每次训练的完整配置、数据 split、模型、tokenizer 和代码 commit；
14. 正式结果至少运行多个 seed；
15. 对 checkpoint 做冻结评测，测试时 DFA 默认只 shadow execution。

### 验收标准

- E1 可稳定复现；
- E3/E4 奖励通过 `action_id` 进入正确动作；
- E5 中 `response_token_ids` 与 `old_logprobs` 长度一致，token span 不越界、不重叠；
- 相同 DFA 决策状态下的替代动作被放入相同 Advantage bucket；
- 训练无 NaN、logprob 错位或 rollout 污染；
- checkpoint 可加载并在冻结 split 上评测；
- 提升不只是来自工具调用次数增加；
- Oracle AP 与 Extracted AP 结果分开报告，不能混为主结果；
- E4 与 E5 的差异只来自信用分配粒度，用于区分奖励收益和动作级 GRPO 收益。

---

## M9：正式 Benchmark、跨域泛化与可选任务规划

### Benchmark 顺序

```text
1. HotpotQA：主训练、主结果和完整消融
2. 2WikiMultiHopQA / MuSiQue：跨问答数据集泛化
3. LongMemEval / MemBench：长期记忆诊断
4. ALFWorld：仅在需要证明任务规划泛化时加入
5. ScienceWorld：更后期的开放式扩展，不属于核心论文最低要求
```

### 完整功能增加顺序

```text
1. DELETE 与过期事实
2. Negative Automata
3. SUMMARY / CLEAR 的 AP/DFA 奖励（FILTER 仅作为后续可选扩展）
4. STM context pressure
5. bounded Trend Shaping
6. 全失败组反事实分析
7. Hard DFA → belief/soft state
8. PostgreSQL + pgvector
```

### 论文结论边界

- 只完成 HotpotQA 时，结论限定为“多跳问答中的可学习记忆管理”；
- 加入 2Wiki/MuSiQue 后，可以讨论跨问答数据集泛化；
- 只有完成 ALFWorld 并控制规划能力等混杂因素后，才讨论通用任务规划。

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
A9：Trajectory Advantage → Action-level Advantage
```

### 11.4 Stage 1/2 反捷径诊断基线

| 阶段 | 固定策略 | 信息权限 | 诊断目的 |
|---|---|---|---|
| Stage 1 | Store-All | 仅公开候选事实 | 检查“全存再检索”捷径 |
| Stage 1 | Store-None | 仅公开候选事实 | 检查零写入是否丢失未来证据 |
| Stage 1 stress | 顺序/长短/opaque-ID/random/entity-chain | 隐藏 task/seed 的公开输入 | 检查顺序、长度、句柄和模板实体链捷径 |
| Stage 1 | Oracle-Safe-Store | 私有 supporting labels | 同预算下的离线可行上界，不可部署 |
| Stage 2 | Always-Keep | 仅公开 query-delayed context | 检查不管理 STM 的预算失败 |
| Stage 2 | Always-Clear | 仅公开 query-delayed context | 检查清空导致 delayed support 丢失 |
| Stage 2 | Opaque-ID control | 仅公开不透明句柄 | 检查“保留最小 ID”这一固定规则不能达到 Oracle；不代表穷尽 ID-only 策略 |
| Stage 2 stress | first/last/length/opaque/random/style | 成对 future 共用的公开输入 | 在严格 0.5 query-blind 上界下审计公开启发式 |
| Stage 2 stress | Pair-Blind Oracle | 私有 pair labels，不知道实际 future | 证明 query-blind 0.5 上界可达，不可部署 |
| Stage 2 stress | Query-Aware Oracle | 私有实际 future labels | 证明预算内 hindsight 解为 1.0，不可部署 |
| Stage 2 | Oracle-Safe-Compress | 私有 segment labels | 同预算下的离线可行上界，不可部署 |

这些基线只验证 benchmark 能否区分所列固定捷径，不代表已训练模型表现，也不替代
后续真实 model policy、生产 tokenizer 和多 seed HotpotQA 评测。

---

## 12. 评价指标

### 12.1 任务指标

- Answer Exact Match；
- Answer F1；
- Supporting-Fact Exact Match / F1；
- Supporting-Fact Coverage；
- 三阶段 Episode Success Rate；
- 平均完成轮数；
- timeout rate；
- invalid tool-call rate。

### 12.2 记忆指标

- Retrieval Recall@k；
- 关键事实保留率；
- 过期记忆率；
- 冲突记忆率；
- update accuracy；
- 平均 memory items；
- 平均 memory tool calls；
- retrieved context tokens；
- active LTM token budget / utilization；
- supporting-fact recall 与 memory precision；
- budget rejection count；
- Stage 2 future-support recall、distractor-removal recall、budget compliance 与 safe success。

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
- action bucket coverage 与零方差 bucket 比例；
- action-level Advantage 的均值、方差和裁剪率；
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

### 13.5 动作奖励与 Return-to-Go 分开保存

每个动作的即时奖励独立计算；最终任务成功通过 Return-to-Go 影响早期记忆动作：

\[
G_{k,t}=\sum_{u=t}^{T_k}\gamma^{u-t}r_{k,u}
\]

要求：

- `reward_breakdown` 保存当前动作的即时奖励；
- `return_to_go` 只作为当前动作的训练权重；
- F1、F2 等 supporting facts 的覆盖增量分别计算；
- 不得把所有 `return_to_go` 再次相加作为轨迹总奖励；
- Action-level Advantage 通过 `(stage_id, dfa_state_before)` 对齐，不按裸 timestep 对齐；
- bucket 样本不足时回退到 stage-level 或 trajectory-level Advantage。

---

## 14. 测试策略

### 14.1 单元测试

必须覆盖：

```text
memory add/update/delete/retrieve
snapshot/restore/reset
rollout isolation
trajectory serialization
action_id uniqueness and deterministic migration
multiple actions in one assistant turn
token span bounds/non-overlap
response_token_ids and old_logprobs length match
ActionEvent ↔ ActionCreditRecord join integrity
triple schema validation
state overwrite/versioning
AP grounding
DFA transition
accept/reject
edge reward once-only
reward replay determinism
critic output validation
ADD/UPDATE/restore budget admission and fail-closed state preservation
UPDATE token-delta accounting and version-history preservation
query-delayed public input hides task/split/private IDs/future query/answer/scenario/oracle role
hard-negative/partial-relevance/delayed-relevance challenge coverage
answer/support grounding, opaque-ID control, payload-budget scope
anti-shortcut report v2 cross-field validation, JSON round trip and byte determinism
```

### 14.2 集成测试

```text
Agent → tool → memory → ToolResponse
Hotpot-style ToyEnv → observation → extractor → state → AP
HotpotQA adapter → three-stage episode → trajectory
AP trace → DFA → step rewards
K rollouts → critic → automaton → replay
Explorer → reward adapter → Buffer
ActionEvent → action return/advantage → token loss mask
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

然后使用 VS Code 的 “Open Folder in WSL”。不要让 Codex 直接复制一个包含未提交修改的工作区；先检查并保存 Git 状态。

### 16.2 当前交给 Codex 的提示词

在 Codex 插件中粘贴：

```text
请完整阅读 PROJECT_HANDOFF.md、STATUS.md、
docs/m8a_terminal_only_preflight.md 和 docs/m8b_autodl_preflight.md。

M0～M7、M8a 和 M8b-prep 已完成。只执行组内远程服务器 M8b GPU smoke，
不要重做准备实现，不要开始 E3/E4/E5，也不要扩大到全量 HotpotQA。

先核对固定代码 commit、`TRINITY_MODEL_REVISION` 和模型 SHA-256 manifest，
再由用户主动运行 `bash scripts/autodl_m8b_smoke.sh`。该脚本必须依次完成
严格 preflight、冻结的 318 项 0-SKIP runtime gate、E0 model-version-0 评测、
E1 单次 optimizer update、Ray 重启、model-version-1 checkpoint 新进程评测
和只读 postflight；不要手工跳过或并行运行阶段。

任何 action_id/token span/old_logprobs/policy version、rollout memory 隔离、
provider 遥测、reward profile、NaN/Inf、receipt、checkpoint shard、LoRA 差异或
`process_execution_id` 门禁失败时立即停止，不继续租卡长跑。只按 metadata usage
与 provider 账单对账，不伪造成本，也不声称端到端无外部模型。
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

### 16.5 M3 提示词（已完成，保留复现用）

```text
请阅读 PROJECT_HANDOFF.md 和 STATUS.md。
只执行 M3：HotpotQA 风格三阶段 Toy Memory Environment。
不要下载完整 HotpotQA，不要调用真实 LLM，不要实现 DFA 或 GRPO。

先检查并复用 M1 的 TrajectoryRecorder 和 M2 的 MemoryStore/rollout 隔离，
然后实现 20～50 条人工两跳事实任务及 gold/error policy。
每条任务必须包含 supporting_fact_ids、distractor_fact_ids、answer，
并覆盖干扰、重复、事实更新、过期事实和关键记忆误删。

运行固定 seed、阶段重置、rollout 隔离、gold success 和错误失败测试，
最后更新 STATUS.md，只汇报 M3 结果，不开始 M4。
```

### 16.6 M4 提示词（已完成，保留复现用）

```text
请阅读 PROJECT_HANDOFF.md 和 STATUS.md。
只执行 M4：Memory Oracle AP + 手工 DFA + 离线奖励。
从 M3 Oracle labels 生成语义 AP，不奖励裸 ADD/RETRIEVE 调用。
实现 once-only progress reward，并测试成功、失败、重复调用、循环和 reward farming。
不要调用真实 LLM，不要接入 HotpotQA 全量数据或 Trinity-RFT。
```

### 16.7 M5 提示词（已完成，保留复现用）

```text
请阅读 PROJECT_HANDOFF.md、STATUS.md 和 AgeMem 的 HotpotQA 数据读取代码。
只执行 M5：真实 HotpotQA 数据适配与 Oracle Benchmark。
先建立小规模 smoke split，确保 supporting facts、答案不可见性和数据 split 有测试。
收集并重放轨迹，生成 Oracle benchmark 报告；不要开始模型训练。
```

### 16.8 M6 提示词（已完成，保留复现用）

```text
请阅读 PROJECT_HANDOFF.md 和 STATUS.md。
用户报告 M0～M5 已完成。只执行 M6，不重做已完成阶段。

第一步先读取 M5 报告和轨迹样本，生成 docs/schema_audit_m6.md，核查：
action_id、stage_id、assistant_turn_id、action_index_in_turn、
token_start/token_end、response_token_ids、old_logprobs、policy_version、
ActionCreditRecord 和 RewardBreakdown。

规则/oracle/error-injector 轨迹的 token/logprob 允许为 None，不得伪造。
需要迁移时保留原文件，使用 schema_version 和确定性迁移测试。

审计通过后再实现严格 Triple/AP schema、mock/LLM extractor、
StateTracker、Markovian overwrite、AP grounding 和 group cache。
每个派生 AP 必须通过 action_id 追溯到原始动作。

比较 Oracle AP 与 Extracted AP，报告 Triple/AP F1、False Accept、
False Reject 和奖励误差传播。不要实现 Group Critic 或 GRPO。
```

### 16.9 M7 提示词（已完成，保留复现用）

```text
请阅读 PROJECT_HANDOFF.md、STATUS.md、M5 Oracle 报告和 M6 抽取报告。
只执行 M7：Group Critic 与自动机离线验证。
保留手工 DFA 为主基线，实现结构化 Critic、validator 和回退机制。
计算 False Accept/Reject、reward farming、稳定性和调用成本。
不要开始 GRPO，直到离线报告通过验收标准。
```

### 16.10 M8 提示词（当前只执行 E1 GPU smoke）

```text
请阅读 PROJECT_HANDOFF.md、STATUS.md、AgeMem、Trinity-RFT 文档和
docs/m8b_autodl_preflight.md。
只执行组内远程服务器 E1 terminal-only 单次更新 smoke。

先核对 `TRINITY_MODEL_REVISION` 与 `.agemem_model_manifest.json`，然后运行
`bash scripts/autodl_m8b_smoke.sh`。不得绕过脚本内的严格 preflight、318 项
0-SKIP runtime gate、E0 model-version-0 receipt、E1 step-1 update/checkpoint、
Ray 重启、model-version-1 checkpoint eval 和 postflight。

postflight 必须验证有限 loss/KL/reward、actor update sentinel、完整非空 checkpoint
shards、训练 LoRA 不同于 dummy、固定 held-out taskset，以及训练与 checkpoint eval
具有不同的 `process_execution_id`。

不要接入 DFA/Extracted AP/Group Critic，不要开始 E3/E4/E5 或全量 benchmark。
若外部 embedding/辅助模型配置未冻结并记录，停止并报告，不自行改变实验环境。
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
| Store-All 捷径 | Stage 1 全存后只依赖 Stage 3 检索 | Stage 1 内 distractor、固定 LTM token budget、Store-All 对照 |
| 主题偏移/全清/ID 捷径 | Stage 2 按主题、固定 ID 删除或直接清空 | query-delayed hard negative/partial/delayed challenge、Always-Keep/Clear 与 min-ID control 对照；不声称穷尽 ID-only 策略 |
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
- 可复现实验配置；
- `docs/anti_shortcut_benchmark.md`（v2 canary）、`docs/anti_shortcut_stress.md`（formal stress）与各自规范 JSON/Markdown artifact。

---

## 21. 项目完成定义

只有同时满足以下条件，才能认为项目核心完成：

1. 新机器按 README 可以复现环境；
2. 所有核心模块有测试；
3. 轨迹可以确定性重放；
4. Oracle AP 与 Extracted AP 结果分别报告；
5. Critic 和 DFA 错误可追溯；
6. Terminal-only baseline 已复现；
7. DFA reward 在 HotpotQA 三阶段记忆 benchmark 上有稳定结果；
8. 提升不依赖测试时自动机干预；
9. False Accept/Reject 被明确报告；
10. 训练配置、seed、模型、commit 和数据 split 完整保存；
11. 没有凭据或敏感信息进入仓库；
12. `ActionEvent` 与 `ActionCreditRecord` 可以通过唯一 `action_id` 完整连接；
13. LLM rollout 的 token IDs、逐 token old logprobs 和动作 span 已通过一致性校验；
14. 轨迹级 Advantage 与动作级 Advantage 的结果分别报告，不能把两者收益混合归因；
15. 论文结论不超过实验实际支持范围。

---

## 22. 当前立即执行阶段

Codex 当前只应执行：

```text
M8b：组内远程 GPU 服务器上的 E1 terminal-only 单次更新与 checkpoint 重载 smoke
```

M0～M7、M8a 与 M8b-prep 已完成，不要重做或覆盖其实现。远程执行前先推送本地已整理提交并轮换本地凭据；进入 `/data/hjx/Age_mem/AgeMem` 后固定最终推送的代码 commit 和模型 revision，生成并核对模型 SHA-256 manifest，再按照 `docs/m8b_autodl_preflight.md` 由用户主动运行 `bash scripts/autodl_m8b_smoke.sh`。真实远程 preflight、E0、E1 和 checkpoint eval 当前全部未执行。

当前及后续顺序是：

```text
M8b：E0 冻结评测 + E1 单次更新 + checkpoint 新进程重载
    ↓（E1 smoke 稳定且外部 provider 已冻结）
E1：小规模 terminal-only 重复运行
    ↓
E3：Oracle AP + 手工 DFA + trajectory advantage
    ↓
E4：Extracted AP + 手工 DFA + trajectory advantage
    ↓
E5：action-level advantage
    ↓
M9：正式 Benchmark + 跨域泛化
```

当前不得提前开始 E3/E4/E5 或全量训练。冻结 runtime gate 必须精确发现 318 项并达到 0 FAIL/ERROR/SKIP；ActionCredit 在线生成器尚未实现，只有 schema/join/buffer validation。只有 postflight 同时证明 E0 model version 0、E1 单次更新/checkpoint、checkpoint eval model version 1 且训练/评测 `process_execution_id` 不同后，才能更新 `STATUS.md` 并决定是否扩大 E1。

---

## 23. 组内远程 GPU 服务器迁移与 1.5B 首次训练执行手册

本节是当前接手者执行真实 M8b 的最短权威路径；更完整的门禁语义和停止条件见
`docs/m8b_autodl_preflight.md`。这里的“训练”只指冻结的 1.5B M8b smoke：6 条
source-train、`K=2`、LoRA rank 16、1 次 optimizer update，以及 2 条 held-out
validation 的 E0/E1 checkpoint 评测。它不是全量训练，也不训练 Critic model。

### 23.1 迁移前的实时状态与硬边界

2026-09-03 的最新本地只读检查确认当前分支为
`feat/m6-extracted-ap-state-tracker`，已推送的组内服务器迁移基线为
`0ca8277bba7ad27f4f96b5ef6c573962b65e3d49`。远端已经能在
`/data/hjx/Age_mem/AgeMem` 进入该仓库，并已建立
`/data/hjx/Age_mem/conda-envs/agemem-m8b` 环境。该 commit 仍含两个在真实环境中才暴露的
问题：`m8b` extra 未显式约束 `mcp<2`，以及 Stage 2 counterfactual fixture 的
`max_context_tokens=18` 无法容纳冻结 Qwen tokenizer 下的一条 19-token support。

本地工作树正在修复这两个问题：为 `m8b` 增加 `mcp>=1.24,<2`，并把两份镜像
counterfactual fixture 的统一预算从 18 调整为 19，同时补依赖契约和 Qwen token-count
snapshot 回归测试。该修复尚未完成全部测试、规范报告重生成、提交和推送，因此
`0ca8277` 不能作为下一次正式 stress/preflight/smoke 的最终实验锁。接手者必须先运行
`git status --short`，完成验证、提交和推送，再用新的完整 40 位 commit 更新
`AGEMEM_EXPECTED_COMMIT`。

当前 M8b 只锁定 `Qwen/Qwen2.5-1.5B-Instruct`。4B 的具体模型尚未冻结，必须使用
独立的模型结构/file manifest、E0/E1/checkpoint-eval YAML、配置 digest、job 名称和
GPU 门禁；不能只替换 `TRINITY_MODEL_PATH` 复用 1.5B lock。暂不进入 7B。

### 23.2 必须迁移的最小文件集合

代码只通过 Git 迁移；不要把 Windows 工作目录整体复制到远程服务器。最终推送的固定
commit 应包含仓库内全部已跟踪输入，包括 `AgeMem_code_agentscope/`、`trinity/`、
`configs/`、`examples/`、`scripts/`、`tests/`、`data/splits/`、`data/annotations/`、
`artifacts/m5_hotpotqa_smoke/`、`artifacts/m6_extraction_benchmark/` 和
`artifacts/m7_group_critic/`。

严格 `all=318` runtime gate 还会读取以下三个被 Git 忽略的规范运行目录，必须把它们
作为私有、带 SHA-256 的最小证据包迁移到云端仓库内的相同相对路径：

```text
runs/m5_hotpotqa_smoke/          120 files，约 2.9 MB
runs/m6_schema_v2/                61 files，约 1.6 MB
runs/m6_extraction_benchmark/    120 files，约 0.9 MB
```

缺少任何一个目录都会令 M5/M6/M7 集成测试 SKIP，而严格门禁把任意 SKIP 判为失败。
“不得迁移整个 `runs/`”仍然成立；这里只允许迁移上述三个规范输入，不迁移其他实验
日志、临时目录或旧 checkpoint。

HotpotQA 必须另行迁移完整 Hugging Face `DatasetDict.save_to_disk` 目录：

```text
Windows: D:\Project\Age-Mem\data\hotpot_qa\fullwiki
Remote: /data/hjx/Age_mem/data/hotpot_qa/fullwiki
```

当前本地 fullwiki 为 11 个文件、645,926,725 bytes，预期 split 大小为
`train=90,447 / validation=7,405 / test=7,405`。不得转成 CSV、只上传 smoke 行或拆散
Arrow/state/metadata 文件。

以下内容禁止迁移：仓库根部 ignored `config`、任意 `.env`、`.venv`、SSH 私钥、
Hugging Face token 文件、数据库、完整 `runs/`、旧 Ray session、历史日志、WandB
缓存和旧 checkpoint。现有云服务 key 应先轮换；远程服务器只通过密钥配置或环境变量
注入 `DASHSCOPE_API_KEY`。

### 23.3 Windows 封版与私有数据打包

在 `D:\Project\Age-Mem\AgeMem` 运行：

```powershell
git status --short
git diff --check
.\.venv\python.exe scripts\agemem_m8b_preflight.py --mode local --no-write

git add -u
git commit -m "feat(agemem): finalize 1.5B remote GPU smoke inputs"
git push -u origin feat/m6-extracted-ap-state-tracker

$commit = git rev-parse HEAD
$commit
git status --short
git ls-remote origin refs/heads/feat/m6-extracted-ap-state-tracker
```

本地预检必须为 0 FAIL；Windows 缺少完整 Ray/vLLM/GPU 造成的 WARN/SKIP 只作诊断，
不能记为远程 GPU 门禁通过。提交后要求工作树干净，并确认远端分支的哈希等于 `$commit`。

制作两个不进入 Git 的传输包：

```powershell
New-Item -ItemType Directory -Force -Path ..\transfer

tar -czf ..\transfer\agemem-runtime-gate-inputs.tar.gz `
  runs/m5_hotpotqa_smoke `
  runs/m6_schema_v2 `
  runs/m6_extraction_benchmark

Push-Location ..
tar -czf transfer\hotpotqa-fullwiki.tar.gz data/hotpot_qa/fullwiki
Pop-Location

Get-FileHash ..\transfer\agemem-runtime-gate-inputs.tar.gz -Algorithm SHA256
Get-FileHash ..\transfer\hotpotqa-fullwiki.tar.gz -Algorithm SHA256
```

保存两个本地 SHA-256，并将压缩包上传到 `/data/hjx/Age_mem/upload/`；轨迹和 reward
证据按私有实验数据处理，不上传公开仓库。

### 23.4 组内服务器固定目录与代码检出

使用以下布局：

```text
/data/hjx/Age_mem/
├── AgeMem/
├── data/hotpot_qa/fullwiki/
├── models/Qwen2.5-1.5B-Instruct/
├── checkpoints/
├── conda-envs/
└── upload/
```

SFTP 的 `remotePath=/data/hjx/Age_mem` 映射本地工作区根目录，因此 Git 仓库必须位于
`/data/hjx/Age_mem/AgeMem`。M7 集成测试按仓库父目录解析 `data/hotpot_qa/fullwiki`；
若把仓库放进 `/data/hjx/Age_mem/code/AgeMem`，默认相对路径会错误地解析到
`/data/hjx/Age_mem/code/data`。

```bash
cd /data/hjx/Age_mem
git clone --branch feat/m6-extracted-ap-state-tracker --single-branch \
  https://github.com/twihujiaxin/symbolicAi-age-mem.git AgeMem
cd /data/hjx/Age_mem/AgeMem

export AGEMEM_EXPECTED_COMMIT=<最终推送的完整40位commit>
git checkout --detach "$AGEMEM_EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$AGEMEM_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

sha256sum /data/hjx/Age_mem/upload/agemem-runtime-gate-inputs.tar.gz
sha256sum /data/hjx/Age_mem/upload/hotpotqa-fullwiki.tar.gz
tar -xzf /data/hjx/Age_mem/upload/agemem-runtime-gate-inputs.tar.gz \
  -C /data/hjx/Age_mem/AgeMem
tar -xzf /data/hjx/Age_mem/upload/hotpotqa-fullwiki.tar.gz \
  -C /data/hjx/Age_mem

find runs/m5_hotpotqa_smoke -type f | wc -l
find runs/m6_schema_v2 -type f | wc -l
find runs/m6_extraction_benchmark -type f | wc -l
find ../data/hotpot_qa/fullwiki -type f | wc -l
```

四个文件计数必须依次为 `120 / 61 / 120 / 11`，云端压缩包 SHA-256 必须与 Windows
记录一致。

### 23.5 环境、GPU、数据与模型冻结

先检查物理 GPU；四卡 A6000 宿主机只显式选择两张实时空闲卡：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free \
  --format=csv
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
```

`1,2` 只沿用 2026-09-02 截图中的空闲选择，执行前必须重新检查。每张选中卡要求总
显存至少 48,000 MiB、空闲显存至少 47,000 MiB；不要终止不属于本实验的进程。
80 GiB 是 checkpoint 持久盘剩余空间门槛，不是 GPU 显存门槛。

建立持久 Conda 环境并按仓库锁安装：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -p /data/hjx/Age_mem/conda-envs/agemem-m8b python=3.10.19 -y
conda activate /data/hjx/Age_mem/conda-envs/agemem-m8b
cd /data/hjx/Age_mem/AgeMem
python -m pip install --upgrade pip
python -m pip install -e ".[m8b,dev]"
python -m pip install "huggingface_hub[cli]"
```

`0ca8277` 的 `m8b` extra 漏掉了 AgentScope 1.x 所需的 MCP 上界，真实安装曾出现
`ImportError: cannot import name 'streamablehttp_client'`。在包含正式修复的新 commit
推送前，远端仅可用下面的环境级兼容命令继续诊断；不要修改远端仓库文件：

```bash
python -m pip install --upgrade "mcp==1.27.2"
python -c 'from mcp.client.streamable_http import streamablehttp_client; import agentscope; print("AgentScope/MCP import: PASS")'
python -m pip check
```

新 commit 的 `m8b` extra 会恢复 `mcp>=1.24,<2` 的解析约束。不要为修复单个依赖错误
而批量升级或放宽锁。验证 DatasetDict：

```bash
python -c "from datasets import load_from_disk; d=load_from_disk('/data/hjx/Age_mem/data/hotpot_qa/fullwiki'); print({k:len(v) for k,v in d.items()})"
```

解析并立即冻结一次 Hugging Face 40 位 revision，之后不得继续使用浮动 `main`：

```bash
export TRINITY_MODEL_REVISION="$(python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen2.5-1.5B-Instruct').sha)")"
[[ "$TRINITY_MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]

export TRINITY_MODEL_PATH=/data/hjx/Age_mem/models/Qwen2.5-1.5B-Instruct
mkdir -p "$(dirname "$TRINITY_MODEL_PATH")"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --revision "$TRINITY_MODEL_REVISION" \
  --local-dir "$TRINITY_MODEL_PATH"

python scripts/agemem_m8b_model_manifest.py \
  --model-path "$TRINITY_MODEL_PATH" \
  --repository-id Qwen/Qwen2.5-1.5B-Instruct \
  --revision "$TRINITY_MODEL_REVISION"
```

不得用 `--force` 掩盖模型目录漂移。设置持久路径和凭据：

```bash
export HOTPOTQA_PATH=/data/hjx/Age_mem/data/hotpot_qa/fullwiki
export TRINITY_CHECKPOINT_ROOT_DIR=/data/hjx/Age_mem/checkpoints
mkdir -p "$TRINITY_CHECKPOINT_ROOT_DIR"
df -h /data/hjx/Age_mem

read -rsp "DASHSCOPE_API_KEY: " DASHSCOPE_API_KEY
printf '\n'
export DASHSCOPE_API_KEY
```

不要把 key 写进 shell history、YAML、JSON、命令参数、报告或仓库文件。

### 23.6 冻结 tokenizer stress、严格预检与训练

先用最终模型 tokenizer 重跑 anti-shortcut stress，证据只写持久盘：

```bash
stress_dir="$TRINITY_CHECKPOINT_ROOT_DIR/anti_shortcut_stress/$AGEMEM_EXPECTED_COMMIT"
python scripts/agemem_anti_shortcut_stress.py \
  --tokenizer-path "$TRINITY_MODEL_PATH" \
  --tokenizer-revision "$TRINITY_MODEL_REVISION" \
  --tokenizer-repository-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir "$stress_dir" \
  --docs-path "$stress_dir/report.md"
```

2026-09-03 在 `0ca8277` 上首次进入该步骤时，AgentScope/MCP 导入问题已通过环境级
`mcp==1.27.2` 兼容安装跨过；随后脚本在真正运行 Stage 2 时以
`counterfactual support exceeds budget in cf-dev-entity-001` 停止。冻结 Qwen tokenizer
的六组实测计数为：

| Pair | 两个 future support | Union | Total | 旧预算 | 可行预算区间 |
|---|---:|---:|---:|---:|---:|
| `cf-dev-entity-001` | 17 / 19 | 36 | 53 | 18 | 19～35 |
| `cf-dev-length-001` | 17 / 17 | 34 | 51 | 18 | 17～33 |
| `cf-dev-style-001` | 11 / 11 | 22 | 33 | 18 | 11～21 |
| `cf-test-entity-001` | 16 / 15 | 31 | 46 | 18 | 16～30 |
| `cf-test-length-001` | 14 / 14 | 28 | 42 | 18 | 14～27 |
| `cf-test-style-001` | 11 / 11 | 22 | 33 | 18 | 11～21 |

统一预算 19 同时满足全部 pair 的
`max(support_A, support_B) <= budget < union`，而且最大 support token gap 为 2，仍满足
现有长度匹配门禁。不得只在远端手改 JSON，也不得用 lexical counter 的 PASS 代替冻结
tokenizer 结果；应先完成本地修复、测试、报告重生成和新 commit，再在新 commit 对应的
全新 `$stress_dir` 重跑本步骤。

检查 shell、确认没有既存 Ray cluster，然后运行只读门禁：

```bash
bash -n scripts/autodl_m8b_preflight.sh
bash -n scripts/autodl_m8b_smoke.sh
ray status
bash scripts/autodl_m8b_preflight.sh
```

只有严格预检和 `318/318 PASS、0 FAIL、0 ERROR、0 SKIP` 全部成立，才执行：

```bash
bash scripts/autodl_m8b_smoke.sh
```

一键脚本必须按以下顺序完成，不得拆开、跳过或并行：

```text
preflight + runtime gate
    → E0 model-version-0 held-out eval
    → E1 single optimizer update
    → global_step_1
    → stop/restart Ray
    → model-version-1 checkpoint eval in a new process
    → read-only postflight
```

成功证据至少包括：

```text
$TRINITY_CHECKPOINT_ROOT_DIR/m8b_preflight/$AGEMEM_EXPECTED_COMMIT/
$TRINITY_CHECKPOINT_ROOT_DIR/m8b_logs/$AGEMEM_EXPECTED_COMMIT/
$TRINITY_CHECKPOINT_ROOT_DIR/m8b_postflight/$AGEMEM_EXPECTED_COMMIT/postflight_report.json
$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8/agemem-e0-terminal-only-frozen-eval/receipts/bench_step_0_model_0.json
$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8/agemem-e1-terminal-only-dry-run/receipts/trainer_step_1.json
$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8/agemem-e1-terminal-only-dry-run/global_step_1/
```

`postflight_report.json` 的顶层 `status` 必须为 `pass`。同时保存 `pip freeze`、完整 GPU/
驱动信息、provider usage 和 DashScope 账单对账记录；模型、checkpoint、私有轨迹和
provider 日志均不得提交进 Git。

### 23.7 失败与后续扩展规则

任一阶段失败都不扩大样本、step、seed 或 GPU 数，不放宽测试数/版本/GPU 门禁，不
删除失败证据后复用固定 job。保留原 checkpoint root；环境修复后使用新的空持久目录
重试，例如 `/data/hjx/Age_mem/checkpoints-attempt-002`。若代码或 YAML 发生变化，必须
产生新 commit、更新相应 lock/digest，并从严格预检重新开始。

M8b 通过后仍按以下顺序推进：

```text
1.5B 小规模 terminal-only 重复运行
    → E3 Oracle AP + 手工 DFA + trajectory advantage
    → E4 Extracted AP + 手工 DFA + trajectory advantage
    → E5 action-level advantage
    → 正式 benchmark
```

当前 Critic 对每个任务可见一条私有完整 HotpotQA record（含 answer、context 和
official supporting facts），但只用于离线 M7 critic prompt/cache；E1 policy
observation 仍不可见该私有记录。在线 `ActionCreditRecord` 生成器未实现，因此不得
把本次 terminal-only smoke 描述成 DFA/AP 奖励训练，也不得提前进入 E3/E4/E5。

---

## 24. 1.5B 远程训练流程与实时进度（2026-09-03）

本节是第 23 节执行手册的进度账。状态只分为“已确认”“部分完成/待复核”“当前阻塞”
和“未开始”；不能根据文件存在或命令曾被输入就推断步骤成功。当前尚未发生任何模型
参数更新，准确表述应为“训练准备进行到步骤十二”，不能表述为“训练已开始”。

### 24.1 顺序流程

| 步骤 | 操作 | 当前状态 | 完成判据或备注 |
|---:|---|---|---|
| 一 | 在组内服务器建立持久目录 | 已确认 | 根目录为 `/data/hjx/Age_mem`；`/data` 已恢复约 2 TiB 可用空间 |
| 二 | 上传两个最小迁移压缩包 | 已确认 | 用户已完成迁移；SHA-256 尚应与 Windows 原包再次对账 |
| 三 | clone Git 仓库到固定路径 | 已确认 | 仓库位于 `/data/hjx/Age_mem/AgeMem` |
| 四 | 固定代码 commit 且保持干净工作树 | 部分完成/待更新 | 远端当前基线是 `0ca8277...`；必须等待本轮修复形成新 commit 后重新 checkout |
| 五 | 解压 runtime gate 输入和 HotpotQA DatasetDict | 部分完成/待复核 | 应重新确认文件计数 `120 / 61 / 120 / 11` 及 split 大小 |
| 六 | 创建并激活 Python 3.10.19 Conda 环境 | 已确认 | 提示符显示 `/data/hjx/Age_mem/conda-envs/agemem-m8b`；不得回退到 base Python 3.9 |
| 七 | 安装 `.[m8b,dev]` 与辅助依赖 | 部分完成/待锁定 | 首次解析暴露 MCP 2.x 兼容问题；环境级 `mcp==1.27.2` 已足以让 stress 进入 Stage 2，新 commit 将补 `mcp<2` |
| 八 | 检查物理 GPU 并过滤为两张空闲 A6000 | 部分完成/待复核 | 已确认宿主机有 4 张 A6000；新 tmux pane/SSH shell 中仍须重设 `CUDA_VISIBLE_DEVICES` 并确认 Python 只看到 2 张 |
| 九 | 设置模型、数据、checkpoint 和 commit 环境变量 | 部分完成/待新 commit | 新 shell 必须重新 export；`AGEMEM_EXPECTED_COMMIT` 必须改成修复后的完整 40 位哈希，不能带尖括号 |
| 十 | 下载并冻结 `Qwen/Qwen2.5-1.5B-Instruct` revision | 已确认到 tokenizer 可加载 | 冻结 tokenizer 已能从本地模型目录运行；仍应在最终运行记录中保存完整 model revision |
| 十一 | 生成并核验模型 file manifest、注入 DashScope key | 部分完成/待复核 | 当前对话未取得 manifest PASS、key 注入和 provider 连通性的完整输出，不能标为完成 |
| **十二** | **使用冻结 tokenizer 重跑反捷径 stress** | **当前阻塞** | 本地 lexical 测试与规范报告已按预算 19 重生成（digest `ae88dc9b64cfab0b1b705ef365e9fccbaa400924a3309346f40b1ff492e1b214`）；修复尚未提交/推送，远程冻结 Qwen tokenizer 重跑仍未执行 |
| 十三 | 检查两个 shell 脚本语法及 Ray 独占状态 | 未开始 | `bash -n` 两脚本均通过，且同一 UNIX 账号下没有其他 Ray 作业后才能继续 |
| 十四 | 运行严格 M8b preflight/runtime gate | 未开始 | 必须 `318/318 PASS`，并且 `FAIL=0 / ERROR=0 / SKIP=0` |
| 十五 | 运行完整 M8b smoke | 未开始 | 顺序为 E0 → E1 一次 optimizer update → `global_step_1` → Ray 重启 → checkpoint E1 eval |
| 十六 | 核验 postflight 与归档证据 | 未开始 | `postflight_report.json.status=pass`，并保存 receipts、日志、GPU/驱动、`pip freeze` 和 provider usage |

### 24.2 步骤十二的故障记录

第一次执行冻结-tokenizer stress 时，脚本在导入 AgentScope 阶段失败：

```text
ImportError: cannot import name 'streamablehttp_client'
```

根因是 `m8b` optional dependency 没有继承 `agent` extra 中的 `mcp>=1.24,<2`；
AgentScope 1.x 的依赖下界允许解析到已删除旧兼容符号的 MCP 2.x。环境级安装
`mcp==1.27.2` 后脚本能够进入实验主体，说明 GPU、模型路径和 stress CLI 参数不是该
错误的根因。

第二次执行在 Stage 2 数据不变量校验处失败：

```text
counterfactual support exceeds budget in cf-dev-entity-001
```

根因是 fixture 的统一预算 18 只在 `unicode-lexical-v1` 下验证过，未在文档要求的冻结
Qwen tokenizer 下做回归；`borealis` support 实测为 19 token。六组数据共同的最小可行
统一预算是 19，而最小 union 是 22，因此改成 19 不会让一对互斥 future 同时装入，也
不会破坏理论 safe-success ceiling 0.5。

### 24.3 当前本地修复状态

截至本次更新，本地 24.4 闭环已完成测试、规范报告重生成，并作为当前提交推送。推送前工作树包含：

```text
M  pyproject.toml
M  data/toy/stage2_counterfactual_pairs.json
M  AgeMem_code_agentscope/toy_hotpotqa/data/stage2_counterfactual_pairs.json
M  tests/common/anti_shortcut_stress_test.py
M  tests/common/m8a_packaging_contract_test.py
M  tests/common/m8b_runtime_gate_test.py
M  configs/m8b_autodl_preflight.json
M  artifacts/anti_shortcut_stress/
M  docs/anti_shortcut_stress.md
M  docs/m8a_terminal_only_preflight.md
M  docs/m8b_autodl_preflight.md
M  docs/project_presentation_materials.md
M  PROJECT_HANDOFF.md
M  STATUS.md
```

其中已经完成并本地核验的修改为：

- 在 `m8b` extra 中补 `mcp>=1.24,<2`；
- 两份必须 byte-identical 的 counterfactual fixture 均把预算 `18 → 19`；
- 原有 lexical stress 断言同步到 19；
- 增加冻结 Qwen token-count snapshot 回归，锁定六个 pair 的实测成本、最大 gap 2、
  ceiling 0.5 和 oracle feasibility；
- 增加 `m8b` extra 必须含 AgentScope/MCP 兼容范围的打包契约；
- `tests.common.anti_shortcut_stress_test` 与 `tests.common.m8a_packaging_contract_test`
  共 10/10 PASS；
- lexical CLI 已重生成 `artifacts/anti_shortcut_stress/` 与
  `docs/anti_shortcut_stress.md`，digest 为
  `ae88dc9b64cfab0b1b705ef365e9fccbaa400924a3309346f40b1ff492e1b214`，
  Stage 2 budgets 为 `(19,)`，公开策略最高 safe success 仍为 `0.372`；
- unittest discovery 实测 `m8a=142`、`all=318`，runtime gate 锁已同步。

尚未完成的工作：远端用本提交的完整 40 位哈希 checkout 后，在全新 `$stress_dir`
用冻结 Qwen tokenizer 重跑步骤十二。完整 318 项 runtime gate 执行、本地
`--mode autodl` 预检、E0/E1 smoke 也尚未执行。本地 lexical PASS 不能改写为
“步骤十二已确认”。

### 24.4 从当前状态继续的唯一顺序

本地 24.4 测试与报告重生成已完成。下一位接手者先检查工作树，再提交并推送：

```powershell
cd D:\Project\Age-Mem\AgeMem
git status --short
git diff --check
```

确认规范 stress JSON/Markdown 已重生成且相关门禁通过后，再提交并推送。记录新的完整
commit 后，远端只做 Git 同步，不手工复制已跟踪文件：

```bash
cd /data/hjx/Age_mem/AgeMem
git fetch origin feat/m6-extracted-ap-state-tracker
export AGEMEM_EXPECTED_COMMIT=REPLACE_WITH_NEW_40_HEX_COMMIT
git checkout --detach "$AGEMEM_EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$AGEMEM_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
```

命令中的 `REPLACE_WITH_NEW_40_HEX_COMMIT` 是占位符，实际输入时必须整体替换为新的完整
40 位哈希。
随后在同一个 tmux pane 中重新激活 Conda、重设全部环境变量，并使用新 commit 对应的
全新 stress 目录重跑步骤十二。只有输出 JSON 的 `status` 为 `pass`，才把步骤十二改成
“已确认”并进入步骤十三；不得跳过到 preflight 或训练。
