# Project Status

## Current milestone

M3：HotpotQA 风格三阶段 Toy Memory Environment

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

## Environment

- OS: Windows NT 10.0.26200.0
- PowerShell: 5.1.26100.8875
- Python: 3.10.20 (`.venv`)
- AgentScope: 1.0.21
- Pydantic: 2.13.4
- PyTorch: 未安装（M3 不需要）
- pytest/Ruff/Flake8: 当前环境未安装；使用标准库 `unittest`

## Git state

- Current branch: `feat/m3-hotpotqa-toy-environment`
- Base: `02536bb feat(agemem): isolate versioned memory stores by rollout`
- `PROJECT_HANDOFF.md` 是用户在 M3 开始前更新的未提交文件，Codex 未修改、未暂存
- M3 未推送远程

## Files changed in M3

- `AgeMem_code_agentscope/toy_hotpotqa/__init__.py`（新增）
- `AgeMem_code_agentscope/toy_hotpotqa/models.py`（新增）
- `AgeMem_code_agentscope/toy_hotpotqa/dataset.py`（新增）
- `AgeMem_code_agentscope/toy_hotpotqa/environment.py`（新增）
- `AgeMem_code_agentscope/toy_hotpotqa/policies.py`（新增）
- `AgeMem_code_agentscope/toy_hotpotqa/runner.py`（新增）
- `data/toy/hotpotqa_memory_tasks.json`（新增）
- `tests/common/toy_hotpotqa_environment_test.py`（新增）
- `AgeMem_code_agentscope/__init__.py`
- `AgeMem_code_agentscope/README.md`
- `STATUS.md`

## Verification

- M3 dataset/environment/trajectory tests：15/15 PASS
- M2 memory regression：11/11 PASS
- M1 trajectory regression：14/14 PASS
- Existing tool-trace regression：28/28 PASS
- Combined unittest suite：68/68 PASS
- Gold task success：30/30
- Python compile/import smoke：PASS
- M3 scoped `git diff --check`：PASS

## Known constraints

- M3 使用人工任务和规则 gold/error policy，不代表真实 HotpotQA 模型表现
- M3 Oracle labels 是结构化监督事件，不是 M4 AP，也没有产生逻辑奖励
- DELETE 不出现在公共 M3 动作空间；仅由错误策略调用现有 M2 soft delete 模拟关键记忆误删
- Stage 3 使用确定性 fact-ID metadata filter，这是 gold/error policy 的 Oracle 行为
- 完整 JSONL 含事实正文和 embedding，必须按敏感数据处理
- 全局 `git diff --check` 仅报告用户更新的 `PROJECT_HANDOFF.md` 两处 Markdown 行尾空格；M3 文件检查通过

## Failures and blockers

- 无未解决的 M3 测试失败
- 无需模型、GPU、API 或外部数据

## Next recommended action

用户验收 M3 后再执行 M4：把 M3 Oracle labels 映射为 Memory Oracle AP，随后实现手工 DFA 和离线 once-only reward。M3 不提前实现这些模块。
