# Project Status

## Current milestone

M1：轨迹记录与可重放

状态：完成

## Completed

### M0

- [x] 已有仓库、分支、远端、环境和用户修改完成接管检查
- [x] standalone demo 由用户完成真实运行并确认
- [x] 六工具基础 smoke test 通过
- [x] 复现记录保存在 `docs/reproduction.md`

### M1

- [x] 实现严格、版本化的 `TrajectoryStep`
- [x] 为 standalone AgentScope `_apply_tool` 增加可选 recorder hook
- [x] 每次工具动作保存 observation、canonical action 和完整 ToolResponse
- [x] 每步保存完整 memory before/after（含 metadata 和 embedding）
- [x] 保存 `env_reward`、`done`、`stage` 和 `old_logprob`
- [x] 使用追加式 UTF-8 JSONL 持久化并在每条记录后 flush/fsync
- [x] 实现无 AgentScope、无 embedding、无 LLM 的 `TrajectoryReplay`
- [x] 实现按 `task_id / rollout_id / timestep` 的组合查询
- [x] 实现相邻 memory 状态连续性与连续 timestep 校验
- [x] 实现确定性 SHA-256 replay digest
- [x] 实现离线 query/replay CLI
- [x] 完整离线 AgentScope demo 可以生成并重放完整 JSONL
- [x] 为损坏 JSON、缺失/额外字段和重复 timestep 增加测试
- [x] 修复注入 chat client 时默认 API client 仍被提前构造的问题
- [x] 保持 recorder 未启用时的原有 standalone 行为
- [x] 未修改 Trinity-RFT 训练、GRPO、MemoryStore、数据库或 LTL/DFA

## Environment

- OS: Windows NT 10.0.26200.0
- PowerShell: 5.1.26100.8875
- Python: 3.10.20 (`.venv`)
- System Python: 3.14.4（不使用，不满足项目 `<3.13` 约束）
- PyTorch: 未安装（M1 不需要）
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB
- NVIDIA driver: 591.94
- AgentScope: 1.0.21
- Pydantic: 2.13.4
- Trinity-RFT project version: 0.3.1
- AgeMem / Trinity-RFT independent upstream commits: 本地仓库未记录

## Git state at M1 start

- Branch: `feat/m1-trajectory-replay`
- Base commits:
  - `0095966 docs: record M0 repository reproduction`
  - `33d9760 feat(agemem): add structured six-tool tracing`
- `main` 在 M1 开始前工作区干净并领先 `origin/main` 2 个提交
- M1 未推送远端

## Files changed in M1

- `AgeMem_code_agentscope/trajectory.py`（新增）
- `AgeMem_code_agentscope/replay.py`（新增）
- `AgeMem_code_agentscope/agent.py`
- `AgeMem_code_agentscope/main.py`
- `AgeMem_code_agentscope/__init__.py`
- `AgeMem_code_agentscope/README.md`
- `tests/common/trajectory_test.py`（新增）
- `STATUS.md`

## Trajectory contract

每个实际执行的工具动作对应一个 timestep，记录：

```text
schema_version / task_id / rollout_id / stage / timestep
observation / action_text / tool_calls / tool_results
memory_before / memory_after / env_reward / done / old_logprob
```

standalone 只有设置 `AGEMEM_TRAJECTORY_PATH` 时才启用记录。轨迹为了确定性重放
不会脱敏或截断，必须视为敏感数据；默认建议写入已被 Git 忽略的 `runs/`。

## Commands run

- `git switch -c feat/m1-trajectory-replay`
- standalone trajectory 模块导入与最小 digest smoke test
- `python -m unittest tests.common.trajectory_test -v`
- `python -m unittest tests.common.trajectory_test tests.common.tool_trace_test -v`
- `git diff --check`
- 变更文件、敏感值和测试临时目录检查

## Tests

- Trajectory schema/serialization：PASS
- 损坏 JSON 与行号诊断：PASS
- 缺失字段：PASS
- 额外字段：PASS
- 重复 timestep：PASS
- 非连续 timestep：PASS
- memory 状态断裂：PASS
- 相同文件重复 replay 与 digest：PASS
- task/rollout/timestep 查询：PASS
- replay 禁止 LLM/embedding 调用：PASS
- AgentScope ADD/UPDATE/RETRIEVE/DELETE hook：PASS
- ToolResponse `env_reward`：PASS
- final response 与 `done=true`：PASS
- 完整离线 AgentScope demo JSONL：PASS
- recorder disabled compatibility：PASS
- 既有 tool trace 回归：PASS

## Failures

- 无未解决的 M1 测试失败。
- Ruff/Flake8 未安装，因此使用 unittest、模块导入和 `git diff --check` 验证；未为
  M1 安装或升级依赖。

## Known issues

- replay JSONL 包含原始观察、工具参数/结果、记忆正文和 embedding，不适合提交到
  Git 或发送给无权访问训练数据的人员。
- M1 recorder 面向单进程 standalone demo；Ray 并发训练仍使用独立的
  `trinity/common/tool_trace.py` 审计记录器。二者不能互换。
- Replay 重建并验证已记录的 memory 状态序列，不调用 MemoryManager，因此不会因
  缺失 API key 而触发 embedding。
- 当前 Codex 进程没有 `DASHSCOPE_API_KEY`，M1 验收使用完全离线的脚本模型运行
  完整 AgentScope 循环，没有重复付费 API 调用。
- AgeMem 与 Trinity-RFT 的独立上游 commit 仍未知。

## User decisions needed

- M1 无阻塞决策。
- 是否提交或合并 M1 分支由用户决定；Codex 不会自行推送。

## Next recommended action

在确认 M1 分支和 JSONL schema 后，只执行 M2：MemoryStore 抽象与 rollout 隔离。
保持现有 AgentScope 工具接口兼容，不提前开始 Toy Environment、LTL/DFA 或 GRPO。
