# Project Status

## Current milestone

M2：MemoryStore 抽象与 rollout 隔离
状态：完成，待提交

## Completed

### M0

- [x] 完成仓库、分支、远程、环境和用户修改的接管检查
- [x] 用户完成 standalone demo 真实运行并确认
- [x] 六工具基础 smoke test 通过
- [x] 复现记录保存在 `docs/reproduction.md`

### M1

- [x] 实现严格、版本化的 `TrajectoryStep`
- [x] 为 standalone AgentScope 工具调用增加 recorder hook
- [x] 保存 observation、action、ToolResponse、memory before/after 和 env reward
- [x] 使用 UTF-8 JSONL 持久化并实现严格 schema validation
- [x] 实现不调用 LLM 或 embedding 的确定性 `TrajectoryReplay`
- [x] 支持按 `task_id / rollout_id / timestep` 查询
- [x] 覆盖损坏 JSON、缺失字段、额外字段、重复 timestep 和状态不连续测试
- [x] M1 已提交为 `f5ceffd feat(agemem): add deterministic trajectory replay`

### M2

- [x] 定义 runtime-checkable `MemoryStore` protocol
- [x] 将 `AgentScopeLongtermMemory` 改为 AgentScope API 与 MemoryStore 的适配器
- [x] 实现线程安全、rollout-scoped 的 `InMemoryStore`
- [x] 实现 add、retrieve、update、delete、snapshot、restore、reset
- [x] 实现 `RolloutMemoryStoreRegistry`，每个 rollout 对应独立 store
- [x] restore 拒绝 rollout_id 或 research_mode 不匹配的 snapshot
- [x] update 追加新版本并将旧 active 版本标记为 `superseded`
- [x] research-mode delete 追加 `discarded` 墓碑，不销毁历史证据
- [x] 普通 get/retrieve 只返回 active 版本，history/snapshot 返回完整版本链
- [x] Agent 初始化时校验 memory 与 rollout_id 一致
- [x] 扩展 M1 memory snapshot schema，记录版本、状态、时间和来源字段
- [x] 保持旧 M1 JSONL 的缺省字段兼容性和无 LLM replay
- [x] 覆盖并行 rollout 隔离、版本审计、soft delete 和 snapshot/restore 测试
- [x] 未修改 Trinity-RFT 训练、GRPO、训练侧 memory store 或工具审计记录器

## Environment

- OS: Windows NT 10.0.26200.0
- PowerShell: 5.1.26100.8875
- Python: 3.10.20 (`.venv`)
- AgentScope: 1.0.21
- Pydantic: 2.13.4
- PyTorch: 未安装（M2 不需要）
- Ruff/pytest: 当前环境未安装；使用标准库 `unittest`、模块导入检查和 `git diff --check`

## Git state

- Current branch: `feat/m2-memory-store`
- M2 base: `f5ceffd feat(agemem): add deterministic trajectory replay`
- M1 branch: `feat/m1-trajectory-replay`
- 未推送远程

## Files changed in M2

- `AgeMem_code_agentscope/memory_store.py`（新增）
- `AgeMem_code_agentscope/memory.py`
- `AgeMem_code_agentscope/agent.py`
- `AgeMem_code_agentscope/trajectory.py`
- `AgeMem_code_agentscope/__init__.py`
- `AgeMem_code_agentscope/README.md`
- `tests/common/memory_store_test.py`（新增）
- `tests/common/trajectory_test.py`
- `STATUS.md`

## Memory contract

每个 `MemoryRecord` 包含：

```text
memory_id / content / metadata / embedding
version / status / created_at / updated_at
source_rollout_id / source_step
```

状态至少包括：

```text
active / superseded / discarded
```

update 与 research-mode delete 的版本链：

```text
v1 active
  -> update: v1 superseded, v2 active
  -> delete: v2 superseded, v3 discarded
```

## Verification

- M2 memory tests：11/11 PASS
- M1 trajectory regression：14/14 PASS
- Existing tool-trace regression：28/28 PASS
- Combined unittest suite：53/53 PASS
- `git diff --check`：PASS
- Python compile/import smoke：PASS

## Known constraints

- 默认 embedding 仍使用 DashScope；测试通过注入 deterministic embedding 完全离线运行
- `InMemoryStore` 面向 standalone 与并行 rollout；持久化数据库后端不属于 M2
- 完整 snapshot 和 M1 JSONL 包含原始记忆正文与 embedding，必须按敏感数据处理
- 训练侧 `trinity/common/workflows/memory_context/memory_store.py` 仍保持原状

## Next recommended action

验收并提交 M2 后进入 M3：Toy Environment。不要在 M2 提前实现 DFA、奖励或训练接入。
