# M6 历史轨迹 Schema 审计

## 结论

M5 原始轨迹是可重放且可无损迁移的，但 **v1 原始 schema 不能直接满足 M6 的动作级归因契约**。本审计因此先判定 raw gate 为 `FAIL`，并要求通过派生的 v2 记录修复；M3～M5 文件保持只读，不就地改写。

审计的权威输入只取自已提交的 M5 报告 `records[*].trajectory_path` 和 `records[*].reward_path`，不对 `runs/m5_hotpotqa_smoke` 做目录通配。该目录包含历史重复运行留下的同内容旧命名文件；它们不是本次 benchmark 的规范样本，也不会被删除。

## 输入与可复现边界

| 项目 | 值 |
|---|---|
| M5 报告 | `artifacts/m5_hotpotqa_smoke/oracle_benchmark.json` |
| 报告内 schema | `1` |
| 报告内 digest | `c18b21b59506733b133ac3510b9c9136c780b79e14af2c96d74b81b6b8d8eef0` |
| 报告文件 SHA-256 | `8f1fd180255d7701d57663857b51d63b45a70b474b26a0a5c9c4ed94913e87ea` |
| 规范 rollout | 30（10 task × 3 policy） |
| 规范文件 | 30 trajectory JSONL + 30 reward JSONL |
| 规范文件哈希清单 digest | `cafbf42b53507e7819c8a73d87a3a0718d93b446514918bfbc791afe5b0a0ffe` |
| trajectory/action 行 | 224 |
| reward 行 | 224 |
| trajectory↔reward 连接 | 224/224 |
| 重复 replay | digest 完全一致（`09480d30fb19137314f9712923286a4f45eb47fc0bed8ddd4c6ce1587fde0453`） |

连接键审计使用 `(task_id, rollout_id, timestep, stage)`。30 组规范文件逐组行数相同，没有缺行或多行；每个 v1 step 恰有一个 tool call 和一个同 ID 的 tool result。

## 字段审计

| M6 字段 | v1 真实情况 | 可用性 | v2 迁移决策 |
|---|---|---|---|
| `action_id` | 顶层缺失；224 个 `tool_calls[0].id` 全局唯一，且全部等于 `tool_results[0].tool_call_id` | 可无损恢复 | 直接采用已有 tool-call ID，禁止重新编号 |
| `stage_id` | 缺失；`stage` 在 224 行完整存在 | 可无损恢复 | `stage_id = stage`；分布为 Stage 1/2/3 = 92/30/102 |
| `assistant_turn_id` | 缺失 | 仅对 M5 规范轨迹可恢复 | M5 runner 每步仅一动作，故取 `timestep`；通用 v1 多动作/轮次输入 fail closed |
| `action_index_in_turn` | 缺失 | 仅对 M5 规范轨迹可恢复 | M5 每步唯一动作取 `0`；不声称恢复一般 AgentScope turn 分组 |
| `token_start/token_end` | 缺失 | 不可恢复 | 规则轨迹保持 `None` |
| `response_token_ids` | 缺失 | 不可恢复 | 规则轨迹保持 `None` |
| `old_logprobs` | 缺失；单值 `old_logprob` 为 224/224 `null` | 不可恢复 | 保持 `None`；禁止把单值复制/扩展为 token logprobs |
| `policy_version` | 缺失 | 不可恢复 | 规则/oracle/error-injector 保持 `None` |
| `source` | 缺失；M5 报告保存 policy | 可由规范报告确定 | `gold → oracle`；`wrong_answer/missing_support → error_injector` |
| `ActionCreditRecord` | 完全缺失 | 需生成派生记录 | 逐 action 生成、以 `action_id` 唯一连接；不写回原轨迹 |
| `RewardBreakdown` | M4 派生 JSONL 存在，但无 `action_id/schema_version/cost` | 可确定性升级 | 与 action 连接；兼容补 `cost=0.0`，保留原五项分量及 total |

### Token/logprob 完整性规则

规则、Oracle 与 error-injector 的 token 归因字段均允许且应当为 `None`。迁移器不得对 action text 重新 tokenize，不得伪造 token span、logprob 或 policy version。未来 LLM action 若带 token 数据，则必须同时满足：

- `response_token_ids` 与 `old_logprobs` 同时存在且等长；
- `0 <= token_start < token_end <= len(response_token_ids)`；
- `policy_version` 非空；
- 同一 assistant turn 的 action span 不重叠。

### Reward 兼容性

224 个 v1 reward 行均满足：

```text
total = env + milestone + violation + trend + format
```

因此 v2 兼容读取只添加 `cost = 0.0`，不会改变总奖励。20/224 个动作在同一步触发两条 DFA edge；为避免信息损失，v2 以有序 `transition_ids` 为权威字段。兼容的单值 `transition_id` 仅在恰好一条 edge 时填写，多 edge 时为 `None`。

## 非破坏迁移契约

迁移输出使用独立、命名空间化的 schema 版本：

- `agemem.action_event.v2`
- `agemem.trajectory_step.v2`
- `agemem.action_credit.v2`

迁移必须满足：

1. 输入文件及其 SHA-256 不变；
2. 输出路径与输入路径不同；
3. 两次迁移产生逐字节相同的 JSONL 与 manifest digest；
4. 每个 `ActionEvent` 恰好连接一个 `ActionCreditRecord`，反向亦然；
5. `action_id` 在 224 个规范动作中唯一且稳定；
6. 无法证明单动作/单 turn 的通用 v1 文件拒绝迁移，而不是猜测；
7. manifest 记录输入相对路径、输入/输出哈希、行数和迁移器版本，支持删除派生输出后回滚；
8. 原始 action、tool result、memory before/after 与 reward payload 保留在派生记录或可由 manifest 定位，不覆盖源记录。

## Gate 状态

| Gate | 状态 |
|---|---|
| 规范 M5 文件存在且 hash 固定 | PASS |
| trajectory/reward 224/224 可连接 | PASS |
| raw v1 直接满足 M6 schema | FAIL（预期） |
| v1→v2 非破坏迁移 | PASS |
| v2 schema validation | PASS |
| ActionEvent↔ActionCreditRecord 唯一连接 | PASS（224/224） |
| 两次迁移确定性 | PASS（逐文件字节一致） |

迁移器版本为 `agemem.migration.m5_v1_to_m6_v2.v1`；本次规范迁移 manifest digest 为 `3615ce1041b47ea30513e81f5ef812da4060df9fb854b843162c171443ac5452`。测试还验证了 20 个双 edge 动作的 `transition_ids` 未丢失、所有模型字段为严格 schema、61 个源文件（报告加 60 个 JSONL）的 hash 前后不变。Schema gate 已通过，可以进入 M6 Triple/AP extractor、StateTracker 与 benchmark。
