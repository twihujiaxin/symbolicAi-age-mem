# Stage 1/2 Anti-Shortcut Benchmark

本报告由确定性 Toy/Oracle sidecar 生成，不调用 LLM、embedding 服务或网络，也不进入训练 buffer。

- Schema: `agemem.anti_shortcut_benchmark.v2`
- Seed: `7`
- Stage 1 task: `toy-train-005`
- Stage 1 task digest: `65ba85b809060ba0ec819d43f48a30876f86833ea659a2854e23123b44f5d51c`
- Stage 1 token counter: `unicode-lexical-v1`
- Stage 1 LTM budget: `15` lexical tokens
- Stage 2 cases: `6`
- Stage 2 private dataset digest: `16cb21c0890ab0ef10c8f49a269fba6343655847f58aa2419680e0c0dcd6c866`
- Stage 2 token counter: `unicode-lexical-v1`
- Stage 2 budget scope: `retained_segment_text_only`
- Real LLM calls: `0`
- Overall gate: `PASS`
- Repeatability checksum (not an authenticity signature): `b5ced8e688194d3d9e7cb3a6b4bd8d256d7cc38610fcb56a1d8c37987a7b952c`

## Stage 1

| Policy | Support recall | Memory precision | Stored tokens | Budget rejects |
|---|---:|---:|---:|---:|
| store-all | 0.500 | 0.500 | 15 | 1 |
| store-none | 0.000 | 0.000 | 0 | 0 |
| oracle-safe-store | 1.000 | 1.000 | 15 | 0 |

`oracle-safe-store` 是使用私有 supporting labels 的离线上界，不是可部署策略。

## Stage 2

| Policy | Future-support recall | Distractor removal | Budget compliance | Safe success |
|---|---:|---:|---:|---:|
| always_keep | 1.000 | 0.000 | 0.000 | 0.000 |
| always_clear | 0.000 | 1.000 | 1.000 | 0.000 |
| opaque_id_control | 0.667 | 0.833 | 1.000 | 0.667 |
| oracle_safe_compress | 1.000 | 1.000 | 1.000 | 1.000 |

`oracle_safe_compress` 是使用私有 segment labels 的离线上界。公开 Stage 2 输入不包含 `task_id`、split、原始消息/segment ID、`future_query` / `future_answer`、场景类型或 Oracle role；每个 seed 使用与角色无关的不透明句柄。`opaque_id_control` 仅测试“保留字典序最小句柄”这一条固定 ID-only 规则，不能代表穷尽所有 ID-only 策略。Supporting message 正文仍可能包含未来答案事实，但当时没有查询可用于判断其相关性。

## Gates

| Gate | Result | Evidence |
|---|---|---|
| store_all_is_not_an_oracle_equivalent | PASS | support_recall=0.500, memory_precision=0.500, budget_rejections=1 |
| store_none_loses_future_support | PASS | support_recall=0.000, stored_tokens=0 |
| oracle_safe_store_is_feasible | PASS | support_recall=1.000, memory_precision=1.000 |
| always_keep_exceeds_context_budget | PASS | support_recall=1.000, budget_rate=0.000 |
| always_clear_loses_delayed_support | PASS | support_recall=0.000, budget_rate=1.000 |
| opaque_id_min_control_is_not_oracle_equivalent | PASS | support_recall=0.667, safe_success=0.667 |
| oracle_safe_compress_is_feasible | PASS | support_recall=1.000, distractor_removal=1.000, budget_rate=1.000 |

## Evidence Boundary

该结果只证明当前构造能暴露 Store-All、Always-Keep、Always-Clear 和当前 min-ID 控制，并证明 Oracle 可行解存在。SHA-256 只用于确定性重复与输入绑定，不提供来源认证。该结果不代表已训练模型表现，也不证明真实 LLM 能达到 Oracle 上界。现有 E1 terminal-only 配置和 M3-M7 artifact 均未改写。
