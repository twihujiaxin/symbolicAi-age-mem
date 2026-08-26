# Stage 1/2 Anti-Shortcut Stress Experiment

这是与 v2 CI canary 分离的扩展离线实验。它不调用 Agent、LLM、embedding 服务或网络，也不改写 E1 配置和训练 buffer。

- Schema: `agemem.anti_shortcut_stress.v1`
- Token counter: unicode-lexical-v1
- Stage 1: `16` 个含噪任务 × `50` seeds × `3` 固定预算；每个策略 `2400` arms
- Stage 1 budgets: `(12, 20, 28)`
- Stage 1 minimum permutation coverage: `1.000`
- Stage 2: `6` 个反事实对 / `12` 个 future variants × `50` seeds；每个策略 `600` arms
- Stage 2 budgets: `(18,)`
- Stage 2 public-input identity: `1.000`
- Real LLM calls: `0`
- Integrity gates: `PASS`
- Repeatability checksum: `385753c1d4d9b0aa8d9398622492e0632618077e921846dc90b88704d3c87b50`

## Reproduce

```bash
python scripts/agemem_anti_shortcut_stress.py

# AutoDL: rerun with the frozen production tokenizer.
stress_dir="$TRINITY_CHECKPOINT_ROOT_DIR/anti_shortcut_stress/$AGEMEM_EXPECTED_COMMIT"
python scripts/agemem_anti_shortcut_stress.py \
  --tokenizer-path "$TRINITY_MODEL_PATH" \
  --tokenizer-revision "$TRINITY_MODEL_REVISION" \
  --tokenizer-repository-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir "$stress_dir" \
  --docs-path "$stress_dir/report.md"
```

## Stage 1: multi-task / multi-seed / fixed-budget

非 Oracle 策略只能看到 `budget_tokens` 和带不透明句柄的公开事实；看不到 task ID、split、seed 或事实角色。`reverse_order` 是 last-in-first 的一轮容量代理；在这个静态一次写入实验中，真正的 LRU 与 FIFO 没有额外访问事件可区分。

| Policy | Support recall | Oracle-normalized recall | Memory precision | Exact support | Oracle equivalent |
|---|---:|---:|---:|---:|---:|
| store_all | 0.670 | 0.782 | 0.671 | 0.114 | 0.217 |
| store_none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| reverse_order | 0.668 | 0.781 | 0.670 | 0.112 | 0.229 |
| shortest_first | 0.637 | 0.732 | 0.621 | 0.085 | 0.274 |
| longest_first | 0.700 | 0.826 | 0.715 | 0.145 | 0.198 |
| opaque_min | 0.670 | 0.780 | 0.669 | 0.121 | 0.263 |
| opaque_max | 0.666 | 0.773 | 0.662 | 0.117 | 0.191 |
| random_hash | 0.667 | 0.776 | 0.665 | 0.116 | 0.228 |
| entity_chain | 0.694 | 0.814 | 0.702 | 0.147 | 0.387 |
| oracle_support | 0.833 | 1.000 | 1.000 | 0.667 | 1.000 |

`entity_chain` 是公开文本启发式，用于显式检测模板/实体链捷径；`oracle_support` 使用私有 supporting labels，只是离线上界。

## Stage 2: paired counterfactual futures

每个 pair 的两个 future query 在决策时共享完全相同的公开上下文，但需要保留互斥的支持段；两组支持无法同时装入预算。因此任何 query-blind 决策在一对上的 safe-success 上界是 `0.500`。目标段最大 token 差为 `0`，最大大写词数量差为 `0`。

| Policy | Future-support recall | Memory precision | Distractor removal | Budget compliance | Safe success |
|---|---:|---:|---:|---:|---:|
| always_keep | 1.000 | 0.333 | 0.000 | 0.000 | 0.000 |
| always_clear | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| first_fit | 0.372 | 0.372 | 0.686 | 1.000 | 0.372 |
| last_fit | 0.313 | 0.313 | 0.657 | 1.000 | 0.313 |
| shortest_first | 0.327 | 0.327 | 0.663 | 1.000 | 0.327 |
| longest_first | 0.327 | 0.327 | 0.663 | 1.000 | 0.327 |
| opaque_min | 0.327 | 0.327 | 0.663 | 1.000 | 0.327 |
| opaque_max | 0.323 | 0.323 | 0.662 | 1.000 | 0.323 |
| random_hash | 0.350 | 0.350 | 0.675 | 1.000 | 0.350 |
| style_density | 0.327 | 0.327 | 0.663 | 1.000 | 0.327 |
| pair_blind_oracle | 0.500 | 0.500 | 0.750 | 1.000 | 0.500 |
| oracle_future | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

`oracle_future` 在查询揭示后使用私有标签，仅用于证明预算内可行解存在。公开策略结果应结合反事实上界解释，不能把 query-blind 任务上的 0.5 当作模型已经学会未来相关性。

## Integrity gates

| Gate | Result | Evidence |
|---|---|---|
| stage1_public_boundary_hides_task_and_seed | PASS | public_fields=('budget_tokens', 'observed_facts') |
| stage1_uses_at_least_50_order_seeds | PASS | seed_count=50 |
| stage1_covers_all_three_fact_permutations | PASS | minimum_coverage=1.000 |
| stage1_uses_three_global_budgets | PASS | budgets=(12, 20, 28) |
| stage1_store_all_is_not_robust | PASS | store_all_recall=0.670, oracle_recall=0.833, store_all_oracle_equivalent=0.217 |
| stage2_public_inputs_are_counterfactually_identical | PASS | identity_rate=1.000 |
| stage2_targets_are_length_and_style_matched | PASS | max_token_gap=0, max_capitalized_gap=0 |
| stage2_public_policies_respect_counterfactual_ceiling | PASS | max_public_safe_success=0.372, ceiling=0.500 |
| stage2_pair_blind_oracle_reaches_exact_ceiling | PASS | safe_success=0.500, ceiling=0.500 |
| stage2_oracle_future_is_feasible | PASS | safe_success=1.000, budget_rate=1.000 |
| offline_run_makes_no_real_llm_calls | PASS | real_llm_call_count=0 |

## Evidence boundary

本报告衡量固定公开基线和数据构造，不是已训练模型结果。若 token counter 为 `unicode-lexical-v1`，它只是一份本机可复现的协议验证；正式上卡结果必须用冻结的 `Qwen/Qwen2.5-1.5B-Instruct` 本地 tokenizer、完整 40 位 revision 和 tokenizer assets digest 重跑。模型策略仍需另报 Answer EM/F1、support F1、memory precision、预算合规率和序列化 token 数。
