# M7 Group Critic Offline Validation

- Report digest: `6d78f7984f3f64cc57863f84d6250d2f6fa3ee65418f2a054723e0d2229642df`
- M6 False Reject audit: `59a582d31396b548c0aa2c9dfc78cb5c93f6d6347a8e073d1ce0d5f291648032`
- Hand replays: `90/90` exact
- Mock critic selected/unavailable: `25/5`
- Validator invalid outputs: `25`; explicit fallbacks (invalid + unavailable): `30`; silent adoptions: `0`
- Milestone evidence coverage: `451/451`
- Stability checks: `150` repeats + `180` permutations, stable=`True`
- Mock calls/cache hits/cache misses: `360/30/30`
- Heuristic input/output tokens (not provider billing): `2046246/627795`
- Hand-DFA reward-farming scenarios: `20/20` passed
- Real LLM calls: `0`; provider tokens and cost: `None`

Critic columns below measure the Critic + explicit fallback pipeline, not the Critic output in isolation.

| AP profile | Hand FA | Hand FR | Critic+fallback FA | Critic+fallback FR | Hand reward MAE | Critic+fallback-vs-hand reward MAE | Pipeline/hand agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oracle` | 0/20 | 0/10 | 0/20 | 0/10 | 0.000000000000 | 0.000000000000 | 1.000 |
| `human_backed_mock` | 0/20 | 0/10 | 0/20 | 0/10 | 0.000000000000 | 0.000000000000 | 1.000 |
| `controlled_error` | 0/20 | 5/10 | 0/20 | 5/10 | 0.056919642857 | 0.000000000000 | 1.000 |

## Attribution and scope

All `5` terminal disagreements are controlled-error False Rejects linked to the M6 `drop_relevant_fact` audit. There are no critic-, state-, or data-attributed failures.

The only measured interference setting is the real smoke configuration: Stage 1 `6` distractors and Stage 2 `3` distractors. No extra interference levels were fabricated.

Reward-farming checks use replay-valid duplicate ADD and two-step RETRIEVE-loop perturbations against the hand-authored DFA only; they do not claim Critic-DFA farming coverage.

The LLM critic is an injected-client adapter only. This benchmark uses the deterministic mock critic, does not call a provider, and does not implement GRPO or training.
