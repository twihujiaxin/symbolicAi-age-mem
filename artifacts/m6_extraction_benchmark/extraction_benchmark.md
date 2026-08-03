# M6 Natural-Language Triple Extraction and Explicit State Benchmark

本报告在 M5 的 30 条规范离线轨迹上比较 Oracle AP 与 Extracted AP。
Triple 抽取使用人工标注支持的确定性 mock；LLM adapter 仅由 fake client 测试，
本报告没有调用真实 LLM，也不代表模型质量。相关性与 coverage 使用独立的人工
Oracle semantic target，未进入 Triple candidate cache。

## Reproducibility

- Report digest: `e803f7752dc9e7357284887cf7716273bbd5396f62db1fc438d7cad95a2f9f92`
- M5 report digest: `c18b21b59506733b133ac3510b9c9136c780b79e14af2c96d74b81b6b8d8eef0`
- Migration manifest digest: `3615ce1041b47ea30513e81f5ef812da4060df9fb854b843162c171443ac5452`
- Annotation corpus digest: `fa74d5098e8dd4040d66ca99ecd76346d4cc799a59ec3f3a4133ba9bab98edd0`
- Canonical rollouts/actions: 30/224
- Real LLM calls: 0 (`not_run`)

## Metric definitions

- Triple exact key: evidence sentence plus normalized subject/category/value; micro spans 37 gold triples and macro spans 34 annotated sentences.
- AP exact key: action_id plus normalized proposition; macro spans every action present on either side.
- False Accept uses Oracle-rejected rollouts as denominator; False Reject uses Oracle-accepted rollouts as denominator.
- Reward error is Extracted minus Oracle over the exact 224-action join; trajectory error aggregates the same values over 30 task/rollout keys.

## Metrics

| Profile | Triple P/R/F1 | AP P/R/F1 | FA | FR | Reward MAE | Trajectory abs error | Accepted | Cache hit/miss |
|---|---|---|---|---|---:|---:|---:|---:|
| human_backed_mock | 1.000/1.000/1.000 | 1.000/0.953/0.976 | 0.000 (0/20) | 0.000 (0/10) | 0.0000 | 0.0000 | 10/30 | 94/130 |
| controlled_error | 0.938/0.811/0.870 | 1.000/0.720/0.837 | 0.000 (0/20) | 0.500 (5/10) | 0.0569 | 7.2500 | 5/30 | 94/130 |

## Error propagation

First downstream-divergence audit rows: 30. 每行使用 action_id 连接该点已有的 Triple/StateFact/AP、DFA edge 和 reward；被漏抽的 Triple 不会伪造 ID，报告也不保存原始句子。

## Limits

- human-backed mock is a Triple-extraction upper bound; downstream AP timing is compared independently with the M4 Oracle
- relevance and required coverage slots are Oracle-derived evaluation targets
- Triple F1 is scored only on the 34 fully annotated sentences
- controlled drop/corrupt errors are synthetic and are not an empirical LLM error distribution
- M5 rule/error trajectories contain no token IDs, token logprobs, or policy version
- missing-support answer correctness is fail-closed from terminal env reward
