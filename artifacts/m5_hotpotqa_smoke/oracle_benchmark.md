# M5 HotpotQA Oracle Benchmark

本报告由本地 `hotpot_qa/fullwiki` smoke split、确定性规则策略、M1 轨迹重放和 M4 Oracle AP/DFA 离线生成。未调用 LLM，未执行模型训练。

## 数据与协议

- Seed：`20260802`
- Manifest digest：`fc875795fcf764d97aa104af9d7f1e3e14a2b94e315abf71670b827f4677af85`
- Smoke config digest：`49c0a36109e3597b228f7fbedd0af37d8a18f93635f31e99b6275ada4dc7945d`
- Reward profile：`terminal_dfa` (`da02223de72cde8f235401890831ee3683e204977452cf98f7e8e9371e3571e0`)
- Report digest：`c18b21b59506733b133ac3510b9c9136c780b79e14af2c96d74b81b6b8d8eef0`
- Source sizes：train=90447，validation=7405，official test=7405
- 官方 test 标签不可见校验：7405 条
- Smoke train 来自 source train；smoke dev/test 是 source validation 的固定互斥子集。官方 test 不进入 Oracle 指标。
- `gold` 是 Oracle 上界；`wrong_answer` 与 `missing_support` 是确定性失败对照，不代表真实 base-model 表现。
- `Retrieval recall@k` 是 Oracle-directed、整条 episode 中多次 top-1 检索结果并集对 supporting facts 的累计召回率；`k` 是唯一返回事实数，不是标准单查询模型 Recall@k。
- `Context tokens` 是所有 timestep 已处理 observation 的、与 tokenizer 无关的累计估算，因此会计入跨步骤重复上下文。
- `Memory precision` 是最终 active memory records 中 supporting records 所占比例。

## 汇总指标

| Split | Policy | N | Success | DFA accept | Answer EM | Answer F1 | Support coverage | Memory precision | Retrieval recall@k | Mean k | Context tokens | Tool calls | Reward |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | gold | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.2 | 1700.0 | 7.3 | 2.000 |
| train | wrong_answer | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 2.2 | 1700.0 | 7.3 | 0.750 |
| train | missing_support | 6 | 0.000 | 0.000 | 1.000 | 1.000 | 0.528 | 1.000 | 0.528 | 1.2 | 1287.8 | 6.3 | 0.250 |
| dev | gold | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.0 | 2358.5 | 9.0 | 2.000 |
| dev | wrong_answer | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 3.0 | 2358.5 | 9.0 | 0.750 |
| dev | missing_support | 2 | 0.000 | 0.000 | 1.000 | 1.000 | 0.667 | 1.000 | 0.667 | 2.0 | 1856.5 | 8.0 | 0.250 |
| test | gold | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.5 | 1866.0 | 8.0 | 2.000 |
| test | wrong_answer | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 2.5 | 1866.0 | 8.0 | 0.750 |
| test | missing_support | 2 | 0.000 | 0.000 | 1.000 | 1.000 | 0.583 | 1.000 | 0.583 | 1.5 | 1456.5 | 7.0 | 0.250 |

## 失败审计

失败/拒绝轨迹：20 / 30。
每条失败记录保存 supporting fact 指针与 ID、最终 memory 版本历史、检索覆盖和自动机状态；不复制完整上下文或答案文本。

## 范围限制

该结果仅验证真实数据适配、Oracle AP 上界、轨迹确定性和离线奖励链路。自然语言 AP 抽取、真实 base model、Critic、GRPO 与训练均不属于 M5。
