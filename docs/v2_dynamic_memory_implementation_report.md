# streaming_dynamic_multiquery_v2 实施报告

## 当前结论

2026-09-16 retry2 已恢复 Experience 持久化，但用户实测 204 responses 全为 invalid_action，ActionEvent=0。完整裸 JSON NEXT 未识别；ADD 示例同时存在多动作、512-token 截断、虚构 source_refs 和重复 memory ID。**这是实际工程失败，不是模型学习结果，也不是 smoke PASS。** vLLM 当前使用 skip_special_tokens=True，标签可能在解码时消失；没有原始 token-ID 审计，不能断言这些响应原本有没有标签。

本次增量修复：V2 执行与 ActionEvent draft 显式共用裸数组解析，原响应不加字符，旧三阶段默认行为不变；V2 只接纳一个完整 JSON action envelope，截断/多动作/额外正文继续判失败。公开输入按生成器的逐句/ref 顺序展示 `[ID] 正文`，manifest validator 对 public line 与 reward-side registry 的正文、history 和 observed_at 做精确连接验证；policy 不读 registry。提示词精简到单动作、唯一 memory_id、复制真实 source ID，全部新增渲染成本仍计 C。动作接口身份新增 `agemem.dynamic.action_interface.v2`，不修改旧 reward/profile 身份。receipt 新增 invalid_response_count 与 admitted_memory_write_count，避免 NEXT-only 被误解为记忆成功。

实测：V2 scope **40/40 PASS**；旧 action/streaming scope运行 34 项，31 PASS、3 环境性 SKIP；diff-check PASS。包含裸 JSON 完整 2×2 fake-policy group 的真实 draft/finalize/validate-on-policy 契约回归、多动作/截断拒绝、可见来源映射及 CPU 初始 prompt 的 C 拒绝测试。新增 CLI `scripts/agemem_dynamic_v2_prompt_preflight.py` 的完整 datasets/Qwen 路径未在本地运行（缺 datasets/模型），helper 使用 debug tokenizer 实测。远端 Qwen tokenizer prompt preflight、GPU retry3、自然来源/正文正确率与 trainer update **未运行**。commands、预算和验收门禁见 `docs/v2_runtime_retry3.md`。初始 prompt 预检不等于带模型动作/handles/receipt 的全部预算验收，后者仍由运行时逐次 enforce。

2026-09-16 远端 retry1 已运行真实 policy 和 4 次 frozen-reader 调用，但 Explorer 仍用旧持久化白名单，因此无可审计 Experience。该次不能记为完整 runtime PASS。现共享 runner/Explorer bench gate，并新增 4 项 CPU 回归（执行实际 `_finish_eval_step` 函数体，边界用 doubles）；动态 V2 总计 34 tests PASS。修复后的 GPU 重跑与落盘审计仍未运行；原产物不覆盖。

P0～P3 的本地代码与 CPU 受控闭环已完成；远端 production Qwen3-4B tokenizer 数据和 CPU replay 也已由用户报告通过。P5 已实现模型 runtime producer、冻结 reader 分支、Trinity workflow 注册、完整 K×m 发布和 ddof=0 advantage operator，并通过 fake-policy 边界测试；尚未在远端 Ray/vLLM/veRL 上验证。P4 冻结模型诊断与 P6 GPU pilot 未运行。当前证据证明受控数据、版本环境、grounding、END/LIFE 数学和本地 runtime 契约可执行，**不证明模型学会动态记忆，也不证明 LIFE 优于 END。**

协议身份：

- protocol：`streaming_dynamic_multiquery_v2`
- public schema：`dynamic_stream_v2`
- time semantics：`monotonic_single_valued_v1`，区间 `[from,to)`
- world Oracle：`agemem.dynamic.world_oracle.v1`
- reward：`dynamic_reward_v2`
- controlled grounder：`agemem.dynamic.controlled_grounder.v1`
- compiled monitor：`agemem.dynamic.compiled_monitor.v1`
- 基线代码提交：`cf80142d72b3009635234d5693fde4acf4a9c839`；本轮运行时为 dirty 工作树，digest 写入 manifest

## P0：当前实现审计

已完成，详见 `docs/v2_current_implementation_audit.md`。没有发现 `AGENTS.md`。v1 的 S0～S4 CPU 工程范围可复用，但 production tokenizer、真实 reader/model rollout 和新协议 trainer step 都未验证；旧自由文本 provenance grounder 未达到 v2 语义准入线。

## P1：动态数据与独立 Oracle

已实现：

- 按 `history_family_id` 先分 train/dev/test，再生成正文；ID 不编码 split/gold。
- public histories、public questions、private events、private query labels、private source registry 和 split audit 分文件。
- A→B→A、当前／历史、多更新、同一时刻两跳 join、预算保留五类候选问题。
- `observed_at` 与 `effective_at` 分离；单调 single-valued 世界，半开区间。
- `WorldOracle` 扫描事件；`independent_validator` 另建区间表重算答案、availability 和支持引用，不调用 Oracle solver。
- 公开 observation 使用白名单；问题只在阅读后使用，private 字段不能由公开 schema 表示。

实际 debug 数据：

| 指标 | 实测 |
|---|---:|
| D0 fixtures | 60 |
| D1 histories | 60（train/dev/test=40/10/10） |
| candidate queries | 600（每 history 10） |
| chunks | 1,260（每 history 21） |
| private events/source records | 19,200 / 19,380（其中重复正文 180） |
| history tokens | 2,564（debug tokenizer） |
| alpha | 5.0078125C（C=512） |
| 可行参考记忆 | 385 token，B=1024 内；检索 cap=400 内 |
| 可行参考 query prompt+output | 最大 483 token，C=512 内 |
| independent Oracle issues | 0 |
| cross-split family overlap | 0 |

这是 debug lexical tokenizer 的受控 D1，不是 Qwen production 数据，也不是规范建议的 400/80/120 完整训练池。

2026-09-16，用户报告在远端提交 `4a134f249b57f29d56cdfacfb16b71a99400228b` 上完成 production Qwen3-4B tokenizer 构建与 CPU replay：build ID=`build_6356da44495c7078566d`，600 histories、6,000 queries，alpha=`5.002197265625C`，独立 Oracle issues=`0`、split overlap=`0`。对应四臂均值为 `V2_T=0.75`、`V2_END=0.90234375`、`V2_LIFE=V2_LIFE_MONITOR=0.875061765491453`；END/LIFE 在 2/4 轨迹不同，direct/compiled 最大差值为 `0.0`。这些远端 manifest/replay 文件尚未迁回本地核对 digest，因此记录为带明确代码/build 身份的用户报告结果，不扩展为本地复验或学习结论。

## P2：版本存储与 grounding

已实现：

- active Agent Memory 与 private Audit Ledger 分离；policy 只能读取当前 active revision。
- ADD/UPDATE/DELETE 原子更新；失败不改变状态或 revision；UPDATE 重新判定新正文，不继承旧 support。
- RETRIEVE 只返回 active revision 的实际 payload；截断后的字节单独评分；source registry 不是 policy RAG。
- B 序列化计 content/title/tags/source_refs/claims/custom；opaque memory/revision ID 不计。
- 每次实际 prompt 继续按 v1 `TokenAccounting` 检查 `prompt+max_new<=C`。
- grounder 对 source、content、time 分别判定；正确 pointer＋错误正文不得分，错误时间不得分；整个 retained set 可联合覆盖，删除冗余副本不降分，删除最后表示会回退。
- 环境不会用 private Oracle 拒绝或修正语义错误写入。

当前只准入 `controlled_semantics` 与 extractive control。没有约 200 个动态自然 free-text 去重审计单元，旧审计指标也不达标，因此 `audited_free_text` 仍 blocked。

## P3：END/LIFE 与 monitor

公式实现：同 history 内先对 m 题任务 EM 平均；END=`mean_j(0.5*u_M_end+0.5*u_E)`；每题在 eligible checkpoints 内先平均再跨题得到 retention；LIFE=`0.5*END+0.5*retention`。`no_retention_opportunity` fail closed，不缩小分母。V2_T/END/LIFE 总奖励分别为 task、task+λEND、task+λLIFE，debug λ=0.25。

四条 CPU fixture（无模型）各 21 checkpoint：

| strategy | task | END | retention | LIFE | V2_T total | V2_END total | V2_LIFE total |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager_timeline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.2500 | 1.2500 |
| late_timeline | 1.0000 | 1.0000 | 0.3300 | 0.6650 | 1.0000 | 1.2500 | 1.1663 |
| current_only | 1.0000 | 0.4375 | 0.2765 | 0.3570 | 1.0000 | 1.1094 | 1.0893 |
| wrong_body | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

END/LIFE 在 2/4 条轨迹上不同。direct evaluator 与 compiled monitor 对 336 个逐 checkpoint/query 状态及 432 个逐 policy-action/query 状态完全一致，最大 utility 差 `0`；`V2_LIFE` 与 `V2_LIFE_MONITOR` 平均 total 均为 `0.8763760653`。因此 compiled monitor 只保留离线交叉验证，不安排重复 GPU 臂。

## P4～P6 状态

- P4：未运行。当前本地没有生产 Qwen tokenizer/model/reader lock，也没有可核验 GPU。
- P5：三臂配置、30-update/10-step-dev 上限、模型 runtime producer、冻结 reader 分支、完整 K×m 发布契约、Trinity workflow 注册和 population-std advantage operator 已实现。本地 fake policy 测试覆盖 2×2 group、真实 `Experience` 字段、ActionEvent join、private observation 隔离与 reader token mask；Ray/vLLM/veRL 真实 trainer 接线仍未验证。
- P6：未运行。不得用 CPU fixture、mock PASS 或参数变化替代学习结果。

`preflight` 对 pilot 配置返回 exit 2，并列出未解析的 v1 runtime lock、初始化 checkpoint、policy/tokenizer/reader revision、retriever/grounder、optimizer、production manifest、frozen test manifest 和 GPU IDs。GPU ID 门禁拒绝 `null`、空列表、重复、负数、布尔值或其他非整数值；`runtime.gpu_ids=[]` 不再被误判为已配置。

## 实际验证命令

通过：

```powershell
python -m unittest discover -s tests\stream_dynamic_v2 -p '*_test.py'
# 40 tests, OK（2026-09-16 动作接口修复后）

python scripts\agemem_dynamic_v2.py build-data --config configs\stream_dynamic_v2\data_debug.yaml
python scripts\agemem_dynamic_v2.py validate-data --manifest runs\dynamic_v2\data_debug\manifest.json
python scripts\agemem_dynamic_v2.py env-smoke --config configs\stream_dynamic_v2\data_debug.yaml
python scripts\agemem_dynamic_v2.py replay --run-dir runs\dynamic_v2\cpu_fixture --profiles V2_T,V2_END,V2_LIFE,V2_LIFE_MONITOR
python scripts\agemem_dynamic_v2.py compare-backends --run-dir runs\dynamic_v2\cpu_fixture
python -m py_compile AgeMem_code_agentscope\streaming_memory\dynamic\environment.py trinity\common\dynamic_multiquery_contract.py trinity\common\workflows\memory_context\dynamic_runtime_producer.py trinity\common\workflows\memory_context\train_dynamic_multiquery.py trinity\algorithm\advantage_fn\dynamic_v2_advantage.py scripts\agemem_dynamic_v2.py
```

预期 blocked：

```powershell
python scripts\agemem_dynamic_v2.py preflight --config configs\stream_dynamic_v2\pilot_terminal.yaml
# exit 2; unresolved production identities/resources
```

环境性失败（没有进入断言，未计为回归失败）：

```powershell
python -m unittest tests.common.stream_mq_data_test tests.common.stream_mq_environment_test tests.common.stream_mq_training_contract_test
# 16 tests, OK
python -m unittest tests.common.m8_action_event_contract_test
# 18 tests, OK (3 environment-dependent skips)
```

未运行：`stream_mq_reward_test`、`memory_store_test` 与 `hotpotqa_oracle_benchmark_test` 在当前基础 Conda 环境分别缺少 `agentscope`／`shortuuid`，导入阶段即阻塞；没有将其记为 PASS。新实现未修改这些旧模块。

## 产物身份

- debug config：`configs/stream_dynamic_v2/data_debug.yaml`
- debug manifest：`runs/dynamic_v2/data_debug/manifest.json`（Git ignored）
- build/config/manifest SHA-256：`build_1c0c6121a9911a92eea9` / `71a67b1a46f33591aa8be8643f73edd14a6e395b468a4be83dee62c9ca4205e0` / `52c74fd08757b27a0ddad9d7b828503d3fc917d885de2ffd99f21e6da8998893`
- manifest 记录的代码/dirty digest：`cf80142d72b3009635234d5693fde4acf4a9c839` / `3fbdf218fb928560bba2123753644b44878e1b1cec1b8e663756e95d8b707715`
- fixture bundle：`runs/dynamic_v2/cpu_fixture/fixture_run.private.json`（Git ignored，含 private labels）
- reward report：`runs/dynamic_v2/cpu_fixture/reward_replay.json`（Git ignored），SHA-256 `0b80eb99558a04b33236cefbc80c1e0d93cceec0db9803c48b7ee82e0420a506`
- model/checkpoint：无；未运行 GPU

## 变更文件

- 动态实现：`AgeMem_code_agentscope/streaming_memory/dynamic/{schema,config,world_generator,world_oracle,independent_validator,environment,temporal_grounder,state_evaluator,lifecycle_monitor,reward_profiles,checkpoint_probe,replay}.py`
- CLI：`scripts/agemem_dynamic_v2.py`
- 训练发布边界：`trinity/common/dynamic_multiquery_contract.py`、`trinity/common/workflows/memory_context/{dynamic_runtime_producer,train_dynamic_multiquery}.py`、`trinity/algorithm/advantage_fn/dynamic_v2_advantage.py` 及相应 registry import；`trinity/explorer/workflow_runner.py` 保留动态 bench Experience 供严格审计
- 配置：`configs/stream_dynamic_v2/{base,data_debug,replay,frozen_diagnostic,pilot_terminal,pilot_end,pilot_life}.yaml`
- 测试：`tests/stream_dynamic_v2/` 下 5 个测试模块
- 文档：本报告、`docs/v2_current_implementation_audit.md`、`docs/v2_baseline_fidelity.md`、`STATUS.md`
- 用户提供的根规范 `AGEMEM_DYNAMIC_LOGIC_INCREMENTAL_SPEC_V2.md` 保持原文，未把它当成已完成状态。

## 研究边界与下一步

当前只能表述“受控动态任务的代码与数学契约通过”。不能表述训练有效、自然语义可靠、LIFE 优于 END、DFA 有额外收益或动态任务泛化。

下一步顺序：

1. 在远端把实际 v1 runtime lock、Qwen policy/tokenizer/reader revision、初始化 checkpoint、optimizer 与 GPU IDs 写入三臂共享锁。
2. 用冻结 Qwen tokenizer 在新 output root 构建 production 5C，并验证参考记忆同时满足 B 和 query prompt C；冻结 test 后不查看。
3. 在远端先运行 runtime/registry 单测，再配置仅 1 个 history、K=2、m=2 的 Trinity bench smoke；核对完整组、ActionEvent、reader token=0、policy version 和预算 receipt。该步需要 GPU，但不做 optimizer update。
4. smoke 通过后，固定 24～40 dev histories 做一次共享冻结 rollout，离线重放 T/END/LIFE；报告自然 grounder unclear、group std、END-LIFE 差异和成本。
5. 自由摘要若要进入语义训练，先完成约 200 个去重的 source/content/time 分层审计并达到预冻结门槛；否则继续 controlled/extractive 臂。
6. 门禁通过后才按每臂最多 30 update、每 10 step dev、seed 7 跑 V2_T/V2_END/V2_LIFE；不跑等价 monitor GPU 臂。
