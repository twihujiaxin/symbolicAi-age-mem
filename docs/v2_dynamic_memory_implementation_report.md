# streaming_dynamic_multiquery_v2 实施报告

## 当前结论

### 2026-09-16 跨chunk/具体例消融本地实现（v3）

用户确认下一步后，新增可选comparison single_vs_no_example_v3，不自动运行GPU。稳定选同一冻结train history首/中/末公开chunk，至少3chunk才接受；每chunk两对seed7/8，总12次独立首动作，不连续阅读、不共享memory、不按gold选chunk。两条件完全相同single-action system，处理仅移除当前chunk具体ADD例，全部公开正文/source ID仍可见。主环境boolean include_add_format_example默认True，不变更旧主协议/配置/reward。v1/v2比较保留，v3锁定chunk_indices/reading_mode/实际cases与12call预算，run/execute重建一致性检查。新增首句/非首句精确匹配、max-token hit和逐chunk写入统计，无奖励或强制非首句。

本地实测V2 **61 PASS**，旧回归 **31 PASS/3环境性SKIP**，diff-check PASS。新增3测试：公开chunk `[0,2,4]`确定选择、paired seed/system相同、仅删例且其他正文不变/其他chunk不可见；12case非首句合法执行和零写入真实计数；chunk选择篡改/不足3chunk拒绝。CPU fixture不是自然语义PASS。远端新CPU部署/prepare待执行，GPU前再次确认，预算单GPU1/12×最多512=6144response tokens，无reader/optimizer/API。命令见 docs/v2_cross_chunk_probe.md。

### 2026-09-16 用户授权物理GPU1后的真实v2探针结果

执行前物理GPU1 A6000占用24MiB、0%且无compute进程；HEAD=a76b8e9、干净工作区，results/run.log不存在。使用已冻结plan（digest bb6a356d24b57d73f8389617d8e36bacbb51395ffd14d9a5d299979345c4dce5），仅8次采样，GPU1/TP1/BF16/utilization0.55/vLLM0.10.2/thinkingFalse。对照task-explicit **3NEXT+1invalid_json(length512)，0写入**；single-action **4ADD/4parse_ok/4admitted/4exact-body**，seed7–10，每条response70tokens、memory68tokens，正文均为“自第 1 日起，星河项目0000的负责人为成员甲0000。”，source ID与公开首句一致。环境revision0→1，无截断；四份独立空memory中m1不构成重复ID。

实际prompt11912tokens、response828tokens；engine_startup20.761172s、sampling8.745864s、total29.794555s。reader/optimizer0，无训练、API或其他GPU启动。对照response文本及token IDs与上轮完全一致。独立复验plan/report提交/digest、代码SHA、config/tokenizer SHA、权重统计、行数、C/B和四条公开正文匹配PASS（非完整权重SHA）。远端Git干净，GPU结束后24MiB/0%，无强制终止其他任务。

结果仍在 `single-action-probe-a76b8e9-20260916-203631/results`。report SHA256=`21d2e6668b009e87f53d0f7633262588d2f36a6070bcaff8f130fa5623beec69`；actions=`db8b003592473afb0dc9304037123144504958b51ec65d3eb7a9272b950c0a68`；run.log=`fd970862eb6e4dec4e9e80783a3f7e414044ca9716112728e4cc46ba60405232`。模型run返回正常并输出GPU_PROBE_COMPLETE，但随后shell最后一条git status因PowerShell管道末尾CRLF报未知开关，整个SSH命令exit1；单独Python调用Git已确认干净，不能将该后置命令错误计为模型失败，也未重复采样。

研究边界：四条ADD都是公开格式例里的同一首句，存在示例照抄混杂；不能由4/4推广为稳定事实选择或整段信息保留。精确正文/time核对不代替一般语义grounder，未验收非示例事实、后续工具交互、跨chunk、memory必要性、END/LIFE奖励、非零advantage或学习有效。建议下一步先冻结小规模跨chunk/非示例事实诊断，不直接启动完整GPU训练；新增GPU运行需用户再次确认。本轮记录仅本地更新，未再提交/改变远端锁定HEAD。

### 2026-09-16 单动作对照远端CPU部署验收（GPU暂停）

提交 `a76b8e911a5dcd9e9d67ec5bffe05b6bbdc62887` 已推送fix/training-protocol并快进同步tx-06，未改变全局GitHub代理。远端实测V2 **58 PASS**、旧协议 **34 PASS，无SKIP**。正式CPU prepare完成，CUDA_VISIBLE_DEVICES空值，未运行模型采样。新plan目录：

`/data/hjx/Age_mem/runtime-v2-smoke-783d785-20260916-112222/single-action-probe-a76b8e9-20260916-203631`

独立只读复验 **INDEPENDENT_CPU_PLAN_VERIFY_PASS**：digest=`bb6a356d24b57d73f8389617d8e36bacbb51395ffd14d9a5d299979345c4dce5`，提交、代码SHA、model/config/tokenizer文件SHA、权重统计、实际prompt IDs重建通过。新旧comparison的history/model/budget/sampling相同，task-explicit对照messages/token IDs/seed与上轮逐项一致。task-explicit1443、新提示1535prompttokens，最大prompt+output2047≤C4096，B2048，max_tokens512，temperature0.6/top_p1/top_k-1，thinkingFalse，四对seed7–10，版本first_action_probe.v2/comparison task_vs_single_v2。远端工作区干净，results/run.log尚不存在，历史目录完整保留。

实际model/reader/optimizer调用0。待用户确认的GPU命令为 `python scripts/agemem_dynamic_v2_first_action_probe.py run --plan <上述目录>/plan.public.json --output-dir <上述目录>/results`，单物理GPU1、TP1/BF16/utilization0.55、8completions/最多4096response tokens、无reader/训练/API；执行前检查卡资源和既有结果。禁止在确认前运行GPU。完整权重SHA、自然语义、写入改善、全流式表现、非零advantage和学习有效均未验证。本段实测记录本地暂未再提交，不使既有plan锁定HEAD失效。

### 2026-09-16 首轮 GPU负结果与独立单动作对照

直接只读远端 report/actions/log，HEAD=c4f7fcb、干净工作区。真实单A6000/vLLM0.10.2/thinkingFalse/temperature0.6：legacy4NEXT；task-explicit3NEXT和1invalid_json，后者512tokens/finish_reason=length，是连续多ADD后截断。成功写入均0，不能推进无信号pilot。只读解码的7个完整ADD与首公开chunk正文/ID/时间一致，不修补、不执行。实际8calls、prompt11052tokens、response595tokens；engine15.403s、sampling8.399s、total24.094s，reader/optimizer0。身份/文件统计/行数/预算核验通过，完整权重hash未核验。

结果文件SHA256：plan=`4a977c2ae2c588dc863b68fef6ee3071d4f2bc09db931fc5f0ab27e382f99264`；report=`e00886974b90aab15c01882c72c358407ca81b3cef371fa18e3fb8b2a9010ae8`；actions=`7a4c94a38aa3d15cf46b598a1829afad48de2728a708e0ceb3845746b095484c`。旧目录与全部结果保持原样。

用户授权修改并执行远端非GPU工作后，新增 `task_vs_single_v2` 可选比较及 v2 probe身份。对照为原task-explicit，处理仅追加一个动作/一个具体事实/等待工具结果的说明，不要求ADD、不强制顺序、不改采样预算或主协议。默认v1两个提示及plan结构仍保持；历史plan重放需原提交。CLI锁定comparison，run/execute重建核对comparison与version，报告仅包含所选两臂。

本地实测：V2 **58 PASS**；旧回归34项 **31 PASS/3环境性SKIP**；diff-check PASS。新增3测试覆盖v1默认保持、新对照非system输入和seed匹配、多动作不修补、独立合法单事实执行、comparison/version篡改拒绝。resource-double不代表自然模型改善。远端新CPU测试/prepare待执行；GPU未运行，需用户再次确认。说明见 `docs/v2_single_action_probe.md`。

### 2026-09-16 正式部署与 CPU prepare 完成

用户授权直接远端执行后，本地提交/推送路径修复与测试记录：`c4f7fcb2f535e46b9e29fa3d8949471e49a64692`。远端检查工作区干净、分支匹配后，fetch + merge --ff-only 并断言完整提交一致。fetch 因全局 GitHub 代理一度延迟，但最终原命令成功；备用增量 bundle 已上传 `/data/hjx/Age_mem/probe-sync-c4f7fcb.bundle`，未用于 fetch，不修改代理配置或其他人的任务。

修复后 CPU 实测：V2 discovery **55 PASS**，旧 action/streaming 回归 **34 PASS，无 SKIP**。禁用 bytecode、设 CUDA_VISIBLE_DEVICES 空值，以既有 retry4 launcher/config/taskset 执行正式 `scripts/agemem_dynamic_v2_first_action_probe.py prepare --gpu-id 1`，新目录：

`/data/hjx/Age_mem/runtime-v2-smoke-783d785-20260916-112222/first-action-probe-c4f7fcb-20260916-202258`

实际输出 **prepared_cpu / FORMAL_CPU_PREPARE_COMPLETE**。plan=`plan.public.json`，thinking=False，sampling temperature0.6/top_p1/top_k-1/n1/max_tokens512；最大 prompt+output=1955≤C4096，预定8-call response上限4096，reader_calls=0、optimizer_updates=0。没有 GPU 模型采样；输出的 model_call_count=8 是未来采样预算，不是实际已执行次数。结束后独立 SSH 再确认完整 HEAD、干净工作区及 plan 文件存在（266098 bytes）。额外独立 plan digest/代码/tokenizer身份复验尝试因 SSH banner 超时未执行，未计 PASS；正常 GPU run 自带这些检查。

下一步保持远端该 HEAD 与锁定代码不变，按 `docs/v2_first_action_probe.md` 用物理 GPU 1 执行已有 plan；本轮未启动。正文/time语义、完整权重密码学身份、全流式记忆保持、非零 advantage 和学习有效性仍未验证。本地本段新实测记录未再次提交，避免让已准备 plan 的锁定 HEAD 失效。

### 2026-09-16 远端 CPU 实测（非部署验收）

SSH 密钥连接 tx-06 后，直接在远端干净工作区 `9e30fd50bc7f067a6b07e4c9de8afadb53721a0d` 执行：

- `python -m unittest discover -s tests/stream_dynamic_v2 -p '*_test.py'` → **55 PASS**。
- `python -m unittest tests.common.m8_action_event_contract_test tests.common.stream_mq_data_test tests.common.stream_mq_environment_test tests.common.stream_mq_training_contract_test` → **34 PASS，无 SKIP**。

使用 `/data/hjx/Age_mem/conda-envs/agemem-m8b/bin/python`，禁用 bytecode 写入。额外只读 Python 检查在内存中替换 `resolve_settings()` 的路径和提示，未修改远端文件；读取真实 `runtime-v2-smoke-783d785-20260916-112222/dynamic_v2_k2m2_smoke_retry4.yaml`、`dynamic_runtime_smoke.json` 和 `taskset`，使用 production tokenizer 实际构建 prompt IDs，得到 **CPU_VALIDATION_PASS**：thinking=False，temperature=0.6/top_p=1/top_k=-1/n=1/max_tokens=512，最大 prompt+output=1955≤C4096，B2048，八次采样预定 response 上限4096。fingerprint=`23a45e57acf0e480`，history=`hist_263ef931bf2c89521710`，单行训练数据与 launcher 模型路径/声明 fingerprint/row ID 核对通过，None/字符串/整数 thinking 值拒绝通过。

实际 model calls / reader calls / optimizer updates 均为 0，未写 plan，未启动 GPU；检查结束远端 Git 仍干净且 HEAD 未变。现有 55 项远端回归属于旧提交，不能声称远端已部署路径修复或完整 prepare 已运行。下一步提交/同步修复，然后正式 CPU prepare；GPU probe 及语义/学习有效性仍未验证。

### 2026-09-16 首动作 probe 配置读取路径修复

用户确认 launcher 已在 `explorer.rollout_model.enable_thinking` 显式配置 False。修正 probe `resolve_settings()` 的读取路径及错误提示，保留严格布尔校验；`model.model_path` 仍按现有接口核对，不修改 launcher 或实验 thinking 条件。回归覆盖嵌套 False/True、缺失、字符串及整数拒绝、顶层冲突不覆盖，以及 CLI resource-double prepare 将 False 写入 plan。

实测：`python -m unittest discover -s tests/stream_dynamic_v2 -p '*_test.py'` → **55 tests PASS**。真实 production prepare 和 GPU run 未运行，本地缺模型/tokenizer/datasets；远端同步后重跑原 prepare 命令，prepare 不需要 GPU。已有 plan 不能手工修改，应重新生成以锁定修复后的代码身份。

### 2026-09-16 retry4：格式已恢复、首动作探针待上卡

远端用户报告 68/68 responses 为 NEXT/ok/admitted=True，配置 max_decisions_per_chunk=3；每块的第一次 NEXT 立即结束，不是解析失败或额外调用额度耗尽。K2×34 chunks 无成功 ADD/UPDATE，reward mean/std=0、nonzero_advantage_rollouts=0。已经恢复的语法/事件前置连接不等于完整 smoke；不删 writes 断言，不强制每块 ADD，也不启动无信号 pilot。原始完整 retry4 文件未迁回，因此仍是用户报告结果，不伪装成本地复验。

本地增量实现独立首动作诊断，不改变主协议 v3 的默认 system/语法、旧 YAML、冻结数据与历史产物：

- `dynamic/first_action_probe.py`：仅首公开 chunk，legacy-v3 与 task-explicit-probe-v1 各 4 样本；新提示是任务说明前缀＋原 v3 system，动作语法/公开 ADD 示例/handles 完全相同。说明有限 context tail 与独立 reader、NEXT 不保存正文、具体实体关系值时间、历史事实保留，不看问题/gold，不规定动作顺序。
- 八次调用分别从空 memory 开始，匹配四个 seed；它们不是完整 K-rollout GRPO 组，不经过 trainer/advantage/reader/Oracle。
- CPU prepare 读取已验证的一份 train taskset，对 bench eval_taskset 的路径、fingerprint/IDs 与采样进行匹配，不错误继承 training rollout_args。explicit enable_thinking 同时用于预算渲染和实际传入 engine 的 prompt IDs。C/B 不放宽，每次输出最多512。
- CPU plan 固定 prompt IDs/源码/tokenizer/config/index SHA、声明 policy revision、权重文件大小/mtime；run 拒绝输入或资源身份改变。完整权重没有独立 SHA 校验，`full_weight_digest_verified=false`，不能把声明 tokenizer-hash revision 当作已核验权重身份。
- GPU 后端是单 GPU 的 standalone direct vLLM，八条 n=1 completion，真实 response IDs/长度/finish reason 记录；并行批次中的环境执行各自独立。不是原 Ray runtime 的完整复现、不是 optimizer 接线证明。
- 后验只将正文与公开可见 source 文字对齐，记录源 ID 可见/精确正文匹配及人工 content/time 复核字段；正确 source ID 不等于正确保存语义。导出实际 prompt/response tokens、engine startup/sampling/runtime 秒、vLLM/GPU 身份；不预填虚假 GPU 成本。

真实本地验证：V2 scope **55 tests PASS**；旧 action/streaming scope **31 PASS / 3 环境性 SKIP**；CLI --help 与 py_compile/diff-check PASS。测试覆盖同一公开输入、未来 chunk 不进入首动作、匹配 seed、空 memory 隔离、原 system 字节保留、source 正确但正文错误不计精确匹配、非法动作不执行、C/token cap 拒绝、实际 bench sampling/thinking 解析，以及完整 prepare/run 函数在 datasets/torch/vLLM 边界 doubles 下的 plan/八行输出/全 NEXT 零写入导出。Windows sandbox 拒绝新临时目录访问，沙箱外运行完成；同时修复了测试读取中文 JSON 的 GBK/UTF-8 问题。上述 mock/double 结果绝不作为学习有效或生产环境 PASS。

未运行：真实 production-tokenizer prepare、单 GPU 8-call 探针、首动作人工语义审计、完整流式 replay/model eval/trainer update。当前本地缺 production 模型/tokenizer/datasets；这些不阻止代码、CPU helper/CLI double 流程和指令交付。下一步在同步代码后按 `docs/v2_first_action_probe.md` 先 CPU prepare 再 GPU run，不直接完整重跑 retry4，也不安排额外等价 monitor GPU 实验。

### 2026-09-16 retry3 当前结论（覆盖下面 retry2 时点状态）

远端用户报告 204 Experiences，203 条非数组 JSON（示例为完整 ACTION-key 对象），1 条 `[{"name":"ACTION","arguments":{"type":"ADD",...}}]` 被日志记为 ACTION/ok。已在本地复现 `arguments.type` 覆盖 name 的实际 bug：该条执行 ADD，但 ActionEvent 标签与 write_count 按 ACTION 记录。不能再用旧 count=0 推断实际完全没有写入，也不能把 pointer 合法/正文看似正确写成语义验收。该次 receipt reward_mean/reward_std=0、nonzero_advantage_rollouts=0，**仍未通过 smoke，不具备已观测的 GRPO 学习信号**。203 条完整对象不符合严格数组协议；当前证据不支持把 retry3 主因归于截断。

本次已实现 action-interface v3（reward/profile 身份不变）：

- 独立 CPU public-action parser 校验完整 JSON 单元素数组与 name/arguments envelope；合法名称仅 ADD/UPDATE/DELETE/RETRIEVE/NEXT，拒绝 arguments.type、重复 JSON keys、非有限常量、多动作和截断。旧 tagged/default parser 未改。
- 执行动作 type 由已校验 name 最后赋值；准备 draft 使用同一原始响应的真实 spans，不加入合成字符。非法响应不产生 ActionEvent。
- 移除 ACTION 占位名称。system 给具体 NEXT 例，当前公开 chunk 给具有真实公开正文/ref 的完整 ADD 例，不使用 private registry/gold，也不强制任何动作顺序。例子是格式条件，不是支持事实提示；所有后续实验臂须统一采用 v3 格式条件。
- 非法响应按 not_array/unknown_action/reserved_type_argument 等分类，返回具体纠错；receipt 的 successful write 与 ActionEvent/执行名称一致。
- `scripts/agemem_dynamic_v2_response_audit.py` 只读分类旧落盘，不修补、不执行动作；`--verify-runtime` 对 v3 K2/m2 检查实际响应名称与 info/event 连接、唯一 action IDs、policy version、快照/问题分支数、成功写入计数和答案正文隔离。JSON exporter 不含原始完整 model tensors，故该脚本不声称重新验收全部 token/logprob；语义质量及学习效果也明确未检查。

实测命令：`python -m unittest discover -s tests/stream_dynamic_v2 -p '*_test.py'` → **47 tests PASS**。旧 action/streaming scope 34 项 → **31 PASS / 3 环境性 SKIP**。包括 retry3 两类真实形状的公开替代 fixture、203+1 只读分类计数、非法响应→具体反馈→正确 ADD 的完整 fake-policy 2×2 group、原始 span/draft/finalize、持久化 audit、标签/计数篡改拒绝。语义测试的 debug fixture C 从 260 调至 400，为新增公开 ADD 例及 receipts 留空间，专门的极小 C 拒绝测试保留；实际 production C/B、旧配置、manifest 和 checkpoint 均未更改。

未验证：真实 retry3 全文件离线审计（本地仅有用户分类/样例）、本地完整 datasets/Qwen CLI、远端 v3 production-tokenizer prompt preflight、GPU retry4、自然正文/source/time 正确率、trainer update。此前 retry3 时点的“GPU未运行”段落为历史；已运行但失败的 retry3 以上述新段为准。下一步命令见 `docs/v2_runtime_retry4.md`，部署前需要提交/同步本次改动。

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
# 55 tests, OK（2026-09-16 独立首动作探针后）

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
