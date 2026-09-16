# 推理输入CPU审计与有限顺序ingest诊断

## CPU实测结论（2026-09-16）

读取首轮v1、v2、v3、v4的冻结plan/actions，逐行核对profile/seed/prompt计数。按实际prompt IDs、采样、seed、model及backend分组：12个重复组中6组response token不同，3组动作或parse类别不同。其余差异只是JSON序列化等输出变化。不是新GPU复现实验，也不能据此确定差异由内核/批处理/随机数实现导致。不同probe批量布局是未隔离条件，跨批次比例不能当作稳定收益。

本地远端production tokenizer加载后，以thinking=False重建四轮所有prompt IDs，全部完全一致。模型config qwen3/Qwen3ForCausalLM，hidden_size2560、36层、vocab151936、BF16、tie_word_embeddings=True；tokenizer长度151669、最大ID151668，EOS151645/PAD151643，与config一致且在embedding范围内。safetensors header embedding shape=[151936,2560]，无独立lm_head符合声明的共享embedding。未发现明显模型/tokenizer/模板错配；不是官方权重真实性证明。

版本：vLLM0.10.2 / torch2.8.0 / transformers4.57.6 / safetensors0.8.0。实际安装源entrypoints/llm.py:1579按request_id排序输出，gpu_model_runner.py:542对seeded采样manual_seed，topk_topp_sampler.py使用按请求generator。没有在CPU源审计中发现显式丢弃seed或输出完成顺序直接错配的证据；已有结果未保存运行时RNG状态/独立seed回执，受控重复GPU复现仍未运行。

已执行CPU sha256sum计算本地三份完整权重文件指纹：

| 分片 | 实际SHA256 |
|---|---|
| model-00001-of-00003.safetensors | 328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223 |
| model-00002-of-00003.safetensors | 6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5 |
| model-00003-of-00003.safetensors | e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1 |

这是观测文件身份，未与官方固定revision/checksum交叉核验；旧plan声明的local-tokenizer-sha256不能作为权重revision。旧报告full_weight_digest_verified=False保留，不追改历史结果。新run仍核验文件统计而非每次重hash，边界如实记录。

## 有限顺序诊断（非主结果、GPU前确认）

不新增/优化提示，使用已有single_action_no_example_v3。沿用冻结v4 source plan的模型/tokenizer/history/budget/sampling，取前3连续公开chunk，不按gold/奖励选数据。两份独立memory以固定serial顺序执行，每chunk最多3次动作；逐动作反馈进入下一次真实prompt，合法NEXT结束chunk，上限耗尽由环境external commit，不伪造NEXT。后续chunk不提前可见，问题/gold/私有index/reader均不加载。

这是短prefix **ingest-only诊断**：不满足原完整长流alpha，不测回答或记忆必要性，不是GRPO组/Experience contract验收，不训练/不计reward/advantage。复用真实DynamicMemoryEnvironment/strict parser，未重建trainer或改变runtime producer/main protocol/reward。每次实际调用检查C、全部active可恢复payload检查B，导出持久memory/checkpoint供人工审计。

最多2×3×3=18calls，每次最多512输出tokens=9216上限，单物理GPU1/TP1/BF16/utilization0.55，temperature0.6/top_p1/top_k-1、thinkingFalse；reader/optimizer/API0。每份rollout seed7/8，call seed=rollout_seed+1000*turn，执行顺序serial固定，但**不因此声称确定性已证明**。GPU前需要用户确认；启动/总耗时未测。

CPU准备（使用新目录）：

```bash
export SMOKE_DIR=/data/hjx/Age_mem/runtime-v2-smoke-783d785-20260916-112222
export SEQ_DIR="$SMOKE_DIR/sequential-ingest-$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES= python scripts/agemem_dynamic_v2_sequential_probe.py prepare \
 --source-plan "$SMOKE_DIR/structure-only-probe-1b87cc6-20260916-205209/plan.public.json" \
 --output-dir "$SEQ_DIR"
```

prepare锁定source plan digest、原代码文件SHA和模型/tokenizer身份；source的历史HEAD不同允许，但原probe源码改变则拒绝。新plan锁定当前HEAD/新增脚本与库SHA。首prompt可CPU实测；未来prompt依赖实际反馈，不能用静态preflight声称全部未来prompt已经验证，run每次检查。

确认GPU后run命令：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
python scripts/agemem_dynamic_v2_sequential_probe.py run \
 --plan "$SEQ_DIR/plan.public.json" --output-dir "$SEQ_DIR/results"
```

先核对HEAD/卡/无既有results及日志。每action写actions.public.jsonl并flush，失败保留已有记录；结束写snapshots.public.json/report.json，不覆盖旧probe。重点人工审计duplicate/missing ID与纠错是否有效、旧记忆是否保留/错误更新、当前source是否支持content/time、逐动作C/B。工程异常保留日志并停止；模型全NEXT/错误/零写入是可解释负结果，不通过反复重采样修饰。受控seed复现、完整多问题分支、GRPO对接、END/LIFE、训练和test主成绩均仍未验证。
