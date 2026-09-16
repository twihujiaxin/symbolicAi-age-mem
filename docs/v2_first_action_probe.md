# V2 首动作短探针：旧／新任务提示

本次新增代码须先提交/推送并在远端同步，再运行下列命令。旧 `3cfd411` 不包含探针。没有自动运行 GPU、外部 API 或远端写入。

目的：判断明确 memory-writer 任务说明是否能缓解 NEXT-only，而不是证明训练有效。两组只改变 system 的任务说明前缀，保留语法、NEXT 例、公开 ADD 例、handles、模型和采样。新提示正确说明 reader 有有限 context tail，不误称 reader 绝对只看 LTM；无需每块 ADD、不强制操作顺序，历史事实不会因当前更新自动失效。

## 1. 同步、CPU 测试（不需要 GPU）

```bash
source /data/conda/etc/profile.d/conda.sh
conda activate /data/hjx/Age_mem/conda-envs/agemem-m8b
cd /data/hjx/Age_mem/AgeMem
git status -sb
# 有未提交改动则先停止，保留修改，不能 reset/stash 覆盖。
git fetch origin
git switch fix/training-protocol
git pull --ff-only origin fix/training-protocol
python -m unittest discover -s tests/stream_dynamic_v2 -p '*_test.py'
```

本地实际55 tests PASS；完整 CLI 流程使用 resource doubles，远端必须以实际结果为准。

## 2. 冻结生产 tokenizer 计划（不需要 GPU，不调用模型）

```bash
export SMOKE_DIR=/data/hjx/Age_mem/runtime-v2-smoke-783d785-20260916-112222
export PROBE_DIR="$SMOKE_DIR/first-action-probe-$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)"

python scripts/agemem_dynamic_v2_first_action_probe.py prepare \
  --config "$SMOKE_DIR/dynamic_runtime_smoke.json" \
  --launcher "$SMOKE_DIR/dynamic_v2_k2m2_smoke_retry4.yaml" \
  --taskset "$SMOKE_DIR/taskset" \
  --gpu-id 1 \
  --output-dir "$PROBE_DIR"
```

预期 `status=prepared_cpu`、model_call_count=8、reader_calls=0、optimizer_updates=0、response_token_upper_bound≤4096。actual sampling 显示冻结 bench 的 temperature/top_p/top_k，不擅自提高温度；若 temperature=0，要记录它是无随机探索条件。max_prompt_plus_output 必须≤冻结 C。prepared 后不得改模型、tokenizer、代码或 plan；变更则在新的 PROBE_DIR 重新 prepare。

选择的是已冻结的一份 train history 的第一个公开 chunk，不打开问题、gold、test 或私有原文索引。计划保存为 `plan.public.json`，含原公开 history 及八份实际 prompt IDs；模型实际只接收首 chunk 的 messages，不接收计划中的后续 chunks。

身份检查包括源 launcher/taskset fingerprint、model path/声明 revision、model config/index 与 tokenizer SHA、权重文件大小/mtime，以及源码/commit。完整权重 digest 未计算，不声称权重密码学身份已验证。CPU 预算渲染使用与 vLLM 相同的 explicit thinking flag，实际直接传 token IDs，避免二次模板化。

## 3. 首动作采样（需要 GPU：仅物理卡 1）

预算：2 条件×4 个独立样本=8 completions，n=1，每次最多512 response tokens，总输出上限4096；四对匹配 seed。同一次 standalone vLLM engine 进行批量采样，TP=1、BF16、GPU utilization=0.55、max_model_len=C。只需一张已分配的 A6000，不启动第二个 reader engine，不调用付费 API，不做参数更新。engine 初次加载/编译仍有启动耗时，实际 cost 记录在 report 中。

先确认卡 1 可用。若 retry4 Ray actors 仍占卡，且当前用户无其他需保留 Ray job，才先 `ray stop --force`；本脚本不会自动终止现有任务。

```bash
nvidia-smi -i 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

(
set -e
test ! -e "$PROBE_DIR/run.log" && test ! -e "$PROBE_DIR/results" || {
  echo "存在既有结果，请保留并使用新的结果目录/计划。"
  exit 1
}

set -o pipefail
python scripts/agemem_dynamic_v2_first_action_probe.py run \
  --plan "$PROBE_DIR/plan.public.json" \
  --output-dir "$PROBE_DIR/results" \
  2>&1 | tee "$PROBE_DIR/run.log"
)
```

存在旧结果时停止，不能接着运行最后一条。建议在已有 tmux 会话中运行；失败保留日志，不增加 max tokens、C 或重新采样直到出现 ADD。空结果目录也属于失败记录，重试使用新目录，不覆盖。

## 4. 查看结果（不需要 GPU）

```bash
cat "$PROBE_DIR/results/report.json"
```

回传 report.json。人工复核文件：`$PROBE_DIR/results/actions.public.json`，逐条含 raw response、name/arguments、response IDs、memory B、action outcome、source_refs_visible、exact_visible_source_body、human_content_review/human_time_review/human_notes。

判断边界：

- ADD 多于旧条件，仅说明此首 chunk 的首动作行为改变，不证明 END/LIFE、学习或任务 F1 改善。
- 正确 pointer＋无关正文不会计精确正文匹配；自由改写与正文时间正确性仍需人工复核，精确匹配不是完整语义 grounder。
- UPDATE/DELETE 对空 memory 报 not-found 是真实结果，不自动转 ADD。NEXT 全部出现时报告零写入，不伪造 probe PASS。
- 八个重复样本来自同一个 chunk，不是八个独立测试任务，不是完整 K 份阅读 GRPO 组；不计算训练 advantage。
- 若新提示能稳定写入且正文/source/time 可信，再单独版本化主协议提示、冻结对照条件后重跑完整 smoke；若仍全 NEXT，保留负结果，先查模型任务理解/实际输入，不直接启动无信号 pilot。
