# V2 retry4：严格动作与真实写入连接验收

本文件的 v3 改动须先提交/推送并在远端同步，不能在旧 `2777a63` 上运行。旧 retry3 与冻结数据保留。以下是准备好的命令，未代替你执行远端/GPU 操作。所有研究臂以后统一 v3 格式条件，不将本次结果与旧格式混作奖励对照。

## 1. 同步与 CPU 门禁（不需要 GPU）

```bash
source /data/conda/etc/profile.d/conda.sh
conda activate /data/hjx/Age_mem/conda-envs/agemem-m8b
cd /data/hjx/Age_mem/AgeMem
git status -sb
# 有未提交改动则先停止，不 reset/stash 覆盖。
git fetch origin
git switch fix/training-protocol
git pull --ff-only origin fix/training-protocol
git rev-parse HEAD
python -m unittest discover -s tests/stream_dynamic_v2 -p '*_test.py'
```

本地实际 47 tests PASS；远端以实际结果为准。

```bash
export SMOKE_DIR=/data/hjx/Age_mem/runtime-v2-smoke-783d785-20260916-112222
python scripts/agemem_dynamic_v2_response_audit.py \
  --experience-file "$SMOKE_DIR/checkpoints-retry3/Trinity-RFT-AgeMem-Dynamic-V2/agemem-dynamic-v2-k2m2-smoke-retry3/buffer/explorer_output.jsonl"

python scripts/agemem_dynamic_v2_prompt_preflight.py \
  --config "$SMOKE_DIR/dynamic_runtime_smoke.json" \
  --taskset "$SMOKE_DIR/taskset"
```

旧响应审计应复核完整 203+1 分类，绝不重放为训练数据。新增公开 ADD 例增加 prompt 成本，必须用实际冻结 tokenizer 再验 C；此预检只覆盖空 memory/无模型 receipt 的 ingest。运行时继续逐次检查带 handles/response/receipt 的实际 C 与全部可恢复 memory 的 B，不放宽预算。

## 2. 创建独立 retry4（不需要 GPU）

```bash
python - "$SMOKE_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
target = root / 'dynamic_v2_k2m2_smoke_retry4.yaml'
checkpoint = root / 'checkpoints-retry4'
assert not target.exists() and not checkpoint.exists(), '禁止覆盖 retry4 产物'
cfg = json.loads((root / 'dynamic_v2_k2m2_smoke_retry3.yaml').read_text())
assert cfg['mode'] == 'bench', '本次不允许启动训练'
cfg['name'] = 'agemem-dynamic-v2-k2m2-smoke-retry4'
cfg['checkpoint_root_dir'] = str(checkpoint)
target.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print(target)
PY
```

仅变更 job/output 身份，不改模型、tokenizer revision、训练题、K/m、temperature 和输出预算。已冻结 production build=`build_6356da44495c7078566d`，实际 manifest/config 身份仍以原 smoke 配置为准；本地尚未迁回核对其 digest。

## 3. bench（需要 GPU：物理卡 1、2；不做 optimizer update）

预算沿用一份 history、K=2、m=2、每 chunk 最多 3 次 policy response、每次最多 512 token、冻结 reader 4 次、每次最多 256 answer token。按旧 34 chunks 上界为 204 次 policy response、104,448 policy response tokens 加 1,024 reader answer tokens；实际 chunk/预算请以配置/taskset 为准，prompt token 与 GPU 耗时另计。两台本地 vLLM engine 沿用旧配置；不调用新的外部付费模型。

确认 1/2 卡获配且可用，当前用户无其他需保留的 Ray job，才执行 ray stop。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export TRINITY_MODEL_PATH=/data/hjx/Age_mem/models/Qwen3-4B
export HOTPOTQA_PATH=/data/hjx/Age_mem/data/hotpot_qa/fullwiki
export TRINITY_CHECKPOINT_ROOT_DIR="$SMOKE_DIR/checkpoints-retry4"
export AGEMEM_EXPECTED_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$SMOKE_DIR/logs"
ray stop --force
set -o pipefail
trinity run --config "$SMOKE_DIR/dynamic_v2_k2m2_smoke_retry4.yaml" \
  2>&1 | tee "$SMOKE_DIR/logs/dynamic_v2_k2m2_smoke_retry4.log"
```

命令非零退出或 C/B 拒绝则停止，保留日志，不去掉断言，不增加预算继续调试。

## 4. 验收（不需要 GPU）

```bash
export JOB="$SMOKE_DIR/checkpoints-retry4/Trinity-RFT-AgeMem-Dynamic-V2/agemem-dynamic-v2-k2m2-smoke-retry4"
python scripts/agemem_dynamic_v2_response_audit.py \
  --experience-file "$JOB/buffer/explorer_output.jsonl" \
  --verify-runtime
```

回传 strict_parse_codes、ActionEvent 数、成功写入数和 receipt。脚本拒绝 ACTION 标签、名称覆盖、漏/重复事件、计数错位与 reader/policy version 混用；不强求所有 response 无错，也不把 group complete、一个成功 ADD 或非零更新当作学习有效。

只有工程门禁通过后，才逐项审计 source/content/time、实际 token/GPU 成本和冻结模型诊断的组内奖励差异，再决定是否投入 pilot。语法/指针正确不代表正文正确。token/logprob 完整性还需原始 pipeline 契约及 tensor 证据，summary JSON 不足以独立复验。
