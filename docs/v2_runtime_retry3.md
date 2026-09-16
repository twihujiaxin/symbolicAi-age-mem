# V2 runtime retry3：直接执行步骤

目标仅为动作接口工程验收；不更新参数、不重跑旧 E1/E3、不覆盖 retry2。已授权双 A6000，仍先执行 CPU 门禁。预算沿用原 1 history、K=2、m=2、每 chunk 最多 3 次 policy response、每次最多 512 response tokens，4 次冻结 reader 调用；实际 chunk 数取冻结任务。该次没有外部付费 API，由现有本地 reader server 回答。

## 1. 同步与 CPU 单测（不需要 GPU）

```bash
source /data/conda/etc/profile.d/conda.sh
conda activate /data/hjx/Age_mem/conda-envs/agemem-m8b
cd /data/hjx/Age_mem/AgeMem
git status -sb
git fetch origin
git switch fix/training-protocol
git pull --ff-only origin fix/training-protocol
git rev-parse HEAD
python -m unittest discover -s tests/stream_dynamic_v2 -p '*_test.py'
```

若有未提交改动或 pull 拒绝，先停止，不 reset/stash 覆盖。本地这组实际运行 40 tests PASS；远端必须以自己的结果为准。

## 2. 冻结 tokenizer 初始 prompt 预检（不需要 GPU）

```bash
export SMOKE_DIR=/data/hjx/Age_mem/runtime-v2-smoke-783d785-20260916-112222
python scripts/agemem_dynamic_v2_prompt_preflight.py \
  --config "$SMOKE_DIR/dynamic_runtime_smoke.json" \
  --taskset "$SMOKE_DIR/taskset"
```

此命令只读公开 taskset，使用已冻结本地 tokenizer，不生成模型 token、不查看 test 问题、不重建数据。必须 PASS 后再上卡。它只验空 memory 的 ingest prompts；模型 response、receipt 和 handles 增长仍由 runtime 每次检查 C。

## 3. 独立 retry3 配置（不需要 GPU）

```bash
python - "$SMOKE_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
source = root / 'dynamic_v2_k2m2_smoke_retry2.yaml'
target = root / 'dynamic_v2_k2m2_smoke_retry3.yaml'
assert not target.exists(), f'保留既有产物：{target}'
cfg = json.loads(source.read_text())
assert cfg['mode'] == 'bench', '本次不允许启动训练'
cfg['name'] = 'agemem-dynamic-v2-k2m2-smoke-retry3'
cfg['checkpoint_root_dir'] = str(root / 'checkpoints-retry3')
assert not Path(cfg['checkpoint_root_dir']).exists(), '禁止覆盖 checkpoint 根'
target.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print(target)
PY
```

复用原动态 config、tokenizer revision、模型、数据 fingerprint、K/m 和预算。新 action-interface 在 receipt 中记录，不能与 retry2 混成相同条件实验。

## 4. 运行 bench（需要 GPU：物理卡 1、2）

确认共享服务器上 1/2 卡可用，且当前用户没有其他需要保留的 Ray job，才执行 ray stop。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export TRINITY_MODEL_PATH=/data/hjx/Age_mem/models/Qwen3-4B
export HOTPOTQA_PATH=/data/hjx/Age_mem/data/hotpot_qa/fullwiki
export TRINITY_CHECKPOINT_ROOT_DIR="$SMOKE_DIR/checkpoints-retry3"
export AGEMEM_EXPECTED_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$SMOKE_DIR/logs"
ray stop --force
set -o pipefail
trinity run --config "$SMOKE_DIR/dynamic_v2_k2m2_smoke_retry3.yaml" \
  2>&1 | tee "$SMOKE_DIR/logs/dynamic_v2_k2m2_smoke_retry3.log"
```

非零退出时保留日志，不继续训练，不通过去掉断言“解决”。

## 5. 落盘动作审计（不需要 GPU）

```bash
export JOB="$SMOKE_DIR/checkpoints-retry3/Trinity-RFT-AgeMem-Dynamic-V2/agemem-dynamic-v2-k2m2-smoke-retry3"
python - "$JOB/buffer/explorer_output.jsonl" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
from trinity.common.action_event_contract import ACTION_EVENTS_KEY
rows = [json.loads(s) for s in Path(sys.argv[1]).read_text().splitlines() if s.strip()]
infos = [r.get('info') or {} for r in rows]
receipts = [i['dynamic_group_receipt'] for i in infos if 'dynamic_group_receipt' in i]
events = [e for i in infos for e in i.get(ACTION_EVENTS_KEY, [])]
writes = [i for i in infos if i.get('dynamic_action_type') in ('ADD','UPDATE') and i.get('dynamic_action_admitted')]
print('experiences:', len(rows))
print('actions:', dict(Counter(i.get('dynamic_action_type') for i in infos)))
print('outcomes:', dict(Counter(i.get('dynamic_action_code') for i in infos)))
print('events:', len(events), 'admitted_memory_writes:', len(writes))
assert len(receipts) == 1, len(receipts)
r = receipts[0]
print('receipt:', json.dumps(r, ensure_ascii=False, indent=2))
assert r['complete_bundle_count'] == 1
assert r['read_rollout_count'] == 2 and r['query_branch_count'] == 4
assert r['model_used'] and r['reader_frozen'] and r['reader_actor_loss_tokens'] == 0
assert r['action_interface_version'] == 'agemem.dynamic.action_interface.v2'
assert events, 'no executable ActionEvent'
ids = [e['action_id'] for e in events]
assert len(ids) == len(set(ids)), 'duplicate action_id'
assert writes, '动作能解析，但尚未产生成功的知识写入'
assert all(i.get('phase') == 'ingest' for i in infos)
assert all(i.get('dynamic_action_interface') == r['action_interface_version'] for i in infos)
assert r['admitted_memory_write_count'] == len(writes)
blob = json.dumps(infos)
assert '"answer_text"' not in blob and '"retrieved_payloads"' not in blob
print('ACTION INTERFACE SMOKE PASS; 不是学习有效/语义正确性验收')
PY
```

把 actions、outcomes、receipt 和成功写入的正文样例发回，下一步才判断 source/content/time 语义、实际预算和冻结模型诊断是否准入。语法有效、来源合法、组完整都不能替代知识正文正确性；该 smoke 更不能证明奖励改善学习。
