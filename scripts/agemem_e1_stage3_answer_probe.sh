#!/usr/bin/env bash
set -euo pipefail

# Frozen 1.5B Stage-3 <answer> probe on the same 6 M5 train rows.
# Does not train, does not reuse M8b smoke or E1 repeat job names.
required_names=(
  AGEMEM_EXPECTED_COMMIT
  CUDA_DEVICE_ORDER
  CUDA_VISIBLE_DEVICES
  TRINITY_MODEL_PATH
  TRINITY_MODEL_REVISION
  HOTPOTQA_PATH
  TRINITY_CHECKPOINT_ROOT_DIR
  DASHSCOPE_API_KEY
)
for name in "${required_names[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'Missing or empty required environment variable: %s\n' "$name" >&2
    exit 2
  fi
done

if [[ ! "$AGEMEM_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'AGEMEM_EXPECTED_COMMIT must be a lowercase 40-character commit ID.\n' >&2
  exit 2
fi
if [[ "$CUDA_DEVICE_ORDER" != "PCI_BUS_ID" ]]; then
  printf 'CUDA_DEVICE_ORDER must be PCI_BUS_ID.\n' >&2
  exit 2
fi
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ ]]; then
  printf 'CUDA_VISIBLE_DEVICES must select exactly two numeric GPU indices.\n' >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGEMEM_PYTHON_BIN:-python}"
log_root="$TRINITY_CHECKPOINT_ROOT_DIR/e1_stage3_answer_probe_logs/$AGEMEM_EXPECTED_COMMIT"
project_dir="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8"
job_dir="$project_dir/agemem-e1-stage3-answer-probe"

if [[ ! -d "$TRINITY_CHECKPOINT_ROOT_DIR" ]]; then
  printf 'Checkpoint root must already exist on persistent storage.\n' >&2
  exit 2
fi
if [[ -e "$project_dir/agemem-e0-terminal-only-frozen-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-dry-run" || \
      -e "$project_dir/agemem-e1-terminal-only-repeat-s7" ]]; then
  printf 'Refusing a checkpoint root that already contains M8b smoke or E1 repeat jobs.\n' >&2
  exit 2
fi
if [[ -e "$job_dir" ]]; then
  printf 'Refusing to reuse existing probe job directory: %s\n' "$job_dir" >&2
  exit 2
fi
if [[ -e "$log_root" ]]; then
  printf 'Refusing to reuse existing Stage-3 answer-probe logs for this commit.\n' >&2
  exit 2
fi

cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to run the Stage-3 answer probe on a dirty worktree.\n' >&2
  exit 2
fi

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before the Stage-3 answer probe.\n' >&2
  exit 2
fi

"$python_bin" -c 'import flash_attn; v=flash_attn.__version__; assert v=="2.8.1", v'
"$python_bin" -m unittest tests.common.e1_stage3_answer_probe_test

mkdir -p "$log_root"
chmod 700 "$log_root"

ray_started=0
cleanup() {
  if [[ "$ray_started" -eq 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

ray_started=1
ray start --head --num-gpus=2 2>&1 | tee "$log_root/ray_start.log"
trinity run --config examples/agemem_hotpotqa/agemem_e1_stage3_answer_probe.yaml \
  2>&1 | tee "$log_root/probe.log"
if [[ ! -s "$job_dir/receipts/bench_step_0_model_0.json" ]]; then
  printf 'Stage-3 answer probe did not persist bench_step_0_model_0.json\n' >&2
  exit 1
fi
ray stop --force 2>&1 | tee "$log_root/ray_stop.log"
ray_started=0

"$python_bin" - "$job_dir" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
job = Path(sys.argv[1])
path = job / "buffer" / "explorer_output.jsonl"
print("== Stage-3 answer probe summary ==")
if not path.is_file():
    print("MISSING", path)
    sys.exit(1)
stage3 = []
with path.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        obj = json.loads(line)
        info = obj.get("info") or {}
        if info.get("trace_stage") != 3:
            continue
        text = obj.get("response_text") or ""
        stage3.append({
            "nudge": bool(info.get("stage3_final_answer_nudge")),
            "found": bool(info.get("found_answer")),
            "has_answer": "<answer>" in text.lower(),
            "has_tool": "<tool_call>" in text,
        })
print("stage3_rows", len(stage3))
print("nudge", Counter(row["nudge"] for row in stage3))
print("found_answer", Counter(row["found"] for row in stage3))
print("has_<answer>", Counter(row["has_answer"] for row in stage3))
print("has_<tool_call>", Counter(row["has_tool"] for row in stage3))
PY

printf 'Stage-3 answer probe finished.\n'
printf 'Logs: %s\n' "$log_root"
