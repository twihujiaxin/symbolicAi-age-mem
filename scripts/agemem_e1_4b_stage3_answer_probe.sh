#!/usr/bin/env bash
set -euo pipefail

# Independent Qwen3-4B Stage-3 <answer> probe on the same 6 M5 train rows.
# Does not train. Does not reuse 1.5B probe/smoke or 4B E1 job names.
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
if [[ ! "$TRINITY_MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'TRINITY_MODEL_REVISION must be a lowercase 40-character revision.\n' >&2
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
if [[ "$TRINITY_MODEL_PATH" == *Qwen2.5-1.5B-Instruct* ]]; then
  printf 'Refusing the 1.5B model path; 4B probe requires the locked Qwen3-4B directory.\n' >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGEMEM_PYTHON_BIN:-python}"
log_root="$TRINITY_CHECKPOINT_ROOT_DIR/e1_4b_stage3_answer_probe_logs/$AGEMEM_EXPECTED_COMMIT"
project_dir="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8"
job_dir="$project_dir/agemem-e1-4b-stage3-answer-probe"

if [[ ! -d "$TRINITY_CHECKPOINT_ROOT_DIR" ]]; then
  printf 'Checkpoint root must already exist on persistent storage.\n' >&2
  exit 2
fi
if [[ -e "$project_dir/agemem-e0-terminal-only-frozen-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-dry-run" || \
      -e "$project_dir/agemem-e1-terminal-only-scale" || \
      -e "$project_dir/agemem-e1-terminal-only-repeat-s7" || \
      -e "$project_dir/agemem-e1-stage3-answer-probe" || \
      -e "$project_dir/agemem-e0-terminal-only-4b-frozen-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-4b-dry-run" || \
      -e "$project_dir/agemem-e0-terminal-only-4b-format-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-4b-format" || \
      -e "$project_dir/agemem-e0-terminal-only-4b-format-var-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-4b-format-var" ]]; then
  printf 'Refusing a checkpoint root that already contains 1.5B, 4B E1, format-conditioned, or probe jobs.\n' >&2
  exit 2
fi
if [[ -e "$job_dir" ]]; then
  printf 'Refusing to reuse existing 4B probe job directory: %s\n' "$job_dir" >&2
  exit 2
fi
if [[ -e "$log_root" ]]; then
  printf 'Refusing to reuse existing 4B Stage-3 answer-probe logs for this commit.\n' >&2
  exit 2
fi

cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to run the 4B Stage-3 answer probe on a dirty worktree.\n' >&2
  exit 2
fi

"$python_bin" - <<'PY'
import json
import os
from pathlib import Path

lock = json.loads(Path("configs/e1_4b_stage3_answer_probe.json").read_text(encoding="utf-8"))
expected = lock["model"]["expected_revision"]
actual = os.environ["TRINITY_MODEL_REVISION"]
if actual != expected:
    raise SystemExit(
        f"TRINITY_MODEL_REVISION {actual} does not match locked 4B revision {expected}"
    )
if lock["model"]["repository_id"] != "Qwen/Qwen3-4B":
    raise SystemExit("4B probe lock repository_id drifted")
if not lock.get("stage3_require_final_answer") or not lock.get("stage3_repair_untagged_answer"):
    raise SystemExit("4B Stage-3 probe must enable the answer nudge.")
if lock.get("trainer_total_steps") != 0:
    raise SystemExit("4B Stage-3 probe must not train.")
PY

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before the 4B Stage-3 answer probe.\n' >&2
  exit 2
fi

"$python_bin" -c 'import flash_attn; v=flash_attn.__version__; assert v=="2.8.1", v'
"$python_bin" -m unittest tests.common.e1_4b_stage3_answer_probe_test

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
trinity run --config examples/agemem_hotpotqa/agemem_e1_4b_stage3_answer_probe.yaml \
  2>&1 | tee "$log_root/probe.log"
if [[ ! -s "$job_dir/receipts/bench_step_0_model_0.json" ]]; then
  printf '4B Stage-3 answer probe did not persist bench_step_0_model_0.json\n' >&2
  exit 1
fi
ray stop --force 2>&1 | tee "$log_root/ray_stop.log"
ray_started=0

"$python_bin" - "$job_dir" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
job = Path(sys.argv[1])
path = job / "trajectories" / "stage3_final_turn.jsonl"
print("== 4B Stage-3 answer probe summary ==")
if not path.is_file() or path.stat().st_size == 0:
    print("MISSING_OR_EMPTY", path)
    sys.exit(1)
rows = []
with path.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(obj)
print("rows", len(rows))
print("round", Counter(row.get("round") for row in rows))
print("nudged", Counter(row.get("nudged") for row in rows))
print("repaired", Counter(row.get("repaired") for row in rows))
print("found_answer", Counter(row.get("found_answer") for row in rows))
print("has_answer_tag", Counter(row.get("has_answer_tag") for row in rows))
print("has_tool_call", Counter(row.get("has_tool_call") for row in rows))
last = [row for row in rows if row.get("nudged") or row.get("repaired") or row.get("round") == 1]
print("last_or_repair_rows", len(last))
repair = [row for row in rows if row.get("repaired")]
print("repair_found_answer", Counter(row.get("found_answer") for row in repair))
print("repair_has_answer_tag", Counter(row.get("has_answer_tag") for row in repair))
for row in last[:8]:
    preview = (row.get("response_preview") or "").replace("\n", " ")[:160]
    print("--- task", row.get("task_id"), "found", row.get("found_answer"), "parsed", row.get("parsed_answer"))
    print("preview", preview)
PY

printf '4B Stage-3 answer probe finished.\n'
printf 'Logs: %s\n' "$log_root"
