#!/usr/bin/env bash
set -euo pipefail

# 1.5B terminal-only E1 scale: 24 train rows, 8 trainer steps, no <answer> nudge.
# Refuses to run until configs/e1_scale.json has frozen 24-row IDs committed.
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
lock_path="$repository_root/configs/e1_scale.json"
log_root="$TRINITY_CHECKPOINT_ROOT_DIR/e1_scale_logs/$AGEMEM_EXPECTED_COMMIT"
project_dir="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8"
job_dir="$project_dir/agemem-e1-terminal-only-scale"
trainer_steps="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["trainer_total_steps"])' "$lock_path")"
trainer_receipt="$job_dir/receipts/trainer_step_${trainer_steps}.json"
eval_receipt="$job_dir/receipts/bench_step_1_model_1.json"

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
if [[ -e "$job_dir" && ! -s "$trainer_receipt" ]]; then
  printf 'Refusing to reuse incomplete scale job directory: %s\n' "$job_dir" >&2
  exit 2
fi

cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to run E1 scale on a dirty worktree.\n' >&2
  exit 2
fi

"$python_bin" - <<'PY'
from trinity.common.e1_scale import load_lock, selection_is_complete
lock = load_lock()
if lock.get("selection_status") != "frozen" or not selection_is_complete(lock):
    raise SystemExit("E1 scale lock is still pending; run scripts/agemem_e1_scale_select.py --write-yaml, commit, then rerun.")
if lock.get("stage3_require_final_answer") or lock.get("stage3_repair_untagged_answer"):
    raise SystemExit("E1 scale must not enable Stage-3 answer nudges.")
PY

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before E1 scale.\n' >&2
  exit 2
fi

"$python_bin" -c 'import flash_attn; v=flash_attn.__version__; assert v=="2.8.1", v'
"$python_bin" -m unittest tests.common.e1_scale_contract_test

mkdir -p "$log_root"
chmod 700 "$log_root"

ray_started=0
cleanup() {
  if [[ "$ray_started" -eq 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

start_ray() {
  ray_started=1
  ray start --head --num-gpus=2 2>&1 | tee "$1"
}

stop_ray() {
  ray stop --force 2>&1 | tee "$1"
  ray_started=0
}

if [[ ! -s "$trainer_receipt" || ! -d "$job_dir/global_step_${trainer_steps}" ]]; then
  start_ray "$log_root/ray_start.log"
  trinity run --config examples/agemem_hotpotqa/agemem_e1_scale.yaml \
    2>&1 | tee "$log_root/train.log"
  if [[ ! -s "$trainer_receipt" ]]; then
    printf 'E1 scale did not persist trainer_step_%s.json\n' "$trainer_steps" >&2
    exit 1
  fi
  if [[ ! -d "$job_dir/global_step_${trainer_steps}" ]]; then
    printf 'E1 scale did not persist global_step_%s\n' "$trainer_steps" >&2
    exit 1
  fi
  stop_ray "$log_root/ray_train_stop.log"
fi

if [[ ! -s "$eval_receipt" ]]; then
  start_ray "$log_root/ray_eval_start.log"
  trinity run --config examples/agemem_hotpotqa/agemem_e1_scale_eval.yaml \
    2>&1 | tee "$log_root/eval.log"
  if [[ ! -s "$eval_receipt" ]]; then
    printf 'E1 scale eval did not persist bench_step_1_model_1.json\n' >&2
    exit 1
  fi
  stop_ray "$log_root/ray_eval_stop.log"
fi

printf 'E1 terminal-only scale finished.\n'
printf 'Logs: %s\n' "$log_root"
