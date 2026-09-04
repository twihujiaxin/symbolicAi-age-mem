#!/usr/bin/env bash
set -euo pipefail

# Format-variance Qwen3-4B GRPO: 6-row K=4, 3 trainer steps, Stage-3 nudge.
# Skips E0 (already measured on the K=2 format arm). Does not reuse that job or vanilla 4B E1.
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
  printf 'Refusing the 1.5B model path; format-variance 4B requires the locked Qwen3-4B directory.\n' >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGEMEM_PYTHON_BIN:-python}"
log_root="$TRINITY_CHECKPOINT_ROOT_DIR/e1_4b_format_var_logs/$AGEMEM_EXPECTED_COMMIT"
preflight_dir="$TRINITY_CHECKPOINT_ROOT_DIR/e1_4b_format_var_preflight/$AGEMEM_EXPECTED_COMMIT"
project_dir="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8"
e1_job="$project_dir/agemem-e1-terminal-only-4b-format-var"
trainer_receipt="$e1_job/receipts/trainer_step_3.json"
eval_receipt="$e1_job/receipts/bench_step_3_model_3.json"

if [[ ! -d "$TRINITY_CHECKPOINT_ROOT_DIR" ]]; then
  printf 'Checkpoint root must already exist on persistent storage.\n' >&2
  exit 2
fi
if [[ -e "$project_dir/agemem-e0-terminal-only-frozen-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-dry-run" || \
      -e "$project_dir/agemem-e1-terminal-only-scale" || \
      -e "$project_dir/agemem-e1-terminal-only-repeat-s7" || \
      -e "$project_dir/agemem-e1-terminal-only-repeat-s17" || \
      -e "$project_dir/agemem-e1-terminal-only-repeat-s27" || \
      -e "$project_dir/agemem-e1-stage3-answer-probe" || \
      -e "$project_dir/agemem-e1-4b-stage3-answer-probe" || \
      -e "$project_dir/agemem-e0-terminal-only-4b-frozen-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-4b-dry-run" || \
      -e "$project_dir/agemem-e0-terminal-only-4b-format-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-4b-format" ]]; then
  printf 'Refusing a checkpoint root that already contains 1.5B, vanilla 4B, probe, or K=2 format jobs.\n' >&2
  exit 2
fi
if [[ -e "$e1_job" && ! -s "$trainer_receipt" ]]; then
  printf 'Refusing to reuse incomplete format-variance 4B job directory: %s\n' "$e1_job" >&2
  exit 2
fi

cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to run format-variance 4B GRPO on a dirty worktree.\n' >&2
  exit 2
fi

"$python_bin" - <<'PY'
import json
import os
from pathlib import Path

lock = json.loads(Path("configs/e1_4b_format_var.json").read_text(encoding="utf-8"))
expected = lock["model"]["expected_revision"]
actual = os.environ["TRINITY_MODEL_REVISION"]
if actual != expected:
    raise SystemExit(
        f"TRINITY_MODEL_REVISION {actual} does not match locked 4B revision {expected}"
    )
if lock["model"]["repository_id"] != "Qwen/Qwen3-4B":
    raise SystemExit("format-variance 4B lock repository_id drifted")
if lock["experiment_id"] != "e1_format_conditioned_4b_group_variance":
    raise SystemExit("format-variance 4B experiment_id drifted")
if not lock.get("stage3_require_final_answer") or not lock.get("stage3_repair_untagged_answer"):
    raise SystemExit("format-variance 4B GRPO must enable Stage-3 answer nudges.")
assertions = {entry["path"]: entry["equals"] for entry in lock["config_assertions"]}
if assertions.get("algorithm.repeat_times") != 4:
    raise SystemExit("format-variance 4B must use K=4")
if assertions.get("trainer.total_steps") != 3:
    raise SystemExit("format-variance 4B must train 3 steps")
PY

"$python_bin" -c 'import flash_attn; v=flash_attn.__version__; assert v=="2.8.1", v'
"$python_bin" -m unittest tests.common.e1_4b_format_var_contract_test

mkdir -p "$preflight_dir" "$log_root"
chmod 700 "$preflight_dir" "$log_root"

"$python_bin" scripts/agemem_m8b_preflight.py \
  --mode autodl \
  --lock configs/e1_4b_format_var.json \
  --config examples/agemem_hotpotqa/agemem_e1_4b_format_var.yaml \
  --expected-commit "$AGEMEM_EXPECTED_COMMIT" \
  --model-path "$TRINITY_MODEL_PATH" \
  --model-revision "$TRINITY_MODEL_REVISION" \
  --dataset-path "$HOTPOTQA_PATH" \
  --checkpoint-root "$TRINITY_CHECKPOINT_ROOT_DIR" \
  --output "$preflight_dir/preflight_report.json"

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before format-variance 4B GRPO.\n' >&2
  exit 2
fi

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

if [[ ! -s "$trainer_receipt" || ! -d "$e1_job/global_step_3" ]]; then
  start_ray "$log_root/ray_e1_start.log"
  trinity run --config examples/agemem_hotpotqa/agemem_e1_4b_format_var.yaml \
    2>&1 | tee "$log_root/e1_format_var_update.log"
  if [[ ! -s "$trainer_receipt" ]]; then
    printf 'Format-variance 4B GRPO did not persist trainer_step_3.json\n' >&2
    exit 1
  fi
  if [[ ! -d "$e1_job/global_step_3" ]]; then
    printf 'Format-variance 4B GRPO did not persist global_step_3\n' >&2
    exit 1
  fi
  stop_ray "$log_root/ray_e1_stop.log"
fi

if [[ ! -s "$eval_receipt" ]]; then
  start_ray "$log_root/ray_eval_start.log"
  trinity run --config examples/agemem_hotpotqa/agemem_e1_4b_format_var_eval.yaml \
    2>&1 | tee "$log_root/e1_format_var_checkpoint_eval.log"
  if [[ ! -s "$eval_receipt" ]]; then
    printf 'Format-variance 4B checkpoint eval did not persist bench_step_3_model_3.json\n' >&2
    exit 1
  fi
  stop_ray "$log_root/ray_eval_stop.log"
fi

printf 'Format-variance 4B GRPO finished.\n'
printf 'Logs: %s\n' "$log_root"
printf 'Preflight: %s\n' "$preflight_dir/preflight_report.json"
