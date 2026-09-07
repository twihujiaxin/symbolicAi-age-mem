#!/usr/bin/env bash
set -euo pipefail

# Format-conditioned 4B 36-step GRPO pilot. Seed 7, 24 train, K=4, consume_put_batch.
# Lock: configs/e1_4b_fc_pilot.json
# Eval frozen 32-dev at steps 0/12/24/36. Independent of the diagnosis checkpoint root.
# Canonical eval jobs: agemem-e1-4b-fc-pilot-eval-s12, agemem-e1-4b-fc-pilot-eval-s24,
# agemem-e1-4b-fc-pilot-eval-s36. E0: agemem-e0-4b-fc-pilot-eval. Train: agemem-e1-4b-fc-pilot.
# Requires trainer_step_36.json plus global_step_12, global_step_24, and global_step_36.
# Train YAML uses gpu_memory_utilization 0.5 and prefix caching off to avoid FSDP OOM at later steps.
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
  printf 'Refusing the 1.5B model path; 36-step pilot requires the locked Qwen3-4B directory.\n' >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGEMEM_PYTHON_BIN:-python}"
log_root="$TRINITY_CHECKPOINT_ROOT_DIR/e1_4b_fc_pilot_logs/$AGEMEM_EXPECTED_COMMIT"
project_dir="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8"
e0_job="$project_dir/agemem-e0-4b-fc-pilot-eval"
train_job="$project_dir/agemem-e1-4b-fc-pilot"
e0_receipt="$e0_job/receipts/bench_step_0_model_0.json"
trainer_receipt="$train_job/receipts/trainer_step_36.json"

if [[ ! -d "$TRINITY_CHECKPOINT_ROOT_DIR" ]]; then
  printf 'Checkpoint root must already exist on persistent storage.\n' >&2
  exit 2
fi
if [[ "$TRINITY_CHECKPOINT_ROOT_DIR" == *checkpoints-e1-4b-format-conditioned ]]; then
  printf 'Refusing the diagnosis checkpoint root; 36-step pilot needs /data/hjx/Age_mem/checkpoints-e1-4b-fc-pilot.\n' >&2
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
      -e "$project_dir/agemem-e1-terminal-only-4b-format" || \
      -e "$project_dir/agemem-e0-terminal-only-4b-format-var-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-4b-format-var" || \
      -e "$project_dir/agemem-e0-terminal-only-4b-format-group-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-4b-format-group" || \
      -e "$project_dir/agemem-e1-4b-fc-signal-diag" || \
      -e "$project_dir/agemem-e1-4b-fc-heldout-regression" || \
      -e "$project_dir/agemem-e1-4b-fc-mem-normal" || \
      -e "$project_dir/agemem-e1-4b-fc-mem-no-retrieve" || \
      -e "$project_dir/agemem-e1-4b-fc-mem-gold-support" || \
      -e "$project_dir/agemem-e1-4b-fc-question-retrieve" ]]; then
  printf 'Refusing a checkpoint root that already contains 1.5B, vanilla 4B, probe, format-group, or format-conditioned diagnosis jobs.\n' >&2
  exit 2
fi
if [[ -e "$train_job" && ! -s "$trainer_receipt" ]]; then
  printf 'Refusing to reuse incomplete 36-step pilot job directory: %s\n' "$train_job" >&2
  exit 2
fi

cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to run the 36-step pilot on a dirty worktree.\n' >&2
  exit 2
fi

eval_dir="$("$python_bin" - <<'PY'
import json
import os
from pathlib import Path

from trinity.common.e1_4b_fc_pilot import (
    EXPECTED_REVISION,
    LOCK_PATH,
    TRAIN_JOB,
    TRAIN_YAML,
    TRAINER_TOTAL_STEPS,
    load_lock,
    write_runtime_eval_yamls,
)
from trinity.common.e1_4b_format_conditioned import load_lock as load_fc_lock
from trinity.common.e1_4b_format_conditioned import selection_is_frozen
from trinity.common.m8b_preflight import _source_digest

lock = load_lock()
fc_lock = load_fc_lock()
if os.environ["TRINITY_MODEL_REVISION"] != EXPECTED_REVISION:
    raise SystemExit("TRINITY_MODEL_REVISION does not match locked 4B revision")
if lock["model"]["repository_id"] != "Qwen/Qwen3-4B":
    raise SystemExit("36-step pilot repository_id drifted")
if lock["experiment_id"] != "e1_format_conditioned_4b_36step_pilot":
    raise SystemExit("36-step pilot experiment_id drifted")
if lock["trainer_total_steps"] != TRAINER_TOTAL_STEPS:
    raise SystemExit("36-step pilot must train 36 steps")
if lock.get("consume_put_batch") is not True:
    raise SystemExit("36-step pilot must consume one explorer put_batch per trainer step")
if not lock.get("stage3_require_final_answer") or not lock.get("stage3_repair_untagged_answer"):
    raise SystemExit("36-step pilot must enable Stage-3 answer nudges.")
if _source_digest(TRAIN_YAML) != lock["source_files"]["train_config"]["sha256"]:
    raise SystemExit("36-step pilot train YAML digest does not match the lock")
train_text = TRAIN_YAML.read_text(encoding="utf-8")
if f'name: "{TRAIN_JOB}"' not in train_text:
    raise SystemExit("36-step pilot train YAML job name drifted")
if "consume_put_batch: true" not in train_text:
    raise SystemExit("36-step pilot train YAML must set consume_put_batch")
if "stage3_disable_ltm_retrieve" in train_text or "stage3_inject_gold_supporting" in train_text:
    raise SystemExit("36-step pilot must not enable diagnosis-only memory flags")
if not selection_is_frozen(fc_lock):
    raise SystemExit(
        "36-step pilot eval requires frozen 32-dev selection; "
        "the format-conditioned lock on this machine must already be frozen."
    )
eval_root = Path(os.environ["TRINITY_CHECKPOINT_ROOT_DIR"]) / "e1_4b_fc_pilot_eval_yaml" / os.environ["AGEMEM_EXPECTED_COMMIT"]
write_runtime_eval_yamls(eval_root, fc_lock)
print(str(eval_root))
PY
)"

"$python_bin" -c 'import flash_attn; v=flash_attn.__version__; assert v=="2.8.1", v'
"$python_bin" -m unittest tests.common.e1_4b_fc_pilot_contract_test

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before the 36-step pilot.\n' >&2
  exit 2
fi

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

if [[ ! -s "$e0_receipt" ]]; then
  start_ray "$log_root/ray_e0_start.log"
  trinity run --config "$eval_dir/agemem_e0_4b_fc_pilot_eval.yaml" \
    2>&1 | tee "$log_root/e0_eval.log"
  if [[ ! -s "$e0_receipt" ]]; then
    printf '36-step pilot E0 did not persist bench_step_0_model_0.json\n' >&2
    exit 1
  fi
  stop_ray "$log_root/ray_e0_stop.log"
fi

if [[ ! -s "$trainer_receipt" || ! -d "$train_job/global_step_36" ]]; then
  start_ray "$log_root/ray_train_start.log"
  trinity run --config examples/agemem_hotpotqa/agemem_e1_4b_fc_pilot.yaml \
    2>&1 | tee "$log_root/e1_pilot_update.log"
  if [[ ! -s "$trainer_receipt" ]]; then
    printf '36-step pilot did not persist trainer_step_36.json\n' >&2
    exit 1
  fi
  for step in 12 24 36; do
    if [[ ! -d "$train_job/global_step_$step" ]]; then
      printf '36-step pilot did not persist global_step_%s\n' "$step" >&2
      exit 1
    fi
  done
  stop_ray "$log_root/ray_train_stop.log"
fi

for step in 12 24 36; do
  eval_job="$project_dir/agemem-e1-4b-fc-pilot-eval-s$step"
  eval_receipt="$eval_job/receipts/bench_step_0_model_0.json"
  if [[ -s "$eval_receipt" ]]; then
    continue
  fi
  start_ray "$log_root/ray_eval_${step}_start.log"
  trinity run --config "$eval_dir/agemem_e1_4b_fc_pilot_eval_s${step}.yaml" \
    2>&1 | tee "$log_root/e1_pilot_eval_s${step}.log"
  if [[ ! -s "$eval_receipt" ]]; then
    printf '36-step pilot checkpoint eval did not persist bench_step_0_model_0.json for step %s\n' "$step" >&2
    exit 1
  fi
  stop_ray "$log_root/ray_eval_${step}_stop.log"
done

printf 'Format-conditioned 4B 36-step pilot finished.\n'
printf 'Logs: %s\n' "$log_root"
