#!/usr/bin/env bash
set -euo pipefail

# Format-conditioned 4B E3: terminal F1 + Oracle AP + hand-authored DFA.
# Seed 7, 24 train, K=4, 12 steps, consume_put_batch, question-retrieve environment.
# Eval shadows DFA on frozen 32-dev at steps 0/12. Independent checkpoint root.
# Lock: configs/e3_4b_fc.json
# Jobs: agemem-e0-4b-fc-e3-eval, agemem-e3-4b-fc, agemem-e3-4b-fc-eval-s12
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
  printf 'Refusing the 1.5B model path; E3 requires the locked Qwen3-4B directory.\n' >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGEMEM_PYTHON_BIN:-python}"
log_root="$TRINITY_CHECKPOINT_ROOT_DIR/e3_4b_fc_logs/$AGEMEM_EXPECTED_COMMIT"
project_dir="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8"
e0_job="$project_dir/agemem-e0-4b-fc-e3-eval"
train_job="$project_dir/agemem-e3-4b-fc"
e0_receipt="$e0_job/receipts/bench_step_0_model_0.json"
trainer_receipt="$train_job/receipts/trainer_step_12.json"

if [[ ! -d "$TRINITY_CHECKPOINT_ROOT_DIR" ]]; then
  printf 'Checkpoint root must already exist on persistent storage.\n' >&2
  exit 2
fi
if [[ "$TRINITY_CHECKPOINT_ROOT_DIR" == *checkpoints-e1-4b-format-conditioned ]]; then
  printf 'Refusing the diagnosis checkpoint root; E3 needs /data/hjx/Age_mem/checkpoints-e3-4b-fc.\n' >&2
  exit 2
fi
if [[ "$TRINITY_CHECKPOINT_ROOT_DIR" == *checkpoints-e1-4b-fc-pilot ]]; then
  printf 'Refusing the 36-step pilot checkpoint root.\n' >&2
  exit 2
fi
if [[ "$TRINITY_CHECKPOINT_ROOT_DIR" == *checkpoints-e1-4b-fc-question-retrieve ]]; then
  printf 'Refusing the question-retrieve checkpoint root.\n' >&2
  exit 2
fi
if [[ "$TRINITY_CHECKPOINT_ROOT_DIR" == *checkpoints-e3-4b-fc-no-qr ]]; then
  printf 'Refusing the no-QR E3 checkpoint root.\n' >&2
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
      -e "$project_dir/agemem-e0-4b-fc-pilot-eval" || \
      -e "$project_dir/agemem-e1-4b-fc-pilot" || \
      -e "$project_dir/agemem-e1-4b-fc-pilot-eval-s12" || \
      -e "$project_dir/agemem-e1-4b-fc-pilot-eval-s24" || \
      -e "$project_dir/agemem-e1-4b-fc-pilot-eval-s36" || \
      -e "$project_dir/agemem-e1-4b-fc-question-retrieve" || \
      -e "$project_dir/agemem-e0-4b-fc-e3-no-qr-eval" || \
      -e "$project_dir/agemem-e3-4b-fc-no-qr" || \
      -e "$project_dir/agemem-e3-4b-fc-no-qr-eval-s12" ]]; then
  printf 'Refusing a checkpoint root that already contains closed 1.5B/4B, diagnosis, pilot, question-retrieve, or no-QR E3 jobs.\n' >&2
  exit 2
fi
if [[ -e "$train_job" && ! -s "$trainer_receipt" ]]; then
  printf 'Refusing to reuse incomplete E3 job directory: %s\n' "$train_job" >&2
  exit 2
fi

cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to run E3 on a dirty worktree.\n' >&2
  exit 2
fi

eval_dir="$("$python_bin" - <<'PY'
import json
import os
from pathlib import Path

from trinity.common.e3_4b_fc import (
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
    raise SystemExit("E3 repository_id drifted")
if lock["experiment_id"] != "e3_format_conditioned_4b_oracle_dfa":
    raise SystemExit("E3 experiment_id drifted")
if lock["trainer_total_steps"] != TRAINER_TOTAL_STEPS:
    raise SystemExit("E3 must train 12 steps")
if lock.get("consume_put_batch") is not True:
    raise SystemExit("E3 must consume one explorer put_batch per trainer step")
if not lock.get("stage3_require_final_answer") or not lock.get("stage3_repair_untagged_answer"):
    raise SystemExit("E3 must enable Stage-3 answer nudges.")
if not lock.get("stage3_question_retrieve") or not lock.get("stage3_index_observed_context"):
    raise SystemExit("E3 must index observed context and retrieve by question")
if lock.get("stage3_inject_gold_supporting"):
    raise SystemExit("E3 must not inject gold supporting facts")
if lock.get("reward_profile") != "terminal_dfa":
    raise SystemExit("E3 reward_profile must be terminal_dfa")
if _source_digest(TRAIN_YAML) != lock["source_files"]["train_config"]["sha256"]:
    raise SystemExit("E3 train YAML digest does not match the lock")
train_text = TRAIN_YAML.read_text(encoding="utf-8")
if f'name: "{TRAIN_JOB}"' not in train_text:
    raise SystemExit("E3 train YAML job name drifted")
if "reward_profile: terminal_dfa" not in train_text:
    raise SystemExit("E3 train YAML must use terminal_dfa")
if "e3_dfa_shadow: false" not in train_text:
    raise SystemExit("E3 train YAML must not shadow DFA")
if "stage3_inject_gold_supporting: true" in train_text:
    raise SystemExit("E3 must not inject gold supporting facts")
if not selection_is_frozen(fc_lock):
    raise SystemExit(
        "E3 requires frozen 32-dev selection; "
        "the format-conditioned lock on this machine must already be frozen."
    )
eval_root = Path(os.environ["TRINITY_CHECKPOINT_ROOT_DIR"]) / "e3_4b_fc_eval_yaml" / os.environ["AGEMEM_EXPECTED_COMMIT"]
write_runtime_eval_yamls(eval_root, fc_lock)
print(str(eval_root))
PY
)"

"$python_bin" -c 'import flash_attn; v=flash_attn.__version__; assert v=="2.8.1", v'
"$python_bin" -m unittest tests.common.e3_4b_fc_contract_test tests.common.e3_oracle_dfa_test

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before E3.\n' >&2
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
  trinity run --config "$eval_dir/agemem_e0_4b_fc_e3_eval.yaml" \
    2>&1 | tee "$log_root/e0_eval.log"
  if [[ ! -s "$e0_receipt" ]]; then
    printf 'E3 E0 did not persist bench_step_0_model_0.json\n' >&2
    exit 1
  fi
  stop_ray "$log_root/ray_e0_stop.log"
fi

if [[ ! -s "$trainer_receipt" || ! -d "$train_job/global_step_12" ]]; then
  start_ray "$log_root/ray_train_start.log"
  trinity run --config examples/agemem_hotpotqa/agemem_e3_4b_fc.yaml \
    2>&1 | tee "$log_root/e3_update.log"
  if [[ ! -s "$trainer_receipt" ]]; then
    printf 'E3 did not persist trainer_step_12.json\n' >&2
    exit 1
  fi
  if [[ ! -d "$train_job/global_step_12" ]]; then
    printf 'E3 did not persist global_step_12\n' >&2
    exit 1
  fi
  stop_ray "$log_root/ray_train_stop.log"
fi

eval_job="$project_dir/agemem-e3-4b-fc-eval-s12"
eval_receipt="$eval_job/receipts/bench_step_0_model_0.json"
if [[ ! -s "$eval_receipt" ]]; then
  start_ray "$log_root/ray_eval_12_start.log"
  trinity run --config "$eval_dir/agemem_e3_4b_fc_eval_s12.yaml" \
    2>&1 | tee "$log_root/e3_eval_s12.log"
  if [[ ! -s "$eval_receipt" ]]; then
    printf 'E3 checkpoint eval did not persist bench_step_0_model_0.json for step 12\n' >&2
    exit 1
  fi
  stop_ray "$log_root/ray_eval_12_stop.log"
fi

printf 'Format-conditioned 4B E3 Oracle DFA finished.\n'
printf 'Logs: %s\n' "$log_root"
