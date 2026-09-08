#!/usr/bin/env bash
set -euo pipefail

# Format-conditioned 4B question-retrieve 32-dev bench. Seed 7, K=1, T=0, nudge.
# Indexes observed Stage-1 sentences into LTM, then hybrid-retrieves by the question.
# Does not inject gold supporting facts. Independent checkpoint root.
# Lock: configs/e1_4b_fc_question_retrieve.json
# Job: agemem-e1-4b-fc-question-retrieve
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
  printf 'Refusing the 1.5B model path; question-retrieve requires the locked Qwen3-4B directory.\n' >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGEMEM_PYTHON_BIN:-python}"
log_root="$TRINITY_CHECKPOINT_ROOT_DIR/e1_4b_fc_question_retrieve_logs/$AGEMEM_EXPECTED_COMMIT"
project_dir="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8"
job_dir="$project_dir/agemem-e1-4b-fc-question-retrieve"
bench_receipt="$job_dir/receipts/bench_step_0_model_0.json"

if [[ ! -d "$TRINITY_CHECKPOINT_ROOT_DIR" ]]; then
  printf 'Checkpoint root must already exist on persistent storage.\n' >&2
  exit 2
fi
if [[ "$TRINITY_CHECKPOINT_ROOT_DIR" == *checkpoints-e1-4b-format-conditioned ]]; then
  printf 'Refusing the diagnosis checkpoint root; question-retrieve needs /data/hjx/Age_mem/checkpoints-e1-4b-fc-question-retrieve.\n' >&2
  exit 2
fi
if [[ "$TRINITY_CHECKPOINT_ROOT_DIR" == *checkpoints-e1-4b-fc-pilot ]]; then
  printf 'Refusing the 36-step pilot checkpoint root.\n' >&2
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
      -e "$project_dir/agemem-e0-4b-fc-e3-eval" || \
      -e "$project_dir/agemem-e3-4b-fc" || \
      -e "$project_dir/agemem-e3-4b-fc-eval-s12" || \
      -e "$project_dir/agemem-e0-4b-fc-e3-no-qr-eval" || \
      -e "$project_dir/agemem-e3-4b-fc-no-qr" || \
      -e "$project_dir/agemem-e3-4b-fc-no-qr-eval-s12" ]]; then
  printf 'Refusing a checkpoint root that already contains closed 1.5B/4B, diagnosis, or pilot jobs.\n' >&2
  exit 2
fi
if [[ -e "$job_dir" && ! -s "$bench_receipt" ]]; then
  printf 'Refusing to reuse incomplete question-retrieve job directory: %s\n' "$job_dir" >&2
  exit 2
fi

cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to run question-retrieve on a dirty worktree.\n' >&2
  exit 2
fi

yaml_path="$("$python_bin" - <<'PY'
import os
from pathlib import Path

from trinity.common.e1_4b_fc_question_retrieve import (
    CHECKPOINT_ROOT,
    EXPECTED_REVISION,
    JOB,
    LOCK_PATH,
    load_lock,
    write_runtime_yaml,
)
from trinity.common.e1_4b_format_conditioned import (
    EXPECTED_REVISION as FC_REVISION,
)
from trinity.common.e1_4b_format_conditioned import load_lock as load_fc_lock
from trinity.common.e1_4b_format_conditioned import selection_is_frozen

lock = load_lock()
fc_lock = load_fc_lock()
if os.environ["TRINITY_MODEL_REVISION"] != EXPECTED_REVISION:
    raise SystemExit("TRINITY_MODEL_REVISION does not match locked 4B revision")
if EXPECTED_REVISION != FC_REVISION:
    raise SystemExit("question-retrieve revision drifted from format-conditioned lock")
if lock["model"]["repository_id"] != "Qwen/Qwen3-4B":
    raise SystemExit("question-retrieve repository_id drifted")
if lock["experiment_id"] != "e1_format_conditioned_4b_question_retrieve":
    raise SystemExit("question-retrieve experiment_id drifted")
if lock.get("stage3_inject_gold_supporting"):
    raise SystemExit("question-retrieve must not inject gold supporting facts")
if not lock.get("stage3_question_retrieve") or not lock.get("stage3_index_observed_context"):
    raise SystemExit("question-retrieve must index observed context and retrieve by question")
if os.environ["TRINITY_CHECKPOINT_ROOT_DIR"] == "/data/hjx/Age_mem/checkpoints-e1-4b-format-conditioned":
    raise SystemExit("question-retrieve must not reuse the diagnosis checkpoint root")
if not selection_is_frozen(fc_lock):
    raise SystemExit(
        "question-retrieve requires frozen 32-dev selection; "
        "the format-conditioned lock on this machine must already be frozen."
    )
eval_root = Path(os.environ["TRINITY_CHECKPOINT_ROOT_DIR"]) / "e1_4b_fc_question_retrieve_yaml" / os.environ["AGEMEM_EXPECTED_COMMIT"]
print(str(write_runtime_yaml(eval_root, fc_lock)))
PY
)"

"$python_bin" -c 'import flash_attn; v=flash_attn.__version__; assert v=="2.8.1", v'
"$python_bin" -m unittest tests.common.e1_4b_fc_question_retrieve_contract_test

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before question-retrieve.\n' >&2
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

if [[ -s "$bench_receipt" ]]; then
  printf 'Question-retrieve bench already has bench_step_0_model_0.json; skipping.\n'
  printf 'Logs: %s\n' "$log_root"
  exit 0
fi

ray_started=1
ray start --head --num-gpus=2 2>&1 | tee "$log_root/ray_start.log"
trinity run --config "$yaml_path" 2>&1 | tee "$log_root/question_retrieve.log"
if [[ ! -s "$bench_receipt" ]]; then
  printf 'question-retrieve did not persist bench_step_0_model_0.json\n' >&2
  exit 1
fi
ray stop --force 2>&1 | tee "$log_root/ray_stop.log"
ray_started=0

printf 'Format-conditioned 4B question-retrieve finished.\n'
printf 'Logs: %s\n' "$log_root"
