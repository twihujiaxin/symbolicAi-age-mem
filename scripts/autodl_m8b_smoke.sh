#!/usr/bin/env bash
set -euo pipefail

# Execute the frozen M8b GPU smoke in fail-closed phase order. This script is
# intentionally separate from autodl_m8b_preflight.sh, which never starts Ray.
required_names=(
  AGEMEM_EXPECTED_COMMIT
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

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGEMEM_PYTHON_BIN:-python}"
e0_job="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8/agemem-e0-terminal-only-frozen-eval"
e1_job="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8/agemem-e1-terminal-only-dry-run"
report_dir="$TRINITY_CHECKPOINT_ROOT_DIR/m8b_postflight/$AGEMEM_EXPECTED_COMMIT"
log_dir="$TRINITY_CHECKPOINT_ROOT_DIR/m8b_logs/$AGEMEM_EXPECTED_COMMIT"

if [[ ! -d "$TRINITY_CHECKPOINT_ROOT_DIR" ]]; then
  printf 'Checkpoint root must already exist on persistent storage.\n' >&2
  exit 2
fi
if [[ -e "$log_dir" || -e "$report_dir" ]]; then
  printf 'Refusing to reuse existing M8b log or postflight evidence.\n' >&2
  exit 2
fi
mkdir -p "$log_dir" "$report_dir"
chmod 700 "$log_dir" "$report_dir"

cd "$repository_root"
bash scripts/autodl_m8b_preflight.sh

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before the isolated M8b smoke.\n' >&2
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

start_ray "$log_dir/ray_e0_start.log"
trinity run \
  --config examples/agemem_hotpotqa/agemem_e0_frozen_eval.yaml \
  2>&1 | tee "$log_dir/e0_frozen_eval.log"
if [[ ! -s "$e0_job/receipts/bench_step_0_model_0.json" ]]; then
  printf 'E0 did not persist its model-version-zero benchmark receipt.\n' >&2
  exit 1
fi

trinity run \
  --config examples/agemem_hotpotqa/agemem_e1_dry_run.yaml \
  2>&1 | tee "$log_dir/e1_single_update.log"
if [[ ! -s "$e1_job/receipts/trainer_step_1.json" ]]; then
  printf 'E1 did not persist its step-one trainer receipt.\n' >&2
  exit 1
fi
if [[ ! -d "$e1_job/global_step_1" ]]; then
  printf 'E1 did not persist global_step_1.\n' >&2
  exit 1
fi

stop_ray "$log_dir/ray_e1_stop.log"
start_ray "$log_dir/ray_checkpoint_eval_start.log"
trinity run \
  --config examples/agemem_hotpotqa/agemem_e1_checkpoint_eval.yaml \
  2>&1 | tee "$log_dir/e1_checkpoint_eval.log"

"$python_bin" scripts/agemem_m8b_postflight.py \
  --checkpoint-root "$TRINITY_CHECKPOINT_ROOT_DIR" \
  --output "$report_dir/postflight_report.json" \
  2>&1 | tee "$log_dir/postflight.log"

printf 'M8b GPU smoke passed every locked preflight and postflight gate.\n'
printf 'Postflight report: %s\n' "$report_dir/postflight_report.json"
printf 'Logs: %s\n' "$log_dir"
