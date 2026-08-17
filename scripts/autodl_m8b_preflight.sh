#!/usr/bin/env bash
set -euo pipefail

# This gate never starts Ray or training. It only validates the uploaded run.
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
if [[ ! "$TRINITY_MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'TRINITY_MODEL_REVISION must be a lowercase 40-character revision.\n' >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGEMEM_PYTHON_BIN:-python}"

if [[ ! -d "$TRINITY_CHECKPOINT_ROOT_DIR" ]]; then
  printf 'Checkpoint root must already exist on persistent storage.\n' >&2
  exit 2
fi

report_dir="$TRINITY_CHECKPOINT_ROOT_DIR/m8b_preflight/$AGEMEM_EXPECTED_COMMIT"
mkdir -p "$report_dir"
chmod 700 "$report_dir"

cd "$repository_root"
"$python_bin" scripts/agemem_m8b_preflight.py \
  --mode autodl \
  --expected-commit "$AGEMEM_EXPECTED_COMMIT" \
  --model-path "$TRINITY_MODEL_PATH" \
  --model-revision "$TRINITY_MODEL_REVISION" \
  --dataset-path "$HOTPOTQA_PATH" \
  --checkpoint-root "$TRINITY_CHECKPOINT_ROOT_DIR" \
  --output "$report_dir/preflight_report.json"

"$python_bin" scripts/agemem_m8b_runtime_gate.py \
  --scope all \
  --output "$report_dir/runtime_gate_report.json"

printf 'M8b preflight gates passed. No Ray process or training was started.\n'
printf 'Reports: %s\n' "$report_dir"
