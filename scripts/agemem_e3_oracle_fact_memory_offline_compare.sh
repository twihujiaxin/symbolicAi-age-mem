#!/usr/bin/env bash
set -euo pipefail

# CPU-only Flat-Oracle/DFA replay over the validated fact-memory diagnosis.
# Gold supporting pointers are consumed only by the reward-side grounder.
# This script does not start Ray, vLLM, CUDA, or a trainer.
export CUDA_VISIBLE_DEVICES=""

required_names=(
  AGEMEM_EXPECTED_COMMIT
  AGEMEM_DIAGNOSIS_ROOT
  AGEMEM_HUMAN_AUDIT_PATH
  HOTPOTQA_PATH
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

repository_root="$(git rev-parse --show-toplevel)"
cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing provenance replay on a dirty worktree.\n' >&2
  git status --porcelain >&2
  exit 2
fi

job="agemem-e1-4b-fc-fact-memory-signal-diag"
job_dir="$AGEMEM_DIAGNOSIS_ROOT/Trinity-RFT-AgeMem-M8/$job"
experience_path="$job_dir/buffer/explorer_output.jsonl"
trace_path="$job_dir/trajectories/tool_calls.jsonl"
stage3_turn_path="$job_dir/trajectories/stage3_final_turn.jsonl"
output_dir="${AGEMEM_ORACLE_REPLAY_OUTPUT_DIR:-$AGEMEM_DIAGNOSIS_ROOT/e3_oracle_fact_memory_provenance_replay/$AGEMEM_EXPECTED_COMMIT}"

for path in "$experience_path" "$trace_path" "$stage3_turn_path"; do
  if [[ ! -s "$path" ]]; then
    printf 'Missing or empty fact-memory diagnosis artifact: %s\n' "$path" >&2
    exit 2
  fi
done
if [[ ! -s "$AGEMEM_HUMAN_AUDIT_PATH" ]]; then
  printf 'Missing or empty private final audit: %s\n' "$AGEMEM_HUMAN_AUDIT_PATH" >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  printf 'Refusing to reuse provenance replay output: %s\n' "$output_dir" >&2
  exit 2
fi

python -m unittest \
  tests.common.fact_memory_contract_test \
  tests.common.e3_oracle_dfa_test \
  tests.common.e3_oracle_offline_compare_test \
  tests.common.e3_human_audit_alignment_test
python scripts/agemem_e3_oracle_offline_compare.py \
  --experience-path "$experience_path" \
  --trace-path "$trace_path" \
  --stage3-turn-path "$stage3_turn_path" \
  --hotpotqa-path "$HOTPOTQA_PATH" \
  --lock-path configs/e1_4b_format_conditioned.json \
  --grounding-mode validated_source_pointer_v1 \
  --output-dir "$output_dir"
python scripts/agemem_e3_human_audit_alignment.py \
  --audit-path "$AGEMEM_HUMAN_AUDIT_PATH" \
  --semantic-audit-path "$output_dir/semantic_audit.jsonl" \
  --output-json "$output_dir/human_alignment.json" \
  --output-md "$output_dir/human_alignment.md"

printf 'Fact-memory provenance Oracle comparison finished.\n'
printf 'Report: %s/report.md\n' "$output_dir"
printf 'Human alignment: %s/human_alignment.md\n' "$output_dir"
