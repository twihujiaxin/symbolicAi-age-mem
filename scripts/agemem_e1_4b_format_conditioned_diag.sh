#!/usr/bin/env bash
set -euo pipefail

# Format-conditioned 4B frozen diagnosis. Bench only; no optimizer.
# Lock: configs/e1_4b_format_conditioned.json
# Canonical jobs: agemem-e1-4b-fc-signal-diag, agemem-e1-4b-fc-heldout-regression,
# agemem-e1-4b-fc-mem-normal, agemem-e1-4b-fc-mem-no-retrieve,
# agemem-e1-4b-fc-mem-gold-support.
# Aliases: signal | heldout | mem-normal | mem-no-retrieve | mem-gold-support
# Memory-necessity jobs require a frozen 32-dev selection.
if [[ "${1:-}" == "" ]]; then
  printf 'Usage: %s <signal|heldout|mem-normal|mem-no-retrieve|mem-gold-support>\n' "$0" >&2
  exit 2
fi
job_alias="$1"
shift || true

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
  printf 'Refusing the 1.5B model path; format-conditioned 4B requires the locked Qwen3-4B directory.\n' >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGEMEM_PYTHON_BIN:-python}"
log_root="$TRINITY_CHECKPOINT_ROOT_DIR/e1_4b_format_conditioned_logs/$AGEMEM_EXPECTED_COMMIT/$job_alias"
project_dir="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8"

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
      -e "$project_dir/agemem-e1-terminal-only-4b-format" || \
      -e "$project_dir/agemem-e0-terminal-only-4b-format-var-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-4b-format-var" || \
      -e "$project_dir/agemem-e0-terminal-only-4b-format-group-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-4b-format-group" ]]; then
  printf 'Refusing a checkpoint root that already contains 1.5B, vanilla 4B, probe, format, format-var, or format-group jobs.\n' >&2
  exit 2
fi

cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to run format-conditioned 4B diagnosis on a dirty worktree.\n' >&2
  printf 'select --write-yaml 会改锁并生成 32-dev YAML，必须先提交这些 freeze 文件。\n' >&2
  git status --porcelain >&2
  exit 2
fi

job_meta="$("$python_bin" - "$job_alias" <<'PY'
import json
import os
import sys
from pathlib import Path

from trinity.common.e1_4b_format_conditioned import (
    HELDOUT_JOB,
    MEM_JOBS,
    SIGNAL_JOB,
    job_requires_frozen_selection,
    load_lock,
    resolve_job_alias,
    selection_is_frozen,
    yaml_path_for_job,
)
from trinity.common.m8b_preflight import _source_digest

alias = sys.argv[1]
try:
    job = resolve_job_alias(alias)
except KeyError as exc:
    raise SystemExit(str(exc)) from exc
lock = load_lock()
expected = lock["model"]["expected_revision"]
actual = os.environ["TRINITY_MODEL_REVISION"]
if actual != expected:
    raise SystemExit(
        f"TRINITY_MODEL_REVISION {actual} does not match locked 4B revision {expected}"
    )
if lock["model"]["repository_id"] != "Qwen/Qwen3-4B":
    raise SystemExit("format-conditioned lock repository_id drifted")
if not lock.get("stage3_require_final_answer") or not lock.get("stage3_repair_untagged_answer"):
    raise SystemExit("format-conditioned diagnosis must enable the answer nudge.")
if lock.get("trainer_total_steps") != 0:
    raise SystemExit("format-conditioned diagnosis must not train.")
if job_requires_frozen_selection(job) and not selection_is_frozen(lock):
    raise SystemExit(
        "memory-necessity jobs require frozen 32-dev selection; "
        "run scripts/agemem_e1_4b_format_conditioned_select.py --write-yaml on the remote."
    )
yaml_path = yaml_path_for_job(job)
if not yaml_path.is_file():
    raise SystemExit(f"missing diagnosis YAML: {yaml_path}")
yaml_text = yaml_path.read_text(encoding="utf-8")
if "consume_put_batch" in yaml_text:
    raise SystemExit("diagnosis YAMLs must not set consume_put_batch")
if job == SIGNAL_JOB:
    digest_key = "signal_config"
elif job == HELDOUT_JOB:
    digest_key = "heldout_config"
elif job == "agemem-e1-4b-fc-mem-normal":
    digest_key = "mem_normal_config"
elif job == "agemem-e1-4b-fc-mem-no-retrieve":
    digest_key = "mem_no_retrieve_config"
else:
    digest_key = "mem_gold_support_config"
expected_digest = (lock.get("source_files") or {}).get(digest_key, {}).get("sha256")
if not expected_digest:
    raise SystemExit(f"lock is missing sha256 for {digest_key}")
if _source_digest(yaml_path) != expected_digest:
    raise SystemExit(f"{yaml_path.name} digest does not match the lock")
if job in MEM_JOBS:
    if "stage3_disable_ltm_retrieve: true" in yaml_text and job != "agemem-e1-4b-fc-mem-no-retrieve":
        raise SystemExit("only mem-no-retrieve may disable Stage-3 LTM retrieve")
    if "stage3_inject_gold_supporting: true" in yaml_text and job != "agemem-e1-4b-fc-mem-gold-support":
        raise SystemExit("only mem-gold-support may inject gold supporting sentences")
print(json.dumps({"job": job, "yaml": str(yaml_path.relative_to(Path(".").resolve()).as_posix())}))
PY
)"
job_name="$(printf '%s' "$job_meta" | "$python_bin" -c 'import json,sys; print(json.load(sys.stdin)["job"])')"
yaml_rel="$(printf '%s' "$job_meta" | "$python_bin" -c 'import json,sys; print(json.load(sys.stdin)["yaml"])')"
job_dir="$project_dir/$job_name"

if [[ -e "$job_dir" ]]; then
  printf 'Refusing to reuse existing format-conditioned diagnosis job directory: %s\n' "$job_dir" >&2
  exit 2
fi
if [[ -e "$log_root" ]]; then
  printf 'Refusing to reuse existing format-conditioned diagnosis logs for this commit and job.\n' >&2
  exit 2
fi

"$python_bin" -c 'import flash_attn; v=flash_attn.__version__; assert v=="2.8.1", v'
"$python_bin" -m unittest tests.common.e1_4b_format_conditioned_contract_test

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before format-conditioned 4B diagnosis.\n' >&2
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

ray_started=1
ray start --head --num-gpus=2 2>&1 | tee "$log_root/ray_start.log"
trinity run --config "$yaml_rel" 2>&1 | tee "$log_root/diag.log"
if [[ ! -s "$job_dir/receipts/bench_step_0_model_0.json" ]]; then
  printf 'format-conditioned diagnosis did not persist bench_step_0_model_0.json\n' >&2
  exit 1
fi
ray stop --force 2>&1 | tee "$log_root/ray_stop.log"
ray_started=0

"$python_bin" scripts/agemem_e1_4b_format_conditioned_diag_report.py \
  --checkpoint-root "$TRINITY_CHECKPOINT_ROOT_DIR" \
  --job "$job_name" \
  2>&1 | tee "$log_root/report.txt"

printf 'Format-conditioned 4B diagnosis finished: %s\n' "$job_name"
printf 'Logs: %s\n' "$log_root"
