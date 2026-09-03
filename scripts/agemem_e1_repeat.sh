#!/usr/bin/env bash
set -euo pipefail

# Sequential E1 terminal-only repeats on the frozen M5 6-row split.
# Does not modify or reuse the M8b dry-run job names.
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
lock_path="$repository_root/configs/e1_repeat.json"
log_root="$TRINITY_CHECKPOINT_ROOT_DIR/e1_repeat_logs/$AGEMEM_EXPECTED_COMMIT"
project_dir="$TRINITY_CHECKPOINT_ROOT_DIR/Trinity-RFT-AgeMem-M8"

if [[ ! -d "$TRINITY_CHECKPOINT_ROOT_DIR" ]]; then
  printf 'Checkpoint root must already exist on persistent storage.\n' >&2
  exit 2
fi
if [[ -e "$project_dir/agemem-e0-terminal-only-frozen-eval" || \
      -e "$project_dir/agemem-e1-terminal-only-dry-run" ]]; then
  printf 'Refusing a checkpoint root that already contains M8b smoke jobs.\n' >&2
  exit 2
fi

cd "$repository_root"
if [[ "$(git rev-parse HEAD)" != "$AGEMEM_EXPECTED_COMMIT" ]]; then
  printf 'HEAD does not match AGEMEM_EXPECTED_COMMIT.\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to run E1 repeats on a dirty worktree.\n' >&2
  exit 2
fi

if ray status >/dev/null 2>&1; then
  printf 'A Ray cluster is already running; stop it before E1 repeats.\n' >&2
  exit 2
fi

"$python_bin" -c 'import flash_attn; v=flash_attn.__version__; assert v=="2.8.1", v'
"$python_bin" -m unittest tests.common.e1_repeat_contract_test

mapfile -t seeds < <(
  "$python_bin" -c 'import json; print("\n".join(str(s) for s in json.load(open("configs/e1_repeat.json"))["seeds"]))'
)
if [[ "${#seeds[@]}" -ne 3 ]]; then
  printf 'E1 repeat lock must list exactly three seeds.\n' >&2
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

for seed in "${seeds[@]}"; do
  export AGEMEM_E1_SEED="$seed"
  AGEMEM_E1_JOB_NAME="$("$python_bin" -c 'import json,os,sys; lock=json.load(open(sys.argv[1])); print(lock["job_name_template"].format(seed=int(os.environ["AGEMEM_E1_SEED"])))' "$lock_path")"
  export AGEMEM_E1_JOB_NAME
  if [[ "$AGEMEM_E1_JOB_NAME" == "agemem-e1-terminal-only-dry-run" ]]; then
    printf 'Repeat job name collided with the frozen M8b smoke job.\n' >&2
    exit 2
  fi
  job_dir="$project_dir/$AGEMEM_E1_JOB_NAME"
  trainer_receipt="$job_dir/receipts/trainer_step_1.json"
  eval_receipt="$job_dir/receipts/bench_step_1_model_1.json"
  seed_log="$log_root/seed_${seed}"
  mkdir -p "$seed_log"

  if [[ -s "$trainer_receipt" && -d "$job_dir/global_step_1" && -s "$eval_receipt" ]]; then
    printf 'Skipping completed E1 repeat seed %s\n' "$seed"
    continue
  fi

  if [[ ! -s "$trainer_receipt" || ! -d "$job_dir/global_step_1" ]]; then
    if [[ -e "$job_dir" ]]; then
      printf 'Refusing to reuse incomplete job directory: %s\n' "$job_dir" >&2
      exit 2
    fi
    start_ray "$seed_log/ray_start.log"
    trinity run --config examples/agemem_hotpotqa/agemem_e1_repeat.yaml \
      2>&1 | tee "$seed_log/train.log"
    if [[ ! -s "$trainer_receipt" ]]; then
      printf 'E1 repeat seed %s did not persist trainer_step_1.json\n' "$seed" >&2
      exit 1
    fi
    if [[ ! -d "$job_dir/global_step_1" ]]; then
      printf 'E1 repeat seed %s did not persist global_step_1\n' "$seed" >&2
      exit 1
    fi
    stop_ray "$seed_log/ray_train_stop.log"
  fi

  if [[ -s "$eval_receipt" ]]; then
    printf 'Skipping completed eval for E1 repeat seed %s\n' "$seed"
    continue
  fi
  start_ray "$seed_log/ray_eval_start.log"
  trinity run --config examples/agemem_hotpotqa/agemem_e1_repeat_eval.yaml \
    2>&1 | tee "$seed_log/eval.log"
  if [[ ! -s "$eval_receipt" ]]; then
    printf 'E1 repeat seed %s did not persist bench_step_1_model_1.json\n' "$seed" >&2
    exit 1
  fi
  stop_ray "$seed_log/ray_eval_stop.log"
done

printf 'E1 terminal-only repeats finished for seeds %s\n' "${seeds[*]}"
printf 'Logs: %s\n' "$log_root"
