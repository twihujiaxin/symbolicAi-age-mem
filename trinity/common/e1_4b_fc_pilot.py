"""Helpers for the format-conditioned 4B 36-step GRPO pilot.

These helpers are not imported by the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trinity.common.e1_4b_format_conditioned import (
    ALL_JOBS as DIAGNOSIS_JOBS,
)
from trinity.common.e1_4b_format_conditioned import (
    LOCK_PATH as FC_LOCK_PATH,
)
from trinity.common.e1_4b_format_conditioned import (
    _WORKFLOW_ARG_LINES,
    _unique_ids,
    _yaml_id_block,
    _yaml_index_block,
    load_lock as load_fc_lock,
    render_bench_yaml,
    selection_is_frozen,
    train_rows_match_scale,
)
from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_4b_fc_pilot.json"
EXAMPLES_DIR = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa"
TRAIN_YAML = EXAMPLES_DIR / "agemem_e1_4b_fc_pilot.yaml"

EXPECTED_REPOSITORY = "Qwen/Qwen3-4B"
EXPECTED_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SCHEMA_VERSION = "agemem.e1_4b_fc_pilot.lock.v1"
EXPERIMENT_ID = "e1_format_conditioned_4b_36step_pilot"
CHECKPOINT_ROOT = "/data/hjx/Age_mem/checkpoints-e1-4b-fc-pilot"

E0_JOB = "agemem-e0-4b-fc-pilot-eval"
TRAIN_JOB = "agemem-e1-4b-fc-pilot"
EVAL_JOB_BY_STEP = {
    12: "agemem-e1-4b-fc-pilot-eval-s12",
    24: "agemem-e1-4b-fc-pilot-eval-s24",
    36: "agemem-e1-4b-fc-pilot-eval-s36",
}
ALL_JOBS = (E0_JOB, TRAIN_JOB, *EVAL_JOB_BY_STEP.values())
EVAL_STEPS = (0, 12, 24, 36)
TRAINER_TOTAL_STEPS = 36
SAVE_INTERVAL = 12
REPEAT_TIMES = 4
BATCH_SIZE = 2
SEED = 7
LATER_SEEDS = (17, 27)

FORBIDDEN_FOREIGN_JOBS = (
    "agemem-e0-terminal-only-frozen-eval",
    "agemem-e1-terminal-only-dry-run",
    "agemem-e1-terminal-only-scale",
    "agemem-e1-terminal-only-repeat-s7",
    "agemem-e1-terminal-only-repeat-s17",
    "agemem-e1-terminal-only-repeat-s27",
    "agemem-e1-stage3-answer-probe",
    "agemem-e1-4b-stage3-answer-probe",
    "agemem-e0-terminal-only-4b-frozen-eval",
    "agemem-e1-terminal-only-4b-dry-run",
    "agemem-e0-terminal-only-4b-format-eval",
    "agemem-e1-terminal-only-4b-format",
    "agemem-e0-terminal-only-4b-format-var-eval",
    "agemem-e1-terminal-only-4b-format-var",
    "agemem-e0-terminal-only-4b-format-group-eval",
    "agemem-e1-terminal-only-4b-format-group",
    *DIAGNOSIS_JOBS,
    "agemem-e1-4b-fc-question-retrieve",
    "agemem-e0-4b-fc-e3-eval",
    "agemem-e3-4b-fc",
    "agemem-e3-4b-fc-eval-s12",
    "agemem-e0-4b-fc-e3-no-qr-eval",
    "agemem-e3-4b-fc-no-qr",
    "agemem-e3-4b-fc-no-qr-eval-s12",
)


def load_lock(path: Path | None = None) -> dict[str, Any]:
    target = path or LOCK_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def _workflow_args() -> str:
    return "\n".join(_WORKFLOW_ARG_LINES)


def render_train_yaml(fc_lock: Mapping[str, Any]) -> str:
    rows = list(fc_lock["fixed_train_rows"])
    if len(rows) != 24:
        raise ValueError("36-step pilot train YAML requires the frozen 24 train rows")
    if not train_rows_match_scale(fc_lock):
        raise ValueError("36-step pilot train rows must copy e1_scale.fixed_train_rows")
    row_ids = _unique_ids(rows)
    fingerprint = str(fc_lock["expected_dataset_fingerprint"])
    workflow = _workflow_args()
    return f"""# Format-conditioned 4B 36-step GRPO pilot. Same 24 train rows as the
# diagnosis lock, K=4, 3 epochs, consume_put_batch, Stage-3 nudge.
# Seed 7 only; 17/27 wait. Do not reuse the diagnosis checkpoint root.
project: "Trinity-RFT-AgeMem-M8"
name: "{TRAIN_JOB}"
mode: both
checkpoint_root_dir: ${{oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}}
continue_from_checkpoint: false

algorithm:
  algorithm_type: multi_step_grpo
  advantage_fn: step_wise_grpo
  repeat_times: {REPEAT_TIMES}

model:
  model_path: ${{oc.env:TRINITY_MODEL_PATH,/data/hjx/Age_mem/models/Qwen3-4B}}
  max_model_len: 5120
  max_prompt_tokens: 4096
  max_response_tokens: 1024
  lora_configs:
  - name: lora
    lora_rank: 16
    lora_alpha: 16
    path: null

cluster:
  node_num: 1
  gpu_per_node: 2

buffer:
  total_epochs: 3
  total_steps: {TRAINER_TOTAL_STEPS}
  batch_size: {BATCH_SIZE}
  # veRL world-size / PPO mini-batch dummy. Trainer reads are one explorer
  # put_batch (consume_put_batch), not 8 flattened steps.
  train_batch_size: 8
  explorer_input:
    taskset:
      name: hotpotqa_fc_train_24
      storage_type: file
      path: ${{oc.env:HOTPOTQA_PATH,/root/autodl-tmp/data/hotpot_qa/fullwiki}}
      split: train
      row_indices: {_yaml_index_block(rows)}
      row_id_key: id
      expected_row_ids:
{_yaml_id_block(row_ids)}
      expected_dataset_fingerprint: {fingerprint}
      format:
        prompt_key: question
        response_key: answer
      rollout_args:
        temperature: 0.6
        max_tokens: 1024
      workflow_args:
{workflow}
    eval_tasksets: []
    default_workflow_type: AgeMem_hotpot_workflow_training
  trainer_input:
    experience_buffer:
      name: agemem_e1_4b_fc_pilot_buffer
      storage_type: queue
      path: null
      consume_put_batch: true

explorer:
  eval_on_startup: false
  eval_interval: {SAVE_INTERVAL}
  runner_per_model: 2
  max_repeat_times_per_runner: {REPEAT_TIMES}
  max_timeout: 7200
  rollout_model:
    engine_num: 1
    tensor_parallel_size: 1
    enable_prefix_caching: false
    enforce_eager: true
    enable_history: true
    dtype: bfloat16
    seed: {SEED}
    gpu_memory_utilization: 0.5
    enable_chunked_prefill: true
    enable_thinking: false

log:
  level: INFO

synchronizer:
  sync_method: checkpoint
  sync_interval: 1
  sync_timeout: 7200

trainer:
  trainer_type: verl
  total_steps: {TRAINER_TOTAL_STEPS}
  save_interval: {SAVE_INTERVAL}
  trainer_config:
    actor_rollout_ref:
      model:
        use_remove_padding: true
        enable_gradient_checkpointing: true
      actor:
        use_dynamic_bsz: true
        ppo_max_token_len_per_gpu: 2304
        ulysses_sequence_parallel_size: 1
        optim:
          lr: 0.000001
      ref:
        log_prob_use_dynamic_bsz: ${{trainer.trainer_config.actor_rollout_ref.actor.use_dynamic_bsz}}
        log_prob_max_token_len_per_gpu: ${{trainer.trainer_config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu}}
        ulysses_sequence_parallel_size: 1
"""


def render_e0_yaml(fc_lock: Mapping[str, Any]) -> str:
    if not selection_is_frozen(fc_lock):
        raise ValueError("pilot E0 eval requires frozen 32-dev selection")
    return render_bench_yaml(
        fc_lock,
        job=E0_JOB,
        rows=list(fc_lock["fixed_dev_rows"]),
        split="validation",
        fingerprint=str(fc_lock["eval_dataset_fingerprint"]),
        temperature=0.0,
        repeat_times=1,
        comment=(
            "# Format-conditioned 4B 36-step pilot E0. Frozen 32-dev, K=1, T=0, "
            "Stage-3 nudge, no optimizer. Independent of the diagnosis checkpoint root."
        ),
        taskset_name="hotpotqa_fc_dev_32",
    )


def render_checkpoint_eval_yaml(fc_lock: Mapping[str, Any], step: int) -> str:
    if step not in EVAL_JOB_BY_STEP:
        raise ValueError(f"unsupported pilot eval step: {step}")
    if not selection_is_frozen(fc_lock):
        raise ValueError("pilot checkpoint eval requires frozen 32-dev selection")
    job = EVAL_JOB_BY_STEP[step]
    rows = list(fc_lock["fixed_dev_rows"])
    row_ids = _unique_ids(rows)
    fingerprint = str(fc_lock["eval_dataset_fingerprint"])
    workflow = _workflow_args()
    lora_path = (
        "${oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}/Trinity-RFT-AgeMem-M8/"
        f"{TRAIN_JOB}/global_step_{step}/actor/lora_adapter"
    )
    return f"""# Format-conditioned 4B 36-step pilot checkpoint eval at global_step_{step}.
# Frozen 32-dev, K=1, T=0, Stage-3 nudge. Loads LoRA from the train job.
project: "Trinity-RFT-AgeMem-M8"
name: "{job}"
mode: bench
checkpoint_root_dir: ${{oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}}
continue_from_checkpoint: false

algorithm:
  algorithm_type: multi_step_grpo
  advantage_fn: step_wise_grpo
  repeat_times: 1

model:
  model_path: ${{oc.env:TRINITY_MODEL_PATH,/data/hjx/Age_mem/models/Qwen3-4B}}
  max_model_len: 5120
  max_prompt_tokens: 4096
  max_response_tokens: 1024
  lora_configs:
  - name: lora
    lora_rank: 16
    lora_alpha: 16
    path: {lora_path}

cluster:
  node_num: 1
  gpu_per_node: 2

buffer:
  total_epochs: 1
  total_steps: 16
  batch_size: 2
  train_batch_size: 4
  explorer_input:
    taskset:
      name: hotpotqa_fc_dev_32
      storage_type: file
      path: ${{oc.env:HOTPOTQA_PATH,/root/autodl-tmp/data/hotpot_qa/fullwiki}}
      split: validation
      row_indices: {_yaml_index_block(rows)}
      row_id_key: id
      expected_row_ids:
{_yaml_id_block(row_ids)}
      expected_dataset_fingerprint: {fingerprint}
      format:
        prompt_key: question
        response_key: answer
      rollout_args:
        temperature: 0.0
        max_tokens: 1024
      workflow_args:
{workflow}
    eval_tasksets:
    - name: hotpotqa_fc_dev_32
      storage_type: file
      path: ${{oc.env:HOTPOTQA_PATH,/root/autodl-tmp/data/hotpot_qa/fullwiki}}
      split: validation
      repeat_times: 1
      row_indices: {_yaml_index_block(rows)}
      row_id_key: id
      expected_row_ids:
{_yaml_id_block(row_ids)}
      expected_dataset_fingerprint: {fingerprint}
      format:
        prompt_key: question
        response_key: answer
      rollout_args:
        temperature: 0.0
        max_tokens: 1024
      workflow_args:
{workflow}
    default_workflow_type: AgeMem_hotpot_workflow_training
    default_eval_workflow_type: AgeMem_hotpot_workflow_training
  trainer_input:
    experience_buffer:
      name: {job.replace("-", "_")}_buffer
      storage_type: queue
      path: null

explorer:
  eval_on_startup: true
  bench_on_latest_checkpoint: false
  eval_interval: 1
  runner_per_model: 2
  max_repeat_times_per_runner: 1
  max_timeout: 1800
  rollout_model:
    engine_num: 1
    tensor_parallel_size: 1
    enable_prefix_caching: true
    enforce_eager: true
    enable_history: true
    dtype: bfloat16
    seed: {SEED}
    gpu_memory_utilization: 0.6
    enable_chunked_prefill: true
    enable_thinking: false

log:
  level: INFO

synchronizer:
  sync_method: checkpoint
  sync_interval: 1
  sync_timeout: 1800

trainer:
  trainer_type: verl
  total_steps: 1
  save_interval: 1
  trainer_config:
    actor_rollout_ref:
      model:
        use_remove_padding: true
        enable_gradient_checkpointing: true
      actor:
        use_dynamic_bsz: true
        ppo_max_token_len_per_gpu: 2304
        ulysses_sequence_parallel_size: 1
        optim:
          lr: 0.000001
      ref:
        log_prob_use_dynamic_bsz: ${{trainer.trainer_config.actor_rollout_ref.actor.use_dynamic_bsz}}
        log_prob_max_token_len_per_gpu: ${{trainer.trainer_config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu}}
        ulysses_sequence_parallel_size: 1
"""


def write_runtime_eval_yamls(directory: Path, fc_lock: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Write E0 and checkpoint-eval YAMLs into a runtime directory."""

    lock = fc_lock or load_fc_lock()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        0: directory / "agemem_e0_4b_fc_pilot_eval.yaml",
        12: directory / "agemem_e1_4b_fc_pilot_eval_s12.yaml",
        24: directory / "agemem_e1_4b_fc_pilot_eval_s24.yaml",
        36: directory / "agemem_e1_4b_fc_pilot_eval_s36.yaml",
    }
    paths[0].write_text(render_e0_yaml(lock), encoding="utf-8", newline="\n")
    for step in (12, 24, 36):
        paths[step].write_text(
            render_checkpoint_eval_yaml(lock, step), encoding="utf-8", newline="\n"
        )
    return paths


def build_lock(fc_lock: Mapping[str, Any] | None = None) -> dict[str, Any]:
    protocol = fc_lock or load_fc_lock()
    TRAIN_YAML.write_text(render_train_yaml(protocol), encoding="utf-8", newline="\n")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "checkpoint_root": CHECKPOINT_ROOT,
        "protocol_lock": FC_LOCK_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "stage3_require_final_answer": True,
        "stage3_repair_untagged_answer": True,
        "reward_profile": "terminal_only",
        "terminal_reward_metric": "hotpotqa_official",
        "seed": SEED,
        "later_seeds": list(LATER_SEEDS),
        "trainer_total_steps": TRAINER_TOTAL_STEPS,
        "save_interval": SAVE_INTERVAL,
        "eval_steps": list(EVAL_STEPS),
        "repeat_times": REPEAT_TIMES,
        "batch_size": BATCH_SIZE,
        "consume_put_batch": True,
        "train_batch_size": 8,
        "model": {
            "repository_id": EXPECTED_REPOSITORY,
            "expected_revision": EXPECTED_REVISION,
        },
        "jobs": {
            "e0": E0_JOB,
            "train": TRAIN_JOB,
            "eval_s12": EVAL_JOB_BY_STEP[12],
            "eval_s24": EVAL_JOB_BY_STEP[24],
            "eval_s36": EVAL_JOB_BY_STEP[36],
        },
        "source_files": {
            "train_config": {
                "path": TRAIN_YAML.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": _source_digest(TRAIN_YAML),
            }
        },
    }


def write_lock_and_train_yaml(fc_lock: Mapping[str, Any] | None = None) -> dict[str, Any]:
    lock = build_lock(fc_lock)
    LOCK_PATH.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return lock
