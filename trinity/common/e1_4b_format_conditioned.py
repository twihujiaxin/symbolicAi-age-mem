"""Helpers for the format-conditioned 4B protocol and frozen diagnosis.

These helpers are not imported by the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_4b_format_conditioned.json"
SCALE_LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_scale.json"
EXAMPLES_DIR = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa"

EXPECTED_REPOSITORY = "Qwen/Qwen3-4B"
EXPECTED_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SCHEMA_VERSION = "agemem.e1_4b_format_conditioned.lock.v1"
EXPERIMENT_ID = "e1_format_conditioned_4b_protocol"
CHECKPOINT_ROOT = "/data/hjx/Age_mem/checkpoints-e1-4b-format-conditioned"
SELECTION_SEED = 20260905
SMOKE_SEED = 20260802
SCALE_SELECTION_SEED = 20260904

SIGNAL_JOB = "agemem-e1-4b-fc-signal-diag"
HELDOUT_JOB = "agemem-e1-4b-fc-heldout-regression"
MEM_NORMAL_JOB = "agemem-e1-4b-fc-mem-normal"
MEM_NO_RETRIEVE_JOB = "agemem-e1-4b-fc-mem-no-retrieve"
MEM_GOLD_JOB = "agemem-e1-4b-fc-mem-gold-support"
ALL_JOBS = (
    SIGNAL_JOB,
    HELDOUT_JOB,
    MEM_NORMAL_JOB,
    MEM_NO_RETRIEVE_JOB,
    MEM_GOLD_JOB,
)
MEM_JOBS = (MEM_NORMAL_JOB, MEM_NO_RETRIEVE_JOB, MEM_GOLD_JOB)

SIGNAL_YAML = EXAMPLES_DIR / "agemem_e1_4b_fc_signal_diag.yaml"
HELDOUT_YAML = EXAMPLES_DIR / "agemem_e1_4b_fc_heldout_regression.yaml"
MEM_NORMAL_YAML = EXAMPLES_DIR / "agemem_e1_4b_fc_mem_normal.yaml"
MEM_NO_RETRIEVE_YAML = EXAMPLES_DIR / "agemem_e1_4b_fc_mem_no_retrieve.yaml"
MEM_GOLD_YAML = EXAMPLES_DIR / "agemem_e1_4b_fc_mem_gold_support.yaml"

JOB_ALIASES = {
    "signal": SIGNAL_JOB,
    "heldout": HELDOUT_JOB,
    "mem-normal": MEM_NORMAL_JOB,
    "mem-no-retrieve": MEM_NO_RETRIEVE_JOB,
    "mem-gold-support": MEM_GOLD_JOB,
}
YAML_BY_JOB = {
    SIGNAL_JOB: SIGNAL_YAML,
    HELDOUT_JOB: HELDOUT_YAML,
    MEM_NORMAL_JOB: MEM_NORMAL_YAML,
    MEM_NO_RETRIEVE_JOB: MEM_NO_RETRIEVE_YAML,
    MEM_GOLD_JOB: MEM_GOLD_YAML,
}
MEM_EXTRA_WORKFLOW_ARGS = {
    MEM_NORMAL_JOB: (),
    MEM_NO_RETRIEVE_JOB: ("stage3_disable_ltm_retrieve: true",),
    MEM_GOLD_JOB: ("stage3_inject_gold_supporting: true",),
}

EXCLUDED_VALIDATION_ROWS = (
    {"hotpot_id": "5ab3d2b7554299233954ffb8", "source_index": 1880},
    {"hotpot_id": "5ab299d6554299449642c926", "source_index": 4920},
    {"hotpot_id": "5ab7c6995542993667794005", "source_index": 4073},
    {"hotpot_id": "5adc8c545542994734353734", "source_index": 5204},
)
DIAGNOSIS_FLAG_STRINGS = (
    "stage3_disable_ltm_retrieve",
    "stage3_inject_gold_supporting",
)
FROZEN_CLEAN_YAMLS = (
    EXAMPLES_DIR / "agemem_e1_dry_run.yaml",
    EXAMPLES_DIR / "agemem_e1_4b_dry_run.yaml",
    EXAMPLES_DIR / "agemem_e0_4b_frozen_eval.yaml",
    EXAMPLES_DIR / "agemem_e1_4b_checkpoint_eval.yaml",
    EXAMPLES_DIR / "agemem_e1_4b_format.yaml",
    EXAMPLES_DIR / "agemem_e1_4b_format_var.yaml",
    EXAMPLES_DIR / "agemem_e1_4b_format_group.yaml",
)

_WORKFLOW_ARG_LINES = (
    "        reward_profile: terminal_only",
    "        terminal_reward_metric: hotpotqa_official",
    "        milestone_reward_enabled: false",
    "        stage3_require_final_answer: true",
    "        stage3_repair_untagged_answer: true",
    "        auxiliary_provider:",
    "          schema_version: agemem.auxiliary_provider.v1",
    "          provider: dashscope",
    "          base_url: https://dashscope.aliyuncs.com/compatible-mode/v1",
    "          embedding_model: text-embedding-v4",
    "          embedding_dimensions: 256",
    "          chat_model: qwen-max",
    "          usage_tracking: true",
    "        auto_summary_threshold: 0.8",
    "        max_tool_rounds_per_turn: 4",
    "        max_context_tokens: 4096",
    "        stage2_distractor_messages: 1",
    "        stage2_distractor_source: fixed",
    "        stage1_max_rounds: 2",
    "        stage2_max_rounds: 2",
    "        stage3_max_rounds: 2",
    "        tool_trace_enabled: true",
    "        tool_trace_console: false",
    "        tool_trace_max_string_chars: 8192",
    "        tool_trace_ray_timeout_seconds: 5.0",
)


def load_lock(path: Path | None = None) -> dict[str, Any]:
    target = path or LOCK_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def load_scale_lock(path: Path | None = None) -> dict[str, Any]:
    target = path or SCALE_LOCK_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def excluded_validation_ids(lock: Mapping[str, Any] | None = None) -> list[str]:
    if lock is None:
        return [str(row["hotpot_id"]) for row in EXCLUDED_VALIDATION_ROWS]
    rows = lock.get("excluded_validation_rows") or EXCLUDED_VALIDATION_ROWS
    return [str(row["hotpot_id"]) for row in rows]


def train_rows_match_scale(lock: Mapping[str, Any], scale: Mapping[str, Any] | None = None) -> bool:
    scale_lock = scale or load_scale_lock()
    return list(lock.get("fixed_train_rows") or []) == list(
        scale_lock.get("fixed_train_rows") or []
    )


def _unique_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["hotpot_id"]) for row in rows]


def selection_is_frozen(lock: Mapping[str, Any]) -> bool:
    if str(lock.get("selection_status") or "") != "frozen":
        return False
    train = list(lock.get("fixed_train_rows") or [])
    dev = list(lock.get("fixed_dev_rows") or [])
    test = list(lock.get("fixed_test_rows") or [])
    if len(train) != int(lock["train_size"]):
        return False
    if len(dev) != int(lock["dev_size"]) or len(test) != int(lock["test_size"]):
        return False
    train_ids = _unique_ids(train)
    dev_ids = _unique_ids(dev)
    test_ids = _unique_ids(test)
    if len(set(train_ids)) != len(train_ids):
        return False
    if len(set(dev_ids)) != len(dev_ids) or len(set(test_ids)) != len(test_ids):
        return False
    excluded = set(excluded_validation_ids(lock))
    if excluded & set(dev_ids) or excluded & set(test_ids):
        return False
    if set(dev_ids) & set(test_ids) or set(dev_ids) & set(train_ids):
        return False
    if set(test_ids) & set(train_ids):
        return False
    return train_rows_match_scale(lock)


def yaml_path_for_job(job: str) -> Path:
    try:
        return YAML_BY_JOB[job]
    except KeyError as exc:
        raise KeyError(f"unknown format-conditioned job: {job}") from exc


def resolve_job_alias(name: str) -> str:
    if name in ALL_JOBS:
        return name
    try:
        return JOB_ALIASES[name]
    except KeyError as exc:
        raise KeyError(f"unknown format-conditioned job alias: {name}") from exc


def job_requires_frozen_selection(job: str) -> bool:
    return resolve_job_alias(job) in MEM_JOBS


def _yaml_id_block(row_ids: Sequence[str]) -> str:
    return "\n".join(f"      - {row_id}" for row_id in row_ids)


def _yaml_index_block(rows: Sequence[Mapping[str, Any]]) -> str:
    return "[" + ", ".join(str(int(row["source_index"])) for row in rows) + "]"


def _workflow_args(extra: Sequence[str] = ()) -> str:
    lines = list(_WORKFLOW_ARG_LINES)
    lines.extend(f"        {item}" for item in extra)
    return "\n".join(lines)


def render_bench_yaml(
    lock: Mapping[str, Any],
    *,
    job: str,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    fingerprint: str,
    temperature: float,
    repeat_times: int,
    extra_workflow_args: Sequence[str] = (),
    comment: str,
    taskset_name: str,
) -> str:
    if not rows:
        raise ValueError(f"cannot render {job} YAML without frozen rows")
    workflow = _workflow_args(extra_workflow_args)
    row_ids = _unique_ids(rows)
    batch_size = int(lock["batch_size"])
    total_steps = max(1, len(rows) // batch_size)
    timeout = 1800 if len(rows) > 2 else 900
    return f"""{comment}
project: "Trinity-RFT-AgeMem-M8"
name: "{job}"
mode: bench
checkpoint_root_dir: ${{oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}}
continue_from_checkpoint: false

algorithm:
  algorithm_type: multi_step_grpo
  advantage_fn: step_wise_grpo
  repeat_times: {int(repeat_times)}

model:
  model_path: ${{oc.env:TRINITY_MODEL_PATH,/data/hjx/Age_mem/models/Qwen3-4B}}
  max_model_len: 5120
  max_prompt_tokens: 4096
  max_response_tokens: 1024

cluster:
  node_num: 1
  gpu_per_node: 2

buffer:
  total_epochs: 1
  total_steps: {total_steps}
  batch_size: {batch_size}
  train_batch_size: 4
  explorer_input:
    taskset:
      name: {taskset_name}
      storage_type: file
      path: ${{oc.env:HOTPOTQA_PATH,/root/autodl-tmp/data/hotpot_qa/fullwiki}}
      split: {split}
      row_indices: {_yaml_index_block(rows)}
      row_id_key: id
      expected_row_ids:
{_yaml_id_block(row_ids)}
      expected_dataset_fingerprint: {fingerprint}
      format:
        prompt_key: question
        response_key: answer
      rollout_args:
        temperature: {temperature}
        max_tokens: 1024
      workflow_args:
{workflow}
    eval_tasksets:
    - name: {taskset_name}
      storage_type: file
      path: ${{oc.env:HOTPOTQA_PATH,/root/autodl-tmp/data/hotpot_qa/fullwiki}}
      split: {split}
      repeat_times: {int(repeat_times)}
      row_indices: {_yaml_index_block(rows)}
      row_id_key: id
      expected_row_ids:
{_yaml_id_block(row_ids)}
      expected_dataset_fingerprint: {fingerprint}
      format:
        prompt_key: question
        response_key: answer
      rollout_args:
        temperature: {temperature}
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
  max_repeat_times_per_runner: {int(repeat_times)}
  max_timeout: {timeout}
  rollout_model:
    engine_num: 1
    tensor_parallel_size: 1
    enable_prefix_caching: true
    enforce_eager: true
    enable_history: true
    dtype: bfloat16
    seed: {int(lock["seed"])}
    gpu_memory_utilization: 0.6
    enable_chunked_prefill: true
    enable_thinking: false

log:
  level: INFO

synchronizer:
  sync_method: checkpoint
  sync_interval: 1
  sync_timeout: {timeout}

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


def render_signal_yaml(lock: Mapping[str, Any]) -> str:
    return render_bench_yaml(
        lock,
        job=SIGNAL_JOB,
        rows=list(lock["fixed_train_rows"]),
        split="train",
        fingerprint=str(lock["expected_dataset_fingerprint"]),
        temperature=float(lock["signal_temperature"]),
        repeat_times=int(lock["signal_repeat_times"]),
        comment=(
            "# Format-conditioned 4B learning-signal diagnosis. 24 frozen train rows, "
            "K=4, T=0.6, Stage-3 nudge, no optimizer."
        ),
        taskset_name="hotpotqa_fc_train_24",
    )


def render_heldout_yaml(lock: Mapping[str, Any]) -> str:
    return render_bench_yaml(
        lock,
        job=HELDOUT_JOB,
        rows=list(lock["held_out_rows"]),
        split="validation",
        fingerprint=str(lock["eval_dataset_fingerprint"]),
        temperature=float(lock["eval_temperature"]),
        repeat_times=int(lock["eval_repeat_times"]),
        comment=(
            "# Format-conditioned 4B held-out regression. Same 2 validation IDs as "
            "closed format-group, K=1, T=0, Stage-3 nudge, no optimizer."
        ),
        taskset_name="hotpotqa_fc_heldout_2",
    )


def render_mem_yaml(lock: Mapping[str, Any], job: str) -> str:
    if not selection_is_frozen(lock):
        raise ValueError(
            "memory-necessity YAMLs require frozen 32-dev / 128-test selection"
        )
    if job not in MEM_JOBS:
        raise ValueError(f"not a memory-necessity job: {job}")
    return render_bench_yaml(
        lock,
        job=job,
        rows=list(lock["fixed_dev_rows"]),
        split="validation",
        fingerprint=str(lock["eval_dataset_fingerprint"]),
        temperature=float(lock["eval_temperature"]),
        repeat_times=int(lock["eval_repeat_times"]),
        extra_workflow_args=MEM_EXTRA_WORKFLOW_ARGS[job],
        comment=(
            f"# Format-conditioned 4B memory-necessity diagnosis ({job}). "
            "Frozen 32-dev rows, K=1, T=0, Stage-3 nudge, no optimizer."
        ),
        taskset_name="hotpotqa_fc_dev_32",
    )


def _source_entry(path: Path, digest: str | None = None) -> dict[str, str | None]:
    relative = path.relative_to(REPOSITORY_ROOT).as_posix()
    return {"path": relative, "sha256": digest}


def build_pending_lock() -> dict[str, Any]:
    scale = load_scale_lock()
    train_rows = list(scale["fixed_train_rows"])
    held_out_rows = list(scale["held_out_rows"])
    if len(train_rows) != 24:
        raise ValueError("e1_scale.json must already contain 24 frozen train rows")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection_status": "pending",
        "checkpoint_root": CHECKPOINT_ROOT,
        "stage3_require_final_answer": True,
        "stage3_repair_untagged_answer": True,
        "stage3_max_rounds": 2,
        "reward_profile": "terminal_only",
        "terminal_reward_metric": "hotpotqa_official",
        "seed": 7,
        "selection_seed": SELECTION_SEED,
        "smoke_seed": SMOKE_SEED,
        "scale_selection_seed": SCALE_SELECTION_SEED,
        "trainer_total_steps": 0,
        "batch_size": 2,
        "train_size": 24,
        "dev_size": 32,
        "test_size": 128,
        "signal_repeat_times": 4,
        "signal_temperature": 0.6,
        "eval_repeat_times": 1,
        "eval_temperature": 0.0,
        "expected_dataset_fingerprint": "c369f1b07b350d37",
        "eval_dataset_fingerprint": "fbe86cb2d14cb199",
        "model": {
            "repository_id": EXPECTED_REPOSITORY,
            "expected_revision": EXPECTED_REVISION,
        },
        "jobs": {
            "signal": SIGNAL_JOB,
            "heldout": HELDOUT_JOB,
            "mem_normal": MEM_NORMAL_JOB,
            "mem_no_retrieve": MEM_NO_RETRIEVE_JOB,
            "mem_gold_support": MEM_GOLD_JOB,
        },
        "excluded_validation_ids": [
            str(row["hotpot_id"]) for row in EXCLUDED_VALIDATION_ROWS
        ],
        "excluded_validation_rows": [dict(row) for row in EXCLUDED_VALIDATION_ROWS],
        "fixed_train_rows": train_rows,
        "held_out_rows": held_out_rows,
        "held_out_row_ids": [str(row["hotpot_id"]) for row in held_out_rows],
        "fixed_dev_rows": [],
        "fixed_test_rows": [],
        "source_files": {
            "signal_config": _source_entry(SIGNAL_YAML),
            "heldout_config": _source_entry(HELDOUT_YAML),
            "mem_normal_config": _source_entry(MEM_NORMAL_YAML),
            "mem_no_retrieve_config": _source_entry(MEM_NO_RETRIEVE_YAML),
            "mem_gold_support_config": _source_entry(MEM_GOLD_YAML),
        },
    }


def write_lock(lock: Mapping[str, Any], path: Path | None = None) -> None:
    target = path or LOCK_PATH
    target.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_known_yamls(lock: Mapping[str, Any]) -> dict[str, Any]:
    SIGNAL_YAML.write_text(render_signal_yaml(lock), encoding="utf-8", newline="\n")
    HELDOUT_YAML.write_text(render_heldout_yaml(lock), encoding="utf-8", newline="\n")
    updated = dict(lock)
    sources = dict(updated.get("source_files") or {})
    sources["signal_config"] = _source_entry(SIGNAL_YAML, _source_digest(SIGNAL_YAML))
    sources["heldout_config"] = _source_entry(HELDOUT_YAML, _source_digest(HELDOUT_YAML))
    updated["source_files"] = sources
    return updated


def write_mem_yamls(lock: Mapping[str, Any]) -> dict[str, Any]:
    if not selection_is_frozen(lock):
        raise ValueError("refusing to write memory-necessity YAMLs before freeze")
    MEM_NORMAL_YAML.write_text(
        render_mem_yaml(lock, MEM_NORMAL_JOB), encoding="utf-8", newline="\n"
    )
    MEM_NO_RETRIEVE_YAML.write_text(
        render_mem_yaml(lock, MEM_NO_RETRIEVE_JOB), encoding="utf-8", newline="\n"
    )
    MEM_GOLD_YAML.write_text(
        render_mem_yaml(lock, MEM_GOLD_JOB), encoding="utf-8", newline="\n"
    )
    updated = dict(lock)
    sources = dict(updated.get("source_files") or {})
    sources["mem_normal_config"] = _source_entry(
        MEM_NORMAL_YAML, _source_digest(MEM_NORMAL_YAML)
    )
    sources["mem_no_retrieve_config"] = _source_entry(
        MEM_NO_RETRIEVE_YAML, _source_digest(MEM_NO_RETRIEVE_YAML)
    )
    sources["mem_gold_support_config"] = _source_entry(
        MEM_GOLD_YAML, _source_digest(MEM_GOLD_YAML)
    )
    updated["source_files"] = sources
    return updated


def write_pending_lock_and_known_yamls() -> dict[str, Any]:
    lock = write_known_yamls(build_pending_lock())
    write_lock(lock)
    return lock
