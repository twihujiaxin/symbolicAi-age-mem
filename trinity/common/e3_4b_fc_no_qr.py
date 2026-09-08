"""Helpers for format-conditioned 4B E3 without question-retrieve.

These helpers are not imported by the frozen M8b 318-count runtime gate.
Do not import e3_oracle_dfa here: that pulls memory_oracle / agentscope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trinity.common.e1_4b_format_conditioned import (
    LOCK_PATH as FC_LOCK_PATH,
)
from trinity.common.e1_4b_format_conditioned import (
    load_lock as load_fc_lock,
)
from trinity.common.e3_4b_fc import (
    ALL_JOBS as QR_E3_JOBS,
    DEFAULT_MAX_STEPS,
    EXPECTED_REPOSITORY,
    EXPECTED_REVISION,
    FORBIDDEN_FOREIGN_JOBS as QR_E3_FOREIGN_JOBS,
    QUESTION_RETRIEVE_TOP_K,
    REPEAT_TIMES,
    REWARD_VERSION,
    SAVE_INTERVAL,
    SEED,
    TRAINER_TOTAL_STEPS,
    render_checkpoint_eval_yaml as render_qr_checkpoint_eval_yaml,
    render_e0_yaml as render_qr_e0_yaml,
    render_train_yaml as render_qr_train_yaml,
    write_runtime_eval_yamls as write_qr_runtime_eval_yamls,
)
from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e3_4b_fc_no_qr.json"
EXAMPLES_DIR = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa"
TRAIN_YAML = EXAMPLES_DIR / "agemem_e3_4b_fc_no_qr.yaml"

SCHEMA_VERSION = "agemem.e3_4b_fc_no_qr.lock.v1"
EXPERIMENT_ID = "e3_format_conditioned_4b_oracle_dfa_no_qr"
CHECKPOINT_ROOT = "/data/hjx/Age_mem/checkpoints-e3-4b-fc-no-qr"

E0_JOB = "agemem-e0-4b-fc-e3-no-qr-eval"
TRAIN_JOB = "agemem-e3-4b-fc-no-qr"
EVAL_JOB_BY_STEP = {12: "agemem-e3-4b-fc-no-qr-eval-s12"}
ALL_JOBS = (E0_JOB, TRAIN_JOB, *EVAL_JOB_BY_STEP.values())
EVAL_STEPS = (0, 12)
BUFFER_NAME = "agemem_e3_4b_fc_no_qr_buffer"

FORBIDDEN_FOREIGN_JOBS = (
    *QR_E3_FOREIGN_JOBS,
    *QR_E3_JOBS,
)


def load_lock(path: Path | None = None) -> dict[str, Any]:
    target = path or LOCK_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def render_train_yaml(fc_lock: Mapping[str, Any]) -> str:
    return render_qr_train_yaml(
        fc_lock,
        question_retrieve=False,
        train_job=TRAIN_JOB,
        buffer_name=BUFFER_NAME,
    )


def render_e0_yaml(fc_lock: Mapping[str, Any]) -> str:
    return render_qr_e0_yaml(fc_lock, question_retrieve=False, job=E0_JOB)


def render_checkpoint_eval_yaml(fc_lock: Mapping[str, Any], step: int) -> str:
    if step not in EVAL_JOB_BY_STEP:
        raise ValueError(f"unsupported E3 no-QR eval step: {step}")
    return render_qr_checkpoint_eval_yaml(
        fc_lock,
        step,
        question_retrieve=False,
        job=EVAL_JOB_BY_STEP[step],
        train_job=TRAIN_JOB,
    )


def write_runtime_eval_yamls(
    directory: Path, fc_lock: Mapping[str, Any] | None = None
) -> dict[int, Path]:
    return write_qr_runtime_eval_yamls(
        directory,
        fc_lock,
        question_retrieve=False,
        e0_job=E0_JOB,
        eval_job=EVAL_JOB_BY_STEP[12],
        train_job=TRAIN_JOB,
        e0_filename="agemem_e0_4b_fc_e3_no_qr_eval.yaml",
        eval_filename="agemem_e3_4b_fc_no_qr_eval_s12.yaml",
    )


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
        "stage3_question_retrieve": False,
        "stage3_index_observed_context": False,
        "stage3_question_retrieve_top_k": QUESTION_RETRIEVE_TOP_K,
        "stage3_inject_gold_supporting": False,
        "reward_profile": "terminal_dfa",
        "terminal_reward_metric": "hotpotqa_official",
        "e3_dfa_shadow_on_eval": True,
        "e3_dfa_max_steps": DEFAULT_MAX_STEPS,
        "reward_version": REWARD_VERSION,
        "seed": SEED,
        "trainer_total_steps": TRAINER_TOTAL_STEPS,
        "save_interval": SAVE_INTERVAL,
        "eval_steps": list(EVAL_STEPS),
        "repeat_times": REPEAT_TIMES,
        "batch_size": 2,
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
