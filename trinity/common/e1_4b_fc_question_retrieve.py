"""Helpers for the format-conditioned 4B question-retrieve 32-dev bench.

These helpers are not imported by the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trinity.common.e1_4b_fc_pilot import ALL_JOBS as PILOT_JOBS
from trinity.common.e1_4b_format_conditioned import (
    ALL_JOBS as DIAGNOSIS_JOBS,
)
from trinity.common.e1_4b_format_conditioned import (
    EXPECTED_REVISION,
    load_lock as load_fc_lock,
    render_bench_yaml,
    selection_is_frozen,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_4b_fc_question_retrieve.json"

SCHEMA_VERSION = "agemem.e1_4b_fc_question_retrieve.lock.v1"
EXPERIMENT_ID = "e1_format_conditioned_4b_question_retrieve"
CHECKPOINT_ROOT = "/data/hjx/Age_mem/checkpoints-e1-4b-fc-question-retrieve"
JOB = "agemem-e1-4b-fc-question-retrieve"
ALL_JOBS = (JOB,)
QUESTION_RETRIEVE_TOP_K = 8
WORKFLOW_EXTRA = (
    "stage3_question_retrieve: true",
    "stage3_index_observed_context: true",
    f"stage3_question_retrieve_top_k: {QUESTION_RETRIEVE_TOP_K}",
)

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
    *PILOT_JOBS,
    "agemem-e0-4b-fc-e3-eval",
    "agemem-e3-4b-fc",
    "agemem-e3-4b-fc-eval-s12",
)


def load_lock(path: Path | None = None) -> dict[str, Any]:
    target = path or LOCK_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def render_bench(fc_lock: Mapping[str, Any]) -> str:
    if not selection_is_frozen(fc_lock):
        raise ValueError("question-retrieve bench requires frozen 32-dev selection")
    return render_bench_yaml(
        fc_lock,
        job=JOB,
        rows=list(fc_lock["fixed_dev_rows"]),
        split="validation",
        fingerprint=str(fc_lock["eval_dataset_fingerprint"]),
        temperature=0.0,
        repeat_times=1,
        extra_workflow_args=WORKFLOW_EXTRA,
        comment=(
            "# Format-conditioned 4B question-retrieve bench. Frozen 32-dev, K=1, T=0, "
            "Stage-3 nudge. Indexes observed Stage-1 sentences into LTM, then hybrid-"
            "retrieves by the question. Does not inject gold supporting facts."
        ),
        taskset_name="hotpotqa_fc_dev_32",
    )


def build_lock() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "checkpoint_root": CHECKPOINT_ROOT,
        "protocol_lock": "configs/e1_4b_format_conditioned.json",
        "stage3_require_final_answer": True,
        "stage3_repair_untagged_answer": True,
        "stage3_question_retrieve": True,
        "stage3_index_observed_context": True,
        "stage3_question_retrieve_top_k": QUESTION_RETRIEVE_TOP_K,
        "stage3_inject_gold_supporting": False,
        "reward_profile": "terminal_only",
        "terminal_reward_metric": "hotpotqa_official",
        "seed": 7,
        "repeat_times": 1,
        "model": {
            "repository_id": "Qwen/Qwen3-4B",
            "expected_revision": EXPECTED_REVISION,
        },
        "jobs": {"bench": JOB},
    }


def write_runtime_yaml(directory: Path, fc_lock: Mapping[str, Any] | None = None) -> Path:
    protocol = fc_lock or load_fc_lock()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "agemem_e1_4b_fc_question_retrieve.yaml"
    path.write_text(render_bench(protocol), encoding="utf-8", newline="\n")
    return path


def write_lock() -> dict[str, Any]:
    lock = build_lock()
    LOCK_PATH.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return lock
