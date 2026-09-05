"""Helpers for the independent 4B terminal-only E1 protocol.

These helpers are not imported by the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_4b.json"
E0_YAML = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e0_4b_frozen_eval.yaml"
E1_YAML = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_dry_run.yaml"
EVAL_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_checkpoint_eval.yaml"
)

EXPECTED_REPOSITORY = "Qwen/Qwen3-4B"
EXPECTED_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SMOKE_E0_JOB = "agemem-e0-terminal-only-frozen-eval"
SMOKE_E1_JOB = "agemem-e1-terminal-only-dry-run"
FORBIDDEN_LEGACY_JOBS = (
    SMOKE_E0_JOB,
    SMOKE_E1_JOB,
    "agemem-e1-terminal-only-scale",
    "agemem-e1-terminal-only-repeat-s7",
    "agemem-e1-terminal-only-repeat-s17",
    "agemem-e1-terminal-only-repeat-s27",
    "agemem-e1-stage3-answer-probe",
    "agemem-e1-4b-stage3-answer-probe",
    "agemem-e0-terminal-only-4b-format-eval",
    "agemem-e1-terminal-only-4b-format",
    "agemem-e0-terminal-only-4b-format-var-eval",
    "agemem-e1-terminal-only-4b-format-var",
    "agemem-e0-terminal-only-4b-format-group-eval",
    "agemem-e1-terminal-only-4b-format-group",
)


def load_lock(path: Path | None = None) -> dict[str, Any]:
    target = path or LOCK_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def yaml_forbids_nudge(text: str) -> bool:
    return (
        "stage3_require_final_answer" not in text
        and "stage3_repair_untagged_answer" not in text
        and "Qwen2.5-1.5B-Instruct" not in text
        and SMOKE_E0_JOB not in text
        and SMOKE_E1_JOB not in text
    )


def job_names(lock: Mapping[str, Any]) -> dict[str, str]:
    e0 = str(lock["paths"]["clean_job_relative_paths"][0]).rsplit("/", 1)[-1]
    e1 = str(lock["paths"]["clean_job_relative_paths"][1]).rsplit("/", 1)[-1]
    return {"e0": e0, "e1": e1}
