"""Helpers for the format-variance 4B GRPO protocol.

These helpers are not imported by the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trinity.common.e1_4b_format import (
    FORMAT_E0_JOB,
    FORMAT_E1_JOB,
    VANILLA_E0_JOB,
    VANILLA_E1_JOB,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_4b_format_var.json"
FORMAT_LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_4b_format.json"
E0_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e0_4b_format_var_eval.yaml"
)
E1_YAML = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_format_var.yaml"
EVAL_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_format_var_eval.yaml"
)

EXPECTED_REPOSITORY = "Qwen/Qwen3-4B"
EXPECTED_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
VAR_E0_JOB = "agemem-e0-terminal-only-4b-format-var-eval"
VAR_E1_JOB = "agemem-e1-terminal-only-4b-format-var"
FORBIDDEN_FOREIGN_JOBS = (
    "agemem-e0-terminal-only-frozen-eval",
    "agemem-e1-terminal-only-dry-run",
    "agemem-e1-terminal-only-scale",
    "agemem-e1-terminal-only-repeat-s7",
    "agemem-e1-terminal-only-repeat-s17",
    "agemem-e1-terminal-only-repeat-s27",
    "agemem-e1-stage3-answer-probe",
    "agemem-e1-4b-stage3-answer-probe",
    VANILLA_E0_JOB,
    VANILLA_E1_JOB,
    FORMAT_E0_JOB,
    FORMAT_E1_JOB,
)


def load_lock(path: Path | None = None) -> dict[str, Any]:
    target = path or LOCK_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def yaml_requires_nudge(text: str) -> bool:
    return (
        "stage3_require_final_answer: true" in text
        and "stage3_repair_untagged_answer: true" in text
        and "stage3_require_final_answer: false" not in text
        and "Qwen2.5-1.5B-Instruct" not in text
        and VANILLA_E0_JOB not in text
        and VANILLA_E1_JOB not in text
        and FORMAT_E0_JOB not in text
        and f'"{FORMAT_E1_JOB}"' not in text
        and f"/{FORMAT_E1_JOB}/" not in text
    )


def job_names(lock: Mapping[str, Any]) -> dict[str, str]:
    e0 = str(lock["paths"]["clean_job_relative_paths"][0]).rsplit("/", 1)[-1]
    e1 = str(lock["paths"]["clean_job_relative_paths"][1]).rsplit("/", 1)[-1]
    return {"e0": e0, "e1": e1}
