"""Small machine-readable receipts for fail-closed training and evaluation."""

from __future__ import annotations

import json
import math
import numbers
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional


TRAINING_RECEIPT_SCHEMA_VERSION = "trinity.training_update.v1"
BENCHMARK_RECEIPT_SCHEMA_VERSION = "trinity.benchmark.v1"
_PROCESS_EXECUTION_ID = uuid.uuid4().hex
_RECEIPT_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def _process_identity() -> dict[str, Any]:
    return {
        "process_id": os.getpid(),
        "process_execution_id": _PROCESS_EXECUTION_ID,
    }


def finite_scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Return sorted finite numeric metrics and reject any non-finite scalar."""

    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if not isinstance(value, numbers.Real):
            item = getattr(value, "item", None)
            if not callable(item):
                continue
            try:
                value = item()
            except Exception:
                continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"metric {key!r} is not finite")
        normalized[str(key)] = numeric
    return {key: normalized[key] for key in sorted(normalized)}


def _write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    text = json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, output)


def write_training_receipt(
    checkpoint_job_dir: str,
    *,
    completed_step: int,
    configured_total_steps: Optional[int],
    metrics: Mapping[str, Any],
) -> Path:
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or completed_step <= 0
    ):
        raise ValueError("completed_step must be positive")
    if configured_total_steps is not None and (
        isinstance(configured_total_steps, bool)
        or not isinstance(configured_total_steps, int)
        or configured_total_steps <= 0
    ):
        raise ValueError("configured_total_steps must be positive or None")
    normalized = finite_scalar_metrics(metrics)
    output = (
        Path(checkpoint_job_dir)
        / "receipts"
        / f"trainer_step_{completed_step}.json"
    )
    _write_json(
        {
            "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
            "status": "completed",
            "completed_step": completed_step,
            "configured_total_steps": configured_total_steps,
            "metrics": normalized,
            **_process_identity(),
        },
        output,
    )
    return output


def write_benchmark_receipt(
    checkpoint_job_dir: str,
    *,
    prefix: str,
    step: int,
    model_version: int,
    task_summaries: list[dict[str, Any]],
    metrics: Mapping[str, Any],
) -> Path:
    if (
        not isinstance(prefix, str)
        or not _RECEIPT_PREFIX_PATTERN.fullmatch(prefix)
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or isinstance(model_version, bool)
        or not isinstance(model_version, int)
        or model_version < 0
    ):
        raise ValueError("benchmark receipt identity is invalid")
    normalized = finite_scalar_metrics(metrics)
    output = (
        Path(checkpoint_job_dir)
        / "receipts"
        / f"{prefix}_step_{step}_model_{model_version}.json"
    )
    _write_json(
        {
            "schema_version": BENCHMARK_RECEIPT_SCHEMA_VERSION,
            "status": "completed",
            "prefix": prefix,
            "step": step,
            "model_version": model_version,
            "task_summaries": task_summaries,
            "metrics": normalized,
            **_process_identity(),
        },
        output,
    )
    return output


__all__ = [
    "BENCHMARK_RECEIPT_SCHEMA_VERSION",
    "TRAINING_RECEIPT_SCHEMA_VERSION",
    "finite_scalar_metrics",
    "write_benchmark_receipt",
    "write_training_receipt",
]
