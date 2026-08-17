"""Read-only postflight validation for the M8b one-update GPU smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from trinity.common.runtime_receipt import (
    BENCHMARK_RECEIPT_SCHEMA_VERSION,
    TRAINING_RECEIPT_SCHEMA_VERSION,
)


POSTFLIGHT_SCHEMA_VERSION = "agemem.m8b_postflight.v1"
E0_JOB_RELATIVE_PATH = "Trinity-RFT-AgeMem-M8/agemem-e0-terminal-only-frozen-eval"
E1_JOB_RELATIVE_PATH = "Trinity-RFT-AgeMem-M8/agemem-e1-terminal-only-dry-run"
EXPECTED_BENCH_TASKSET = "hotpotqa_m5_heldout_smoke"
EXPECTED_BENCH_TASK_COUNT = 2

PASS = "pass"
FAIL = "fail"

_SHARD_PATTERN = re.compile(
    r"^(model|optim|extra_state)_world_size_([1-9][0-9]*)_rank_([0-9]+)\.pt$"
)
_PROCESS_EXECUTION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class CheckResult:
    """One machine-readable postflight decision."""

    name: str
    status: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": dict(self.details),
        }


class CheckBook:
    """Collect all postflight failures instead of stopping at the first one."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(
        self,
        name: str,
        status: str,
        message: str,
        **details: Any,
    ) -> None:
        if status not in {PASS, FAIL}:
            raise ValueError(f"unsupported postflight status: {status}")
        self.results.append(CheckResult(name, status, message, details))

    @property
    def passed(self) -> bool:
        return all(result.status == PASS for result in self.results)


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_metrics(value: Any) -> tuple[dict[str, float], list[str]]:
    if not isinstance(value, Mapping) or not value:
        return {}, ["metrics must be a non-empty object"]
    metrics: dict[str, float] = {}
    errors: list[str] = []
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            errors.append("metric keys must be non-empty strings")
            continue
        if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
            errors.append(f"metric {key!r} is not numeric")
            continue
        numeric = float(raw)
        if not math.isfinite(numeric):
            errors.append(f"metric {key!r} is not finite")
            continue
        metrics[key] = numeric
    return metrics, errors


def _metric_has_token(key: str, *expected: str) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", key.lower()))
    return bool(tokens.intersection(expected))


def _validate_task_summaries(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["task_summaries must be a non-empty list"]
    errors: list[str] = []
    if len(value) != 1:
        errors.append("task_summaries must contain exactly one held-out taskset")
    for index, summary in enumerate(value):
        if not isinstance(summary, Mapping):
            errors.append(f"task_summaries[{index}] must be an object")
            continue
        taskset = summary.get("taskset")
        task_count = summary.get("task_count")
        failed_count = summary.get("failed_count")
        if not isinstance(taskset, str) or not taskset:
            errors.append(f"task_summaries[{index}].taskset is invalid")
        elif taskset != EXPECTED_BENCH_TASKSET:
            errors.append(
                f"task_summaries[{index}].taskset must equal "
                f"{EXPECTED_BENCH_TASKSET!r}"
            )
        if (
            isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count <= 0
        ):
            errors.append(f"task_summaries[{index}].task_count is invalid")
        elif task_count != EXPECTED_BENCH_TASK_COUNT:
            errors.append(
                f"task_summaries[{index}].task_count must equal "
                f"{EXPECTED_BENCH_TASK_COUNT}"
            )
        if (
            isinstance(failed_count, bool)
            or not isinstance(failed_count, int)
            or failed_count != 0
        ):
            errors.append(f"task_summaries[{index}] contains failed tasks")
    return errors


def _validate_process_identity(payload: Mapping[str, Any]) -> list[str]:
    process_id = payload.get("process_id")
    process_execution_id = payload.get("process_execution_id")
    errors: list[str] = []
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        errors.append("process_id must be a positive integer")
    if (
        not isinstance(process_execution_id, str)
        or not _PROCESS_EXECUTION_ID_PATTERN.fullmatch(process_execution_id)
    ):
        errors.append("process_execution_id must be a lowercase 32-character ID")
    return errors


def _identity_matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, int) and not isinstance(expected, bool):
        return (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed == expected
        )
    return observed == expected


def _read_benchmark_receipt(
    path: Path,
    *,
    expected_step: int,
    expected_model_version: int,
) -> tuple[Optional[dict[str, Any]], dict[str, float], list[str]]:
    try:
        payload = _load_json_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, {}, [f"unable to load strict JSON receipt: {type(exc).__name__}"]

    errors: list[str] = []
    expected = {
        "schema_version": BENCHMARK_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "prefix": "bench",
        "step": expected_step,
        "model_version": expected_model_version,
    }
    for key, expected_value in expected.items():
        if not _identity_matches(payload.get(key), expected_value):
            errors.append(f"{key} must equal {expected_value!r}")
    errors.extend(_validate_process_identity(payload))
    errors.extend(_validate_task_summaries(payload.get("task_summaries")))
    metrics, metric_errors = _finite_metrics(payload.get("metrics"))
    errors.extend(metric_errors)
    score_key = f"bench/{EXPECTED_BENCH_TASKSET}/task_score/mean"
    score = metrics.get(score_key)
    if score is None or not 0.0 <= score <= 1.0:
        errors.append(
            f"metrics must contain finite {score_key!r} in the range [0, 1]"
        )
    return payload, metrics, errors


def _check_benchmark_receipt(
    checks: CheckBook,
    *,
    name: str,
    path: Path,
    expected_step: int,
    expected_model_version: int,
) -> dict[str, Any]:
    payload, metrics, errors = _read_benchmark_receipt(
        path,
        expected_step=expected_step,
        expected_model_version=expected_model_version,
    )
    if errors:
        checks.add(
            name,
            FAIL,
            "benchmark completion receipt is invalid",
            path=str(path),
            errors=errors,
        )
    else:
        checks.add(
            name,
            PASS,
            "benchmark completed with no failed tasks at the expected model version",
            path=str(path),
            metric_keys=sorted(metrics),
            process_id=payload["process_id"],
            process_execution_id=payload["process_execution_id"],
        )
    return {
        "path": str(path),
        "schema_version": payload.get("schema_version") if payload else None,
        "step": payload.get("step") if payload else None,
        "model_version": payload.get("model_version") if payload else None,
        "metric_keys": sorted(metrics),
        "process_id": payload.get("process_id") if payload else None,
        "process_execution_id": (
            payload.get("process_execution_id") if payload else None
        ),
    }


def _check_training_receipt(
    checks: CheckBook,
    path: Path,
    *,
    expected_step: int,
) -> dict[str, Any]:
    try:
        payload = _load_json_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        checks.add(
            "e1.training_receipt",
            FAIL,
            "training completion receipt is unreadable",
            path=str(path),
            error_type=type(exc).__name__,
        )
        return {"path": str(path), "metric_keys": []}

    errors: list[str] = []
    expected = {
        "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "completed_step": expected_step,
        "configured_total_steps": expected_step,
    }
    for key, expected_value in expected.items():
        if not _identity_matches(payload.get(key), expected_value):
            errors.append(f"{key} must equal {expected_value!r}")
    errors.extend(_validate_process_identity(payload))
    metrics, metric_errors = _finite_metrics(payload.get("metrics"))
    errors.extend(metric_errors)
    if errors:
        checks.add(
            "e1.training_receipt",
            FAIL,
            "training completion receipt is invalid",
            path=str(path),
            errors=errors,
        )
    else:
        checks.add(
            "e1.training_receipt",
            PASS,
            "one finite trainer update is recorded",
            path=str(path),
            metric_count=len(metrics),
        )

    sentinel = metrics.get("training/actor_update_completed")
    checks.add(
        "e1.actor_update",
        PASS if sentinel == 1.0 else FAIL,
        "actor update completion sentinel is present"
        if sentinel == 1.0
        else "actor update completion sentinel must equal 1.0",
        observed=sentinel,
    )

    category_keys = {
        "loss": sorted(key for key in metrics if _metric_has_token(key, "loss")),
        "kl": sorted(key for key in metrics if _metric_has_token(key, "kl")),
        "reward": sorted(
            key for key in metrics if _metric_has_token(key, "reward", "rewards")
        ),
    }
    missing_categories = [
        category for category, keys in category_keys.items() if not keys
    ]
    checks.add(
        "e1.training_metrics",
        PASS if not missing_categories else FAIL,
        "finite loss, KL, and reward metrics are recorded"
        if not missing_categories
        else "training receipt is missing required metric categories",
        matched_keys=category_keys,
        missing_categories=missing_categories,
    )
    return {
        "path": str(path),
        "completed_step": payload.get("completed_step"),
        "metric_keys": sorted(metrics),
        "required_metric_keys": category_keys,
        "process_id": payload.get("process_id"),
        "process_execution_id": payload.get("process_execution_id"),
    }


def _check_trainer_state(
    checks: CheckBook,
    job_dir: Path,
    *,
    expected_step: int,
) -> dict[str, Any]:
    meta_path = job_dir / "trainer_meta.json"
    try:
        meta = _load_json_object(meta_path)
        latest_iteration = meta.get("latest_iteration")
        latest_exp_index = meta.get("latest_exp_index")
        valid_meta = (
            latest_iteration == expected_step
            and not isinstance(latest_iteration, bool)
            and isinstance(latest_exp_index, int)
            and not isinstance(latest_exp_index, bool)
            and latest_exp_index > 0
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        meta = {}
        valid_meta = False
        meta_error = type(exc).__name__
    else:
        meta_error = None
    checks.add(
        "e1.trainer_meta",
        PASS if valid_meta else FAIL,
        "trainer state records completed step one"
        if valid_meta
        else "trainer_meta.json does not prove completed step one",
        path=str(meta_path),
        latest_iteration=meta.get("latest_iteration"),
        latest_exp_index=meta.get("latest_exp_index"),
        error_type=meta_error,
    )

    iteration_path = job_dir / "latest_checkpointed_iteration.txt"
    try:
        if iteration_path.is_symlink() or not iteration_path.is_file():
            raise FileNotFoundError(iteration_path)
        iteration_text = iteration_path.read_text(encoding="utf-8").strip()
        checkpoint_iteration = int(iteration_text)
        valid_iteration = (
            iteration_text == str(expected_step)
            and checkpoint_iteration == expected_step
        )
    except (OSError, UnicodeError, ValueError) as exc:
        checkpoint_iteration = None
        valid_iteration = False
        iteration_error = type(exc).__name__
    else:
        iteration_error = None
    checks.add(
        "e1.latest_checkpoint",
        PASS if valid_iteration else FAIL,
        "latest checkpoint marker is exactly step one"
        if valid_iteration
        else "latest checkpoint marker must be exactly step one",
        path=str(iteration_path),
        observed=checkpoint_iteration,
        error_type=iteration_error,
    )
    return {
        "trainer_meta_path": str(meta_path),
        "latest_iteration": meta.get("latest_iteration"),
        "latest_exp_index": meta.get("latest_exp_index"),
        "checkpoint_marker_path": str(iteration_path),
        "checkpoint_iteration": checkpoint_iteration,
    }


def _check_checkpoint_shards(
    checks: CheckBook,
    actor_dir: Path,
) -> dict[str, Any]:
    groups: dict[str, list[tuple[int, int, Path]]] = {
        "model": [],
        "optim": [],
        "extra_state": [],
    }
    errors: list[str] = []
    if actor_dir.is_symlink() or not actor_dir.is_dir():
        errors.append("actor checkpoint directory is missing or is a symlink")
    else:
        try:
            entries = sorted(actor_dir.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            entries = []
            errors.append(f"unable to list actor checkpoint: {type(exc).__name__}")
        for entry in entries:
            match = _SHARD_PATTERN.fullmatch(entry.name)
            if match is None:
                continue
            kind, world_size_text, rank_text = match.groups()
            if entry.is_symlink() or not entry.is_file():
                errors.append(f"{entry.name} is not a regular shard")
                continue
            try:
                size = entry.stat().st_size
            except OSError as exc:
                errors.append(f"unable to stat {entry.name}: {type(exc).__name__}")
                continue
            if size <= 0:
                errors.append(f"{entry.name} is empty")
            groups[kind].append((int(world_size_text), int(rank_text), entry))

    expected_world_size: Optional[int] = None
    inventory: dict[str, Any] = {}
    for kind, shards in groups.items():
        worlds = {world_size for world_size, _rank, _path in shards}
        ranks = {rank for _world_size, rank, _path in shards}
        if not shards:
            errors.append(f"no {kind} checkpoint shards found")
        elif len(worlds) != 1:
            errors.append(f"{kind} shards contain inconsistent world sizes")
        else:
            world_size = next(iter(worlds))
            if ranks != set(range(world_size)) or len(shards) != world_size:
                errors.append(f"{kind} shards do not cover every rank exactly once")
            if expected_world_size is None:
                expected_world_size = world_size
            elif expected_world_size != world_size:
                errors.append("checkpoint shard categories use different world sizes")
        inventory[kind] = [
            {
                "path": str(path),
                "world_size": world_size,
                "rank": rank,
                "size_bytes": path.stat().st_size
                if path.is_file() and not path.is_symlink()
                else None,
            }
            for world_size, rank, path in shards
        ]

    checks.add(
        "e1.checkpoint_shards",
        PASS if not errors else FAIL,
        "model, optimizer, and extra-state shard sets are complete and non-empty"
        if not errors
        else "step-one checkpoint shards are incomplete or invalid",
        actor_dir=str(actor_dir),
        errors=errors,
        world_size=expected_world_size,
    )
    return inventory


def _regular_nonempty_file(path: Path) -> bool:
    try:
        return not path.is_symlink() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _check_lora(
    checks: CheckBook,
    *,
    actor_dir: Path,
    dummy_dir: Path,
) -> dict[str, Any]:
    trained_dir = actor_dir / "lora_adapter"
    trained_weights = trained_dir / "adapter_model.safetensors"
    trained_config = trained_dir / "adapter_config.json"
    dummy_weights = dummy_dir / "adapter_model.safetensors"
    dummy_config = dummy_dir / "adapter_config.json"
    required = {
        "trained_weights": trained_weights,
        "trained_config": trained_config,
        "dummy_weights": dummy_weights,
        "dummy_config": dummy_config,
    }
    errors = [
        name for name, path in required.items() if not _regular_nonempty_file(path)
    ]
    for name, path in (
        ("trained_config", trained_config),
        ("dummy_config", dummy_config),
    ):
        if name in errors:
            continue
        try:
            _load_json_object(path)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            errors.append(f"{name}_invalid_json")

    trained_digest = None
    dummy_digest = None
    if not errors:
        try:
            trained_digest = _sha256_file(trained_weights)
            dummy_digest = _sha256_file(dummy_weights)
        except OSError as exc:
            errors.append(f"unable to hash adapter weights: {type(exc).__name__}")
    if not errors and trained_digest == dummy_digest:
        errors.append("trained adapter weights are byte-identical to dummy_lora")

    checks.add(
        "e1.lora_adapter",
        PASS if not errors else FAIL,
        "step-one LoRA adapter exists and differs from dummy_lora"
        if not errors
        else "step-one LoRA adapter is missing, invalid, or unchanged",
        trained_dir=str(trained_dir),
        dummy_dir=str(dummy_dir),
        errors=errors,
        trained_weights_sha256=trained_digest,
        dummy_weights_sha256=dummy_digest,
    )
    return {
        "trained_dir": str(trained_dir),
        "dummy_dir": str(dummy_dir),
        "trained_weights_sha256": trained_digest,
        "dummy_weights_sha256": dummy_digest,
    }


def build_postflight_report(
    *,
    checkpoint_root: Path,
    e0_job_dir: Optional[Path] = None,
    e1_job_dir: Optional[Path] = None,
    expected_step: int = 1,
) -> dict[str, Any]:
    """Validate persisted M8b evidence without importing Ray or model code."""

    if expected_step != 1:
        raise ValueError("M8b postflight only supports the frozen single update")
    root = checkpoint_root.expanduser().resolve()
    e0_job = (
        e0_job_dir.expanduser().resolve()
        if e0_job_dir is not None
        else (root / E0_JOB_RELATIVE_PATH).resolve()
    )
    e1_job = (
        e1_job_dir.expanduser().resolve()
        if e1_job_dir is not None
        else (root / E1_JOB_RELATIVE_PATH).resolve()
    )
    checks = CheckBook()
    evidence: dict[str, Any] = {}

    evidence["e0_benchmark"] = _check_benchmark_receipt(
        checks,
        name="e0.benchmark_receipt",
        path=e0_job / "receipts" / "bench_step_0_model_0.json",
        expected_step=0,
        expected_model_version=0,
    )
    evidence["training"] = _check_training_receipt(
        checks,
        e1_job / "receipts" / f"trainer_step_{expected_step}.json",
        expected_step=expected_step,
    )
    evidence["trainer_state"] = _check_trainer_state(
        checks,
        e1_job,
        expected_step=expected_step,
    )
    actor_dir = e1_job / f"global_step_{expected_step}" / "actor"
    evidence["checkpoint_shards"] = _check_checkpoint_shards(checks, actor_dir)
    evidence["lora"] = _check_lora(
        checks,
        actor_dir=actor_dir,
        dummy_dir=e1_job / "dummy_lora",
    )
    evidence["checkpoint_evaluation"] = _check_benchmark_receipt(
        checks,
        name="e1.checkpoint_eval_receipt",
        path=(
            e1_job
            / "receipts"
            / f"bench_step_{expected_step}_model_{expected_step}.json"
        ),
        expected_step=expected_step,
        expected_model_version=expected_step,
    )

    training_execution_id = evidence["training"].get("process_execution_id")
    evaluation_execution_id = evidence["checkpoint_evaluation"].get(
        "process_execution_id"
    )
    distinct_process = bool(
        training_execution_id
        and evaluation_execution_id
        and training_execution_id != evaluation_execution_id
    )
    checks.add(
        "e1.checkpoint_eval_new_process",
        PASS if distinct_process else FAIL,
        "checkpoint evaluation ran in a distinct process"
        if distinct_process
        else "checkpoint evaluation must run in a process distinct from training",
        training_process_execution_id=training_execution_id,
        evaluation_process_execution_id=evaluation_execution_id,
    )
    evidence["checkpoint_evaluation"]["new_process_proof"] = (
        "distinct_process_execution_id" if distinct_process else None
    )
    return {
        "schema_version": POSTFLIGHT_SCHEMA_VERSION,
        "status": PASS if checks.passed else FAIL,
        "expected_step": expected_step,
        "paths": {
            "checkpoint_root": str(root),
            "e0_job_dir": str(e0_job),
            "e1_job_dir": str(e1_job),
        },
        "checks": [result.to_dict() for result in checks.results],
        "evidence": evidence,
    }


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Atomically write the postflight report with owner-only permissions."""

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    payload = (
        json.dumps(
            report,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, output)


def _summary(report: Mapping[str, Any], output_path: Optional[Path]) -> str:
    checks = report.get("checks", [])
    passed = sum(check.get("status") == PASS for check in checks)
    failed = sum(check.get("status") == FAIL for check in checks)
    destination = f"; report={output_path.resolve()}" if output_path else ""
    return (
        f"M8b postflight {str(report.get('status')).upper()}: "
        f"pass={passed} fail={failed}{destination}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate persisted evidence from the M8b AutoDL GPU smoke."
    )
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--e0-job-dir", default=None)
    parser.add_argument("--e1-job-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    arguments = parser.parse_args(argv)

    raw_root = arguments.checkpoint_root or os.environ.get(
        "TRINITY_CHECKPOINT_ROOT_DIR"
    )
    if not raw_root:
        parser.error("--checkpoint-root or TRINITY_CHECKPOINT_ROOT_DIR is required")
    checkpoint_root = Path(raw_root)
    report = build_postflight_report(
        checkpoint_root=checkpoint_root,
        e0_job_dir=Path(arguments.e0_job_dir) if arguments.e0_job_dir else None,
        e1_job_dir=Path(arguments.e1_job_dir) if arguments.e1_job_dir else None,
    )

    output_path: Optional[Path] = None
    if not arguments.no_write:
        output_path = (
            Path(arguments.output)
            if arguments.output
            else checkpoint_root / "m8b_postflight" / "postflight_report.json"
        )
        write_report(report, output_path)

    if arguments.print_json:
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(_summary(report, output_path))
        for check in report["checks"]:
            if check["status"] == FAIL:
                print(f"[FAIL] {check['name']}: {check['message']}")
    return 0 if report["status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "E0_JOB_RELATIVE_PATH",
    "E1_JOB_RELATIVE_PATH",
    "POSTFLIGHT_SCHEMA_VERSION",
    "build_postflight_report",
    "main",
    "write_report",
]
