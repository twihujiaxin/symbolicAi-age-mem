"""Fail-closed preflight checks for the M8b AutoDL smoke run.

The checker is deliberately read-only apart from its JSON report.  It never
starts Ray, imports a model, contacts a provider, or launches training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence


PREFLIGHT_SCHEMA_VERSION = "agemem.m8b_preflight.v1"
LOCK_SCHEMA_VERSION = "agemem.m8b_preflight_lock.v1"
DEFAULT_CONFIG = "examples/agemem_hotpotqa/agemem_e1_dry_run.yaml"
DEFAULT_LOCK = "configs/m8b_autodl_preflight.json"
DEFAULT_REPORT = "runs/m8b_preflight/preflight_report.json"

PASS = "pass"
FAIL = "fail"
WARN = "warn"
SKIP = "skip"

_TEXT_DIGEST_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
_FULL_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_MODEL_MANIFEST_SCHEMA_VERSION = "agemem.model_manifest.v1"
_MODEL_MANIFEST_IGNORED_DIRECTORIES = {".cache", ".git", "__pycache__"}


@dataclass(frozen=True)
class GateResult:
    """One machine-readable preflight decision."""

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


class GateBook:
    """Collect ordered checks without aborting after the first failure."""

    def __init__(self) -> None:
        self.results: list[GateResult] = []

    def add(
        self,
        name: str,
        status: str,
        message: str,
        **details: Any,
    ) -> None:
        if status not in {PASS, FAIL, WARN, SKIP}:
            raise ValueError(f"unsupported gate status: {status}")
        self.results.append(GateResult(name, status, message, details))

    @property
    def passed(self) -> bool:
        return all(result.status != FAIL for result in self.results)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_digest(path: Path) -> str:
    """Hash text sources with canonical LF line endings across checkouts."""

    if path.suffix.lower() not in _TEXT_DIGEST_SUFFIXES:
        return sha256_file(path)
    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _load_yaml_object(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised on bare hosts
        raise RuntimeError("PyYAML is required to validate the E1 config") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return payload


def _value_at_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for component in dotted_path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                raise KeyError(dotted_path)
            current = current[component]
            continue
        if isinstance(current, list):
            try:
                current = current[int(component)]
            except (ValueError, IndexError) as exc:
                raise KeyError(dotted_path) from exc
            continue
        raise KeyError(dotted_path)
    return current


def _run_command(
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 20.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _secret_key_present(environment: Mapping[str, str], key: str) -> bool:
    # Only availability is retained. The secret value is never returned or logged.
    value = environment.get(key)
    return isinstance(value, str) and bool(value.strip())


def _check_lock(
    lock_path: Path,
    config_path: Path,
    repository_root: Path,
    gates: GateBook,
) -> Optional[dict[str, Any]]:
    try:
        lock = _load_json_object(lock_path)
    except Exception as exc:
        gates.add("lock.load", FAIL, f"unable to load preflight lock: {type(exc).__name__}")
        return None

    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        gates.add(
            "lock.schema",
            FAIL,
            "preflight lock schema is not supported",
            actual=lock.get("schema_version"),
            expected=LOCK_SCHEMA_VERSION,
        )
        return None
    gates.add("lock.schema", PASS, "preflight lock schema is supported")

    source_files = lock.get("source_files")
    if not isinstance(source_files, dict):
        gates.add("lock.sources", FAIL, "source_files must be a mapping")
        return None

    required_sources = (
        "config",
        "smoke_manifest",
        "e0_config",
        "checkpoint_eval_config",
    )
    for source_name in required_sources:
        source = source_files.get(source_name)
        if not isinstance(source, dict):
            gates.add(
                f"lock.source.{source_name}",
                FAIL,
                "locked source declaration is missing",
            )
            continue
        relative_path = source.get("path")
        expected_digest = source.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            gates.add(
                f"lock.source.{source_name}",
                FAIL,
                "locked source path and digest must be strings",
            )
            continue
        actual_path = (repository_root / relative_path).resolve()
        if not _is_relative_to(actual_path, repository_root) or not actual_path.is_file():
            gates.add(
                f"lock.source.{source_name}",
                FAIL,
                "locked source is missing or outside the repository",
                path=relative_path,
            )
            continue
        actual_digest = _source_digest(actual_path)
        if actual_digest != expected_digest:
            gates.add(
                f"lock.source.{source_name}",
                FAIL,
                "locked source digest changed",
                path=relative_path,
                actual_sha256=actual_digest,
                expected_sha256=expected_digest,
            )
            continue
        if source_name == "config" and actual_path != config_path.resolve():
            gates.add(
                "lock.source.config",
                FAIL,
                "the requested config is not the locked E1 config",
                requested=str(config_path),
                locked=relative_path,
            )
            continue
        gates.add(
            f"lock.source.{source_name}",
            PASS,
            "locked source digest matches",
            path=relative_path,
            sha256=actual_digest,
        )
    return lock


def _check_config(
    config_path: Path,
    lock: Mapping[str, Any],
    gates: GateBook,
) -> Optional[dict[str, Any]]:
    try:
        config = _load_yaml_object(config_path)
    except Exception as exc:
        gates.add(
            "config.parse",
            FAIL,
            f"unable to parse E1 YAML: {type(exc).__name__}",
        )
        return None
    gates.add("config.parse", PASS, "E1 YAML parsed without interpolation or imports")

    assertions = lock.get("config_assertions")
    if not isinstance(assertions, list) or not assertions:
        gates.add("config.contract", FAIL, "config_assertions must be a non-empty list")
        return config

    errors: list[dict[str, Any]] = []
    for assertion in assertions:
        if not isinstance(assertion, dict):
            errors.append({"path": None, "error": "assertion is not an object"})
            continue
        dotted_path = assertion.get("path")
        if not isinstance(dotted_path, str) or "equals" not in assertion:
            errors.append({"path": dotted_path, "error": "invalid assertion"})
            continue
        try:
            actual = _value_at_path(config, dotted_path)
        except KeyError:
            errors.append({"path": dotted_path, "error": "missing"})
            continue
        expected = assertion["equals"]
        if actual != expected:
            errors.append(
                {
                    "path": dotted_path,
                    "error": "mismatch",
                    "actual": actual,
                    "expected": expected,
                }
            )

    if errors:
        gates.add(
            "config.contract",
            FAIL,
            "E1 config does not match the frozen terminal-only contract",
            errors=errors,
        )
    else:
        gates.add(
            "config.contract",
            PASS,
            "E1 config matches every frozen terminal-only assertion",
            assertion_count=len(assertions),
        )
    return config


def _check_manifest_consistency(
    repository_root: Path,
    lock: Mapping[str, Any],
    config: Optional[Mapping[str, Any]],
    gates: GateBook,
) -> Optional[dict[str, Any]]:
    source = lock.get("source_files", {}).get("smoke_manifest", {})
    relative_path = source.get("path")
    if not isinstance(relative_path, str):
        gates.add("manifest.contract", FAIL, "smoke manifest path is missing from lock")
        return None
    try:
        manifest = _load_json_object(repository_root / relative_path)
    except Exception as exc:
        gates.add(
            "manifest.contract",
            FAIL,
            f"unable to load smoke manifest: {type(exc).__name__}",
        )
        return None

    dataset_lock = lock.get("dataset")
    if not isinstance(dataset_lock, dict):
        gates.add("manifest.contract", FAIL, "dataset lock is missing")
        return manifest
    expected_fingerprints = dataset_lock.get("source_fingerprints")
    expected_rows = dataset_lock.get("fixed_train_rows")
    expected_eval_rows = dataset_lock.get("fixed_eval_rows")
    actual_rows = [
        {
            "source_index": selection.get("source_index"),
            "hotpot_id": selection.get("hotpot_id"),
        }
        for selection in manifest.get("selections", [])
        if selection.get("benchmark_split") == "train"
    ]
    actual_eval_rows = [
        {
            "source_index": selection.get("source_index"),
            "hotpot_id": selection.get("hotpot_id"),
        }
        for selection in manifest.get("selections", [])
        if selection.get("benchmark_split") == "test"
    ]
    expected_row_identity = [
        {
            "source_index": row.get("source_index"),
            "hotpot_id": row.get("hotpot_id"),
        }
        for row in (expected_rows or [])
        if isinstance(row, Mapping)
    ]
    expected_eval_identity = [
        {
            "source_index": row.get("source_index"),
            "hotpot_id": row.get("hotpot_id"),
        }
        for row in (expected_eval_rows or [])
        if isinstance(row, Mapping)
    ]
    errors: list[str] = []
    if manifest.get("source_fingerprints") != expected_fingerprints:
        errors.append("source_fingerprints")
    if actual_rows != expected_row_identity:
        errors.append("fixed_train_rows")
    if actual_eval_rows != expected_eval_identity:
        errors.append("fixed_eval_rows")
    if manifest.get("split_sizes", {}).get("train") != len(expected_rows or []):
        errors.append("smoke_train_size")

    if config is not None:
        try:
            taskset = _value_at_path(config, "buffer.explorer_input.taskset")
        except KeyError:
            taskset = None
        if isinstance(taskset, Mapping):
            if taskset.get("row_indices") != [row["source_index"] for row in expected_rows or []]:
                errors.append("config_row_indices")
            if taskset.get("expected_row_ids") != [row["hotpot_id"] for row in expected_rows or []]:
                errors.append("config_expected_row_ids")
            if taskset.get("expected_dataset_fingerprint") != (expected_fingerprints or {}).get("train"):
                errors.append("config_train_fingerprint")
        else:
            errors.append("config_taskset")

    if errors:
        gates.add(
            "manifest.contract",
            FAIL,
            "M5 manifest, lock, and E1 config are not aligned",
            mismatches=errors,
        )
    else:
        gates.add(
            "manifest.contract",
            PASS,
            "M5 manifest, lock, and configs select the same train and held-out rows",
            fixed_train_rows=len(actual_rows),
            fixed_eval_rows=len(actual_eval_rows),
        )
    return manifest


def _check_git(
    repository_root: Path,
    *,
    mode: str,
    expected_commit: Optional[str],
    gates: GateBook,
) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "commit": None,
        "branch": None,
        "dirty": None,
        "dirty_entry_count": None,
        "expected_commit": expected_commit,
    }
    try:
        commit_result = _run_command(
            ["git", "rev-parse", "HEAD"], cwd=repository_root
        )
        branch_result = _run_command(
            ["git", "branch", "--show-current"], cwd=repository_root
        )
        status_result = _run_command(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        gates.add("git.state", FAIL, f"unable to inspect Git: {type(exc).__name__}")
        return inventory

    if any(result.returncode != 0 for result in (commit_result, branch_result, status_result)):
        gates.add("git.state", FAIL, "repository Git inspection failed")
        return inventory

    commit = commit_result.stdout.strip()
    branch = branch_result.stdout.strip()
    dirty_lines = [line for line in status_result.stdout.splitlines() if line.strip()]
    inventory.update(
        {
            "commit": commit,
            "branch": branch,
            "dirty": bool(dirty_lines),
            "dirty_entry_count": len(dirty_lines),
        }
    )
    dirty_status = FAIL if mode == "autodl" and dirty_lines else (WARN if dirty_lines else PASS)
    gates.add(
        "git.clean",
        dirty_status,
        "worktree is clean" if not dirty_lines else "worktree contains uncommitted files",
        dirty_entry_count=len(dirty_lines),
    )

    if expected_commit is None:
        gates.add(
            "git.commit",
            FAIL if mode == "autodl" else SKIP,
            "an expected commit is required on AutoDL" if mode == "autodl" else "no expected commit supplied for local audit",
            actual=commit,
        )
    elif not _FULL_COMMIT_PATTERN.fullmatch(expected_commit):
        gates.add(
            "git.commit",
            FAIL,
            "expected commit must be a lowercase 40-character hexadecimal ID",
            actual=commit,
        )
    elif commit != expected_commit:
        gates.add(
            "git.commit",
            FAIL,
            "checked-out commit does not match the frozen source commit",
            actual=commit,
            expected=expected_commit,
        )
    else:
        gates.add(
            "git.commit",
            PASS,
            "checked-out commit matches the frozen source commit",
            commit=commit,
        )
    return inventory


def _check_credential_isolation(
    repository_root: Path,
    lock: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    mode: str,
    gates: GateBook,
) -> dict[str, bool]:
    credentials = lock.get("credentials")
    if not isinstance(credentials, dict):
        gates.add("credentials.lock", FAIL, "credential policy is missing from lock")
        return {}
    required_env = credentials.get("required_env")
    forbidden_paths = credentials.get("forbidden_repo_paths")
    if not isinstance(required_env, list) or not isinstance(forbidden_paths, list):
        gates.add("credentials.lock", FAIL, "credential policy has invalid fields")
        return {}

    presence = {
        str(key): _secret_key_present(environment, str(key)) for key in required_env
    }
    missing = sorted(key for key, present in presence.items() if not present)
    if missing:
        gates.add(
            "credentials.environment",
            FAIL if mode == "autodl" else WARN,
            "required secret environment variables are not present",
            missing=missing,
            presence=presence,
        )
    else:
        gates.add(
            "credentials.environment",
            PASS,
            "required secret names are present in the environment",
            presence=presence,
        )

    existing: list[str] = []
    for relative_path in forbidden_paths:
        relative_text = str(relative_path)
        if relative_text == ".env":
            matches = (
                candidate
                for candidate in repository_root.rglob(".env")
                if ".git" not in candidate.relative_to(repository_root).parts
            )
        else:
            matches = iter((repository_root / relative_text,))
        for candidate in matches:
            if candidate.exists() or candidate.is_symlink():
                existing.append(
                    candidate.relative_to(repository_root).as_posix()
                )
    existing = sorted(set(existing))
    if existing:
        gates.add(
            "credentials.repository",
            FAIL if mode == "autodl" else WARN,
            "local credential files exist and must not be transferred to AutoDL",
            paths=existing,
        )
    else:
        gates.add(
            "credentials.repository",
            PASS,
            "no forbidden credential file exists in the checkout",
        )
    return presence


def _path_inventory(path: Optional[Path]) -> Optional[str]:
    return str(path.resolve()) if path is not None else None


def _check_model(
    model_path: Optional[Path],
    model_revision: Optional[str],
    lock: Mapping[str, Any],
    *,
    mode: str,
    gates: GateBook,
) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "repository_id": None,
        "revision": model_revision,
        "manifest_sha256": None,
        "file_count": 0,
        "total_size_bytes": 0,
    }
    if model_path is None or not model_path.is_dir():
        return inventory

    model_lock = lock.get("model")
    if not isinstance(model_lock, Mapping):
        gates.add("model.identity", FAIL, "model identity lock is missing")
        return inventory

    expected_repository = model_lock.get("repository_id")
    expected_config = model_lock.get("config_assertions")
    required_files = model_lock.get("required_files")
    minimum_weight_bytes = model_lock.get("minimum_weight_bytes")
    manifest_filename = model_lock.get(
        "manifest_filename", ".agemem_model_manifest.json"
    )
    if (
        not isinstance(expected_repository, str)
        or not expected_repository
        or not isinstance(expected_config, Mapping)
        or not isinstance(required_files, list)
        or not all(isinstance(item, str) and item for item in required_files)
        or not isinstance(minimum_weight_bytes, int)
        or isinstance(minimum_weight_bytes, bool)
        or minimum_weight_bytes <= 0
        or not isinstance(manifest_filename, str)
        or not manifest_filename
    ):
        gates.add("model.identity", FAIL, "model identity lock is invalid")
        return inventory

    errors: list[str] = []
    try:
        model_config = _load_json_object(model_path / "config.json")
        tokenizer_config = _load_json_object(
            model_path / "tokenizer_config.json"
        )
    except Exception as exc:
        gates.add(
            "model.identity",
            FAIL if mode == "autodl" else WARN,
            f"unable to parse model metadata: {type(exc).__name__}",
        )
        return inventory

    for dotted_path, expected in expected_config.items():
        try:
            actual = _value_at_path(model_config, str(dotted_path))
        except KeyError:
            errors.append(f"config:{dotted_path}:missing")
            continue
        if actual != expected:
            errors.append(f"config:{dotted_path}:mismatch")
    if not isinstance(tokenizer_config.get("chat_template"), str) or not tokenizer_config[
        "chat_template"
    ].strip():
        errors.append("tokenizer_config:chat_template")
    for relative_path in required_files:
        artifact = (model_path / relative_path).resolve()
        if not _is_relative_to(artifact, model_path.resolve()) or not artifact.is_file():
            errors.append(f"required_file:{relative_path}")

    manifest_path = (model_path / manifest_filename).resolve()
    if not _is_relative_to(manifest_path, model_path.resolve()):
        errors.append("manifest:path")
        manifest = None
    else:
        try:
            manifest = _load_json_object(manifest_path)
        except Exception as exc:
            errors.append(f"manifest:{type(exc).__name__}")
            manifest = None

    if manifest is not None:
        inventory["manifest_sha256"] = sha256_file(manifest_path)
        inventory["repository_id"] = manifest.get("repository_id")
        inventory["revision"] = manifest.get("revision")
        if manifest.get("schema_version") != _MODEL_MANIFEST_SCHEMA_VERSION:
            errors.append("manifest:schema_version")
        if manifest.get("repository_id") != expected_repository:
            errors.append("manifest:repository_id")
        manifest_revision = manifest.get("revision")
        if (
            not isinstance(manifest_revision, str)
            or not _FULL_COMMIT_PATTERN.fullmatch(manifest_revision)
        ):
            errors.append("manifest:revision")
        if model_revision != manifest_revision:
            errors.append("manifest:expected_revision")

        file_entries = manifest.get("files")
        if not isinstance(file_entries, list) or not file_entries:
            errors.append("manifest:files")
            file_entries = []
        seen_paths: set[str] = set()
        total_size = 0
        weight_size = 0
        weight_paths: set[str] = set()
        for entry in file_entries:
            if not isinstance(entry, Mapping):
                errors.append("manifest:file_entry")
                continue
            relative_path = entry.get("path")
            expected_size = entry.get("size_bytes")
            expected_sha256 = entry.get("sha256")
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or relative_path in seen_paths
                or not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size <= 0
                or not isinstance(expected_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            ):
                errors.append("manifest:file_entry")
                continue
            seen_paths.add(relative_path)
            artifact = (model_path / relative_path).resolve()
            if (
                not _is_relative_to(artifact, model_path.resolve())
                or not artifact.is_file()
                or artifact.stat().st_size != expected_size
                or sha256_file(artifact) != expected_sha256
            ):
                errors.append(f"manifest:file:{relative_path}")
                continue
            total_size += expected_size
            if relative_path.endswith(".safetensors"):
                weight_size += expected_size
                weight_paths.add(relative_path)
        try:
            actual_paths = {
                candidate.relative_to(model_path).as_posix()
                for candidate in model_path.rglob("*")
                if candidate.resolve() != manifest_path
                and not any(
                    part in _MODEL_MANIFEST_IGNORED_DIRECTORIES
                    for part in candidate.relative_to(model_path).parts
                )
                and candidate.is_file()
            }
        except OSError:
            actual_paths = set()
            errors.append("manifest:file_inventory")
        if actual_paths != seen_paths:
            errors.append("manifest:file_inventory")
        if not set(required_files).issubset(seen_paths):
            errors.append("manifest:required_files")
        if manifest.get("file_count") != len(seen_paths):
            errors.append("manifest:file_count")
        if manifest.get("total_size_bytes") != total_size:
            errors.append("manifest:total_size_bytes")
        inventory["file_count"] = len(seen_paths)
        inventory["total_size_bytes"] = total_size
        if weight_size < minimum_weight_bytes:
            errors.append("manifest:minimum_weight_bytes")

        index_path = model_path / "model.safetensors.index.json"
        if index_path.is_file():
            try:
                index = _load_json_object(index_path)
                indexed_shards = set(index.get("weight_map", {}).values())
            except Exception:
                indexed_shards = set()
            if not indexed_shards or not indexed_shards.issubset(weight_paths):
                errors.append("manifest:weight_index")
        elif "model.safetensors" not in weight_paths:
            errors.append("manifest:weight_index")

    status = FAIL if errors and mode == "autodl" else (WARN if errors else PASS)
    gates.add(
        "model.identity",
        status,
        "model provenance, structure, and file hashes match"
        if not errors
        else "model identity or content does not match the frozen contract",
        repository_id=inventory["repository_id"],
        revision=inventory["revision"],
        mismatches=sorted(set(errors)),
    )
    return inventory


def _check_paths(
    *,
    mode: str,
    model_path: Optional[Path],
    model_revision: Optional[str],
    dataset_path: Optional[Path],
    checkpoint_root: Optional[Path],
    lock: Mapping[str, Any],
    gates: GateBook,
) -> dict[str, Any]:
    path_lock = lock.get("paths") if isinstance(lock.get("paths"), dict) else {}
    persistent_prefix_raw = path_lock.get("autodl_persistent_prefix")
    persistent_prefix = (
        Path(persistent_prefix_raw).resolve()
        if isinstance(persistent_prefix_raw, str)
        else None
    )
    inventory = {
        "model": _path_inventory(model_path),
        "dataset": _path_inventory(dataset_path),
        "checkpoint_root": _path_inventory(checkpoint_root),
        "autodl_persistent_prefix": (
            str(persistent_prefix) if persistent_prefix is not None else None
        ),
    }

    for name, path in (
        ("model", model_path),
        ("dataset", dataset_path),
        ("checkpoint_root", checkpoint_root),
    ):
        if path is None:
            gates.add(
                f"path.{name}",
                FAIL if mode == "autodl" or name == "dataset" else SKIP,
                f"{name} path is not configured",
            )
            continue
        resolved = path.resolve()
        if not resolved.is_dir():
            gates.add(
                f"path.{name}",
                FAIL if mode == "autodl" or name == "dataset" else WARN,
                f"{name} path is not an existing directory",
                path=str(resolved),
            )
            continue
        if mode == "autodl" and (
            persistent_prefix is None or not _is_relative_to(resolved, persistent_prefix)
        ):
            gates.add(
                f"path.{name}",
                FAIL,
                f"{name} path is outside the frozen AutoDL persistent root",
                path=str(resolved),
                persistent_prefix=str(persistent_prefix),
            )
            continue
        gates.add(
            f"path.{name}",
            PASS,
            f"{name} path exists" + (" on persistent storage" if mode == "autodl" else ""),
            path=str(resolved),
        )

    inventory["model_identity"] = _check_model(
        model_path,
        model_revision,
        lock,
        mode=mode,
        gates=gates,
    )

    if checkpoint_root is not None and checkpoint_root.is_dir():
        writable = os.access(checkpoint_root, os.W_OK)
        free_gib = round(shutil.disk_usage(checkpoint_root).free / (1024**3), 3)
        minimum_free = float(path_lock.get("minimum_checkpoint_free_gib", 0))
        if not writable or free_gib < minimum_free:
            gates.add(
                "checkpoint.capacity",
                FAIL if mode == "autodl" else WARN,
                "checkpoint root is not writable or lacks free space",
                writable=writable,
                free_gib=free_gib,
                minimum_free_gib=minimum_free,
            )
        else:
            gates.add(
                "checkpoint.capacity",
                PASS,
                "checkpoint root is writable and has sufficient free space",
                free_gib=free_gib,
                minimum_free_gib=minimum_free,
            )
        clean_jobs = path_lock.get("clean_job_relative_paths", [])
        if not isinstance(clean_jobs, list) or any(
            not isinstance(relative_path, str) or not relative_path
            for relative_path in clean_jobs
        ):
            gates.add(
                "checkpoint.jobs",
                FAIL,
                "clean_job_relative_paths must be a list of non-empty strings",
            )
        else:
            nonempty_jobs = []
            for relative_path in clean_jobs:
                job_candidate = checkpoint_root / relative_path
                current = checkpoint_root
                has_symlink_component = False
                for component in Path(relative_path).parts:
                    current = current / component
                    if current.is_symlink():
                        has_symlink_component = True
                        break
                job_path = job_candidate.resolve()
                if not _is_relative_to(job_path, checkpoint_root.resolve()):
                    nonempty_jobs.append(relative_path)
                    continue
                if has_symlink_component or (
                    job_candidate.exists()
                    and (
                        not job_candidate.is_dir()
                        or any(job_candidate.iterdir())
                    )
                ):
                    nonempty_jobs.append(relative_path)
            status = FAIL if mode == "autodl" and nonempty_jobs else (
                WARN if nonempty_jobs else PASS
            )
            gates.add(
                "checkpoint.jobs",
                status,
                "smoke job directories are clean"
                if not nonempty_jobs
                else "a smoke job directory is non-empty and could trigger an implicit rename or stale reload",
                nonempty_jobs=nonempty_jobs,
            )
    return inventory


def _check_dataset(
    dataset_path: Optional[Path],
    lock: Mapping[str, Any],
    gates: GateBook,
) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "kind": None,
        "split_sizes": {},
        "source_fingerprints": {},
        "selected_train_ids": [],
        "selected_eval_ids": [],
        "selected_content_sha256": {},
    }
    if dataset_path is None or not dataset_path.is_dir():
        gates.add("dataset.identity", FAIL, "HotpotQA DatasetDict is unavailable")
        return inventory
    if not (dataset_path / "dataset_dict.json").is_file():
        gates.add(
            "dataset.identity",
            FAIL,
            "dataset path is not a Hugging Face saved DatasetDict",
        )
        return inventory

    try:
        from datasets import DatasetDict, load_from_disk

        dataset = load_from_disk(str(dataset_path))
    except Exception as exc:
        gates.add(
            "dataset.identity",
            FAIL,
            f"unable to load saved HotpotQA data: {type(exc).__name__}",
        )
        return inventory

    if not isinstance(dataset, DatasetDict):
        gates.add("dataset.identity", FAIL, "saved HotpotQA data is not a DatasetDict")
        return inventory

    dataset_lock = lock.get("dataset") if isinstance(lock.get("dataset"), dict) else {}
    expected_sizes = dataset_lock.get("source_split_sizes", {})
    expected_fingerprints = dataset_lock.get("source_fingerprints", {})
    expected_rows = dataset_lock.get("fixed_train_rows", [])
    expected_eval_rows = dataset_lock.get("fixed_eval_rows", [])
    inventory["kind"] = "DatasetDict"
    inventory["split_sizes"] = {
        split: len(dataset[split]) for split in sorted(dataset.keys())
    }
    inventory["source_fingerprints"] = {
        split: getattr(dataset[split], "_fingerprint", None)
        for split in sorted(dataset.keys())
    }

    errors: list[str] = []
    if inventory["split_sizes"] != expected_sizes:
        errors.append("split_sizes")
    if inventory["source_fingerprints"] != expected_fingerprints:
        errors.append("source_fingerprints")
    for label, split, rows in (
        ("train", "train", expected_rows),
        ("eval", "validation", expected_eval_rows),
    ):
        try:
            if not isinstance(rows, list) or not rows:
                raise TypeError("row lock must be a non-empty list")
            indices = [int(row["source_index"]) for row in rows]
            expected_ids = [str(row["hotpot_id"]) for row in rows]
            expected_hashes = [str(row["content_sha256"]) for row in rows]
            selected = dataset[split].select(indices)
            selected_records = [selected[index] for index in range(len(selected))]
            selected_ids = [str(record["id"]) for record in selected_records]
            selected_hashes = [
                _canonical_json_sha256(record) for record in selected_records
            ]
        except Exception as exc:
            errors.append(f"fixed_{label}_rows:{type(exc).__name__}")
            selected_ids = []
            selected_hashes = []
            expected_ids = [
                str(row.get("hotpot_id"))
                for row in rows
                if isinstance(row, Mapping)
            ]
            expected_hashes = [
                str(row.get("content_sha256"))
                for row in rows
                if isinstance(row, Mapping)
            ]
        inventory[f"selected_{label}_ids"] = selected_ids
        inventory["selected_content_sha256"][label] = selected_hashes
        if selected_ids != expected_ids:
            errors.append(f"fixed_{label}_ids")
        if selected_hashes != expected_hashes:
            errors.append(f"fixed_{label}_content")

    if errors:
        gates.add(
            "dataset.identity",
            FAIL,
            "HotpotQA source identity does not match the frozen M5 manifest",
            mismatches=errors,
        )
    else:
        gates.add(
            "dataset.identity",
            PASS,
            "HotpotQA splits, fingerprints, IDs, and selected row contents match",
            train_rows=len(dataset["train"]),
            selected_train_rows=len(inventory["selected_train_ids"]),
            selected_eval_rows=len(inventory["selected_eval_ids"]),
        )
    return inventory


def _check_python(mode: str, gates: GateBook) -> dict[str, Any]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    supported = (3, 10) <= sys.version_info[:2] < (3, 13)
    gates.add(
        "runtime.python",
        PASS if supported else FAIL,
        "Python version is supported" if supported else "Python must be >=3.10,<3.13",
        version=version,
        executable=sys.executable,
        mode=mode,
    )
    return {"version": version, "executable": sys.executable}


def _version_satisfies(version: str, specifier: str) -> bool:
    if not specifier:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError as exc:  # pragma: no cover - pip installs packaging
        raise RuntimeError("packaging is required for version checks") from exc
    return Version(version) in SpecifierSet(specifier)


def _probe_import(import_name: str, repository_root: Path) -> bool:
    code = (
        "import importlib,sys; "
        "importlib.import_module(sys.argv[1]); "
        "print('AGEMEM_IMPORT_OK')"
    )
    try:
        result = _run_command(
            [sys.executable, "-c", code, import_name],
            cwd=repository_root,
            timeout=60.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "AGEMEM_IMPORT_OK" in result.stdout


def _check_packages(
    repository_root: Path,
    lock: Mapping[str, Any],
    *,
    mode: str,
    gates: GateBook,
) -> dict[str, Any]:
    runtime_lock = lock.get("runtime") if isinstance(lock.get("runtime"), dict) else {}
    packages = runtime_lock.get("packages")
    inventory: dict[str, Any] = {}
    if not isinstance(packages, list):
        gates.add("runtime.packages", FAIL, "runtime package lock is missing")
        return inventory

    for package in packages:
        if not isinstance(package, dict):
            gates.add("runtime.package.invalid", FAIL, "runtime package entry is invalid")
            continue
        distribution = str(package.get("distribution", ""))
        import_name = str(package.get("import_name", ""))
        specifier = str(package.get("specifier", ""))
        required_mode = str(package.get("required_mode", "all"))
        required = required_mode == "all" or mode == required_mode
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        inventory[distribution] = {
            "version": version,
            "specifier": specifier,
            "import_name": import_name,
            "required": required,
            "import_ok": None,
        }
        if version is None:
            gates.add(
                f"runtime.package.{distribution}",
                FAIL if required else SKIP,
                "required distribution is not installed" if required else "AutoDL-only distribution is not installed locally",
                specifier=specifier,
            )
            continue
        try:
            satisfies = _version_satisfies(version, specifier)
        except Exception as exc:
            gates.add(
                f"runtime.package.{distribution}",
                FAIL,
                f"unable to compare package version: {type(exc).__name__}",
            )
            continue
        if not satisfies:
            gates.add(
                f"runtime.package.{distribution}",
                FAIL if required else WARN,
                "installed version is outside the frozen range",
                version=version,
                specifier=specifier,
            )
            continue
        import_ok: Optional[bool] = None
        if required and import_name:
            import_ok = _probe_import(import_name, repository_root)
            inventory[distribution]["import_ok"] = import_ok
        if import_ok is False:
            gates.add(
                f"runtime.package.{distribution}",
                FAIL,
                "distribution metadata is valid but its module cannot be imported",
                version=version,
                import_name=import_name,
            )
        else:
            gates.add(
                f"runtime.package.{distribution}",
                PASS,
                "package version and import gate passed",
                version=version,
                specifier=specifier,
                import_checked=bool(import_name and required),
            )
    return inventory


def _check_trinity_config_schema(
    repository_root: Path,
    lock: Mapping[str, Any],
    *,
    mode: str,
    gates: GateBook,
) -> dict[str, Any]:
    """Validate all three YAML files against Trinity's structured schema."""

    source_files = lock.get("source_files")
    source_names = ("config", "e0_config", "checkpoint_eval_config")
    inventory: dict[str, Any] = {}
    if not isinstance(source_files, Mapping):
        gates.add("config.trinity_schema", FAIL, "source lock is unavailable")
        return inventory
    try:
        from trinity.common.config import load_config
    except Exception as exc:
        gates.add(
            "config.trinity_schema",
            FAIL if mode == "autodl" else SKIP,
            f"Trinity Config cannot be imported: {type(exc).__name__}",
        )
        return inventory

    errors: list[dict[str, str]] = []
    for source_name in source_names:
        declaration = source_files.get(source_name)
        relative_path = (
            declaration.get("path")
            if isinstance(declaration, Mapping)
            else None
        )
        if not isinstance(relative_path, str):
            errors.append({"source": source_name, "error": "missing_path"})
            continue
        path = repository_root / relative_path
        try:
            parsed = load_config(str(path))
        except Exception as exc:
            errors.append(
                {"source": source_name, "error": type(exc).__name__}
            )
            continue
        inventory[source_name] = {
            "mode": parsed.mode,
            "project": parsed.project,
            "name": parsed.name,
        }

    gates.add(
        "config.trinity_schema",
        FAIL if errors else PASS,
        "all run configs satisfy Trinity's structured schema"
        if not errors
        else "a run config does not satisfy Trinity's structured schema",
        errors=errors,
    )
    return inventory


def _query_nvidia_smi(repository_root: Path) -> list[dict[str, Any]]:
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        cwd=repository_root,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi returned a non-zero status")
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=4)]
        if len(parts) != 5:
            raise ValueError("unexpected nvidia-smi CSV")
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_mib": int(parts[3]),
                "free_memory_mib": int(parts[4]),
            }
        )
    return gpus


def _query_torch_cuda(repository_root: Path) -> dict[str, Any]:
    code = (
        "import json,torch; "
        "devices=[torch.cuda.get_device_properties(i) "
        "for i in range(torch.cuda.device_count())]; "
        "print('AGEMEM_CUDA_JSON=' + json.dumps({"
        "'torch_version':torch.__version__,"
        "'torch_cuda_version':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),"
        "'device_count':torch.cuda.device_count(),"
        "'devices':[{'index':i,'name':p.name,"
        "'uuid':str(getattr(p,'uuid','')) or None,"
        "'memory_mib':int(p.total_memory // (1024 * 1024))} "
        "for i,p in enumerate(devices)]}))"
    )
    result = _run_command(
        [sys.executable, "-c", code], cwd=repository_root, timeout=60.0
    )
    if result.returncode != 0:
        raise RuntimeError("PyTorch CUDA probe failed")
    prefix = "AGEMEM_CUDA_JSON="
    lines = [line for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError("PyTorch CUDA probe did not return structured output")
    payload = json.loads(lines[0][len(prefix) :])
    if not isinstance(payload, dict):
        raise TypeError("PyTorch CUDA probe returned an invalid object")
    return payload


def _check_gpu(
    repository_root: Path,
    lock: Mapping[str, Any],
    *,
    mode: str,
    gates: GateBook,
) -> dict[str, Any]:
    if mode != "autodl":
        gates.add("gpu.topology", SKIP, "GPU checks are deferred to AutoDL mode")
        return {"nvidia_smi": [], "torch_cuda": None}

    gpu_lock = lock.get("gpu") if isinstance(lock.get("gpu"), dict) else {}
    minimum_count = int(gpu_lock.get("minimum_count", 2))
    minimum_memory_mib = int(gpu_lock.get("minimum_memory_mib", 0))
    minimum_free_memory_mib = int(
        gpu_lock.get("minimum_free_memory_mib", 0)
    )
    exact_count = bool(gpu_lock.get("require_exact_count", True))
    try:
        gpus = _query_nvidia_smi(repository_root)
    except Exception as exc:
        gates.add("gpu.nvidia_smi", FAIL, f"GPU inventory failed: {type(exc).__name__}")
        gpus = []
    else:
        count_ok = (
            len(gpus) == minimum_count if exact_count else len(gpus) >= minimum_count
        )
        enough = count_ok and all(
            gpu["memory_mib"] >= minimum_memory_mib
            and gpu["free_memory_mib"] >= minimum_free_memory_mib
            for gpu in gpus
        )
        gates.add(
            "gpu.nvidia_smi",
            PASS if enough else FAIL,
            "GPU count and memory meet the two-card smoke lock" if enough else "GPU count or memory is below the two-card smoke lock",
            gpu_count=len(gpus),
            minimum_count=minimum_count,
            minimum_memory_mib=minimum_memory_mib,
            minimum_free_memory_mib=minimum_free_memory_mib,
        )

    try:
        torch_cuda = _query_torch_cuda(repository_root)
    except Exception as exc:
        gates.add("gpu.torch_cuda", FAIL, f"PyTorch CUDA probe failed: {type(exc).__name__}")
        torch_cuda = None
    else:
        torch_devices = torch_cuda.get("devices", [])
        torch_count = int(torch_cuda.get("device_count", 0))
        count_ok = (
            torch_count == minimum_count
            if exact_count
            else torch_count >= minimum_count
        )
        nvidia_by_uuid = {
            str(gpu.get("uuid", "")).lower(): gpu for gpu in gpus
        }
        mapped_devices = []
        for device in torch_devices if isinstance(torch_devices, list) else []:
            uuid_value = str(device.get("uuid", "")).lower()
            gpu = nvidia_by_uuid.get(uuid_value)
            mapped_devices.append((device, gpu))
        valid = (
            bool(torch_cuda.get("cuda_available"))
            and count_ok
            and len(mapped_devices) == torch_count
            and all(
                gpu is not None
                and int(device.get("memory_mib", 0)) >= minimum_memory_mib
                and int(gpu.get("free_memory_mib", 0))
                >= minimum_free_memory_mib
                for device, gpu in mapped_devices
            )
        )
        gates.add(
            "gpu.torch_cuda",
            PASS if valid else FAIL,
            "PyTorch sees the required CUDA devices" if valid else "PyTorch cannot see the required CUDA devices",
            **torch_cuda,
            mapped_nvidia_uuids=[
                gpu.get("uuid") if gpu is not None else None
                for _device, gpu in mapped_devices
            ],
        )
    return {"nvidia_smi": gpus, "torch_cuda": torch_cuda}


def build_preflight_report(
    *,
    repository_root: Path,
    config_path: Path,
    lock_path: Path,
    mode: str,
    expected_commit: Optional[str],
    model_path: Optional[Path],
    model_revision: Optional[str],
    dataset_path: Optional[Path],
    checkpoint_root: Optional[Path],
    environment: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Run all read-only checks and return a deterministic JSON-compatible report."""

    if mode not in {"local", "autodl"}:
        raise ValueError("mode must be 'local' or 'autodl'")
    root = repository_root.resolve()
    config = config_path.resolve()
    lock_file = lock_path.resolve()
    env = os.environ if environment is None else environment
    gates = GateBook()
    inventory: MutableMapping[str, Any] = {}

    inventory["git"] = _check_git(
        root, mode=mode, expected_commit=expected_commit, gates=gates
    )
    lock = _check_lock(lock_file, config, root, gates)
    if lock is None:
        lock = {}
        parsed_config = None
        manifest = None
    else:
        parsed_config = _check_config(config, lock, gates)
        manifest = _check_manifest_consistency(root, lock, parsed_config, gates)
    inventory["config"] = {
        "path": str(config),
        "sha256": _source_digest(config) if config.is_file() else None,
        "digest_mode": "utf8_lf",
    }
    inventory["lock"] = {
        "path": str(lock_file),
        "sha256": sha256_file(lock_file) if lock_file.is_file() else None,
    }
    inventory["smoke_manifest_schema_version"] = (
        manifest.get("schema_version") if isinstance(manifest, dict) else None
    )
    inventory["credential_presence"] = _check_credential_isolation(
        root, lock, env, mode=mode, gates=gates
    )
    inventory["paths"] = _check_paths(
        mode=mode,
        model_path=model_path,
        model_revision=model_revision,
        dataset_path=dataset_path,
        checkpoint_root=checkpoint_root,
        lock=lock,
        gates=gates,
    )
    inventory["dataset"] = _check_dataset(dataset_path, lock, gates)
    inventory["python"] = _check_python(mode, gates)
    inventory["packages"] = _check_packages(
        root, lock, mode=mode, gates=gates
    )
    inventory["trinity_config_schema"] = _check_trinity_config_schema(
        root,
        lock,
        mode=mode,
        gates=gates,
    )
    inventory["gpu"] = _check_gpu(root, lock, mode=mode, gates=gates)

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "mode": mode,
        "status": PASS if gates.passed else FAIL,
        "checks": [result.to_dict() for result in gates.results],
        "inventory": dict(inventory),
    }


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Atomically write a report with owner-only permissions where supported."""

    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        report,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, output)


def _resolve_optional_path(
    explicit: Optional[str],
    environment: Mapping[str, str],
    environment_key: str,
    fallback: Optional[Path] = None,
) -> Optional[Path]:
    if explicit:
        return Path(explicit).expanduser()
    raw = environment.get(environment_key)
    if raw:
        return Path(raw).expanduser()
    if fallback is not None and fallback.is_dir():
        return fallback
    return None


def _summarize(report: Mapping[str, Any], output_path: Optional[Path]) -> str:
    checks = report.get("checks", [])
    counts = {
        status: sum(1 for check in checks if check.get("status") == status)
        for status in (PASS, FAIL, WARN, SKIP)
    }
    destination = f"; report={output_path.resolve()}" if output_path else ""
    return (
        f"M8b preflight {str(report.get('status')).upper()}: "
        f"pass={counts[PASS]} fail={counts[FAIL]} "
        f"warn={counts[WARN]} skip={counts[SKIP]}{destination}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run read-only M8b source, data, runtime, and GPU gates."
    )
    parser.add_argument("--mode", choices=("local", "autodl"), default="local")
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--output", default=DEFAULT_REPORT)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    arguments = parser.parse_args(argv)

    inferred_root = Path(__file__).resolve().parents[2]
    repository_root = Path(arguments.repository_root or inferred_root).resolve()
    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    lock_path = Path(arguments.lock)
    if not lock_path.is_absolute():
        lock_path = repository_root / lock_path

    environment = os.environ
    model_path = _resolve_optional_path(
        arguments.model_path, environment, "TRINITY_MODEL_PATH"
    )
    model_revision = arguments.model_revision or environment.get(
        "TRINITY_MODEL_REVISION"
    )
    dataset_path = _resolve_optional_path(
        arguments.dataset_path,
        environment,
        "HOTPOTQA_PATH",
        repository_root.parent / "data" / "hotpot_qa" / "fullwiki",
    )
    checkpoint_root = _resolve_optional_path(
        arguments.checkpoint_root, environment, "TRINITY_CHECKPOINT_ROOT_DIR"
    )

    report = build_preflight_report(
        repository_root=repository_root,
        config_path=config_path,
        lock_path=lock_path,
        mode=arguments.mode,
        expected_commit=arguments.expected_commit,
        model_path=model_path,
        model_revision=model_revision,
        dataset_path=dataset_path,
        checkpoint_root=checkpoint_root,
        environment=environment,
    )
    output_path: Optional[Path] = None
    if not arguments.no_write:
        output_path = Path(arguments.output)
        if not output_path.is_absolute():
            output_path = repository_root / output_path
        write_report(report, output_path)

    if arguments.print_json:
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(_summarize(report, output_path))
        for check in report["checks"]:
            if check["status"] in {FAIL, WARN}:
                print(f"[{check['status'].upper()}] {check['name']}: {check['message']}")
    return 0 if report["status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LOCK_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "build_preflight_report",
    "main",
    "sha256_file",
    "write_report",
]
