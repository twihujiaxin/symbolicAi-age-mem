"""Frozen auxiliary-provider contract and payload-free usage telemetry."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TypeVar
from urllib.parse import urlsplit

from trinity.common.constants import LOG_DIR_ENV_VAR


AUXILIARY_PROVIDER_SCHEMA_VERSION = "agemem.auxiliary_provider.v1"
AUXILIARY_PROVIDER_CALL_SCHEMA_VERSION = "agemem.auxiliary_provider_call.v1"
AUXILIARY_PROVIDER_TELEMETRY_PATH_ENV_VAR = (
    "AGEMEM_AUXILIARY_PROVIDER_TELEMETRY_PATH"
)
DEFAULT_DASHSCOPE_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSIONS = 256
DEFAULT_CHAT_MODEL = "qwen-max"

_T = TypeVar("_T")
_R = TypeVar("_R")
_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
)
_LOCAL_PATH_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_PATH_LOCKS_GUARD = threading.Lock()
_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|"
    r"secret|credential|authorization|cookie)\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)
_BEARER_SECRET_PATTERN = re.compile(
    r"(?i)\b(bearer\s+)[a-z0-9._~+/=-]+"
)
_COMMON_API_TOKEN_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:sk|rk|pk)-[a-z0-9][a-z0-9._-]{7,}"
)
_JWT_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9_-])eyJ[a-zA-Z0-9_-]{5,}\."
    r"[a-zA-Z0-9_-]{5,}\.[a-zA-Z0-9_-]{5,}"
)


class AuxiliaryProviderCallError(RuntimeError):
    """A provider failure whose message cannot contain provider payloads."""


class AuxiliaryProviderResponseError(AuxiliaryProviderCallError):
    """A provider returned a response that does not satisfy its contract."""


class AuxiliaryProviderTelemetryError(RuntimeError):
    """A metadata-only provider call record could not be persisted."""


def safe_exception_text(exc: BaseException, max_chars: int = 256) -> str:
    """Return bounded exception text with common credential forms removed."""

    value = str(exc)
    value = _INLINE_SECRET_PATTERN.sub(r"\1[REDACTED]", value)
    value = _BEARER_SECRET_PATTERN.sub(r"\1[REDACTED]", value)
    value = _COMMON_API_TOKEN_PATTERN.sub("[REDACTED]", value)
    value = _JWT_PATTERN.sub("[REDACTED]", value)
    return value[: max(0, int(max_chars))]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_name(value: Any, field: str, *, max_chars: int = 256) -> str:
    if value is None:
        raise ValueError(f"{field} must be a non-empty value")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty value")
    if len(normalized) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    if safe_exception_text(normalized, max_chars) != normalized:
        raise ValueError(f"{field} contains credential-shaped text")
    return normalized


def _safe_error_type(exc: BaseException) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", type(exc).__name__)[:128]
    return name or "Exception"


@dataclass(frozen=True)
class AuxiliaryProviderConfig:
    schema_version: str
    provider: str
    base_url: str
    embedding_model: str
    embedding_dimensions: int
    chat_model: str
    usage_tracking: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "base_url": self.base_url,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "chat_model": self.chat_model,
            "usage_tracking": self.usage_tracking,
        }


@dataclass(frozen=True)
class AuxiliaryProviderCallContext:
    task_id: str
    rollout_id: str
    execution_id: str
    is_eval: bool

    @classmethod
    def create(
        cls,
        *,
        task_id: Any,
        rollout_id: Any,
        execution_id: Any,
        is_eval: bool,
    ) -> "AuxiliaryProviderCallContext":
        if not isinstance(is_eval, bool):
            raise TypeError("is_eval must be a boolean")
        return cls(
            task_id=_bounded_name(task_id, "task_id"),
            rollout_id=_bounded_name(rollout_id, "rollout_id"),
            execution_id=_bounded_name(execution_id, "execution_id"),
            is_eval=is_eval,
        )


def default_auxiliary_provider_config() -> AuxiliaryProviderConfig:
    """Return the existing upstream DashScope behavior as an explicit object."""

    return AuxiliaryProviderConfig(
        schema_version=AUXILIARY_PROVIDER_SCHEMA_VERSION,
        provider="dashscope",
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        chat_model=DEFAULT_CHAT_MODEL,
        usage_tracking=True,
    )


def load_auxiliary_provider_config(
    workflow_args: Mapping[str, Any],
    *,
    required: bool,
) -> AuxiliaryProviderConfig:
    """Parse a strict provider lock without accepting credentials in config."""

    raw = workflow_args.get("auxiliary_provider")
    if raw is None:
        if required:
            raise ValueError(
                "terminal_only requires an explicit auxiliary_provider lock"
            )
        return default_auxiliary_provider_config()
    if not isinstance(raw, Mapping):
        raise TypeError("auxiliary_provider must be a mapping")

    payload = dict(raw)
    required_keys = {
        "schema_version",
        "provider",
        "base_url",
        "embedding_model",
        "embedding_dimensions",
        "chat_model",
        "usage_tracking",
    }
    missing = sorted(required_keys.difference(payload))
    unexpected = sorted(set(payload).difference(required_keys))
    if missing:
        raise ValueError(
            "auxiliary_provider is missing field(s): " + ", ".join(missing)
        )
    if unexpected:
        raise ValueError(
            "auxiliary_provider contains unknown field(s): "
            + ", ".join(unexpected)
        )
    if payload["schema_version"] != AUXILIARY_PROVIDER_SCHEMA_VERSION:
        raise ValueError("unsupported auxiliary_provider schema_version")
    if payload["provider"] != "dashscope":
        raise ValueError("M8b freezes auxiliary_provider.provider to 'dashscope'")

    for key in ("base_url", "embedding_model", "chat_model"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise TypeError(f"auxiliary_provider.{key} must be a non-empty string")
    parsed_url = urlsplit(payload["base_url"])
    if parsed_url.username is not None or parsed_url.password is not None:
        raise ValueError("auxiliary_provider.base_url must not contain credentials")
    dimensions = payload["embedding_dimensions"]
    if (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions <= 0
    ):
        raise TypeError(
            "auxiliary_provider.embedding_dimensions must be a positive integer"
        )
    if not isinstance(payload["usage_tracking"], bool):
        raise TypeError("auxiliary_provider.usage_tracking must be a boolean")
    if required and payload["usage_tracking"] is not True:
        raise ValueError("M8b terminal_only requires auxiliary usage tracking")

    if required:
        frozen_values = {
            "base_url": DEFAULT_DASHSCOPE_BASE_URL,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "embedding_dimensions": DEFAULT_EMBEDDING_DIMENSIONS,
            "chat_model": DEFAULT_CHAT_MODEL,
        }
        mismatches = [
            key for key, value in frozen_values.items() if payload[key] != value
        ]
        if mismatches:
            raise ValueError(
                "M8b terminal_only auxiliary_provider differs from the frozen "
                "contract: " + ", ".join(sorted(mismatches))
            )

    return AuxiliaryProviderConfig(
        schema_version=payload["schema_version"],
        provider=payload["provider"],
        base_url=payload["base_url"],
        embedding_model=payload["embedding_model"],
        embedding_dimensions=dimensions,
        chat_model=payload["chat_model"],
        usage_tracking=payload["usage_tracking"],
    )


def resolve_auxiliary_provider_telemetry_path(
    workflow_args: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Resolve a JSONL path independent from Experience and tool-call traces."""

    args = workflow_args or {}
    explicit_path = args.get("auxiliary_provider_telemetry_path") or os.getenv(
        AUXILIARY_PROVIDER_TELEMETRY_PATH_ENV_VAR
    )
    if explicit_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(explicit_path))))
        if path.suffix.lower() != ".jsonl":
            path = path / "auxiliary_provider_calls.jsonl"
        return str(path.resolve())

    tool_trace_path = args.get("tool_trace_path")
    if tool_trace_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(tool_trace_path))))
        directory = path.parent if path.suffix.lower() == ".jsonl" else path
        return str((directory / "auxiliary_provider_calls.jsonl").resolve())

    log_dir = os.getenv(LOG_DIR_ENV_VAR)
    if not log_dir:
        return None
    log_path = Path(log_dir).resolve()
    job_dir = log_path.parent if log_path.name.lower() == "log" else log_path
    return str(job_dir / "trajectories" / "auxiliary_provider_calls.jsonl")


def _get_local_path_lock(path: str) -> threading.Lock:
    normalized_path = os.path.normcase(os.path.abspath(path))
    with _LOCAL_PATH_LOCKS_GUARD:
        return _LOCAL_PATH_LOCKS.setdefault(normalized_path, threading.Lock())


@contextmanager
def _interprocess_file_lock(path: str):
    lock_path = f"{path}.lock"
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class AuxiliaryProviderTelemetryRecorder:
    """Synchronously append one metadata-only, fsynced JSONL record per call."""

    def __init__(self, path: str) -> None:
        self.path = str(Path(path).resolve())
        self._local_lock = _get_local_path_lock(self.path)
        self._state_lock = threading.Lock()
        self._last_write_error_type: Optional[str] = None
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            with self._state_lock:
                self._last_write_error_type = type(exc).__name__[:128]
            raise AuxiliaryProviderTelemetryError(
                "unable to prepare auxiliary-provider telemetry"
            ) from None

    @property
    def last_write_error_type(self) -> Optional[str]:
        with self._state_lock:
            return self._last_write_error_type

    def record(self, event: Mapping[str, Any]) -> None:
        try:
            line = (
                json.dumps(
                    dict(event),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            with self._local_lock:
                with _interprocess_file_lock(self.path):
                    descriptor = os.open(
                        self.path,
                        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                        0o600,
                    )
                    try:
                        try:
                            os.chmod(self.path, 0o600)
                        except OSError:
                            pass
                        written = 0
                        while written < len(line):
                            count = os.write(descriptor, line[written:])
                            if count <= 0:
                                raise OSError("append made no progress")
                            written += count
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
        except Exception as exc:
            with self._state_lock:
                self._last_write_error_type = type(exc).__name__[:128]
            raise AuxiliaryProviderTelemetryError(
                "unable to persist auxiliary-provider telemetry"
            ) from None


def _usage_mapping(response: Any) -> tuple[Optional[Mapping[str, Any]], str]:
    try:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, Mapping):
            usage = response.get("usage")
        if usage is None:
            return None, "missing"
        if isinstance(usage, Mapping):
            return usage, "reported"
        model_dump = getattr(usage, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            return (
                (dumped, "reported")
                if isinstance(dumped, Mapping)
                else (None, "malformed")
            )
        as_dict = getattr(usage, "dict", None)
        if callable(as_dict):
            dumped = as_dict()
            return (
                (dumped, "reported")
                if isinstance(dumped, Mapping)
                else (None, "malformed")
            )
        values = {
            key: getattr(usage, key)
            for key in _USAGE_KEYS
            if getattr(usage, key, None) is not None
        }
        return (values, "reported") if values else (None, "malformed")
    except Exception:
        return None, "malformed"


def _normalized_usage(response: Any) -> tuple[dict[str, int], str]:
    raw, status = _usage_mapping(response)
    if raw is None:
        return {}, status
    result: dict[str, int] = {}
    malformed = False
    for key in _USAGE_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[key] = value
        else:
            malformed = True
    return result, "malformed" if malformed else status


class AuxiliaryProviderUsageTracker:
    """Track metadata only; prompts, responses, headers, and keys are excluded."""

    def __init__(
        self,
        config: AuxiliaryProviderConfig,
        *,
        telemetry_path: Optional[str] = None,
        telemetry_recorder: Optional[AuxiliaryProviderTelemetryRecorder] = None,
    ) -> None:
        if telemetry_path is not None and telemetry_recorder is not None:
            raise ValueError(
                "provide telemetry_path or telemetry_recorder, not both"
            )
        self.config = config
        self._lock = threading.RLock()
        self._generation = 0
        self._calls: list[dict[str, Any]] = []
        self._next_call_index = 0
        self._context: Optional[AuxiliaryProviderCallContext] = None
        self._telemetry_recorder = telemetry_recorder
        if telemetry_path is not None:
            self._telemetry_recorder = AuxiliaryProviderTelemetryRecorder(
                telemetry_path
            )

    @property
    def telemetry_enabled(self) -> bool:
        return self._telemetry_recorder is not None

    def reset(self) -> None:
        """Atomically discard the current in-memory rollout aggregate."""

        with self._lock:
            self._generation += 1
            self._calls = []
            self._next_call_index = 0
            self._context = None

    def start_rollout(
        self,
        *,
        task_id: Any,
        rollout_id: Any,
        execution_id: Any,
        is_eval: bool,
    ) -> None:
        """Atomically bind a new auditable call-index namespace."""

        context = AuxiliaryProviderCallContext.create(
            task_id=task_id,
            rollout_id=rollout_id,
            execution_id=execution_id,
            is_eval=is_eval,
        )
        with self._lock:
            self._generation += 1
            self._calls = []
            self._next_call_index = 0
            self._context = context

    def _reserve_call(
        self, operation: str, model: str
    ) -> tuple[int, int, Optional[AuxiliaryProviderCallContext]]:
        with self._lock:
            if self._telemetry_recorder is not None and self._context is None:
                raise RuntimeError(
                    "start_rollout must bind provider telemetry before a call"
                )
            call_index = self._next_call_index
            self._next_call_index += 1
            return self._generation, call_index, self._context

    def _event(
        self,
        *,
        context: Optional[AuxiliaryProviderCallContext],
        call_index: int,
        operation: str,
        model: str,
        success: bool,
        outcome: str,
        latency_ms: float,
        usage: Mapping[str, int],
        usage_status: str,
        error_type: Optional[str],
    ) -> dict[str, Any]:
        return {
            "schema_version": AUXILIARY_PROVIDER_CALL_SCHEMA_VERSION,
            "timestamp": _utc_now(),
            "task_id": context.task_id if context is not None else None,
            "rollout_id": context.rollout_id if context is not None else None,
            "execution_id": context.execution_id if context is not None else None,
            "is_eval": context.is_eval if context is not None else None,
            "call_index": call_index,
            "provider": self.config.provider,
            "operation": operation,
            "model": model,
            "success": success,
            "outcome": outcome,
            "latency_ms": latency_ms,
            "usage": dict(usage),
            "usage_status": usage_status,
            "error_type": error_type,
            "cost": {
                "amount": None,
                "currency": None,
                "source": "not_reported_by_openai_compatible_api",
            },
        }

    def _complete_call(self, generation: int, event: dict[str, Any]) -> None:
        with self._lock:
            if generation == self._generation:
                self._calls.append(event)
            recorder = self._telemetry_recorder
        if recorder is not None:
            try:
                recorder.record(event)
            except Exception:
                raise AuxiliaryProviderTelemetryError(
                    "unable to persist auxiliary-provider telemetry"
                ) from None

    def call(
        self,
        *,
        operation: str,
        model: str,
        invoke: Callable[[], _T],
        response_parser: Optional[Callable[[_T], _R]] = None,
    ) -> _T | _R:
        operation = _bounded_name(operation, "operation", max_chars=128)
        model = _bounded_name(model, "model", max_chars=128)
        if not self.config.usage_tracking:
            try:
                response = invoke()
                return response_parser(response) if response_parser else response
            except AuxiliaryProviderResponseError as exc:
                raise exc from None
            except Exception as exc:
                raise AuxiliaryProviderCallError(
                    f"auxiliary provider {operation} failed "
                    f"({_safe_error_type(exc)})"
                ) from None

        generation, call_index, context = self._reserve_call(operation, model)
        started = time.perf_counter()
        response: Any = None
        response_received = False
        usage: dict[str, int] = {}
        usage_status = "missing"
        try:
            response = invoke()
            response_received = True
            usage, usage_status = _normalized_usage(response)
            result = response_parser(response) if response_parser else response
        except Exception as exc:
            outcome = "malformed_response" if response_received else "provider_error"
            error_type = _safe_error_type(exc)
            event = self._event(
                context=context,
                call_index=call_index,
                operation=operation,
                model=model,
                success=False,
                outcome=outcome,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                usage=usage,
                usage_status=usage_status,
                error_type=error_type,
            )
            self._complete_call(generation, event)
            if isinstance(exc, AuxiliaryProviderResponseError):
                raise exc from None
            raise AuxiliaryProviderCallError(
                f"auxiliary provider {operation} failed ({error_type})"
            ) from None

        event = self._event(
            context=context,
            call_index=call_index,
            operation=operation,
            model=model,
            success=True,
            outcome="success",
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            usage=usage,
            usage_status=usage_status,
            error_type=None,
        )
        self._complete_call(generation, event)
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            calls = copy.deepcopy(
                sorted(self._calls, key=lambda item: item["call_index"])
            )
            recorder = self._telemetry_recorder

        usage_totals = {key: 0 for key in _USAGE_KEYS}
        by_operation: dict[str, dict[str, Any]] = {}
        total_latency = 0.0
        failures = 0
        for call in calls:
            total_latency += float(call["latency_ms"])
            failures += int(not call["success"])
            operation = call["operation"]
            aggregate = by_operation.setdefault(
                operation,
                {
                    "calls": 0,
                    "failures": 0,
                    "latency_ms": 0.0,
                    "usage": {key: 0 for key in _USAGE_KEYS},
                },
            )
            aggregate["calls"] += 1
            aggregate["failures"] += int(not call["success"])
            aggregate["latency_ms"] = round(
                aggregate["latency_ms"] + float(call["latency_ms"]), 3
            )
            for key, value in call["usage"].items():
                usage_totals[key] += value
                aggregate["usage"][key] += value

        return {
            "schema_version": AUXILIARY_PROVIDER_SCHEMA_VERSION,
            "provider": self.config.provider,
            "base_url": self.config.base_url,
            "embedding_model": self.config.embedding_model,
            "chat_model": self.config.chat_model,
            "tracking_enabled": self.config.usage_tracking,
            "telemetry_enabled": recorder is not None,
            "telemetry_last_write_error_type": (
                recorder.last_write_error_type if recorder is not None else None
            ),
            "total_calls": len(calls),
            "failed_calls": failures,
            "total_latency_ms": round(total_latency, 3),
            "usage": usage_totals,
            "by_operation": {
                key: by_operation[key] for key in sorted(by_operation)
            },
            "calls": calls,
            "cost": {
                "amount": None,
                "currency": None,
                "source": "not_reported_by_openai_compatible_api",
            },
        }


__all__ = [
    "AUXILIARY_PROVIDER_CALL_SCHEMA_VERSION",
    "AUXILIARY_PROVIDER_SCHEMA_VERSION",
    "AUXILIARY_PROVIDER_TELEMETRY_PATH_ENV_VAR",
    "AuxiliaryProviderCallError",
    "AuxiliaryProviderConfig",
    "AuxiliaryProviderResponseError",
    "AuxiliaryProviderTelemetryError",
    "AuxiliaryProviderTelemetryRecorder",
    "AuxiliaryProviderUsageTracker",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_DASHSCOPE_BASE_URL",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "default_auxiliary_provider_config",
    "load_auxiliary_provider_config",
    "resolve_auxiliary_provider_telemetry_path",
    "safe_exception_text",
]
