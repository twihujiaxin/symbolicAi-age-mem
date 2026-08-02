"""Structured, append-only tool-call tracing.

The recorder is intentionally independent from workflow implementations. Calls
emit start/finish records; later observations such as retrieval use emit a
linked usage record. In a Ray runtime, all workers that target the same path
share one named writer actor so JSONL records cannot interleave. Outside Ray
(for local debugging and unit tests), records are appended with ``os.write``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from trinity.common.constants import LOG_DIR_ENV_VAR

TOOL_TRACE_PATH_ENV_VAR = "AGEMEM_TOOL_TRACE_PATH"
TOOL_TRACE_SCHEMA_VERSION = 2

_LOCAL_PATH_LOCKS: Dict[str, threading.Lock] = {}
_LOCAL_PATH_LOCKS_GUARD = threading.Lock()

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "secret",
    "credential",
    "cookie",
    "token",
}
_INLINE_SECRET_PATTERN = re.compile(
    r"""(?ix)
    (
        ["']?
        (?:[a-z0-9]+[_-]+)*
        (?:api[_-]?key|access[_-]?token|refresh[_-]?token|
           password|secret|credential|token)
        ["']?
        \s*[:=]\s*
    )
    (?:"[^"]*"|'[^']*'|[^\s,;}]+)
    """
)
_AUTH_SECRET_PATTERN = re.compile(r"(?i)\b(authorization|cookie)(\s*[:=]\s*)([^\r\n]+)")
_BEARER_SECRET_PATTERN = re.compile(r"(?i)\b(bearer\s+)[a-z0-9._~+/=-]+")
_COMMON_API_TOKEN_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:sk|rk|pk)-[a-z0-9][a-z0-9._-]{7,}"
)
_GITHUB_TOKEN_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:gh[pousr]_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,})"
)
_JWT_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9_-])eyJ[a-zA-Z0-9_-]{5,}\."
    r"[a-zA-Z0-9_-]{5,}\.[a-zA-Z0-9_-]{5,}"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_inline_secrets(value: str) -> str:
    value = _AUTH_SECRET_PATTERN.sub(r"\1\2[REDACTED]", value)
    value = _INLINE_SECRET_PATTERN.sub(r"\1[REDACTED]", value)
    value = _BEARER_SECRET_PATTERN.sub(r"\1[REDACTED]", value)
    value = _COMMON_API_TOKEN_PATTERN.sub("[REDACTED]", value)
    value = _GITHUB_TOKEN_PATTERN.sub("[REDACTED]", value)
    return _JWT_PATTERN.sub("[REDACTED]", value)


def _safe_exception_text(exc: Exception, max_chars: int = 1024) -> str:
    """Return a bounded, redacted exception string for logs and metadata."""
    return _redact_inline_secrets(str(exc))[:max_chars]


def _is_sensitive_key(key_hint: Optional[str]) -> bool:
    if not key_hint:
        return False
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key_hint.strip())
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", snake_case).strip("_").lower()
    compact = normalized.replace("_", "")
    return any(
        normalized == sensitive_key
        or normalized.endswith(f"_{sensitive_key}")
        or compact.endswith(sensitive_key.replace("_", ""))
        for sensitive_key in _SENSITIVE_KEYS
    )


def _safe_dict_key(key: Any, existing_keys: set[str]) -> str:
    """Redact secret-shaped dictionary keys without silently merging them."""
    base_key = _redact_inline_secrets(str(key))
    safe_key = base_key
    suffix = 2
    while safe_key in existing_keys:
        safe_key = f"{base_key}#{suffix}"
        suffix += 1
    return safe_key


def _json_safe(
    value: Any,
    *,
    max_string_chars: int,
    key_hint: Optional[str] = None,
) -> Any:
    """Return a bounded, JSON-serializable and secret-aware representation."""
    if _is_sensitive_key(key_hint):
        return "[REDACTED]"

    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, str):
        redacted = _redact_inline_secrets(value)
        if len(redacted) <= max_string_chars:
            return redacted
        return {
            "_truncated": True,
            "length": len(redacted),
            "sha256": hashlib.sha256(redacted.encode("utf-8")).hexdigest(),
            "preview": redacted[:max_string_chars],
        }

    if isinstance(value, dict):
        safe_dict: Dict[str, Any] = {}
        for key, item in value.items():
            raw_key = str(key)
            safe_key = _safe_dict_key(key, set(safe_dict))
            safe_dict[safe_key] = _json_safe(
                item,
                max_string_chars=max_string_chars,
                key_hint=raw_key,
            )
        return safe_dict

    if isinstance(value, set):
        value = sorted(value, key=repr)

    if isinstance(value, (list, tuple)):
        return [_json_safe(item, max_string_chars=max_string_chars) for item in value]

    return _json_safe(
        repr(value),
        max_string_chars=max_string_chars,
        key_hint=key_hint,
    )


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _get_local_path_lock(path: Optional[str]) -> threading.Lock:
    if not path:
        return threading.Lock()
    normalized_path = os.path.normcase(os.path.abspath(path))
    with _LOCAL_PATH_LOCKS_GUARD:
        return _LOCAL_PATH_LOCKS.setdefault(
            normalized_path,
            threading.Lock(),
        )


@contextmanager
def _interprocess_file_lock(path: str):
    """Serialize non-Ray appenders across local processes."""
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


def resolve_tool_trace_path(
    workflow_args: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve the JSONL path from workflow args, env, or Trinity's log dir."""
    workflow_args = workflow_args or {}
    explicit_path = workflow_args.get("tool_trace_path") or os.getenv(
        TOOL_TRACE_PATH_ENV_VAR
    )

    if explicit_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(explicit_path))))
        if path.suffix.lower() != ".jsonl":
            path = path / "tool_calls.jsonl"
        return str(path.resolve())

    log_dir = os.getenv(LOG_DIR_ENV_VAR)
    if not log_dir:
        return None

    log_path = Path(log_dir).resolve()
    job_dir = log_path.parent if log_path.name.lower() == "log" else log_path
    return str(job_dir / "trajectories" / "tool_calls.jsonl")


class _ToolTraceWriter:
    """Single-concurrency Ray actor implementation."""

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        os.close(descriptor)
        os.chmod(path, 0o600)

    def append_line(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8", newline="\n") as file:
            file.write(line)
            file.flush()


class ToolTraceRecorder:
    """Emit start/finish and linked usage events to append-only JSONL."""

    def __init__(
        self,
        path: Optional[str],
        *,
        enabled: bool = True,
        max_string_chars: int = 8192,
        ray_write_timeout_seconds: float = 5.0,
    ) -> None:
        self.path = str(Path(path).resolve()) if path else None
        self.enabled = bool(enabled and self.path)
        self.max_string_chars = max(256, int(max_string_chars))
        self.ray_write_timeout_seconds = max(
            0.1,
            float(ray_write_timeout_seconds),
        )
        self._local_lock = _get_local_path_lock(self.path)
        self._ray = None
        self._ray_writer = None
        self._use_process_fallback = False
        self._warned_write_failure = False
        self._warned_dropped_record = False
        self.dropped_record_count = 0
        self.last_write_error: Optional[str] = None

        if self.enabled:
            try:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                self._initialize_ray_writer()
            except Exception as exc:
                self.enabled = False
                logging.getLogger(__name__).warning(
                    "Unable to prepare tool tracing at %s; tracing is disabled: %s",
                    self.path,
                    exc,
                )

    @classmethod
    def from_workflow_args(
        cls,
        workflow_args: Optional[Dict[str, Any]],
    ) -> "ToolTraceRecorder":
        args = workflow_args or {}
        try:
            return cls(
                resolve_tool_trace_path(args),
                enabled=_as_bool(args.get("tool_trace_enabled"), True),
                max_string_chars=int(args.get("tool_trace_max_string_chars", 8192)),
                ray_write_timeout_seconds=float(
                    args.get("tool_trace_ray_timeout_seconds", 5.0)
                ),
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Invalid tool-trace configuration; tracing is disabled: %s",
                exc,
            )
            return cls(None, enabled=False)

    def _initialize_ray_writer(self) -> None:
        try:
            import ray
        except ImportError:
            return

        try:
            if not ray.is_initialized():
                return

            path_hash = hashlib.sha256(self.path.encode("utf-8")).hexdigest()[:16]
            actor_name = f"agemem_tool_trace_{path_hash}"
            namespace = ray.get_runtime_context().namespace
            options: Dict[str, Any] = {
                "name": actor_name,
                "get_if_exists": True,
                "lifetime": "detached",
                "num_cpus": 0,
            }
            if namespace:
                options["namespace"] = namespace

            self._ray = ray
            self._ray_writer = (
                ray.remote(_ToolTraceWriter).options(**options).remote(self.path)
            )
        except Exception as exc:
            self._ray = None
            self._ray_writer = None
            self._use_process_fallback = True
            logging.getLogger(__name__).warning(
                "Unable to initialize the shared tool-trace writer; "
                "using a process-specific fallback: %s",
                _safe_exception_text(exc),
            )

    def _fallback_path(self) -> str:
        path = Path(self.path)
        host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
        return str(
            path.with_name(f"{path.stem}.{host}.{os.getpid()}.fallback{path.suffix}")
        )

    @property
    def fallback_path(self) -> Optional[str]:
        """Return the active process fallback path, if one is in use."""
        if not self.enabled or not self._use_process_fallback:
            return None
        return self._fallback_path()

    @staticmethod
    def _append_line_locally(path: str, line: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = line.encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        with _interprocess_file_lock(path):
            descriptor = os.open(path, flags, 0o600)
            try:
                os.chmod(path, 0o600)
                total_written = 0
                while total_written < len(data):
                    written = os.write(descriptor, data[total_written:])
                    if written <= 0:
                        raise OSError(
                            "Unable to make progress while appending JSONL data"
                        )
                    total_written += written
            finally:
                os.close(descriptor)

    def _write(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        line = (
            json.dumps(
                event,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )

        try:
            if self._ray is not None and self._ray_writer is not None:
                object_ref = self._ray_writer.append_line.remote(line)
                try:
                    self._ray.get(
                        object_ref,
                        timeout=self.ray_write_timeout_seconds,
                    )
                except Exception:
                    cancel = getattr(self._ray, "cancel", None)
                    if cancel is not None:
                        try:
                            cancel(object_ref, force=False)
                        except Exception:
                            pass
                    raise
                return

            local_path = (
                self._fallback_path() if self._use_process_fallback else self.path
            )
            with self._local_lock:
                self._append_line_locally(local_path, line)
        except Exception as exc:
            primary_error = _safe_exception_text(exc)
            self._ray = None
            self._ray_writer = None
            self._use_process_fallback = True
            try:
                with self._local_lock:
                    self._append_line_locally(self._fallback_path(), line)
            except Exception as fallback_exc:
                fallback_error = _safe_exception_text(fallback_exc)
                self.dropped_record_count += 1
                self.last_write_error = (
                    f"primary={type(exc).__name__}: {primary_error}; "
                    f"fallback={type(fallback_exc).__name__}: {fallback_error}"
                )
            should_warn = (
                self.last_write_error is not None and not self._warned_dropped_record
            ) or not self._warned_write_failure
            if should_warn:
                logging.getLogger(__name__).warning(
                    "Unable to append the primary tool trace %s: %s%s",
                    self.path,
                    primary_error,
                    (
                        f"; record was dropped ({self.last_write_error})"
                        if self.last_write_error
                        else "; switched to the process fallback"
                    ),
                )
                self._warned_write_failure = True
                if self.last_write_error is not None:
                    self._warned_dropped_record = True

    def _base_event(
        self,
        *,
        call_id: str,
        phase: str,
        status: str,
        context: Dict[str, Any],
        tool_name: str,
        tool_index: int,
    ) -> Dict[str, Any]:
        return {
            "schema_version": TOOL_TRACE_SCHEMA_VERSION,
            "record_id": str(uuid.uuid4()),
            "call_id": call_id,
            "timestamp": _utc_now(),
            "phase": phase,
            "status": status,
            "batch_id": context.get("batch_id"),
            "task_id": context.get("task_id"),
            "run_id": context.get("run_id"),
            "rollout_id": context.get("rollout_id"),
            "execution_id": context.get("execution_id"),
            "stage": context.get("stage"),
            "round": context.get("round"),
            "step": context.get("step"),
            "turn": context.get("turn"),
            "tool_index": tool_index,
            "tool_name": tool_name,
            "process": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            },
        }

    def record_start(
        self,
        *,
        context: Dict[str, Any],
        tool_name: str,
        tool_index: int,
        arguments: Any,
        state_before: Dict[str, Any],
    ) -> Tuple[str, float]:
        call_id = str(uuid.uuid4())
        event = self._base_event(
            call_id=call_id,
            phase="start",
            status="started",
            context=context,
            tool_name=tool_name,
            tool_index=tool_index,
        )
        event["arguments"] = arguments
        event["state_before"] = state_before
        safe_event = _json_safe(
            event,
            max_string_chars=self.max_string_chars,
        )
        self._write(safe_event)
        return call_id, time.perf_counter()

    def record_usage(
        self,
        *,
        call_id: str,
        context: Dict[str, Any],
        tool_name: str,
        tool_index: int,
        usage: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Append a later observation linked to an already-finished call."""
        event = self._base_event(
            call_id=call_id,
            phase="usage",
            status="used",
            context=context,
            tool_name=tool_name,
            tool_index=tool_index,
        )
        event["usage"] = usage
        safe_event = _json_safe(
            event,
            max_string_chars=self.max_string_chars,
        )
        self._write(safe_event)
        return safe_event

    def record_finish(
        self,
        *,
        call_id: str,
        started_at: float,
        context: Dict[str, Any],
        tool_name: str,
        tool_index: int,
        arguments: Any,
        status: str,
        result: Any,
        state_after: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = self._base_event(
            call_id=call_id,
            phase="finish",
            status=status,
            context=context,
            tool_name=tool_name,
            tool_index=tool_index,
        )
        event.update(
            {
                "arguments": arguments,
                "result": result,
                "error": error,
                "latency_ms": round(
                    (time.perf_counter() - started_at) * 1000,
                    3,
                ),
                "state_after": state_after,
            }
        )
        safe_event = _json_safe(
            event,
            max_string_chars=self.max_string_chars,
        )
        self._write(safe_event)
        return safe_event
