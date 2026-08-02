# -*- coding: utf-8 -*-
"""Deterministic JSONL trajectory recording and replay for standalone AgeMem.

Unlike the training-side tool audit trace, this module stores complete memory
snapshots and never truncates or redacts fields.  The resulting files can
therefore be replayed deterministically, but they must be treated as sensitive
artifacts because they may contain raw conversation and memory content.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


TRAJECTORY_SCHEMA_VERSION = 1


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory file or step violates the M1 contract."""


def to_jsonable(value: Any, *, field_name: str = "value") -> Any:
    """Convert supported runtime values to deterministic JSON-compatible data.

    Unsupported objects are rejected instead of being stringified with ``repr``;
    repr output can contain memory addresses and would break deterministic replay.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrajectoryValidationError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item, field_name=f"{field_name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"), field_name=field_name)
    if is_dataclass(value):
        return to_jsonable(asdict(value), field_name=field_name)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict(), field_name=field_name)
    raise TrajectoryValidationError(
        f"{field_name} contains unsupported type {type(value).__name__}"
    )


class MemorySnapshotItem(BaseModel):
    """A replayable standalone long-term-memory item."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_finite(
        cls, embedding: Optional[List[float]]
    ) -> Optional[List[float]]:
        if embedding is not None and any(not math.isfinite(value) for value in embedding):
            raise ValueError("embedding values must be finite")
        return embedding

    def canonical_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class ToolCallSnapshot(BaseModel):
    """The normalized AgentScope tool-use block for one action."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input: Dict[str, Any] = Field(default_factory=dict)


class ToolResultSnapshot(BaseModel):
    """One accumulated ToolResponse chunk returned by AgentScope."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    content: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    is_last: bool = True
    is_interrupted: bool = False


class TrajectoryStep(BaseModel):
    """Strict, versioned representation of one executed Agent action."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[TRAJECTORY_SCHEMA_VERSION] = TRAJECTORY_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage: int = Field(default=0, ge=0)
    timestep: int = Field(ge=0)
    observation: str
    action_text: str
    tool_calls: List[ToolCallSnapshot] = Field(min_length=1)
    tool_results: List[ToolResultSnapshot] = Field(min_length=1)
    memory_before: List[MemorySnapshotItem]
    memory_after: List[MemorySnapshotItem]
    env_reward: float = 0.0
    done: bool = False
    old_logprob: Optional[float] = None

    @field_validator("env_reward")
    @classmethod
    def reward_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("env_reward must be finite")
        return value

    @field_validator("old_logprob")
    @classmethod
    def logprob_must_be_finite(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("old_logprob must be finite")
        return value

    def key(self) -> Tuple[str, str, int]:
        return self.task_id, self.rollout_id, self.timestep

    def canonical_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json_line(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class ReplayResult(BaseModel):
    """Deterministic replay output for one task rollout."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    rollout_id: str
    steps: List[TrajectoryStep]
    memory_states: List[List[MemorySnapshotItem]]
    final_memory: List[MemorySnapshotItem]
    done: bool
    digest: str


def snapshot_memory(memory: Any) -> List[MemorySnapshotItem]:
    """Capture and validate a complete memory snapshot without calling an LLM."""

    state_dict = getattr(memory, "state_dict", None)
    if not callable(state_dict):
        raise TrajectoryValidationError(
            "trajectory recording requires memory.state_dict()"
        )
    state = to_jsonable(state_dict(), field_name="memory.state_dict")
    if not isinstance(state, dict) or not isinstance(state.get("content"), list):
        raise TrajectoryValidationError(
            "memory.state_dict() must return {'content': [...]}"
        )
    try:
        return [MemorySnapshotItem.model_validate(item) for item in state["content"]]
    except ValidationError as exc:
        raise TrajectoryValidationError(f"invalid memory snapshot: {exc}") from exc


_PATH_LOCKS: Dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    resolved = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(resolved, threading.Lock())


class TrajectoryRecorder:
    """Append validated trajectory steps to a local JSONL file."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _path_lock(self.path)
        self._known_keys: set[Tuple[str, str, int]] = set()
        if self.path.exists() and self.path.stat().st_size:
            existing = TrajectoryReplay.from_jsonl(self.path)
            self._known_keys.update(step.key() for step in existing.steps)

    def record(self, step: TrajectoryStep | Dict[str, Any]) -> TrajectoryStep:
        try:
            raw_step = (
                step.model_dump(mode="python")
                if isinstance(step, TrajectoryStep)
                else step
            )
            validated = TrajectoryStep.model_validate(
                to_jsonable(raw_step, field_name="step")
            )
        except (ValidationError, TrajectoryValidationError) as exc:
            raise TrajectoryValidationError(f"invalid trajectory step: {exc}") from exc

        payload = (validated.to_json_line() + "\n").encode("utf-8")
        key = validated.key()
        with self._lock:
            if key in self._known_keys:
                raise TrajectoryValidationError(
                    "duplicate trajectory timestep for "
                    f"task_id={key[0]!r}, rollout_id={key[1]!r}, timestep={key[2]}"
                )
            fd = os.open(
                str(self.path),
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(fd, payload[offset:])
                    if written <= 0:
                        raise OSError("unable to make progress while writing trajectory")
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)
            self._known_keys.add(key)
        return validated


class TrajectoryReplay:
    """Strict JSONL loader, query index, and deterministic memory replay."""

    def __init__(self, steps: Iterable[TrajectoryStep]) -> None:
        self.steps = list(steps)
        self._index: Dict[Tuple[str, str, int], TrajectoryStep] = {}
        for step in self.steps:
            key = step.key()
            if key in self._index:
                raise TrajectoryValidationError(
                    "duplicate trajectory timestep for "
                    f"task_id={key[0]!r}, rollout_id={key[1]!r}, timestep={key[2]}"
                )
            self._index[key] = step

    @classmethod
    def from_jsonl(cls, path: str | os.PathLike[str]) -> "TrajectoryReplay":
        trajectory_path = Path(path).expanduser().resolve()
        if not trajectory_path.is_file():
            raise TrajectoryValidationError(
                f"trajectory file does not exist: {trajectory_path}"
            )

        steps: List[TrajectoryStep] = []
        with trajectory_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise TrajectoryValidationError(
                        f"blank JSONL record at line {line_number}"
                    )
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise TrajectoryValidationError(
                        f"invalid JSON at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(data, dict):
                    raise TrajectoryValidationError(
                        f"trajectory record at line {line_number} must be an object"
                    )
                try:
                    steps.append(TrajectoryStep.model_validate(data))
                except ValidationError as exc:
                    raise TrajectoryValidationError(
                        f"schema validation failed at line {line_number}: {exc}"
                    ) from exc
        return cls(steps)

    def query(
        self,
        *,
        task_id: Optional[str] = None,
        rollout_id: Optional[str] = None,
        timestep: Optional[int] = None,
    ) -> List[TrajectoryStep]:
        """Query steps by any combination of task, rollout, and timestep."""

        if task_id is not None and rollout_id is not None and timestep is not None:
            step = self._index.get((task_id, rollout_id, timestep))
            return [step] if step is not None else []
        return [
            step
            for step in self.steps
            if (task_id is None or step.task_id == task_id)
            and (rollout_id is None or step.rollout_id == rollout_id)
            and (timestep is None or step.timestep == timestep)
        ]

    def available_rollouts(self) -> List[Tuple[str, str]]:
        return sorted({(step.task_id, step.rollout_id) for step in self.steps})

    def replay(
        self,
        *,
        task_id: str,
        rollout_id: str,
        require_complete: bool = False,
    ) -> ReplayResult:
        """Rebuild the recorded memory state sequence without model calls."""

        selected = sorted(
            self.query(task_id=task_id, rollout_id=rollout_id),
            key=lambda step: step.timestep,
        )
        if not selected:
            raise TrajectoryValidationError(
                f"no trajectory for task_id={task_id!r}, rollout_id={rollout_id!r}"
            )

        actual_timesteps = [step.timestep for step in selected]
        expected_timesteps = list(range(len(selected)))
        if actual_timesteps != expected_timesteps:
            raise TrajectoryValidationError(
                "trajectory timesteps must be contiguous and start at zero: "
                f"expected {expected_timesteps}, got {actual_timesteps}"
            )

        memory_states: List[List[MemorySnapshotItem]] = [
            [item.model_copy(deep=True) for item in selected[0].memory_before]
        ]
        for previous, current in zip(selected, selected[1:]):
            previous_after = [item.canonical_dict() for item in previous.memory_after]
            current_before = [item.canonical_dict() for item in current.memory_before]
            if previous_after != current_before:
                raise TrajectoryValidationError(
                    "memory state discontinuity between timesteps "
                    f"{previous.timestep} and {current.timestep}"
                )
        for step in selected:
            memory_states.append(
                [item.model_copy(deep=True) for item in step.memory_after]
            )

        done = selected[-1].done
        if require_complete and not done:
            raise TrajectoryValidationError(
                "trajectory is incomplete because its final step is not done"
            )

        canonical_payload = json.dumps(
            [step.canonical_dict() for step in selected],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(canonical_payload).hexdigest()
        final_memory = [item.model_copy(deep=True) for item in selected[-1].memory_after]
        return ReplayResult(
            task_id=task_id,
            rollout_id=rollout_id,
            steps=[step.model_copy(deep=True) for step in selected],
            memory_states=memory_states,
            final_memory=final_memory,
            done=done,
            digest=digest,
        )


__all__ = [
    "TRAJECTORY_SCHEMA_VERSION",
    "MemorySnapshotItem",
    "ReplayResult",
    "ToolCallSnapshot",
    "ToolResultSnapshot",
    "TrajectoryRecorder",
    "TrajectoryReplay",
    "TrajectoryStep",
    "TrajectoryValidationError",
    "snapshot_memory",
    "to_jsonable",
]
