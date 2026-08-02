"""Loader and split validation for the synthetic M3 task fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import TypeAdapter, ValidationError

from .models import Split, ToyMemoryTask


def default_task_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "toy" / "hotpotqa_memory_tasks.json"


class ToyTaskDataset:
    """Strict, immutable collection of artificial two-hop tasks."""

    def __init__(self, tasks: Iterable[ToyMemoryTask]) -> None:
        self._tasks = tuple(tasks)
        if not 20 <= len(self._tasks) <= 50:
            raise ValueError("M3 requires between 20 and 50 synthetic tasks")
        ids = [task.task_id for task in self._tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task_id values must be unique")
        if {task.split for task in self._tasks} != {"train", "dev", "test"}:
            raise ValueError("dataset must contain train, dev, and test splits")
        train_signatures = {
            task.entity_signature() for task in self._tasks if task.split == "train"
        }
        test_signatures = {
            task.entity_signature() for task in self._tasks if task.split == "test"
        }
        if train_signatures & test_signatures:
            raise ValueError("test contains an entity combination seen in train")
        self._by_id: Dict[str, ToyMemoryTask] = {
            task.task_id: task for task in self._tasks
        }

    @classmethod
    def from_json(cls, path: Optional[str | Path] = None) -> "ToyTaskDataset":
        task_path = Path(path) if path is not None else default_task_path()
        try:
            raw = json.loads(task_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid toy task JSON: {exc}") from exc
        try:
            tasks = TypeAdapter(List[ToyMemoryTask]).validate_python(raw)
        except ValidationError as exc:
            raise ValueError(f"invalid toy task schema: {exc}") from exc
        return cls(tasks)

    def all(self) -> List[ToyMemoryTask]:
        return [task.model_copy(deep=True) for task in self._tasks]

    def split(self, split: Split) -> List[ToyMemoryTask]:
        return [
            task.model_copy(deep=True)
            for task in self._tasks
            if task.split == split
        ]

    def get(self, task_id: str) -> ToyMemoryTask:
        try:
            return self._by_id[task_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"unknown toy task_id {task_id!r}") from exc

    def __len__(self) -> int:
        return len(self._tasks)


__all__ = ["ToyTaskDataset", "default_task_path"]
