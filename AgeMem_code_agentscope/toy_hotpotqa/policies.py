"""Offline deterministic policies for M3 success and failure trajectories."""

from __future__ import annotations

from typing import List, Literal, Protocol

from .models import ToyAction, ToyMemoryTask


class ToyPolicy(Protocol):
    """Policy contract used by the model-free M3 episode runner."""

    def actions(self, task: ToyMemoryTask, seed: int) -> List[ToyAction]: ...


class GoldMemoryPolicy:
    """Oracle policy that stores, updates, retrieves, and answers correctly."""

    def actions(self, task: ToyMemoryTask, seed: int) -> List[ToyAction]:
        del seed
        actions: List[ToyAction] = []
        supporting = set(task.supporting_fact_ids)

        # Store ordinary current supporting facts first.
        for fact in task.facts:
            if fact.fact_id in supporting and fact.replaces_fact_id is None:
                actions.append(ToyAction(kind="add", fact_id=fact.fact_id))

        # Fact-update tasks first observe/store the stale version, then replace it
        # through M2's versioned update semantics.
        for stale_id in task.stale_fact_ids:
            actions.append(ToyAction(kind="add", fact_id=stale_id))
            replacement = next(
                fact for fact in task.facts if fact.replaces_fact_id == stale_id
            )
            actions.append(
                ToyAction(
                    kind="update",
                    fact_id=replacement.fact_id,
                    target_fact_id=stale_id,
                )
            )

        actions.extend([ToyAction(kind="advance"), ToyAction(kind="advance")])
        actions.extend(
            ToyAction(kind="retrieve", fact_id=fact_id)
            for fact_id in task.supporting_fact_ids
        )
        actions.append(ToyAction(kind="answer", answer=task.answer))
        return actions


ErrorMode = Literal[
    "wrong_answer",
    "missing_support",
    "stale_retrieval",
    "delete_support",
    "duplicate_add",
]


class ErrorMemoryPolicy:
    """Deterministic negative policy with one explicit failure mode."""

    def __init__(self, mode: ErrorMode = "wrong_answer") -> None:
        self.mode = mode

    def actions(self, task: ToyMemoryTask, seed: int) -> List[ToyAction]:
        actions = GoldMemoryPolicy().actions(task, seed)

        if self.mode == "wrong_answer":
            actions[-1] = ToyAction(kind="answer", answer="definitely incorrect")
            return actions

        if self.mode == "missing_support":
            missing_id = task.supporting_fact_ids[1]
            missing_fact = task.fact(missing_id)
            actions = [
                action
                for action in actions
                if not (
                    (action.kind == "add" and action.fact_id == missing_id)
                    or (action.kind == "update" and action.fact_id == missing_id)
                    or (
                        missing_fact.replaces_fact_id is not None
                        and action.kind == "add"
                        and action.fact_id == missing_fact.replaces_fact_id
                    )
                )
            ]
            return actions

        if self.mode == "stale_retrieval":
            if not task.stale_fact_ids:
                raise ValueError("stale_retrieval requires a fact-update task")
            stale_id = task.stale_fact_ids[0]
            current_id = next(
                fact.fact_id for fact in task.facts if fact.replaces_fact_id == stale_id
            )
            actions = [
                action
                for action in actions
                if not (action.kind == "update" and action.fact_id == current_id)
            ]
            actions = [
                ToyAction(kind="retrieve", fact_id=stale_id)
                if action.kind == "retrieve" and action.fact_id == current_id
                else action
                for action in actions
            ]
            return actions

        if self.mode == "delete_support":
            first_advance = next(
                index for index, action in enumerate(actions) if action.kind == "advance"
            )
            actions.insert(
                first_advance + 1,
                ToyAction(kind="delete", fact_id=task.supporting_fact_ids[0]),
            )
            return actions

        if self.mode == "duplicate_add":
            if not task.duplicate_fact_ids:
                raise ValueError("duplicate_add requires a duplicate-fact task")
            first_advance = next(
                index for index, action in enumerate(actions) if action.kind == "advance"
            )
            original_id = task.fact(task.duplicate_fact_ids[0]).duplicate_of_fact_id
            actions.insert(
                first_advance,
                ToyAction(kind="add", fact_id=original_id),
            )
            return actions

        raise ValueError(f"unsupported error policy mode {self.mode!r}")


__all__ = ["ErrorMemoryPolicy", "ErrorMode", "GoldMemoryPolicy", "ToyPolicy"]
