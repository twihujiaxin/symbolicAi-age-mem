"""Strict M3 data contracts for synthetic two-hop memory episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..memory_store import MemoryStoreSnapshot


Split = Literal["train", "dev", "test"]
Difficulty = Literal[
    "clean",
    "distractor",
    "duplicate",
    "fact_update",
    "stale_fact",
    "critical_delete",
]
ActionKind = Literal["add", "update", "retrieve", "delete", "advance", "answer"]


class ToyFact(BaseModel):
    """One synthetic context sentence with stable internal identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    sentence: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    object: str = Field(min_length=1)
    stage: Literal[1, 2] = 1
    replaces_fact_id: Optional[str] = None
    duplicate_of_fact_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_links(self) -> "ToyFact":
        if self.replaces_fact_id and self.duplicate_of_fact_id:
            raise ValueError("a fact cannot both replace and duplicate another fact")
        if self.replaces_fact_id == self.fact_id:
            raise ValueError("a fact cannot replace itself")
        if self.duplicate_of_fact_id == self.fact_id:
            raise ValueError("a fact cannot duplicate itself")
        return self


class ToyMemoryTask(BaseModel):
    """Private task specification containing answers and Oracle annotations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    split: Split
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    facts: Tuple[ToyFact, ...] = Field(min_length=2)
    supporting_fact_ids: Tuple[str, str]
    distractor_fact_ids: Tuple[str, ...] = ()
    stale_fact_ids: Tuple[str, ...] = ()
    duplicate_fact_ids: Tuple[str, ...] = ()
    difficulty: Tuple[Difficulty, ...] = ("clean",)

    @model_validator(mode="after")
    def validate_fact_annotations(self) -> "ToyMemoryTask":
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact_id values must be unique within a task")
        known = set(fact_ids)
        annotation_sets = {
            "supporting": set(self.supporting_fact_ids),
            "distractor": set(self.distractor_fact_ids),
            "stale": set(self.stale_fact_ids),
            "duplicate": set(self.duplicate_fact_ids),
        }
        for name, values in annotation_sets.items():
            missing = values - known
            if missing:
                raise ValueError(f"{name} fact IDs do not exist: {sorted(missing)}")
        if len(set(self.supporting_fact_ids)) != 2:
            raise ValueError("a two-hop task must have exactly two supporting facts")
        protected_sets = [
            annotation_sets["supporting"],
            annotation_sets["distractor"],
            annotation_sets["stale"],
            annotation_sets["duplicate"],
        ]
        for index, left in enumerate(protected_sets):
            for right in protected_sets[index + 1 :]:
                if left & right:
                    raise ValueError("fact annotation sets must be disjoint")

        by_id = {fact.fact_id: fact for fact in self.facts}
        for stale_id in self.stale_fact_ids:
            replacements = [
                fact for fact in self.facts if fact.replaces_fact_id == stale_id
            ]
            if len(replacements) != 1:
                raise ValueError(
                    f"stale fact {stale_id!r} must have exactly one replacement"
                )
            if replacements[0].fact_id not in annotation_sets["supporting"]:
                raise ValueError("a stale fact replacement must be a supporting fact")
        for duplicate_id in self.duplicate_fact_ids:
            original_id = by_id[duplicate_id].duplicate_of_fact_id
            if original_id is None or original_id not in known:
                raise ValueError("duplicate facts must reference an existing fact")
        if not self.difficulty:
            raise ValueError("difficulty must not be empty")
        return self

    def fact(self, fact_id: str) -> ToyFact:
        for fact in self.facts:
            if fact.fact_id == fact_id:
                return fact
        raise KeyError(f"unknown fact_id {fact_id!r} for task {self.task_id!r}")

    def entity_signature(self) -> Tuple[str, str, str]:
        first = self.fact(self.supporting_fact_ids[0])
        second = self.fact(self.supporting_fact_ids[1])
        return first.subject, first.object, second.object


class StageInput(BaseModel):
    """The public, answer-free view presented to a policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    rollout_id: str
    seed: int = Field(ge=0)
    stage: Literal[1, 2, 3]
    observation: str
    allowed_actions: Tuple[str, ...]


class ToyAction(BaseModel):
    """One deterministic policy action in the M3 environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ActionKind
    fact_id: Optional[str] = None
    target_fact_id: Optional[str] = None
    answer: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_arguments(self) -> "ToyAction":
        if self.kind in {"add", "retrieve", "delete"} and not self.fact_id:
            raise ValueError(f"{self.kind} requires fact_id")
        if self.kind == "update" and (not self.fact_id or not self.target_fact_id):
            raise ValueError("update requires fact_id and target_fact_id")
        if self.kind == "answer" and self.answer is None:
            raise ValueError("answer action requires answer")
        if self.kind == "advance" and any(
            value is not None
            for value in (self.fact_id, self.target_fact_id, self.answer)
        ):
            raise ValueError("advance does not accept action arguments")
        return self


class OracleLabels(BaseModel):
    """M3 supervision labels; these are not M4 atomic propositions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observed_fact_ids: Tuple[str, ...] = ()
    stored_supporting_fact_ids: Tuple[str, ...] = ()
    stored_distractor_fact_ids: Tuple[str, ...] = ()
    ignored_duplicate_fact_ids: Tuple[str, ...] = ()
    updated_stale_fact_ids: Tuple[str, ...] = ()
    retrieved_supporting_fact_ids: Tuple[str, ...] = ()
    retrieved_distractor_fact_ids: Tuple[str, ...] = ()
    retrieved_stale_fact_ids: Tuple[str, ...] = ()
    deleted_supporting_fact_ids: Tuple[str, ...] = ()
    supporting_coverage_complete: bool = False
    answer_correct: Optional[bool] = None


class EpisodeStepResult(BaseModel):
    """Validated environment result returned for one action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    rollout_id: str
    seed: int = Field(ge=0)
    timestep: int = Field(ge=0)
    stage_before: Literal[1, 2, 3]
    stage_after: Literal[1, 2, 3]
    action: ToyAction
    success: bool
    message: str
    labels: OracleLabels
    task_reward: float = 0.0
    done: bool = False
    episode_success: bool = False


class MemoryEpisode(BaseModel):
    """Serializable public episode status without private task annotations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    rollout_id: str
    seed: int = Field(ge=0)
    split: Split
    stage: Literal[1, 2, 3]
    timestep: int = Field(ge=0)
    stm: Tuple[str, ...]
    retrieved_supporting_fact_ids: Tuple[str, ...]
    done: bool
    success: bool


@dataclass(frozen=True)
class EpisodeSnapshot:
    """Complete checkpoint used to verify deterministic environment restore."""

    task_id: str
    rollout_id: str
    seed: int
    stage: int
    timestep: int
    stm: Tuple[str, ...]
    pending_observed_fact_ids: Tuple[str, ...]
    retrieved_supporting_fact_ids: Tuple[str, ...]
    memory_ids_by_fact_id: Tuple[Tuple[str, str], ...]
    done: bool
    success: bool
    memory: MemoryStoreSnapshot


def sorted_tuple(values: Set[str] | List[str]) -> Tuple[str, ...]:
    """Return a deterministic representation for label and state sets."""

    return tuple(sorted(values))


__all__ = [
    "ActionKind",
    "Difficulty",
    "EpisodeSnapshot",
    "EpisodeStepResult",
    "MemoryEpisode",
    "OracleLabels",
    "Split",
    "StageInput",
    "ToyAction",
    "ToyFact",
    "ToyMemoryTask",
    "sorted_tuple",
]
