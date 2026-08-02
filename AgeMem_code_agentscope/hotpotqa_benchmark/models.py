"""Strict M5 contracts for real HotpotQA adaptation and smoke splits."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


SourceSplit = Literal["train", "validation", "test"]
BenchmarkSplit = Literal["train", "dev", "test"]
HotpotType = Literal["bridge", "comparison"]
HotpotLevel = Literal["easy", "medium", "hard"]
PolicyName = Literal["gold", "wrong_answer", "missing_support"]


class HotpotContext(BaseModel):
    """Hugging Face fullwiki context represented as parallel arrays."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: Tuple[str, ...]
    sentences: Tuple[Tuple[str, ...], ...]

    @model_validator(mode="after")
    def validate_parallel_arrays(self) -> "HotpotContext":
        if not self.title or len(self.title) != len(self.sentences):
            raise ValueError("context title/sentences must be non-empty parallel arrays")
        if any(not title.strip() for title in self.title):
            raise ValueError("context titles must be non-empty")
        if any(not paragraph for paragraph in self.sentences):
            raise ValueError("context paragraphs must be non-empty")
        return self


class HotpotSupportingFacts(BaseModel):
    """Official sentence pointers as exact (title, sent_id) pairs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: Tuple[str, ...] = ()
    sent_id: Tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_parallel_arrays(self) -> "HotpotSupportingFacts":
        if len(self.title) != len(self.sent_id):
            raise ValueError("supporting_facts title/sent_id lengths must match")
        if any(not title.strip() for title in self.title):
            raise ValueError("supporting-fact titles must be non-empty")
        if any(sent_id < 0 for sent_id in self.sent_id):
            raise ValueError("supporting-fact sentence IDs must be non-negative")
        if len(set(zip(self.title, self.sent_id))) != len(self.title):
            raise ValueError("supporting-fact pointers must be unique")
        return self

    def pairs(self) -> Tuple[Tuple[str, int], ...]:
        return tuple(zip(self.title, self.sent_id))


class HotpotQARow(BaseModel):
    """One real row from the Hugging Face HotpotQA fullwiki DatasetDict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: Optional[str] = None
    type: Optional[HotpotType] = None
    level: Optional[HotpotLevel] = None
    supporting_facts: HotpotSupportingFacts
    context: HotpotContext


class HotpotFact(BaseModel):
    """One addressable source sentence; no M6 semantic triple is invented."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    sent_id: int = Field(ge=0)
    sentence: str = Field(min_length=1)
    stage: Literal[1, 2]
    role: Literal["supporting", "distractor"]

    @property
    def replaces_fact_id(self) -> None:
        return None

    @property
    def duplicate_of_fact_id(self) -> None:
        return None


class SourceFactPointer(BaseModel):
    """Auditable link from a stable fact ID to the official annotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    sent_id: int = Field(ge=0)


class HotpotQAMemoryTask(BaseModel):
    """Private real-data task compatible with the deterministic M3 runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    split: BenchmarkSplit
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    facts: Tuple[HotpotFact, ...] = Field(min_length=2)
    supporting_fact_ids: Tuple[str, ...] = Field(min_length=2)
    distractor_fact_ids: Tuple[str, ...] = ()
    stale_fact_ids: Tuple[str, ...] = ()
    duplicate_fact_ids: Tuple[str, ...] = ()
    hotpot_id: str = Field(min_length=1)
    source_split: Literal["train", "validation"]
    source_index: int = Field(ge=0)
    hotpot_type: HotpotType
    level: HotpotLevel
    supporting_fact_pointers: Tuple[SourceFactPointer, ...]
    source_context_sentence_count: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_annotations(self) -> "HotpotQAMemoryTask":
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("adapted fact IDs must be unique")
        known = set(fact_ids)
        supporting = set(self.supporting_fact_ids)
        distractors = set(self.distractor_fact_ids)
        if len(supporting) != len(self.supporting_fact_ids):
            raise ValueError("supporting fact IDs must be unique")
        if len(distractors) != len(self.distractor_fact_ids):
            raise ValueError("distractor fact IDs must be unique")
        if supporting & distractors:
            raise ValueError("supporting and distractor fact IDs must be disjoint")
        if supporting | distractors != known:
            raise ValueError("every adapted fact must have exactly one Oracle role")
        if self.stale_fact_ids or self.duplicate_fact_ids:
            raise ValueError("M5 does not synthesize stale or duplicate facts")
        pointer_ids = [pointer.fact_id for pointer in self.supporting_fact_pointers]
        if len(pointer_ids) != len(set(pointer_ids)):
            raise ValueError("supporting pointers must be unique")
        if set(pointer_ids) != supporting:
            raise ValueError("supporting pointers must cover all supporting fact IDs")
        by_id = {fact.fact_id: fact for fact in self.facts}
        if any(by_id[fact_id].stage != 1 for fact_id in supporting):
            raise ValueError("all Oracle supporting facts must be visible in Stage 1")
        for fact in self.facts:
            expected_role = "supporting" if fact.fact_id in supporting else "distractor"
            if fact.role != expected_role:
                raise ValueError(
                    f"fact {fact.fact_id!r} role does not match Oracle ID lists"
                )
        for pointer in self.supporting_fact_pointers:
            fact = by_id[pointer.fact_id]
            if (pointer.title, pointer.sent_id) != (fact.title, fact.sent_id):
                raise ValueError(
                    f"supporting pointer for {pointer.fact_id!r} does not match fact"
                )
        return self

    def fact(self, fact_id: str) -> HotpotFact:
        for fact in self.facts:
            if fact.fact_id == fact_id:
                return fact
        raise KeyError(f"unknown fact_id {fact_id!r} for task {self.task_id!r}")


class HotpotQASmokeConfig(BaseModel):
    """Externalized deterministic selection and rollout settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    seed: int = Field(ge=0)
    train_size: int = Field(ge=2)
    dev_size: int = Field(ge=2)
    test_size: int = Field(ge=2)
    min_supporting_facts: int = Field(default=2, ge=2)
    max_supporting_facts: int = Field(default=4, ge=2)
    stage1_distractors: int = Field(ge=0)
    stage2_distractors: int = Field(ge=0)
    policies: Tuple[PolicyName, ...] = (
        "gold",
        "wrong_answer",
        "missing_support",
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> "HotpotQASmokeConfig":
        if self.max_supporting_facts < self.min_supporting_facts:
            raise ValueError("max_supporting_facts must be >= min_supporting_facts")
        if self.max_supporting_facts < 3:
            raise ValueError(
                "smoke balancing requires max_supporting_facts >= 3"
            )
        if not self.policies or "gold" not in self.policies:
            raise ValueError("smoke policies must be non-empty and include gold")
        if len(self.policies) != len(set(self.policies)):
            raise ValueError("policies must be unique")
        return self


class SmokeSelection(BaseModel):
    """One immutable source-row assignment in the M5 smoke split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_split: BenchmarkSplit
    source_split: Literal["train", "validation"]
    source_index: int = Field(ge=0)
    hotpot_id: str = Field(min_length=1)
    hotpot_type: HotpotType
    level: HotpotLevel
    supporting_fact_count: int = Field(ge=2)


class HotpotQASmokeManifest(BaseModel):
    """Reproducible split manifest tied to source Dataset fingerprints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    dataset_name: Literal["hotpot_qa"] = "hotpot_qa"
    dataset_config: Literal["fullwiki"] = "fullwiki"
    source_fingerprints: Dict[SourceSplit, str]
    seed: int = Field(ge=0)
    smoke_config_digest: str = Field(min_length=64, max_length=64)
    split_sizes: Dict[BenchmarkSplit, int]
    selections: Tuple[SmokeSelection, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> "HotpotQASmokeManifest":
        if set(self.source_fingerprints) != {"train", "validation", "test"}:
            raise ValueError("manifest requires all three source fingerprints")
        if set(self.split_sizes) != {"train", "dev", "test"}:
            raise ValueError("manifest requires train/dev/test sizes")
        if any(not value for value in self.source_fingerprints.values()):
            raise ValueError("source fingerprints must be non-empty")
        if any(value < 1 for value in self.split_sizes.values()):
            raise ValueError("manifest split sizes must be positive")
        ids = [item.hotpot_id for item in self.selections]
        if len(ids) != len(set(ids)):
            raise ValueError("a source task cannot appear in multiple smoke splits")
        for item in self.selections:
            expected_source = (
                "train" if item.benchmark_split == "train" else "validation"
            )
            if item.source_split != expected_source:
                raise ValueError(
                    f"{item.benchmark_split} must derive from {expected_source}"
                )
        actual = {
            split: sum(item.benchmark_split == split for item in self.selections)
            for split in ("train", "dev", "test")
        }
        if actual != self.split_sizes:
            raise ValueError(
                f"manifest split counts do not match: expected {self.split_sizes}, got {actual}"
            )
        return self


__all__ = [
    "BenchmarkSplit",
    "HotpotContext",
    "HotpotFact",
    "HotpotLevel",
    "HotpotQAMemoryTask",
    "HotpotQARow",
    "HotpotQASmokeConfig",
    "HotpotQASmokeManifest",
    "HotpotSupportingFacts",
    "HotpotType",
    "PolicyName",
    "SmokeSelection",
    "SourceFactPointer",
    "SourceSplit",
]
