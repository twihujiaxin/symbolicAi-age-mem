"""Strict, text-free contracts for the M6 HotpotQA manual annotations.

The checked-in annotation files contain only source pointers, hashes, and
human-authored semantic labels.  Source sentences are resolved only while an
explicit validation pass is running; they are never retained in a model or in
the returned validation summary.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from AgeMem_code_agentscope.hotpotqa_benchmark.adapter import stable_fact_id


MANUAL_TRIPLES_SCHEMA_VERSION = "agemem.hotpotqa_manual_triples.v1"
SEMANTIC_TARGETS_SCHEMA_VERSION = "agemem.hotpotqa_semantic_targets.v1"
ANNOTATION_CORPUS_SCHEMA_VERSION = "agemem.hotpotqa_annotation_corpus.v1"
ANNOTATION_VALIDATION_SCHEMA_VERSION = "agemem.annotation_validation.v1"

BenchmarkSplit = Literal["train", "dev", "test"]
SourceSplit = Literal["train", "validation"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FACT_ID_RE = re.compile(r"^hp-([0-9a-f]+)-([0-9a-f]{16})$")


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _require_nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("text fields must not be blank")
    return value


class ManualTriple(BaseModel):
    """One human-authored semantic triple; no source text is embedded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.manual_triple.v1"] = "agemem.manual_triple.v1"
    subject: str = Field(min_length=1)
    category: str = Field(min_length=1)
    value: str = Field(min_length=1)

    @field_validator("subject", "category", "value")
    @classmethod
    def semantic_text_must_not_be_blank(cls, value: str) -> str:
        return _require_nonblank(value)

    @property
    def normalized_key(self) -> Tuple[str, str, str]:
        return tuple(
            _normalized_text(value)
            for value in (self.subject, self.category, self.value)
        )  # type: ignore[return-value]


class ManualTripleAnnotation(BaseModel):
    """Triples attached to one exact HotpotQA sentence pointer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.manual_triple_annotation.v1"] = (
        "agemem.manual_triple_annotation.v1"
    )
    annotation_id: str = Field(min_length=1)
    hotpot_id: str = Field(min_length=1)
    benchmark_split: BenchmarkSplit
    source_split: SourceSplit
    source_index: int = Field(ge=0)
    fact_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    sent_id: int = Field(ge=0)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    triples: Tuple[ManualTriple, ...] = Field(min_length=1)

    @field_validator("annotation_id", "hotpot_id", "fact_id", "title")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        return _require_nonblank(value)

    @model_validator(mode="after")
    def validate_identity(self) -> "ManualTripleAnnotation":
        match = _FACT_ID_RE.fullmatch(self.fact_id)
        if match is None or match.group(1) != self.hotpot_id:
            raise ValueError("fact_id must contain the matching hotpot_id")
        if self.annotation_id != f"m6-{self.fact_id}":
            raise ValueError("annotation_id must be 'm6-' followed by fact_id")
        keys = [triple.normalized_key for triple in self.triples]
        if len(keys) != len(set(keys)):
            raise ValueError("triples must be unique after evaluation normalization")
        return self


class ManualTripleCorpus(BaseModel):
    """The versioned manual-triple annotation artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[MANUAL_TRIPLES_SCHEMA_VERSION]
    annotation_method: Literal["human_manual"]
    guidelines_version: Literal["agemem.m6.triple_guidelines.v1"]
    normalization: Literal[
        "NFKC + casefold + collapsed whitespace for exact-set scoring"
    ]
    records: Tuple[ManualTripleAnnotation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_records(self) -> "ManualTripleCorpus":
        for name, values in (
            ("annotation_id", [record.annotation_id for record in self.records]),
            ("fact_id", [record.fact_id for record in self.records]),
            (
                "source pointer",
                [
                    (
                        record.source_split,
                        record.source_index,
                        record.title,
                        record.sent_id,
                    )
                    for record in self.records
                ],
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"manual annotation {name} values must be unique")

        by_task: dict[str, Tuple[BenchmarkSplit, SourceSplit, int]] = {}
        for record in self.records:
            coordinates = (
                record.benchmark_split,
                record.source_split,
                record.source_index,
            )
            previous = by_task.setdefault(record.hotpot_id, coordinates)
            if previous != coordinates:
                raise ValueError(
                    "one hotpot_id cannot map to multiple benchmark/source rows"
                )
        return self


class SemanticTarget(BaseModel):
    """Oracle relevance partition for one annotated HotpotQA task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.semantic_target.v1"] = "agemem.semantic_target.v1"
    hotpot_id: str = Field(min_length=1)
    relevant_fact_ids: Tuple[str, ...] = Field(min_length=1)
    irrelevant_fact_ids: Tuple[str, ...] = Field(min_length=1)

    @field_validator("hotpot_id")
    @classmethod
    def hotpot_id_must_not_be_blank(cls, value: str) -> str:
        return _require_nonblank(value)

    @model_validator(mode="after")
    def validate_partition(self) -> "SemanticTarget":
        relevant = set(self.relevant_fact_ids)
        irrelevant = set(self.irrelevant_fact_ids)
        if len(relevant) != len(self.relevant_fact_ids):
            raise ValueError("relevant_fact_ids must be unique")
        if len(irrelevant) != len(self.irrelevant_fact_ids):
            raise ValueError("irrelevant_fact_ids must be unique")
        if relevant & irrelevant:
            raise ValueError("relevant and irrelevant fact IDs must be disjoint")
        for fact_id in relevant | irrelevant:
            match = _FACT_ID_RE.fullmatch(fact_id)
            if match is None or match.group(1) != self.hotpot_id:
                raise ValueError("target fact IDs must contain their hotpot_id")
        return self


class SemanticTargetCorpus(BaseModel):
    """The evaluation-only Oracle target artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SEMANTIC_TARGETS_SCHEMA_VERSION]
    annotation_method: Literal["human_oracle_target"]
    usage: Literal["evaluation_only_not_extractor_input"]
    tasks: Tuple[SemanticTarget, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def task_ids_must_be_unique(self) -> "SemanticTargetCorpus":
        ids = [task.hotpot_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic target hotpot_ids must be unique")
        return self


class AnnotationValidationSummary(BaseModel):
    """Text-free result of validating source pointers and annotation hashes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[ANNOTATION_VALIDATION_SCHEMA_VERSION] = (
        ANNOTATION_VALIDATION_SCHEMA_VERSION
    )
    corpus_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    record_count: int = Field(ge=1)
    triple_count: int = Field(ge=1)
    relevant_fact_count: int = Field(ge=1)
    irrelevant_fact_count: int = Field(ge=1)
    source_rows_checked: int = Field(ge=1)


class AnnotationCorpus(BaseModel):
    """Joined manual triples and relevance targets with full-cover semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[ANNOTATION_CORPUS_SCHEMA_VERSION] = (
        ANNOTATION_CORPUS_SCHEMA_VERSION
    )
    manual: ManualTripleCorpus
    targets: SemanticTargetCorpus

    @model_validator(mode="after")
    def validate_full_cover(self) -> "AnnotationCorpus":
        records_by_fact = {record.fact_id: record for record in self.manual.records}
        covered: set[str] = set()
        for target in self.targets.tasks:
            task_facts = set(target.relevant_fact_ids) | set(target.irrelevant_fact_ids)
            if task_facts & covered:
                raise ValueError("a fact cannot occur in multiple semantic targets")
            covered.update(task_facts)
            for fact_id in task_facts:
                record = records_by_fact.get(fact_id)
                if record is None:
                    raise ValueError("semantic targets contain an unannotated fact_id")
                if record.hotpot_id != target.hotpot_id:
                    raise ValueError("semantic target hotpot_id does not match record")

        if covered != set(records_by_fact):
            raise ValueError(
                "semantic targets must provide disjoint full cover of annotations"
            )
        task_ids = {record.hotpot_id for record in self.manual.records}
        if task_ids != {target.hotpot_id for target in self.targets.tasks}:
            raise ValueError("manual and target task sets must match")
        return self

    @property
    def relevant_fact_ids(self) -> frozenset[str]:
        return frozenset(
            fact_id for task in self.targets.tasks for fact_id in task.relevant_fact_ids
        )

    @property
    def irrelevant_fact_ids(self) -> frozenset[str]:
        return frozenset(
            fact_id
            for task in self.targets.tasks
            for fact_id in task.irrelevant_fact_ids
        )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_release_contract(self) -> None:
        """Validate the fixed M6 benchmark size promised by the handoff."""

        record_count = len(self.manual.records)
        triple_count = sum(len(record.triples) for record in self.manual.records)
        counts = (
            record_count,
            triple_count,
            len(self.relevant_fact_ids),
            len(self.irrelevant_fact_ids),
            len(self.targets.tasks),
        )
        if counts != (34, 37, 24, 10, 10):
            raise ValueError(
                "M6 release annotations require exactly 34 records, 37 triples, "
                "24 relevant facts, 10 irrelevant facts, and 10 tasks"
            )

    def validate_against_adapter(self, adapter: Any) -> AnnotationValidationSummary:
        """Resolve every pointer against an explicit local-data adapter.

        The adapter must expose ``row(source_split, source_index)`` and may be a
        :class:`HotpotQADataAdapter` backed by a fake in-memory DatasetDict.
        No implicit dataset loading or network access occurs here.
        """

        checked_rows: set[Tuple[str, int]] = set()
        for record in self.manual.records:
            row = adapter.row(record.source_split, record.source_index)
            checked_rows.add((record.source_split, record.source_index))
            if row.id != record.hotpot_id:
                raise ValueError(
                    f"hotpot_id mismatch at {record.source_split}[{record.source_index}]"
                )

            title_matches = [
                index
                for index, title in enumerate(row.context.title)
                if title == record.title
            ]
            if len(title_matches) != 1:
                raise ValueError(
                    f"title pointer mismatch for annotation {record.annotation_id}"
                )
            paragraph = row.context.sentences[title_matches[0]]
            if record.sent_id >= len(paragraph):
                raise ValueError(
                    f"sent_id out of range for annotation {record.annotation_id}"
                )
            sentence = paragraph[record.sent_id].strip()
            if not sentence:
                raise ValueError(
                    f"empty source sentence for annotation {record.annotation_id}"
                )
            actual_digest = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
            if actual_digest != record.text_sha256:
                raise ValueError(
                    f"text SHA-256 mismatch for annotation {record.annotation_id}"
                )
            actual_fact_id = stable_fact_id(
                record.hotpot_id,
                record.title,
                record.sent_id,
                sentence,
            )
            if actual_fact_id != record.fact_id:
                raise ValueError(
                    f"stable_fact_id mismatch for annotation {record.annotation_id}"
                )

            try:
                supporting_pairs = set(row.supporting_facts.pairs())
            except AttributeError as exc:
                raise ValueError(
                    "adapter rows must expose exact supporting_facts pairs"
                ) from exc
            pointer = (record.title, record.sent_id)
            if record.fact_id in self.relevant_fact_ids:
                if pointer not in supporting_pairs:
                    raise ValueError(
                        f"relevant annotation {record.annotation_id} is not an "
                        "official supporting fact"
                    )
            elif pointer in supporting_pairs:
                raise ValueError(
                    f"irrelevant annotation {record.annotation_id} overlaps an "
                    "official supporting fact"
                )

        return AnnotationValidationSummary(
            corpus_digest=self.digest,
            task_count=len(self.targets.tasks),
            record_count=len(self.manual.records),
            triple_count=sum(len(record.triples) for record in self.manual.records),
            relevant_fact_count=len(self.relevant_fact_ids),
            irrelevant_fact_count=len(self.irrelevant_fact_ids),
            source_rows_checked=len(checked_rows),
        )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manual_triples_path() -> Path:
    return (
        repository_root() / "data" / "annotations" / "m6_hotpotqa_manual_triples.json"
    )


def default_semantic_targets_path() -> Path:
    return (
        repository_root() / "data" / "annotations" / "m6_hotpotqa_semantic_targets.json"
    )


def load_annotation_corpus(
    manual_path: Optional[str | Path] = None,
    targets_path: Optional[str | Path] = None,
) -> AnnotationCorpus:
    """Load and validate the fixed, checked-in M6 annotation release."""

    manual = ManualTripleCorpus.model_validate_json(
        Path(manual_path or default_manual_triples_path()).read_text(encoding="utf-8")
    )
    targets = SemanticTargetCorpus.model_validate_json(
        Path(targets_path or default_semantic_targets_path()).read_text(
            encoding="utf-8"
        )
    )
    corpus = AnnotationCorpus(manual=manual, targets=targets)
    corpus.validate_release_contract()
    return corpus


__all__ = [
    "ANNOTATION_CORPUS_SCHEMA_VERSION",
    "ANNOTATION_VALIDATION_SCHEMA_VERSION",
    "MANUAL_TRIPLES_SCHEMA_VERSION",
    "SEMANTIC_TARGETS_SCHEMA_VERSION",
    "AnnotationCorpus",
    "AnnotationValidationSummary",
    "ManualTriple",
    "ManualTripleAnnotation",
    "ManualTripleCorpus",
    "SemanticTarget",
    "SemanticTargetCorpus",
    "default_manual_triples_path",
    "default_semantic_targets_path",
    "load_annotation_corpus",
]
