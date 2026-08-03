"""Strict, immutable contracts for M6 semantic extraction.

The key separation in this module is intentional: :class:`TripleCandidate`
contains action-independent extraction output and is therefore safe to cache,
while :class:`TripleRecord`, :class:`RelevanceDecision`, and
:class:`APRecord` are action-bound provenance records and must never be cached
as group extraction output.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Dict, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EXTRACTION_SCHEMA_VERSION = "agemem.extraction_request.v1"
EXTRACTOR_OUTPUT_SCHEMA_VERSION = "agemem.extractor_output.v1"
TRIPLE_CANDIDATE_SCHEMA_VERSION = "agemem.triple_candidate.v1"
TRIPLE_RECORD_SCHEMA_VERSION = "agemem.triple_record.v1"
AP_RECORD_SCHEMA_VERSION = "agemem.ap_record.v1"

EvidenceSource = Literal[
    "observation",
    "action",
    "tool_result",
    "memory_before",
    "memory_after",
    "question",
]
ExtractorKind = Literal["mock", "llm", "rule"]
SemanticRole = Literal["relevant", "irrelevant", "abstain"]
QuarantineReason = Literal[
    "invalid_json",
    "invalid_schema",
    "invalid_evidence",
    "unknown_subject",
    "unknown_category",
    "duplicate_candidate",
    "extractor_error",
]

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_digest(value: object) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_digest(text: str) -> str:
    """Hash source text without normalizing it; offsets refer to exact text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_digest(value: str, name: str) -> str:
    if not _HEX_64.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _normalize_label(value: str) -> str:
    return " ".join(value.split()).casefold()


class EvidenceSpan(BaseModel):
    """An exact character span in one named source document."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source: EvidenceSource
    source_digest: str = Field(min_length=64, max_length=64)
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("source_digest")
    @classmethod
    def digest_must_be_sha256(cls, value: str) -> str:
        return _require_digest(value, "source_digest")

    @model_validator(mode="after")
    def validate_interval(self) -> "EvidenceSpan":
        if self.end <= self.start:
            raise ValueError("evidence end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("evidence text length must equal end - start")
        return self

    @classmethod
    def from_source(
        cls,
        *,
        source: EvidenceSource,
        source_text: str,
        start: int,
        end: int,
    ) -> "EvidenceSpan":
        if start < 0 or end <= start or end > len(source_text):
            raise ValueError("evidence span is outside its source text")
        return cls(
            source=source,
            source_digest=text_digest(source_text),
            text=source_text[start:end],
            start=start,
            end=end,
        )

    def validate_against(self, source_text: str) -> None:
        """Fail unless both the document hash and exact slice match."""

        if text_digest(source_text) != self.source_digest:
            raise ValueError("evidence source digest does not match source text")
        if (
            self.end > len(source_text)
            or source_text[self.start : self.end] != self.text
        ):
            raise ValueError("evidence text does not equal source_text[start:end]")


class ExtractionRequest(BaseModel):
    """Action-independent natural-language input for one group extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[EXTRACTION_SCHEMA_VERSION] = EXTRACTION_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    split_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    observation: str = Field(min_length=1)
    question: Optional[str] = None
    known_subjects: Tuple[str, ...] = Field(min_length=1)
    allowed_categories: Tuple[str, ...] = Field(min_length=1)

    @field_validator("task_id", "split_id", "rollout_id", "group_id", "observation")
    @classmethod
    def required_text_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required text fields must not be whitespace")
        return value

    @field_validator("question")
    @classmethod
    def optional_question_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("question must be non-blank when supplied")
        return value

    @model_validator(mode="after")
    def validate_registries(self) -> "ExtractionRequest":
        for name, values in (
            ("known_subjects", self.known_subjects),
            ("allowed_categories", self.allowed_categories),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"{name} entries must be non-blank")
            normalized = [_normalize_label(value) for value in values]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} entries must be unique after normalization")
        return self

    @property
    def observation_digest(self) -> str:
        return text_digest(self.observation)

    @property
    def question_digest(self) -> Optional[str]:
        return text_digest(self.question) if self.question is not None else None

    @property
    def constraint_digest(self) -> str:
        return canonical_digest(
            {
                # Preserve exact spelling and order because both are rendered
                # into an LLM prompt and may change model output.
                "known_subjects": self.known_subjects,
                "allowed_categories": self.allowed_categories,
            }
        )

    @property
    def request_digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))

    def source_text(self, source: EvidenceSource) -> str:
        """Resolve extractor-visible source text.

        Observation extraction deliberately cannot inspect action/tool/memory
        text. Those action-specific sources are handled by deterministic rules
        and therefore cannot contaminate a shared observation cache.
        """

        if source == "observation":
            return self.observation
        if source == "question" and self.question is not None:
            return self.question
        raise KeyError(f"source {source!r} is unavailable to observation extractor")

    def accepts_subject(self, subject: str) -> bool:
        expected = {_normalize_label(value) for value in self.known_subjects}
        return _normalize_label(subject) in expected

    def accepts_category(self, category: str) -> bool:
        expected = {_normalize_label(value) for value in self.allowed_categories}
        return _normalize_label(category) in expected


class TripleCandidate(BaseModel):
    """Validated semantic output with no action identity or relevance role."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[TRIPLE_CANDIDATE_SCHEMA_VERSION] = (
        TRIPLE_CANDIDATE_SCHEMA_VERSION
    )
    candidate_id: str = Field(min_length=64, max_length=64)
    subject: str = Field(min_length=1)
    category: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float
    evidence: Tuple[EvidenceSpan, ...] = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    extractor_kind: ExtractorKind
    model_version: str = Field(min_length=1)

    @field_validator("candidate_id")
    @classmethod
    def candidate_id_must_be_sha256(cls, value: str) -> str:
        return _require_digest(value, "candidate_id")

    @field_validator(
        "subject", "category", "value", "extractor_version", "model_version"
    )
    @classmethod
    def semantic_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("semantic and version fields must not be whitespace")
        return value

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_finite_unit_interval(cls, value: float) -> float:
        if (
            isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_identity_and_evidence(self) -> "TripleCandidate":
        if len(self.evidence) != len(set(self.evidence)):
            raise ValueError("evidence spans must be unique")
        if self.candidate_id != self.expected_candidate_id():
            raise ValueError("candidate_id does not match canonical candidate content")
        return self

    def identity_payload(self) -> Dict[str, object]:
        return {
            "subject": self.subject,
            "category": self.category,
            "value": self.value,
            "confidence": self.confidence,
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
            "extractor_version": self.extractor_version,
            "extractor_kind": self.extractor_kind,
            "model_version": self.model_version,
        }

    def expected_candidate_id(self) -> str:
        return canonical_digest(
            {"namespace": TRIPLE_CANDIDATE_SCHEMA_VERSION, **self.identity_payload()}
        )

    @classmethod
    def create(
        cls,
        *,
        subject: str,
        category: str,
        value: str,
        confidence: float,
        evidence: Tuple[EvidenceSpan, ...],
        extractor_version: str,
        extractor_kind: ExtractorKind,
        model_version: str,
    ) -> "TripleCandidate":
        if (
            isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        payload: Dict[str, object] = {
            "subject": subject,
            "category": category,
            "value": value,
            "confidence": confidence,
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "extractor_version": extractor_version,
            "extractor_kind": extractor_kind,
            "model_version": model_version,
        }
        candidate_id = canonical_digest(
            {"namespace": TRIPLE_CANDIDATE_SCHEMA_VERSION, **payload}
        )
        return cls(
            candidate_id=candidate_id,
            evidence=evidence,
            **{key: value for key, value in payload.items() if key != "evidence"},
        )


class ActionBinding(BaseModel):
    """The immutable trajectory coordinates used to bind cached candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    assistant_turn_id: int = Field(ge=0)
    action_index_in_turn: int = Field(ge=0)

    @field_validator("task_id", "rollout_id", "action_id")
    @classmethod
    def identity_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("action identity fields must not be whitespace")
        return value


class TripleRecord(BaseModel):
    """One candidate materialized for exactly one original action."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[TRIPLE_RECORD_SCHEMA_VERSION] = TRIPLE_RECORD_SCHEMA_VERSION
    triple_id: str = Field(min_length=64, max_length=64)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    assistant_turn_id: int = Field(ge=0)
    action_index_in_turn: int = Field(ge=0)
    candidate_id: str = Field(min_length=64, max_length=64)
    subject: str = Field(min_length=1)
    category: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float
    evidence: Tuple[EvidenceSpan, ...] = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    extractor_kind: ExtractorKind
    model_version: str = Field(min_length=1)

    @field_validator("triple_id", "candidate_id")
    @classmethod
    def ids_must_be_sha256(cls, value: str) -> str:
        return _require_digest(value, "record ID")

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_finite_unit_interval(cls, value: float) -> float:
        if (
            isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "TripleRecord":
        candidate = self.as_candidate()
        if self.candidate_id != candidate.candidate_id:
            raise ValueError("candidate_id does not match triple semantic content")
        if self.triple_id != self.expected_triple_id():
            raise ValueError("triple_id does not match action-bound content")
        return self

    def as_candidate(self) -> TripleCandidate:
        return TripleCandidate(
            candidate_id=self.candidate_id,
            subject=self.subject,
            category=self.category,
            value=self.value,
            confidence=self.confidence,
            evidence=self.evidence,
            extractor_version=self.extractor_version,
            extractor_kind=self.extractor_kind,
            model_version=self.model_version,
        )

    def expected_triple_id(self) -> str:
        return canonical_digest(
            {
                "namespace": TRIPLE_RECORD_SCHEMA_VERSION,
                "task_id": self.task_id,
                "rollout_id": self.rollout_id,
                "stage_id": self.stage_id,
                "timestep": self.timestep,
                "action_id": self.action_id,
                "assistant_turn_id": self.assistant_turn_id,
                "action_index_in_turn": self.action_index_in_turn,
                "candidate_id": self.candidate_id,
            }
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: TripleCandidate,
        binding: ActionBinding,
        *,
        source_texts: Mapping[EvidenceSource, str],
    ) -> "TripleRecord":
        for evidence in candidate.evidence:
            try:
                source_text = source_texts[evidence.source]
            except KeyError as exc:
                raise ValueError(
                    f"missing source text for evidence source {evidence.source!r}"
                ) from exc
            evidence.validate_against(source_text)
        identity = {
            "namespace": TRIPLE_RECORD_SCHEMA_VERSION,
            "task_id": binding.task_id,
            "rollout_id": binding.rollout_id,
            "stage_id": binding.stage_id,
            "timestep": binding.timestep,
            "action_id": binding.action_id,
            "assistant_turn_id": binding.assistant_turn_id,
            "action_index_in_turn": binding.action_index_in_turn,
            "candidate_id": candidate.candidate_id,
        }
        return cls(
            triple_id=canonical_digest(identity),
            **binding.model_dump(mode="python"),
            **candidate.model_dump(mode="python", exclude={"schema_version"}),
        )


class RelevanceDecision(BaseModel):
    """Action-bound relevance label kept outside cacheable triple content."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["agemem.relevance_decision.v1"] = (
        "agemem.relevance_decision.v1"
    )
    decision_id: str = Field(min_length=64, max_length=64)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    triple_id: str = Field(min_length=64, max_length=64)
    role: SemanticRole
    confidence: float
    decision_version: str = Field(min_length=1)

    @field_validator("decision_id", "triple_id")
    @classmethod
    def ids_must_be_sha256(cls, value: str) -> str:
        return _require_digest(value, "decision ID")

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_finite_unit_interval(cls, value: float) -> float:
        if (
            isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "RelevanceDecision":
        if self.decision_id != self.expected_decision_id():
            raise ValueError("decision_id does not match decision content")
        return self

    def expected_decision_id(self) -> str:
        return canonical_digest(
            {
                "namespace": self.schema_version,
                "task_id": self.task_id,
                "rollout_id": self.rollout_id,
                "stage_id": self.stage_id,
                "timestep": self.timestep,
                "action_id": self.action_id,
                "triple_id": self.triple_id,
                "role": self.role,
                "confidence": self.confidence,
                "decision_version": self.decision_version,
            }
        )

    @classmethod
    def create(
        cls,
        triple: TripleRecord,
        *,
        role: SemanticRole,
        confidence: float,
        decision_version: str,
    ) -> "RelevanceDecision":
        payload = {
            "task_id": triple.task_id,
            "rollout_id": triple.rollout_id,
            "stage_id": triple.stage_id,
            "timestep": triple.timestep,
            "action_id": triple.action_id,
            "triple_id": triple.triple_id,
            "role": role,
            "confidence": confidence,
            "decision_version": decision_version,
        }
        return cls(
            decision_id=canonical_digest(
                {"namespace": "agemem.relevance_decision.v1", **payload}
            ),
            **payload,
        )


class APRecord(BaseModel):
    """A grounded atomic proposition with explicit action provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[AP_RECORD_SCHEMA_VERSION] = AP_RECORD_SCHEMA_VERSION
    ap_id: str = Field(min_length=64, max_length=64)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    proposition: str = Field(min_length=1)
    confidence: float
    evidence_triple_ids: Tuple[str, ...] = ()
    evidence_state_fact_ids: Tuple[str, ...] = ()
    evidence_memory_ids: Tuple[str, ...] = ()
    evidence_action_ids: Tuple[str, ...] = Field(min_length=1)
    grounder_version: str = Field(min_length=1)

    @field_validator("ap_id")
    @classmethod
    def ap_id_must_be_sha256(cls, value: str) -> str:
        return _require_digest(value, "ap_id")

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_finite_unit_interval(cls, value: float) -> float:
        if (
            isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_provenance_and_identity(self) -> "APRecord":
        evidence_groups = (
            self.evidence_triple_ids,
            self.evidence_state_fact_ids,
            self.evidence_memory_ids,
            self.evidence_action_ids,
        )
        if any(len(values) != len(set(values)) for values in evidence_groups):
            raise ValueError("AP evidence IDs must be unique within each evidence type")
        if self.action_id not in self.evidence_action_ids:
            raise ValueError("evidence_action_ids must include the source action_id")
        if self.ap_id != self.expected_ap_id():
            raise ValueError("ap_id does not match AP content")
        return self

    def identity_payload(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "stage_id": self.stage_id,
            "timestep": self.timestep,
            "action_id": self.action_id,
            "proposition": self.proposition,
            "confidence": self.confidence,
            "evidence_triple_ids": self.evidence_triple_ids,
            "evidence_state_fact_ids": self.evidence_state_fact_ids,
            "evidence_memory_ids": self.evidence_memory_ids,
            "evidence_action_ids": self.evidence_action_ids,
            "grounder_version": self.grounder_version,
        }

    def expected_ap_id(self) -> str:
        return canonical_digest(
            {"namespace": AP_RECORD_SCHEMA_VERSION, **self.identity_payload()}
        )

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        rollout_id: str,
        stage_id: int,
        timestep: int,
        action_id: str,
        proposition: str,
        confidence: float,
        evidence_triple_ids: Tuple[str, ...] = (),
        evidence_state_fact_ids: Tuple[str, ...] = (),
        evidence_memory_ids: Tuple[str, ...] = (),
        evidence_action_ids: Optional[Tuple[str, ...]] = None,
        grounder_version: str,
    ) -> "APRecord":
        payload: Dict[str, object] = {
            "task_id": task_id,
            "rollout_id": rollout_id,
            "stage_id": stage_id,
            "timestep": timestep,
            "action_id": action_id,
            "proposition": proposition,
            "confidence": confidence,
            "evidence_triple_ids": evidence_triple_ids,
            "evidence_state_fact_ids": evidence_state_fact_ids,
            "evidence_memory_ids": evidence_memory_ids,
            "evidence_action_ids": evidence_action_ids or (action_id,),
            "grounder_version": grounder_version,
        }
        return cls(
            ap_id=canonical_digest({"namespace": AP_RECORD_SCHEMA_VERSION, **payload}),
            **payload,
        )


class QuarantinedCandidate(BaseModel):
    """Non-executable audit record for rejected extractor output."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    reason: QuarantineReason
    candidate_index: Optional[int] = Field(default=None, ge=0)
    raw_digest: str = Field(min_length=64, max_length=64)
    subject: Optional[str] = None
    category: Optional[str] = None
    value: Optional[str] = None
    message: str = Field(min_length=1)

    @field_validator("raw_digest")
    @classmethod
    def raw_digest_must_be_sha256(cls, value: str) -> str:
        return _require_digest(value, "raw_digest")


class ExtractionDiagnostics(BaseModel):
    """Deterministic accepted/quarantined counts and audit details."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    accepted_count: int = Field(ge=0)
    quarantined_count: int = Field(ge=0)
    quarantine: Tuple[QuarantinedCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_count(self) -> "ExtractionDiagnostics":
        if self.quarantined_count != len(self.quarantine):
            raise ValueError("quarantined_count must equal quarantine length")
        return self


class ExtractionResult(BaseModel):
    """Validated action-independent output from one extractor invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["agemem.extraction_result.v1"] = (
        "agemem.extraction_result.v1"
    )
    request_digest: str = Field(min_length=64, max_length=64)
    extractor_version: str = Field(min_length=1)
    extractor_kind: ExtractorKind
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    candidates: Tuple[TripleCandidate, ...] = ()
    diagnostics: ExtractionDiagnostics

    @field_validator("request_digest")
    @classmethod
    def request_digest_must_be_sha256(cls, value: str) -> str:
        return _require_digest(value, "request_digest")

    @model_validator(mode="after")
    def validate_result(self) -> "ExtractionResult":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("extraction candidate IDs must be unique")
        if self.diagnostics.accepted_count != len(self.candidates):
            raise ValueError("accepted_count must equal candidates length")
        for candidate in self.candidates:
            identity = (
                candidate.extractor_version,
                candidate.extractor_kind,
                candidate.model_version,
            )
            if identity != (
                self.extractor_version,
                self.extractor_kind,
                self.model_version,
            ):
                raise ValueError("candidate extractor provenance mismatches result")
        return self


__all__ = [
    "AP_RECORD_SCHEMA_VERSION",
    "EXTRACTION_SCHEMA_VERSION",
    "EXTRACTOR_OUTPUT_SCHEMA_VERSION",
    "TRIPLE_CANDIDATE_SCHEMA_VERSION",
    "TRIPLE_RECORD_SCHEMA_VERSION",
    "APRecord",
    "ActionBinding",
    "EvidenceSource",
    "EvidenceSpan",
    "ExtractionDiagnostics",
    "ExtractionRequest",
    "ExtractionResult",
    "ExtractorKind",
    "QuarantineReason",
    "QuarantinedCandidate",
    "RelevanceDecision",
    "SemanticRole",
    "TripleCandidate",
    "TripleRecord",
    "canonical_digest",
    "text_digest",
]
