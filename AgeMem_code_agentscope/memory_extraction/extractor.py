"""Fail-closed M6 triple extractor interfaces and deterministic adapters."""

from __future__ import annotations

import json
import math
import threading
from typing import (
    Callable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .models import (
    EXTRACTOR_OUTPUT_SCHEMA_VERSION,
    EvidenceSource,
    EvidenceSpan,
    ExtractionDiagnostics,
    ExtractionRequest,
    ExtractionResult,
    ExtractorKind,
    QuarantineReason,
    QuarantinedCandidate,
    TripleCandidate,
    canonical_digest,
    text_digest,
)


class RawEvidenceSpan(BaseModel):
    """Strict wire representation accepted from an untrusted extractor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source: EvidenceSource
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class RawTriple(BaseModel):
    """Strict wire triple before registry and evidence grounding."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    subject: str = Field(min_length=1)
    category: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float
    evidence: Tuple[RawEvidenceSpan, ...] = Field(min_length=1)

    @field_validator("subject", "category", "value")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("triple text fields must not be whitespace")
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


class RawExtractionPayload(BaseModel):
    """The only JSON object shape accepted from mock or injected clients."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[EXTRACTOR_OUTPUT_SCHEMA_VERSION]
    triples: Tuple[RawTriple, ...]


@runtime_checkable
class TripleExtractor(Protocol):
    """Minimal synchronous interface used by offline extraction and cache."""

    extractor_kind: ExtractorKind
    extractor_version: str
    model_version: str
    prompt_version: str

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """Extract action-independent candidates without mutating state."""


class InjectedCompletionClient(Protocol):
    """Narrow protocol for a caller-owned LLM or deterministic fake client."""

    def complete(self, *, prompt: str) -> str:
        """Return exactly one JSON object as text."""


def _raw_digest(raw: object) -> str:
    if isinstance(raw, str):
        return text_digest(raw)
    try:
        return canonical_digest(raw)
    except (TypeError, ValueError):
        return text_digest(
            f"unserializable:{type(raw).__module__}.{type(raw).__name__}"
        )


def _quarantine(
    *,
    reason: QuarantineReason,
    raw: object,
    message: str,
    candidate_index: Optional[int] = None,
    subject: Optional[str] = None,
    category: Optional[str] = None,
    value: Optional[str] = None,
) -> QuarantinedCandidate:
    return QuarantinedCandidate(
        reason=reason,
        candidate_index=candidate_index,
        raw_digest=_raw_digest(raw),
        subject=subject,
        category=category,
        value=value,
        message=message,
    )


def _empty_result(
    request: ExtractionRequest,
    *,
    extractor_kind: ExtractorKind,
    extractor_version: str,
    model_version: str,
    prompt_version: str,
    quarantine: Tuple[QuarantinedCandidate, ...],
) -> ExtractionResult:
    return ExtractionResult(
        request_digest=request.request_digest,
        extractor_kind=extractor_kind,
        extractor_version=extractor_version,
        model_version=model_version,
        prompt_version=prompt_version,
        candidates=(),
        diagnostics=ExtractionDiagnostics(
            accepted_count=0,
            quarantined_count=len(quarantine),
            quarantine=quarantine,
        ),
    )


def _validate_payload(
    request: ExtractionRequest,
    raw_payload: object,
    *,
    extractor_kind: ExtractorKind,
    extractor_version: str,
    model_version: str,
    prompt_version: str,
) -> ExtractionResult:
    """Validate structure, registries, and exact evidence before acceptance."""

    try:
        strict_json = json.dumps(
            raw_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload = RawExtractionPayload.model_validate_json(strict_json)
    except (TypeError, ValueError, ValidationError) as exc:
        quarantine = (
            _quarantine(
                reason="invalid_schema",
                raw=raw_payload,
                message=(
                    "extractor output failed schema validation: "
                    f"{getattr(exc, 'title', type(exc).__name__)}"
                ),
            ),
        )
        return _empty_result(
            request,
            extractor_kind=extractor_kind,
            extractor_version=extractor_version,
            model_version=model_version,
            prompt_version=prompt_version,
            quarantine=quarantine,
        )

    accepted = []
    quarantined = []
    known_candidate_ids = set()
    for index, raw_triple in enumerate(payload.triples):
        raw_dict = raw_triple.model_dump(mode="json")
        identity = {
            "candidate_index": index,
            "subject": raw_triple.subject,
            "category": raw_triple.category,
            "value": raw_triple.value,
        }
        if not request.accepts_subject(raw_triple.subject):
            quarantined.append(
                _quarantine(
                    reason="unknown_subject",
                    raw=raw_dict,
                    message="subject is not present in the request subject registry",
                    **identity,
                )
            )
            continue
        if not request.accepts_category(raw_triple.category):
            quarantined.append(
                _quarantine(
                    reason="unknown_category",
                    raw=raw_dict,
                    message="category is not present in the request category registry",
                    **identity,
                )
            )
            continue

        evidence = []
        try:
            for raw_span in raw_triple.evidence:
                source_text = request.source_text(raw_span.source)
                span = EvidenceSpan.from_source(
                    source=raw_span.source,
                    source_text=source_text,
                    start=raw_span.start,
                    end=raw_span.end,
                )
                if raw_span.text != span.text:
                    raise ValueError(
                        "declared evidence text does not equal source_text[start:end]"
                    )
                evidence.append(span)
        except (KeyError, ValueError) as exc:
            quarantined.append(
                _quarantine(
                    reason="invalid_evidence",
                    raw=raw_dict,
                    message=str(exc),
                    **identity,
                )
            )
            continue

        candidate = TripleCandidate.create(
            subject=raw_triple.subject,
            category=raw_triple.category,
            value=raw_triple.value,
            confidence=raw_triple.confidence,
            evidence=tuple(evidence),
            extractor_version=extractor_version,
            extractor_kind=extractor_kind,
            model_version=model_version,
        )
        if candidate.candidate_id in known_candidate_ids:
            quarantined.append(
                _quarantine(
                    reason="duplicate_candidate",
                    raw=raw_dict,
                    message="duplicate canonical candidate in one extractor response",
                    **identity,
                )
            )
            continue
        known_candidate_ids.add(candidate.candidate_id)
        accepted.append(candidate)

    return ExtractionResult(
        request_digest=request.request_digest,
        extractor_kind=extractor_kind,
        extractor_version=extractor_version,
        model_version=model_version,
        prompt_version=prompt_version,
        candidates=tuple(accepted),
        diagnostics=ExtractionDiagnostics(
            accepted_count=len(accepted),
            quarantined_count=len(quarantined),
            quarantine=tuple(quarantined),
        ),
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _decode_single_json_object(raw: str) -> Mapping[str, object]:
    """Parse exactly one RFC-compatible JSON object and reject NaN/Infinity."""

    if not isinstance(raw, str):
        raise TypeError("client output must be a string")
    parsed = json.loads(raw, parse_constant=_reject_json_constant)
    if not isinstance(parsed, dict):
        raise ValueError("client output must be one JSON object")
    return parsed


class MockTripleExtractor:
    """Deterministic fixture extractor; it never invokes a model or network."""

    extractor_kind: Literal["mock"] = "mock"

    def __init__(
        self,
        responses: Mapping[str, object],
        *,
        extractor_version: str = "mock-v1",
        model_version: str = "fixture-v1",
        prompt_version: str = "triple-prompt-v1",
    ) -> None:
        if (
            not extractor_version.strip()
            or not model_version.strip()
            or not prompt_version.strip()
        ):
            raise ValueError("extractor, model, and prompt versions must be non-blank")
        self._responses = dict(responses)
        self.extractor_version = extractor_version
        self.model_version = model_version
        self.prompt_version = prompt_version
        self._lock = threading.Lock()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        with self._lock:
            self._call_count += 1
        raw = self._responses.get(
            request.observation_digest,
            self._responses.get(
                request.observation,
                {"schema_version": EXTRACTOR_OUTPUT_SCHEMA_VERSION, "triples": []},
            ),
        )
        if isinstance(raw, str):
            try:
                raw = _decode_single_json_object(raw)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return _empty_result(
                    request,
                    extractor_kind=self.extractor_kind,
                    extractor_version=self.extractor_version,
                    model_version=self.model_version,
                    prompt_version=self.prompt_version,
                    quarantine=(
                        _quarantine(
                            reason="invalid_json",
                            raw=raw,
                            message=f"mock output is not strict JSON: {exc}",
                        ),
                    ),
                )
        return _validate_payload(
            request,
            raw,
            extractor_kind=self.extractor_kind,
            extractor_version=self.extractor_version,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
        )


class LLMTripleExtractor:
    """Strict adapter around a caller-injected client; no SDK client is created."""

    extractor_kind: Literal["llm"] = "llm"

    def __init__(
        self,
        client: InjectedCompletionClient | Callable[[str], str],
        *,
        extractor_version: str,
        model_version: str,
        prompt_version: str,
    ) -> None:
        if client is None:
            raise ValueError("an injected client or callable is required")
        if (
            not extractor_version.strip()
            or not model_version.strip()
            or not prompt_version.strip()
        ):
            raise ValueError("extractor, model, and prompt versions must be non-blank")
        self._client = client
        self.extractor_version = extractor_version
        self.model_version = model_version
        self.prompt_version = prompt_version

    def build_prompt(self, request: ExtractionRequest) -> str:
        """Build a deterministic prompt whose output contract is explicit."""

        contract = {
            "schema_version": EXTRACTOR_OUTPUT_SCHEMA_VERSION,
            "triples": [
                {
                    "subject": "one exact registered subject",
                    "category": "one exact registered category",
                    "value": "grounded value",
                    "confidence": 0.0,
                    "evidence": [
                        {
                            "source": "observation",
                            "text": "exact source slice",
                            "start": 0,
                            "end": 1,
                        }
                    ],
                }
            ],
        }
        sections = [
            "Return exactly one JSON object and no prose or markdown.",
            f"Schema example: {json.dumps(contract, ensure_ascii=False, sort_keys=True)}",
            f"Known subjects: {json.dumps(request.known_subjects, ensure_ascii=False)}",
            f"Allowed categories: {json.dumps(request.allowed_categories, ensure_ascii=False)}",
            f"Observation: {request.observation}",
        ]
        if request.question is not None:
            sections.append(f"Question: {request.question}")
        return "\n".join(sections)

    def _complete(self, prompt: str) -> str:
        complete = getattr(self._client, "complete", None)
        if callable(complete):
            return complete(prompt=prompt)
        if callable(self._client):
            return self._client(prompt)
        raise TypeError(
            "injected client must be callable or expose complete(prompt=...)"
        )

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        prompt = self.build_prompt(request)
        try:
            raw = self._complete(prompt)
        except Exception as exc:  # injected boundary is intentionally fail-closed
            return _empty_result(
                request,
                extractor_kind=self.extractor_kind,
                extractor_version=self.extractor_version,
                model_version=self.model_version,
                prompt_version=self.prompt_version,
                quarantine=(
                    _quarantine(
                        reason="extractor_error",
                        raw=f"{type(exc).__module__}.{type(exc).__name__}",
                        message=f"injected client failed with {type(exc).__name__}",
                    ),
                ),
            )
        try:
            payload = _decode_single_json_object(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return _empty_result(
                request,
                extractor_kind=self.extractor_kind,
                extractor_version=self.extractor_version,
                model_version=self.model_version,
                prompt_version=self.prompt_version,
                quarantine=(
                    _quarantine(
                        reason="invalid_json",
                        raw=raw,
                        message=f"client output is not strict single-object JSON: {exc}",
                    ),
                ),
            )
        return _validate_payload(
            request,
            payload,
            extractor_kind=self.extractor_kind,
            extractor_version=self.extractor_version,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
        )


__all__ = [
    "InjectedCompletionClient",
    "LLMTripleExtractor",
    "MockTripleExtractor",
    "RawEvidenceSpan",
    "RawExtractionPayload",
    "RawTriple",
    "TripleExtractor",
]
