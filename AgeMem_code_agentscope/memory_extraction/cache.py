"""Thread-safe, provenance-preserving group cache for M6 extraction."""

from __future__ import annotations

import threading
from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .extractor import TripleExtractor
from .models import (
    ActionBinding,
    ExtractionDiagnostics,
    ExtractionRequest,
    ExtractionResult,
    ExtractorKind,
    TripleCandidate,
    TripleRecord,
    canonical_digest,
)


class ExtractionCacheError(ValueError):
    """Raised when an extractor result or materialization violates provenance."""


class ExtractionCacheKey(BaseModel):
    """All inputs that may change cached action-independent candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["agemem.group_cache_key.v1"] = "agemem.group_cache_key.v1"
    task_id: str = Field(min_length=1)
    split_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    observation_digest: str = Field(min_length=64, max_length=64)
    question_digest: Optional[str] = Field(default=None, min_length=64, max_length=64)
    constraint_digest: str = Field(min_length=64, max_length=64)
    request_schema_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    extractor_kind: ExtractorKind
    extractor_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)

    @field_validator("observation_digest", "question_digest", "constraint_digest")
    @classmethod
    def digests_must_be_lowercase_sha256(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("cache digests must be lowercase SHA-256 values")
        return value

    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))

    @classmethod
    def from_request(
        cls, request: ExtractionRequest, extractor: TripleExtractor
    ) -> "ExtractionCacheKey":
        return cls(
            task_id=request.task_id,
            split_id=request.split_id,
            group_id=request.group_id,
            stage_id=request.stage_id,
            observation_digest=request.observation_digest,
            question_digest=request.question_digest,
            constraint_digest=request.constraint_digest,
            request_schema_version=request.schema_version,
            prompt_version=extractor.prompt_version,
            extractor_kind=extractor.extractor_kind,
            extractor_version=extractor.extractor_version,
            model_version=extractor.model_version,
        )


class CacheLookup(BaseModel):
    """One cache lookup result; candidates remain action-independent."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cache_key_digest: str = Field(min_length=64, max_length=64)
    cache_hit: bool
    result: ExtractionResult


class MaterializedExtraction(BaseModel):
    """Lookup metadata plus records rebound to one original action."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cache_key_digest: str = Field(min_length=64, max_length=64)
    cache_hit: bool
    candidates: Tuple[TripleCandidate, ...]
    records: Tuple[TripleRecord, ...]


class GroupExtractionCache:
    """Cache only candidates, never APs, relevance roles, or action records.

    Extraction runs under the cache lock so concurrent identical lookups invoke
    the injected extractor once. This is intentionally conservative: M6 smoke
    groups are small, and provenance determinism is more important than
    parallel calls for a single key.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Tuple[TripleCandidate, ...]] = {}
        self._diagnostics: Dict[str, ExtractionDiagnostics] = {}
        self._keys: Dict[str, ExtractionCacheKey] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @staticmethod
    def _validate_candidates(
        request: ExtractionRequest, candidates: Tuple[TripleCandidate, ...]
    ) -> None:
        candidate_ids = set()
        for candidate in candidates:
            if candidate.candidate_id in candidate_ids:
                raise ExtractionCacheError("extractor returned duplicate candidate IDs")
            candidate_ids.add(candidate.candidate_id)
            if not request.accepts_subject(candidate.subject):
                raise ExtractionCacheError(
                    "extractor bypassed unknown-subject quarantine"
                )
            if not request.accepts_category(candidate.category):
                raise ExtractionCacheError(
                    "extractor bypassed unknown-category quarantine"
                )
            for evidence in candidate.evidence:
                try:
                    source_text = request.source_text(evidence.source)
                except KeyError as exc:
                    raise ExtractionCacheError(
                        f"candidate references unavailable source {evidence.source!r}"
                    ) from exc
                try:
                    evidence.validate_against(source_text)
                except ValueError as exc:
                    raise ExtractionCacheError(
                        "candidate evidence failed exact source validation"
                    ) from exc

    @staticmethod
    def _validate_result(
        request: ExtractionRequest,
        key: ExtractionCacheKey,
        result: ExtractionResult,
    ) -> None:
        if result.request_digest != request.request_digest:
            raise ExtractionCacheError("extractor result request digest mismatch")
        identity = (
            result.extractor_kind,
            result.extractor_version,
            result.model_version,
            result.prompt_version,
        )
        expected = (
            key.extractor_kind,
            key.extractor_version,
            key.model_version,
            key.prompt_version,
        )
        if identity != expected:
            raise ExtractionCacheError("extractor result version identity mismatch")
        GroupExtractionCache._validate_candidates(request, result.candidates)

    def get_or_extract(
        self, request: ExtractionRequest, extractor: TripleExtractor
    ) -> CacheLookup:
        """Return cached candidates or perform one validated extraction."""

        key = ExtractionCacheKey.from_request(request, extractor)
        key_digest = key.digest
        with self._lock:
            candidates = self._entries.get(key_digest)
            if candidates is not None:
                self._hits += 1
                result = ExtractionResult(
                    request_digest=request.request_digest,
                    extractor_kind=key.extractor_kind,
                    extractor_version=key.extractor_version,
                    model_version=key.model_version,
                    prompt_version=key.prompt_version,
                    candidates=tuple(
                        candidate.model_copy(deep=True) for candidate in candidates
                    ),
                    diagnostics=self._diagnostics[key_digest].model_copy(deep=True),
                )
                return CacheLookup(
                    cache_key_digest=key_digest,
                    cache_hit=True,
                    result=result,
                )

            self._misses += 1
            result = extractor.extract(request)
            self._validate_result(request, key, result)
            # Only immutable, action-independent candidates and their audit
            # diagnostics enter the cache. No action IDs, relevance roles,
            # TripleRecords, or APRecords are retained here.
            self._entries[key_digest] = tuple(
                candidate.model_copy(deep=True) for candidate in result.candidates
            )
            self._diagnostics[key_digest] = result.diagnostics.model_copy(deep=True)
            self._keys[key_digest] = key
            return CacheLookup(
                cache_key_digest=key_digest,
                cache_hit=False,
                result=result.model_copy(deep=True),
            )

    def materialize(
        self,
        request: ExtractionRequest,
        candidates: Tuple[TripleCandidate, ...],
        binding: ActionBinding,
    ) -> Tuple[TripleRecord, ...]:
        """Revalidate source spans and bind candidates to exactly one action."""

        if (binding.task_id, binding.rollout_id, binding.stage_id) != (
            request.task_id,
            request.rollout_id,
            request.stage_id,
        ):
            raise ExtractionCacheError(
                "action binding task/rollout/stage does not match extraction request"
            )
        self._validate_candidates(request, candidates)
        source_texts = {"observation": request.observation}
        if request.question is not None:
            source_texts["question"] = request.question
        try:
            return tuple(
                TripleRecord.from_candidate(
                    candidate,
                    binding,
                    source_texts=source_texts,
                )
                for candidate in candidates
            )
        except ValueError as exc:
            raise ExtractionCacheError("candidate materialization failed") from exc

    def get_or_extract_and_materialize(
        self,
        request: ExtractionRequest,
        extractor: TripleExtractor,
        binding: ActionBinding,
    ) -> MaterializedExtraction:
        lookup = self.get_or_extract(request, extractor)
        records = self.materialize(request, lookup.result.candidates, binding)
        return MaterializedExtraction(
            cache_key_digest=lookup.cache_key_digest,
            cache_hit=lookup.cache_hit,
            candidates=lookup.result.candidates,
            records=records,
        )

    def invalidate(
        self,
        *,
        extractor_version: Optional[str] = None,
        model_version: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> int:
        """Remove matching versions; omitted filters clear all entries."""

        with self._lock:
            doomed = [
                digest
                for digest, key in self._keys.items()
                if (
                    extractor_version is None
                    or key.extractor_version == extractor_version
                )
                and (model_version is None or key.model_version == model_version)
                and (prompt_version is None or key.prompt_version == prompt_version)
            ]
            for digest in doomed:
                del self._entries[digest]
                del self._diagnostics[digest]
                del self._keys[digest]
            return len(doomed)


__all__ = [
    "CacheLookup",
    "ExtractionCacheError",
    "ExtractionCacheKey",
    "GroupExtractionCache",
    "MaterializedExtraction",
]
