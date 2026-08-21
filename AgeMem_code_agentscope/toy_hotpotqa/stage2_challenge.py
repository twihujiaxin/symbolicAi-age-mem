"""Deterministic Stage-2 benchmark for query-delayed context control.

This module is intentionally separate from the frozen M3 environment and M5
adapter.  It evaluates context-selection policies offline with private Oracle
segment labels; it does not add actions to ``HotpotQAToyEnvironment`` and does
not call an LLM or tokenizer service.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Literal, Optional, Protocol, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)


ChallengeSplit = Literal["dev", "test"]
ChallengeScenario = Literal[
    "hard_negative",
    "partial_relevance",
    "delayed_relevance",
]
SegmentRole = Literal["future_support", "distractor"]
MessageKind = Literal[
    "hard_negative",
    "partial_relevance",
    "delayed_relevance",
    "simple_distractor",
]
Stage2PolicyName = Literal[
    "always_keep",
    "always_clear",
    "opaque_id_control",
    "oracle_safe_compress",
]

MIN_CHALLENGE_CASES = 3
MAX_CHALLENGE_CASES = 12
MAX_MESSAGES_PER_CASE = 8
MAX_SEGMENTS_PER_MESSAGE = 4
MAX_CONTEXT_TOKENS = 128
STAGE2_TOKEN_COUNTER_NAME = "unicode-lexical-v1"
STAGE2_BUDGET_SCOPE = "retained_segment_text_only"
STAGE2_PUBLIC_HANDLE_VERSION = "stage2-opaque-handles-v1"


def deterministic_token_count(text: str) -> int:
    """Return a local token proxy used consistently by all benchmark arms."""

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _opaque_handle(task_id: str, seed: int, kind: str, private_id: str) -> str:
    material = (
        f"{STAGE2_PUBLIC_HANDLE_VERSION}\0{task_id}\0{seed}\0{kind}\0{private_id}"
    ).encode("utf-8")
    return f"{kind}-{hashlib.sha256(material).hexdigest()[:24]}"


class Stage2ChallengeSegment(BaseModel):
    """One indivisible unit for deterministic keep/drop evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    oracle_role: SegmentRole


class Stage2ChallengeMessage(BaseModel):
    """One Stage-2 message with private, segment-level challenge annotations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str = Field(min_length=1)
    kind: MessageKind
    segments: Tuple[Stage2ChallengeSegment, ...] = Field(
        min_length=1,
        max_length=MAX_SEGMENTS_PER_MESSAGE,
    )
    shared_terms: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_kind(self) -> "Stage2ChallengeMessage":
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment_id values must be unique within a message")
        roles = {segment.oracle_role for segment in self.segments}
        if self.kind in {"hard_negative", "simple_distractor"}:
            if roles != {"distractor"}:
                raise ValueError(f"{self.kind} messages must contain only distractors")
        elif self.kind == "delayed_relevance":
            if roles != {"future_support"}:
                raise ValueError(
                    "delayed_relevance messages must contain only future support"
                )
        elif roles != {"future_support", "distractor"}:
            raise ValueError(
                "partial_relevance messages must mix future support and distractors"
            )
        if self.kind == "hard_negative":
            if not self.shared_terms:
                raise ValueError("hard_negative messages require shared_terms")
        elif self.shared_terms:
            raise ValueError("shared_terms are only valid for hard_negative messages")
        if len(self.shared_terms) != len(
            set(term.casefold() for term in self.shared_terms)
        ):
            raise ValueError("shared_terms must be unique case-insensitively")
        return self


class PublicStage2Segment(BaseModel):
    """Segment visible without future-query fields or Oracle annotations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str
    text: str


class PublicStage2Message(BaseModel):
    """Message visible without future-query fields or Oracle annotations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str
    segments: Tuple[PublicStage2Segment, ...]


class PublicStage2Input(BaseModel):
    """Stage-2 policy input; the future query is deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal[2] = 2
    protocol: Literal["query_delayed"] = "query_delayed"
    budget_scope: Literal["retained_segment_text_only"] = STAGE2_BUDGET_SCOPE
    seed: int = Field(ge=0)
    max_context_tokens: int = Field(ge=1, le=MAX_CONTEXT_TOKENS)
    messages: Tuple[PublicStage2Message, ...]
    observation: str

    def segment_ids(self) -> Tuple[str, ...]:
        return tuple(
            segment.segment_id
            for message in self.messages
            for segment in message.segments
        )


class Stage2ChallengeCase(BaseModel):
    """Private benchmark case with a query revealed only after Stage 2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    task_id: str = Field(min_length=1)
    split: ChallengeSplit
    scenario: ChallengeScenario
    protocol: Literal["query_delayed"] = "query_delayed"
    budget_scope: Literal["retained_segment_text_only"] = STAGE2_BUDGET_SCOPE
    future_query: str = Field(min_length=1)
    future_answer: str = Field(min_length=1)
    max_context_tokens: int = Field(ge=1, le=MAX_CONTEXT_TOKENS)
    messages: Tuple[Stage2ChallengeMessage, ...] = Field(
        min_length=2,
        max_length=MAX_MESSAGES_PER_CASE,
    )

    @model_validator(mode="after")
    def validate_annotations_and_budget(self) -> "Stage2ChallengeCase":
        message_ids = [message.message_id for message in self.messages]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("message_id values must be unique within a case")
        segments = [
            segment for message in self.messages for segment in message.segments
        ]
        segment_ids = [segment.segment_id for segment in segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment_id values must be unique within a case")
        roles = {segment.oracle_role for segment in segments}
        if roles != {"future_support", "distractor"}:
            raise ValueError(
                "each challenge case requires support and distractor segments"
            )
        kinds = {message.kind for message in self.messages}
        if self.scenario not in kinds:
            raise ValueError(
                f"scenario {self.scenario!r} requires a matching annotated message"
            )

        query = self.future_query.casefold()
        for message in self.messages:
            if message.kind != "hard_negative":
                continue
            text = " ".join(segment.text for segment in message.segments).casefold()
            for term in message.shared_terms:
                folded = term.casefold().strip()
                if not folded or folded not in query or folded not in text:
                    raise ValueError(
                        "each hard-negative shared term must occur in both the "
                        "future query and the distractor text"
                    )

        support_tokens = sum(
            deterministic_token_count(segment.text)
            for segment in segments
            if segment.oracle_role == "future_support"
        )
        total_tokens = sum(
            deterministic_token_count(segment.text) for segment in segments
        )
        if support_tokens > self.max_context_tokens:
            raise ValueError("Oracle future support does not fit the context budget")
        if total_tokens <= self.max_context_tokens:
            raise ValueError("challenge input must exceed its context budget")

        support_text = _normalized_text(
            " ".join(
                segment.text
                for segment in segments
                if segment.oracle_role == "future_support"
            )
        )
        distractor_text = _normalized_text(
            " ".join(
                segment.text
                for segment in segments
                if segment.oracle_role == "distractor"
            )
        )
        answer = _normalized_text(self.future_answer)
        if answer not in support_text:
            raise ValueError("future_answer must be grounded in future-support text")
        if answer in distractor_text:
            raise ValueError("future_answer must not occur in distractor text")
        return self

    def public_input(self, seed: int) -> PublicStage2Input:
        """Build a deterministically shuffled view without future labels."""

        if seed < 0:
            raise ValueError("seed must be non-negative")
        messages = list(self.messages)
        seed_material = f"{self.task_id}:{seed}:stage2".encode("utf-8")
        local_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        random.Random(local_seed).shuffle(messages)
        public_messages = []
        for message in messages:
            public_segments = tuple(
                PublicStage2Segment(
                    segment_id=_opaque_handle(
                        self.task_id,
                        seed,
                        "segment",
                        segment.segment_id,
                    ),
                    text=segment.text,
                )
                for segment in sorted(
                    message.segments,
                    key=lambda item: _opaque_handle(
                        self.task_id,
                        seed,
                        "segment",
                        item.segment_id,
                    ),
                )
            )
            public_messages.append(
                PublicStage2Message(
                    message_id=_opaque_handle(
                        self.task_id,
                        seed,
                        "message",
                        message.message_id,
                    ),
                    segments=public_segments,
                )
            )
        public_messages = tuple(public_messages)
        lines = [
            "Stage 2 - Query-delayed context control.",
            "The future query is unavailable. Manage the messages within "
            f"a {self.max_context_tokens}-token retained-segment-text budget; "
            "handles, formatting, and control text are excluded.",
        ]
        for message in public_messages:
            lines.append(f"[{message.message_id}]")
            lines.extend(
                f"- [{segment.segment_id}] {segment.text}"
                for segment in message.segments
            )
        return PublicStage2Input(
            seed=seed,
            max_context_tokens=self.max_context_tokens,
            messages=public_messages,
            observation="\n".join(lines),
        )

    def segments(self) -> Tuple[Stage2ChallengeSegment, ...]:
        return tuple(
            segment for message in self.messages for segment in message.segments
        )


def default_stage2_challenge_path() -> Path:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "toy"
        / "stage2_context_challenges.json"
    )
    if source_path.exists():
        return source_path
    packaged_path = Path(__file__).with_name("data") / "stage2_context_challenges.json"
    if not packaged_path.is_file():
        raise FileNotFoundError(
            "Stage-2 challenge fixture is missing from source and package data"
        )
    return packaged_path


class Stage2ChallengeDataset:
    """Small, immutable and coverage-checked Stage-2 challenge collection."""

    def __init__(self, cases: Iterable[Stage2ChallengeCase]) -> None:
        self._cases = tuple(cases)
        if not MIN_CHALLENGE_CASES <= len(self._cases) <= MAX_CHALLENGE_CASES:
            raise ValueError(
                f"Stage-2 challenge requires {MIN_CHALLENGE_CASES} to "
                f"{MAX_CHALLENGE_CASES} cases"
            )
        task_ids = [case.task_id for case in self._cases]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Stage-2 challenge task IDs must be unique")
        if {case.split for case in self._cases} != {"dev", "test"}:
            raise ValueError("Stage-2 challenge requires dev and test cases")
        required = {"hard_negative", "partial_relevance", "delayed_relevance"}
        for split in ("dev", "test"):
            scenarios = {case.scenario for case in self._cases if case.split == split}
            if scenarios != required:
                raise ValueError(
                    f"Stage-2 {split} split must contain every required scenario"
                )
        self._by_id = {case.task_id: case for case in self._cases}

    @classmethod
    def from_json(cls, path: Optional[str | Path] = None) -> "Stage2ChallengeDataset":
        challenge_path = (
            Path(path) if path is not None else default_stage2_challenge_path()
        )
        try:
            raw = json.loads(challenge_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Stage-2 challenge JSON: {exc}") from exc
        try:
            cases = TypeAdapter(List[Stage2ChallengeCase]).validate_python(raw)
        except ValidationError as exc:
            raise ValueError(f"invalid Stage-2 challenge schema: {exc}") from exc
        return cls(cases)

    def all(self) -> List[Stage2ChallengeCase]:
        return [case.model_copy(deep=True) for case in self._cases]

    def split(self, split: ChallengeSplit) -> List[Stage2ChallengeCase]:
        return [
            case.model_copy(deep=True) for case in self._cases if case.split == split
        ]

    def get(self, task_id: str) -> Stage2ChallengeCase:
        try:
            return self._by_id[task_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"unknown Stage-2 challenge task_id {task_id!r}") from exc

    def __len__(self) -> int:
        return len(self._cases)

    def digest(self) -> str:
        """Bind reports to the complete private query/answer/label fixture."""

        payload = [
            case.model_dump(mode="json")
            for case in sorted(self._cases, key=lambda item: item.task_id)
        ]
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Stage2CompressionDecision(BaseModel):
    """Segment selection made before the future query is revealed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Stage2PolicyName
    kept_segment_ids: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "Stage2CompressionDecision":
        if len(self.kept_segment_ids) != len(set(self.kept_segment_ids)):
            raise ValueError("kept_segment_ids must be unique")
        return self


class Stage2ChallengePolicy(Protocol):
    """Offline fixed-policy contract used by the benchmark."""

    name: Stage2PolicyName

    def decide(
        self,
        public_input: PublicStage2Input,
        *,
        oracle_case: Optional[Stage2ChallengeCase] = None,
    ) -> Stage2CompressionDecision: ...


class AlwaysKeepPolicy:
    """Retain every visible segment without consulting Oracle labels."""

    name: Stage2PolicyName = "always_keep"

    def decide(
        self,
        public_input: PublicStage2Input,
        *,
        oracle_case: Optional[Stage2ChallengeCase] = None,
    ) -> Stage2CompressionDecision:
        del oracle_case
        return Stage2CompressionDecision(
            policy=self.name,
            kept_segment_ids=public_input.segment_ids(),
        )


class AlwaysClearPolicy:
    """Drop every visible segment without consulting Oracle labels."""

    name: Stage2PolicyName = "always_clear"

    def decide(
        self,
        public_input: PublicStage2Input,
        *,
        oracle_case: Optional[Stage2ChallengeCase] = None,
    ) -> Stage2CompressionDecision:
        del public_input, oracle_case
        return Stage2CompressionDecision(policy=self.name)


class OpaqueIdControlPolicy:
    """ID-only sanity control over role-independent, per-seed handles."""

    name: Stage2PolicyName = "opaque_id_control"

    def decide(
        self,
        public_input: PublicStage2Input,
        *,
        oracle_case: Optional[Stage2ChallengeCase] = None,
    ) -> Stage2CompressionDecision:
        del oracle_case
        segment_ids = public_input.segment_ids()
        return Stage2CompressionDecision(
            policy=self.name,
            kept_segment_ids=(min(segment_ids),) if segment_ids else (),
        )


class OracleSafeCompressPolicy:
    """Offline upper bound retaining only future-support segments."""

    name: Stage2PolicyName = "oracle_safe_compress"

    def decide(
        self,
        public_input: PublicStage2Input,
        *,
        oracle_case: Optional[Stage2ChallengeCase] = None,
    ) -> Stage2CompressionDecision:
        if oracle_case is None:
            raise ValueError("oracle_safe_compress requires the matching private case")
        expected_public = oracle_case.public_input(public_input.seed)
        if expected_public != public_input:
            raise ValueError("oracle_safe_compress requires the matching private case")
        support_ids = {
            _opaque_handle(
                oracle_case.task_id,
                public_input.seed,
                "segment",
                segment.segment_id,
            )
            for segment in oracle_case.segments()
            if segment.oracle_role == "future_support"
        }
        return Stage2CompressionDecision(
            policy=self.name,
            kept_segment_ids=tuple(
                segment_id
                for segment_id in public_input.segment_ids()
                if segment_id in support_ids
            ),
        )


class Stage2CaseMetrics(BaseModel):
    """Auditable per-case metrics calculated after query reveal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    split: ChallengeSplit
    scenario: ChallengeScenario
    policy: Stage2PolicyName
    budget_scope: Literal["retained_segment_text_only"] = STAGE2_BUDGET_SCOPE
    total_tokens: int = Field(ge=1)
    kept_tokens: int = Field(ge=0)
    max_context_tokens: int = Field(ge=1)
    total_support_segments: int = Field(ge=1)
    retained_support_segments: int = Field(ge=0)
    total_distractor_segments: int = Field(ge=1)
    removed_distractor_segments: int = Field(ge=0)
    removed_segment_count: int = Field(ge=0)
    future_support_recall: float = Field(ge=0.0, le=1.0)
    distractor_removal_recall: float = Field(ge=0.0, le=1.0)
    removal_precision: float = Field(ge=0.0, le=1.0)
    token_reduction: float = Field(ge=0.0, le=1.0)
    budget_compliant: bool
    future_support_recoverable: bool
    safe_success: bool

    @model_validator(mode="after")
    def validate_derived_metrics(self) -> "Stage2CaseMetrics":
        if self.kept_tokens > self.total_tokens:
            raise ValueError("kept_tokens cannot exceed total_tokens")
        if self.retained_support_segments > self.total_support_segments:
            raise ValueError("retained support count exceeds its total")
        if self.removed_distractor_segments > self.total_distractor_segments:
            raise ValueError("removed distractor count exceeds its total")
        expected_removed = (
            self.total_support_segments
            - self.retained_support_segments
            + self.removed_distractor_segments
        )
        if self.removed_segment_count != expected_removed:
            raise ValueError("removed segment count is inconsistent")
        expected_values = {
            "future_support_recall": _ratio(
                self.retained_support_segments,
                self.total_support_segments,
            ),
            "distractor_removal_recall": _ratio(
                self.removed_distractor_segments,
                self.total_distractor_segments,
            ),
            "removal_precision": _ratio(
                self.removed_distractor_segments,
                self.removed_segment_count,
            ),
            "token_reduction": (self.total_tokens - self.kept_tokens)
            / self.total_tokens,
        }
        for field_name, expected in expected_values.items():
            if abs(getattr(self, field_name) - expected) > 1e-12:
                raise ValueError(f"{field_name} is inconsistent with case counts")
        budget_compliant = self.kept_tokens <= self.max_context_tokens
        support_recoverable = (
            self.retained_support_segments == self.total_support_segments
        )
        if self.budget_compliant != budget_compliant:
            raise ValueError("budget_compliant is inconsistent with token counts")
        if self.future_support_recoverable != support_recoverable:
            raise ValueError(
                "future_support_recoverable is inconsistent with support counts"
            )
        if self.safe_success != (budget_compliant and support_recoverable):
            raise ValueError("safe_success is inconsistent with its components")
        return self


class Stage2AggregateMetrics(BaseModel):
    """Macro averages for one fixed policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Stage2PolicyName
    case_count: int = Field(ge=1)
    future_support_recall: float = Field(ge=0.0, le=1.0)
    distractor_removal_recall: float = Field(ge=0.0, le=1.0)
    removal_precision: float = Field(ge=0.0, le=1.0)
    token_reduction: float = Field(ge=0.0, le=1.0)
    budget_compliance_rate: float = Field(ge=0.0, le=1.0)
    safe_success_rate: float = Field(ge=0.0, le=1.0)


class Stage2BenchmarkReport(BaseModel):
    """Deterministic report for all case/policy combinations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    token_counter: Literal["unicode-lexical-v1"]
    budget_scope: Literal["retained_segment_text_only"] = STAGE2_BUDGET_SCOPE
    seed: int = Field(ge=0)
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=1)
    rows: Tuple[Stage2CaseMetrics, ...] = Field(min_length=1)
    aggregates: Dict[Stage2PolicyName, Stage2AggregateMetrics]

    @model_validator(mode="after")
    def validate_rows_and_aggregates(self) -> "Stage2BenchmarkReport":
        keys = [(row.task_id, row.policy) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Stage-2 rows must be unique by task_id and policy")
        task_ids = {row.task_id for row in self.rows}
        policies = {row.policy for row in self.rows}
        if len(task_ids) != self.case_count:
            raise ValueError("case_count does not match Stage-2 rows")
        if set(self.aggregates) != policies:
            raise ValueError("aggregate keys must match row policies")
        if len(self.rows) != self.case_count * len(policies):
            raise ValueError("each Stage-2 policy must cover every case exactly once")

        for task_id in task_ids:
            task_rows = [row for row in self.rows if row.task_id == task_id]
            if {row.policy for row in task_rows} != policies:
                raise ValueError("each Stage-2 task must contain every policy")
            references = {
                (
                    row.task_id,
                    row.split,
                    row.scenario,
                    row.budget_scope,
                    row.total_tokens,
                    row.max_context_tokens,
                    row.total_support_segments,
                    row.total_distractor_segments,
                )
                for row in task_rows
            }
            if len(references) != 1:
                raise ValueError("policy-independent Stage-2 case fields disagree")

        aggregate_fields = (
            "future_support_recall",
            "distractor_removal_recall",
            "removal_precision",
            "token_reduction",
            "budget_compliance_rate",
            "safe_success_rate",
        )
        for policy in policies:
            aggregate = self.aggregates[policy]
            policy_rows = [row for row in self.rows if row.policy == policy]
            if aggregate.policy != policy:
                raise ValueError("aggregate policy must match its map key")
            if aggregate.case_count != self.case_count:
                raise ValueError("aggregate case_count does not match report")
            expected = {
                "future_support_recall": fmean(
                    row.future_support_recall for row in policy_rows
                ),
                "distractor_removal_recall": fmean(
                    row.distractor_removal_recall for row in policy_rows
                ),
                "removal_precision": fmean(
                    row.removal_precision for row in policy_rows
                ),
                "token_reduction": fmean(row.token_reduction for row in policy_rows),
                "budget_compliance_rate": fmean(
                    float(row.budget_compliant) for row in policy_rows
                ),
                "safe_success_rate": fmean(
                    float(row.safe_success) for row in policy_rows
                ),
            }
            for field_name in aggregate_fields:
                if abs(getattr(aggregate, field_name) - expected[field_name]) > 1e-12:
                    raise ValueError(
                        f"aggregate {field_name} does not match Stage-2 rows"
                    )
        return self


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def evaluate_stage2_decision(
    case: Stage2ChallengeCase,
    decision: Stage2CompressionDecision,
    *,
    public_input: PublicStage2Input,
) -> Stage2CaseMetrics:
    """Reveal private labels and score one already-made Stage-2 decision."""

    expected_public = case.public_input(public_input.seed)
    if public_input != expected_public:
        raise ValueError("public_input does not match the private challenge case")
    segments = {
        _opaque_handle(
            case.task_id,
            public_input.seed,
            "segment",
            segment.segment_id,
        ): segment
        for segment in case.segments()
    }
    kept = set(decision.kept_segment_ids)
    unknown = kept - set(segments)
    if unknown:
        raise ValueError(f"decision contains unknown segment IDs: {sorted(unknown)}")
    support = {
        segment_id
        for segment_id, segment in segments.items()
        if segment.oracle_role == "future_support"
    }
    distractors = set(segments) - support
    removed = set(segments) - kept
    retained_support = support & kept
    removed_distractors = distractors & removed
    total_tokens = sum(
        deterministic_token_count(segment.text) for segment in segments.values()
    )
    kept_tokens = sum(deterministic_token_count(segments[item].text) for item in kept)
    support_recall = _ratio(len(retained_support), len(support))
    distractor_recall = _ratio(len(removed_distractors), len(distractors))
    removal_precision = _ratio(len(removed_distractors), len(removed))
    budget_compliant = kept_tokens <= case.max_context_tokens
    support_recoverable = retained_support == support
    return Stage2CaseMetrics(
        task_id=case.task_id,
        split=case.split,
        scenario=case.scenario,
        policy=decision.policy,
        budget_scope=case.budget_scope,
        total_tokens=total_tokens,
        kept_tokens=kept_tokens,
        max_context_tokens=case.max_context_tokens,
        total_support_segments=len(support),
        retained_support_segments=len(retained_support),
        total_distractor_segments=len(distractors),
        removed_distractor_segments=len(removed_distractors),
        removed_segment_count=len(removed),
        future_support_recall=support_recall,
        distractor_removal_recall=distractor_recall,
        removal_precision=removal_precision,
        token_reduction=(total_tokens - kept_tokens) / total_tokens,
        budget_compliant=budget_compliant,
        future_support_recoverable=support_recoverable,
        safe_success=budget_compliant and support_recoverable,
    )


def run_stage2_challenge_benchmark(
    dataset: Optional[Stage2ChallengeDataset] = None,
    *,
    seed: int = 2026,
    policies: Optional[Iterable[Stage2ChallengePolicy]] = None,
) -> Stage2BenchmarkReport:
    """Run fixed offline baselines without an LLM or the M3 runtime."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    selected_dataset = dataset or Stage2ChallengeDataset.from_json()
    selected_policies = (
        (
            AlwaysKeepPolicy(),
            AlwaysClearPolicy(),
            OpaqueIdControlPolicy(),
            OracleSafeCompressPolicy(),
        )
        if policies is None
        else tuple(policies)
    )
    names = [policy.name for policy in selected_policies]
    if not names or len(names) != len(set(names)):
        raise ValueError("benchmark policies must be non-empty and uniquely named")

    rows: List[Stage2CaseMetrics] = []
    for case in sorted(selected_dataset.all(), key=lambda item: item.task_id):
        public_input = case.public_input(seed)
        for policy in selected_policies:
            oracle_case = case if type(policy) is OracleSafeCompressPolicy else None
            decision = policy.decide(public_input, oracle_case=oracle_case)
            if decision.policy != policy.name:
                raise ValueError("policy returned a decision under a different name")
            rows.append(
                evaluate_stage2_decision(
                    case,
                    decision,
                    public_input=public_input,
                )
            )

    aggregates: Dict[Stage2PolicyName, Stage2AggregateMetrics] = {}
    for policy in selected_policies:
        policy_rows = [row for row in rows if row.policy == policy.name]
        aggregates[policy.name] = Stage2AggregateMetrics(
            policy=policy.name,
            case_count=len(policy_rows),
            future_support_recall=fmean(
                row.future_support_recall for row in policy_rows
            ),
            distractor_removal_recall=fmean(
                row.distractor_removal_recall for row in policy_rows
            ),
            removal_precision=fmean(row.removal_precision for row in policy_rows),
            token_reduction=fmean(row.token_reduction for row in policy_rows),
            budget_compliance_rate=fmean(
                float(row.budget_compliant) for row in policy_rows
            ),
            safe_success_rate=fmean(float(row.safe_success) for row in policy_rows),
        )
    return Stage2BenchmarkReport(
        token_counter=STAGE2_TOKEN_COUNTER_NAME,
        budget_scope=STAGE2_BUDGET_SCOPE,
        seed=seed,
        dataset_digest=selected_dataset.digest(),
        case_count=len(selected_dataset),
        rows=tuple(rows),
        aggregates=aggregates,
    )


def stage2_report_digest(report: Stage2BenchmarkReport) -> str:
    """Return a stable digest suitable for repeatability assertions."""

    payload = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AlwaysClearPolicy",
    "AlwaysKeepPolicy",
    "ChallengeScenario",
    "ChallengeSplit",
    "MAX_CHALLENGE_CASES",
    "MAX_CONTEXT_TOKENS",
    "MAX_MESSAGES_PER_CASE",
    "MAX_SEGMENTS_PER_MESSAGE",
    "MIN_CHALLENGE_CASES",
    "MessageKind",
    "OpaqueIdControlPolicy",
    "OracleSafeCompressPolicy",
    "PublicStage2Input",
    "PublicStage2Message",
    "PublicStage2Segment",
    "SegmentRole",
    "Stage2AggregateMetrics",
    "Stage2BenchmarkReport",
    "Stage2CaseMetrics",
    "Stage2ChallengeCase",
    "Stage2ChallengeDataset",
    "Stage2ChallengeMessage",
    "Stage2ChallengePolicy",
    "Stage2ChallengeSegment",
    "Stage2CompressionDecision",
    "Stage2PolicyName",
    "STAGE2_TOKEN_COUNTER_NAME",
    "STAGE2_BUDGET_SCOPE",
    "STAGE2_PUBLIC_HANDLE_VERSION",
    "default_stage2_challenge_path",
    "deterministic_token_count",
    "evaluate_stage2_decision",
    "run_stage2_challenge_benchmark",
    "stage2_report_digest",
]
