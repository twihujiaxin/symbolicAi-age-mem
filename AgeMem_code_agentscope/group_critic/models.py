"""Strict, immutable contracts for the M7 group-level logic critic.

The contracts deliberately keep critic suggestions separate from executable
automata.  A :class:`CriticOutput` becomes reward-bearing only after the
fail-closed validator and deterministic compiler accept it.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..hotpotqa_benchmark.models import HotpotContext, HotpotSupportingFacts
from ..memory_oracle.models import APName, AP_ORDER, AutomatonSpec


EVIDENCE_STEP_REF_SCHEMA_VERSION = "agemem.critic_evidence_step_ref.v1"
MEMORY_EVENT_SCHEMA_VERSION = "agemem.critic_memory_event.v1"
ACTION_AP_TRACE_SCHEMA_VERSION = "agemem.critic_action_ap_trace.v1"
ROLLOUT_TRACE_SCHEMA_VERSION = "agemem.critic_rollout_trace.v1"
CRITIC_HOTPOTQA_REFERENCE_SCHEMA_VERSION = "agemem.critic_hotpotqa_reference.v1"
GROUP_INPUT_SCHEMA_VERSION = "agemem.critic_group_input.v2"
MILESTONE_SCHEMA_VERSION = "agemem.critic_milestone.v1"
DEPENDENCY_SCHEMA_VERSION = "agemem.critic_dependency.v1"
BAD_BEHAVIOR_SCHEMA_VERSION = "agemem.critic_bad_behavior.v1"
COUNTERFACTUAL_SCHEMA_VERSION = "agemem.critic_counterfactual.v1"
CRITIC_OUTPUT_SCHEMA_VERSION = "agemem.critic_output.v1"
VALIDATION_ISSUE_SCHEMA_VERSION = "agemem.critic_validation_issue.v1"
VALIDATION_REPORT_SCHEMA_VERSION = "agemem.critic_validation_report.v1"
CALL_USAGE_SCHEMA_VERSION = "agemem.critic_call_usage.v1"
INVOCATION_SCHEMA_VERSION = "agemem.critic_invocation.v1"
DECISION_SCHEMA_VERSION = "agemem.critic_decision.v1"
CACHE_KEY_SCHEMA_VERSION = "agemem.critic_cache_key.v1"
CACHE_LOOKUP_SCHEMA_VERSION = "agemem.critic_cache_lookup.v1"

CriticKind = Literal["mock", "llm"]
APProfile = Literal["oracle", "human_backed_mock", "controlled_error"]
TerminalOutcome = Literal["success", "failure"]
DecisionSource = Literal["critic", "hand_authored", "terminal_only"]
PositiveMilestoneAP = Literal[
    "stored_supporting_fact",
    "updated_stale_fact",
    "supporting_coverage_complete",
    "retrieved_supporting_fact",
    "answered_correctly",
]
POSITIVE_MILESTONE_APS: Tuple[PositiveMilestoneAP, ...] = (
    "stored_supporting_fact",
    "updated_stale_fact",
    "supporting_coverage_complete",
    "retrieved_supporting_fact",
    "answered_correctly",
)
BadBehaviorTag = Literal[
    "stored_irrelevant_fact",
    "retrieved_irrelevant_fact",
    "deleted_supporting_fact",
    "duplicate_tool_call",
    "action_loop",
    "reward_farming",
]


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


def raw_text_digest(value: str) -> str:
    """Hash the exact UTF-8 bytes returned by a completion boundary."""

    if not isinstance(value, str):
        raise TypeError("raw completion payload must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvidenceStepRef(_StrictFrozenModel):
    """Globally unambiguous reference to one original trajectory action."""

    schema_version: Literal[EVIDENCE_STEP_REF_SCHEMA_VERSION] = (
        EVIDENCE_STEP_REF_SCHEMA_VERSION
    )
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    assistant_turn_id: int = Field(ge=0)
    action_index_in_turn: int = Field(ge=0)
    ap_evidence_ids: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_reference(self) -> "EvidenceStepRef":
        if len(self.ap_evidence_ids) != len(set(self.ap_evidence_ids)):
            raise ValueError("ap_evidence_ids must be unique")
        return self

    @property
    def action_key(self) -> Tuple[str, str, int, int, str, int, int]:
        return (
            self.task_id,
            self.rollout_id,
            self.stage_id,
            self.timestep,
            self.action_id,
            self.assistant_turn_id,
            self.action_index_in_turn,
        )


class MemoryEvent(_StrictFrozenModel):
    """Small, backend-independent memory-event view supplied to the critic."""

    schema_version: Literal[MEMORY_EVENT_SCHEMA_VERSION] = MEMORY_EVENT_SCHEMA_VERSION
    event_id: str = Field(min_length=1)
    event_type: Literal["add", "retrieve", "update", "delete", "answer", "other"]
    memory_ids: Tuple[str, ...] = ()
    payload_digest: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_memory_ids(self) -> "MemoryEvent":
        if len(self.memory_ids) != len(set(self.memory_ids)):
            raise ValueError("memory_ids must be unique")
        return self


class ActionAPTrace(_StrictFrozenModel):
    """One action and its already-grounded M6 atomic propositions."""

    schema_version: Literal[ACTION_AP_TRACE_SCHEMA_VERSION] = (
        ACTION_AP_TRACE_SCHEMA_VERSION
    )
    evidence: EvidenceStepRef
    action_type: str = Field(min_length=1)
    memory_events: Tuple[MemoryEvent, ...] = ()
    propositions: Tuple[APName, ...] = ()
    atomic_proposition_evidence: Dict[APName, Tuple[str, ...]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_aps(self) -> "ActionAPTrace":
        if len(self.propositions) != len(set(self.propositions)):
            raise ValueError("propositions must be unique")
        expected = tuple(ap for ap in AP_ORDER if ap in self.propositions)
        if self.propositions != expected:
            raise ValueError("propositions must follow canonical AP_ORDER")
        if set(self.atomic_proposition_evidence) != set(self.propositions):
            raise ValueError(
                "atomic_proposition_evidence keys must exactly match propositions"
            )
        for values in self.atomic_proposition_evidence.values():
            # M6 answer APs legitimately use the action coordinate itself as
            # provenance and therefore have no APRecord/triple evidence ID.
            if len(values) != len(set(values)):
                raise ValueError("proposition evidence IDs must be unique")
        expected_evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for values in self.atomic_proposition_evidence.values()
                    for evidence_id in values
                }
            )
        )
        if self.evidence.ap_evidence_ids != expected_evidence_ids:
            raise ValueError(
                "action evidence AP IDs must equal the proposition evidence union"
            )
        event_ids = [item.event_id for item in self.memory_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("memory event IDs must be unique within an action")
        return self


class CriticRolloutTrace(_StrictFrozenModel):
    """Action-grounded critic view of exactly one rollout."""

    schema_version: Literal[ROLLOUT_TRACE_SCHEMA_VERSION] = ROLLOUT_TRACE_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    terminal_outcome: TerminalOutcome
    actions: Tuple[ActionAPTrace, ...] = Field(min_length=1)
    source_trajectory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    ap_trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_actions(self) -> "CriticRolloutTrace":
        action_ids = []
        previous_position: Optional[Tuple[int, int, int]] = None
        for action in self.actions:
            ref = action.evidence
            if (ref.task_id, ref.rollout_id) != (self.task_id, self.rollout_id):
                raise ValueError("action evidence must belong to its rollout")
            if ref.action_id in action_ids:
                raise ValueError("action_id values must be unique within a rollout")
            action_ids.append(ref.action_id)
            position = (
                ref.timestep,
                ref.assistant_turn_id,
                ref.action_index_in_turn,
            )
            if previous_position is not None and position <= previous_position:
                raise ValueError("rollout actions must be strictly ordered")
            previous_position = position
        return self


class CriticHotpotQAPrivateReference(_StrictFrozenModel):
    """One complete labeled HotpotQA row visible only to the group critic."""

    schema_version: Literal[CRITIC_HOTPOTQA_REFERENCE_SCHEMA_VERSION] = (
        CRITIC_HOTPOTQA_REFERENCE_SCHEMA_VERSION
    )
    visibility: Literal["critic_only_privileged"] = "critic_only_privileged"
    scope: Literal["current_task_only"] = "current_task_only"
    dataset_name: Literal["hotpot_qa"] = "hotpot_qa"
    dataset_config: Literal["fullwiki"] = "fullwiki"
    hotpot_id: str = Field(min_length=1)
    source_split: Literal["train", "validation"]
    source_index: int = Field(ge=0)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    hotpot_type: Literal["bridge", "comparison"]
    level: Literal["easy", "medium", "hard"]
    context: HotpotContext
    supporting_facts: HotpotSupportingFacts

    @model_validator(mode="after")
    def validate_complete_labeled_row(self) -> "CriticHotpotQAPrivateReference":
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("critic HotpotQA question and answer must be non-blank")
        if not self.supporting_facts.title:
            raise ValueError("critic HotpotQA reference requires supporting facts")

        def canonical_title(value: str) -> str:
            return " ".join(value.split()).casefold()

        context_indices: Dict[str, Tuple[int, ...]] = {}
        for index, title in enumerate(self.context.title):
            key = canonical_title(title)
            context_indices[key] = (*context_indices.get(key, ()), index)
        for title, sent_id in self.supporting_facts.pairs():
            indices = context_indices.get(canonical_title(title), ())
            if len(indices) != 1:
                raise ValueError(
                    "each supporting-fact title must resolve to exactly one context"
                )
            if sent_id >= len(self.context.sentences[indices[0]]):
                raise ValueError("supporting-fact sentence ID is outside the context")
        return self


class CriticGroupInput(_StrictFrozenModel):
    """Complete K-rollout input used to derive one critic cache key."""

    schema_version: Literal[GROUP_INPUT_SCHEMA_VERSION] = GROUP_INPUT_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    split_id: str = Field(min_length=1)
    task_description: str = Field(min_length=1)
    critic_only_reference: CriticHotpotQAPrivateReference
    ap_profile: APProfile
    rollouts: Tuple[CriticRolloutTrace, ...] = Field(min_length=1)
    source_report_digests: Tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_group(self) -> "CriticGroupInput":
        rollout_ids = [item.rollout_id for item in self.rollouts]
        if len(rollout_ids) != len(set(rollout_ids)):
            raise ValueError("rollout IDs must be unique within a critic group")
        if any(item.task_id != self.task_id for item in self.rollouts):
            raise ValueError("all rollouts must belong to group task_id")
        reference = self.critic_only_reference
        if self.task_id != f"hotpot-{reference.hotpot_id}":
            raise ValueError("critic reference Hotpot ID must match group task_id")
        if self.task_description != reference.question:
            raise ValueError("task_description must match the private HotpotQA question")
        expected_source_split = "train" if self.split_id == "train" else "validation"
        if self.split_id in {"train", "dev", "test"} and (
            reference.source_split != expected_source_split
        ):
            raise ValueError("critic reference source split must match benchmark split")
        if len(self.source_report_digests) != len(set(self.source_report_digests)):
            raise ValueError("source report digests must be unique")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.source_report_digests
        ):
            raise ValueError("source report digests must be lowercase SHA-256 values")
        action_ids = [
            action.evidence.action_id
            for rollout in self.rollouts
            for action in rollout.actions
        ]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action_id values must be globally unique within a group")
        return self

    @property
    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        # A group is a mathematical set of rollouts.  Canonicalize only that
        # outer ordering; action order within each rollout remains semantic.
        payload["rollouts"] = sorted(
            payload["rollouts"], key=lambda item: item["rollout_id"]
        )
        payload["source_report_digests"] = sorted(payload["source_report_digests"])
        return canonical_digest(payload)

    @property
    def all_failed(self) -> bool:
        return all(item.terminal_outcome == "failure" for item in self.rollouts)


class Milestone(_StrictFrozenModel):
    """One evidence-grounded semantic proposition proposed by the critic."""

    schema_version: Literal[MILESTONE_SCHEMA_VERSION] = MILESTONE_SCHEMA_VERSION
    milestone_id: str = Field(min_length=1)
    # Only the positive M4 main chain and its optional update progress may
    # become reward-bearing milestones.  Negative/audit APs remain available
    # to ActionAPTrace and BadBehaviorRecord, but cannot enter this contract.
    proposition: PositiveMilestoneAP
    description: str = Field(min_length=1)
    evidence_steps: Tuple[EvidenceStepRef, ...] = Field(min_length=1)
    confidence: float

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if (
            isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> "Milestone":
        keys = [item.action_key for item in self.evidence_steps]
        if len(keys) != len(set(keys)):
            raise ValueError("milestone evidence steps must be unique")
        return self


class MilestoneDependency(_StrictFrozenModel):
    """Directed prerequisite edge between two milestone IDs."""

    schema_version: Literal[DEPENDENCY_SCHEMA_VERSION] = DEPENDENCY_SCHEMA_VERSION
    prerequisite_id: str = Field(min_length=1)
    dependent_id: str = Field(min_length=1)


class BadBehaviorRecord(_StrictFrozenModel):
    """Audit-only behavior tag; it can never compile into a negative reward."""

    schema_version: Literal[BAD_BEHAVIOR_SCHEMA_VERSION] = BAD_BEHAVIOR_SCHEMA_VERSION
    tag: BadBehaviorTag
    description: str = Field(min_length=1)
    evidence_steps: Tuple[EvidenceStepRef, ...] = Field(min_length=1)
    confidence: float
    audit_only: Literal[True] = True

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if (
            isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> "BadBehaviorRecord":
        keys = [item.action_key for item in self.evidence_steps]
        if len(keys) != len(set(keys)):
            raise ValueError("bad-behavior evidence steps must be unique")
        return self


class CounterfactualSuggestion(_StrictFrozenModel):
    """Non-executable suggestion, especially for all-failure groups."""

    schema_version: Literal[COUNTERFACTUAL_SCHEMA_VERSION] = (
        COUNTERFACTUAL_SCHEMA_VERSION
    )
    suggestion_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_steps: Tuple[EvidenceStepRef, ...] = ()
    confidence: float
    reward_eligible: Literal[False] = False

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if (
            isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> "CounterfactualSuggestion":
        keys = [item.action_key for item in self.evidence_steps]
        if len(keys) != len(set(keys)):
            raise ValueError("counterfactual evidence steps must be unique")
        return self


class CriticOutput(_StrictFrozenModel):
    """Structured critic output; validation is mandatory before compilation."""

    schema_version: Literal[CRITIC_OUTPUT_SCHEMA_VERSION] = CRITIC_OUTPUT_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    milestones: Tuple[Milestone, ...] = ()
    dependencies: Tuple[MilestoneDependency, ...] = ()
    bad_behaviors: Tuple[BadBehaviorRecord, ...] = ()
    counterfactual_suggestions: Tuple[CounterfactualSuggestion, ...] = ()
    counterfactual_used: bool = False
    warnings: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_local_uniqueness(self) -> "CriticOutput":
        milestone_ids = [item.milestone_id for item in self.milestones]
        propositions = [item.proposition for item in self.milestones]
        if len(milestone_ids) != len(set(milestone_ids)):
            raise ValueError("milestone IDs must be unique")
        if len(propositions) != len(set(propositions)):
            raise ValueError("milestone propositions must be unique")
        dependencies = [
            (item.prerequisite_id, item.dependent_id) for item in self.dependencies
        ]
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("dependencies must be unique")
        suggestion_ids = [
            item.suggestion_id for item in self.counterfactual_suggestions
        ]
        if len(suggestion_ids) != len(set(suggestion_ids)):
            raise ValueError("counterfactual suggestion IDs must be unique")
        if self.counterfactual_used != bool(self.counterfactual_suggestions):
            raise ValueError(
                "counterfactual_used must exactly reflect counterfactual suggestions"
            )
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("warnings must be unique")
        return self

    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


class CriticValidationIssue(_StrictFrozenModel):
    schema_version: Literal[VALIDATION_ISSUE_SCHEMA_VERSION] = (
        VALIDATION_ISSUE_SCHEMA_VERSION
    )
    severity: Literal["error", "warning"]
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    milestone_id: Optional[str] = Field(default=None, min_length=1)
    action_id: Optional[str] = Field(default=None, min_length=1)


class CriticValidationReport(_StrictFrozenModel):
    schema_version: Literal[VALIDATION_REPORT_SCHEMA_VERSION] = (
        VALIDATION_REPORT_SCHEMA_VERSION
    )
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    valid: bool
    automaton_compilable: bool
    evidence_reference_count: int = Field(ge=0)
    reachable_progress_state_count: int = Field(ge=0)
    issues: Tuple[CriticValidationIssue, ...] = ()

    @model_validator(mode="after")
    def validate_summary(self) -> "CriticValidationReport":
        has_error = any(item.severity == "error" for item in self.issues)
        if self.valid == has_error:
            raise ValueError("valid must be true exactly when there are no errors")
        if self.automaton_compilable and (
            not self.valid or not self.reachable_progress_state_count
        ):
            raise ValueError("compilable output must be valid with reachable states")
        return self


class CriticCallUsage(_StrictFrozenModel):
    schema_version: Literal[CALL_USAGE_SCHEMA_VERSION] = CALL_USAGE_SCHEMA_VERSION
    call_count: int = Field(ge=0)
    cache_hit: bool = False
    input_chars: int = Field(ge=0)
    output_chars: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    provider_input_tokens: Optional[int] = Field(default=None, ge=0)
    provider_output_tokens: Optional[int] = Field(default=None, ge=0)
    provider_cost: Optional[float] = Field(default=None, ge=0.0)
    latency_ms: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("provider_cost", "latency_ms")
    @classmethod
    def validate_optional_finite(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError("usage values must be finite")
        return value


class CriticInvocationResult(_StrictFrozenModel):
    schema_version: Literal[INVOCATION_SCHEMA_VERSION] = INVOCATION_SCHEMA_VERSION
    critic_kind: CriticKind
    critic_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    output: Optional[CriticOutput] = None
    output_digest: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_output_digest: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error: Optional[str] = Field(default=None, min_length=1)
    usage: CriticCallUsage

    @model_validator(mode="after")
    def validate_result(self) -> "CriticInvocationResult":
        if (self.output is None) == (self.error is None):
            raise ValueError("exactly one of output and error must be provided")
        if self.output is not None and self.output_digest != self.output.digest:
            raise ValueError("structured output digest must match output")
        if self.output is not None and self.raw_output_digest is None:
            raise ValueError("successful invocation requires raw_output_digest")
        if self.output is None and self.output_digest is not None:
            raise ValueError("failed invocation cannot declare output_digest")
        return self


class CriticDecision(_StrictFrozenModel):
    """Auditable selection of a critic DFA or an explicit fallback."""

    schema_version: Literal[DECISION_SCHEMA_VERSION] = DECISION_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    selected_source: DecisionSource
    invocation: CriticInvocationResult
    validation: Optional[CriticValidationReport] = None
    automaton_spec: Optional[AutomatonSpec] = None
    fallback_reason: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> "CriticDecision":
        if self.selected_source == "critic":
            if (
                self.invocation.output is None
                or self.validation is None
                or not self.validation.automaton_compilable
                or self.automaton_spec is None
                or self.fallback_reason is not None
            ):
                raise ValueError("critic selection requires validated compiled output")
        elif self.selected_source == "hand_authored":
            if self.automaton_spec is None or self.fallback_reason is None:
                raise ValueError("hand-authored fallback requires spec and reason")
        elif self.automaton_spec is not None or self.fallback_reason is None:
            raise ValueError("terminal-only fallback requires reason and no DFA")
        return self


class GroupCriticCacheKey(_StrictFrozenModel):
    """Cache key covering every group, schema, prompt, model and source input."""

    schema_version: Literal[CACHE_KEY_SCHEMA_VERSION] = CACHE_KEY_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    ap_profile: APProfile
    group_input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_schema_version: Literal[GROUP_INPUT_SCHEMA_VERSION] = (
        GROUP_INPUT_SCHEMA_VERSION
    )
    output_schema_version: Literal[CRITIC_OUTPUT_SCHEMA_VERSION] = (
        CRITIC_OUTPUT_SCHEMA_VERSION
    )
    prompt_version: str = Field(min_length=1)
    critic_kind: CriticKind
    critic_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    source_report_digests: Tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_digests(self) -> "GroupCriticCacheKey":
        if len(self.source_report_digests) != len(set(self.source_report_digests)):
            raise ValueError("source report digests must be unique")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.source_report_digests
        ):
            raise ValueError("source report digests must be lowercase SHA-256 values")
        return self

    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


class GroupCriticCacheLookup(_StrictFrozenModel):
    schema_version: Literal[CACHE_LOOKUP_SCHEMA_VERSION] = CACHE_LOOKUP_SCHEMA_VERSION
    cache_key_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_hit: bool
    result: CriticInvocationResult


__all__ = [
    "ACTION_AP_TRACE_SCHEMA_VERSION",
    "APProfile",
    "ActionAPTrace",
    "BAD_BEHAVIOR_SCHEMA_VERSION",
    "BadBehaviorTag",
    "BadBehaviorRecord",
    "CACHE_KEY_SCHEMA_VERSION",
    "CACHE_LOOKUP_SCHEMA_VERSION",
    "CALL_USAGE_SCHEMA_VERSION",
    "CRITIC_HOTPOTQA_REFERENCE_SCHEMA_VERSION",
    "COUNTERFACTUAL_SCHEMA_VERSION",
    "CRITIC_OUTPUT_SCHEMA_VERSION",
    "CounterfactualSuggestion",
    "CriticCallUsage",
    "CriticDecision",
    "CriticGroupInput",
    "CriticHotpotQAPrivateReference",
    "CriticInvocationResult",
    "CriticKind",
    "CriticOutput",
    "CriticRolloutTrace",
    "CriticValidationIssue",
    "CriticValidationReport",
    "DECISION_SCHEMA_VERSION",
    "DEPENDENCY_SCHEMA_VERSION",
    "DecisionSource",
    "EVIDENCE_STEP_REF_SCHEMA_VERSION",
    "EvidenceStepRef",
    "GROUP_INPUT_SCHEMA_VERSION",
    "GroupCriticCacheKey",
    "GroupCriticCacheLookup",
    "INVOCATION_SCHEMA_VERSION",
    "MILESTONE_SCHEMA_VERSION",
    "MEMORY_EVENT_SCHEMA_VERSION",
    "MemoryEvent",
    "Milestone",
    "MilestoneDependency",
    "POSITIVE_MILESTONE_APS",
    "PositiveMilestoneAP",
    "ROLLOUT_TRACE_SCHEMA_VERSION",
    "TerminalOutcome",
    "VALIDATION_ISSUE_SCHEMA_VERSION",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "canonical_digest",
    "raw_text_digest",
]
