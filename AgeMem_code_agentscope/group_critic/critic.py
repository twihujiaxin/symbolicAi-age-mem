"""Structured M7 group critics, provenance-safe cache, and explicit fallback."""

from __future__ import annotations

import json
import threading
import time
from typing import Dict, Literal, Optional, Protocol, Tuple, runtime_checkable

from pydantic import ValidationError

from ..memory_oracle.automaton import hand_authored_memory_dfa
from .compiler import CriticCompilationError, compile_critic_output
from .models import (
    BadBehaviorRecord,
    CounterfactualSuggestion,
    CriticCallUsage,
    CriticDecision,
    CriticGroupInput,
    CriticInvocationResult,
    CriticKind,
    CriticOutput,
    EvidenceStepRef,
    GroupCriticCacheKey,
    GroupCriticCacheLookup,
    Milestone,
    MilestoneDependency,
    raw_text_digest,
)
from .validator import DEFAULT_AUTOMATON_STATE_CAP, validate_critic_output


DEFAULT_MOCK_CRITIC_VERSION = "agemem.group_critic.mock.v1"
DEFAULT_MOCK_MODEL_VERSION = "deterministic-no-llm"
DEFAULT_CRITIC_PROMPT_VERSION = "agemem.group_critic.prompt.v1"

_MILESTONE_CHAIN: Tuple[Tuple[str, str, str], ...] = (
    (
        "m_store_support",
        "stored_supporting_fact",
        "At least one supporting fact was semantically stored.",
    ),
    (
        "m_support_coverage",
        "supporting_coverage_complete",
        "The active memory reached complete supporting-fact coverage.",
    ),
    (
        "m_retrieve_support",
        "retrieved_supporting_fact",
        "Supporting memory was semantically retrieved before answering.",
    ),
    (
        "m_answer_correct",
        "answered_correctly",
        "The final answer was correct.",
    ),
)

_BAD_BEHAVIOR_APS: Tuple[Tuple[str, str], ...] = (
    ("stored_irrelevant_fact", "stored_irrelevant_fact"),
    ("retrieved_irrelevant_fact", "retrieved_irrelevant_fact"),
    ("deleted_supporting_fact", "deleted_supporting_fact"),
)


@runtime_checkable
class GroupCritic(Protocol):
    """Minimal synchronous group critic interface used by offline M7."""

    critic_kind: CriticKind
    critic_version: str
    model_version: str
    prompt_version: str

    def critique(self, group_input: CriticGroupInput) -> CriticInvocationResult:
        """Return structured output or an explicit, non-throwing invocation error."""


class InjectedCriticCompletionClient(Protocol):
    """Caller-owned completion client; tests inject deterministic fakes."""

    def complete(self, *, prompt: str) -> str:
        """Return exactly one CriticOutput JSON object as text."""


def _estimated_tokens(character_count: int) -> int:
    """Clearly labelled tokenizer-independent estimate used only for audit cost."""

    return (character_count + 3) // 4


def _reference_for_proposition(action, proposition: str) -> EvidenceStepRef:
    source = action.evidence
    return EvidenceStepRef(
        task_id=source.task_id,
        rollout_id=source.rollout_id,
        stage_id=source.stage_id,
        timestep=source.timestep,
        action_id=source.action_id,
        assistant_turn_id=source.assistant_turn_id,
        action_index_in_turn=source.action_index_in_turn,
        ap_evidence_ids=action.atomic_proposition_evidence[proposition],
    )


class MockGroupCritic:
    """Deterministic, evidence-grounded critic equivalent to the M4 main chain."""

    critic_kind: Literal["mock"] = "mock"

    def __init__(
        self,
        *,
        critic_version: str = DEFAULT_MOCK_CRITIC_VERSION,
        model_version: str = DEFAULT_MOCK_MODEL_VERSION,
        prompt_version: str = DEFAULT_CRITIC_PROMPT_VERSION,
    ) -> None:
        for value in (critic_version, model_version, prompt_version):
            if not value.strip():
                raise ValueError("critic identity values must be non-blank")
        self.critic_version = critic_version
        self.model_version = model_version
        self.prompt_version = prompt_version

    def _output(self, group_input: CriticGroupInput) -> CriticOutput:
        all_actions = sorted(
            (action for rollout in group_input.rollouts for action in rollout.actions),
            key=lambda action: action.evidence.action_key,
        )
        if group_input.all_failed:
            evidence = tuple(action.evidence for action in all_actions[-1:])
            return CriticOutput(
                task_id=group_input.task_id,
                group_id=group_input.group_id,
                counterfactual_suggestions=(
                    CounterfactualSuggestion(
                        suggestion_id="cf_obtain_successful_memory_path",
                        description=(
                            "No observed success establishes reward milestones; "
                            "collect a successful memory trajectory for validation."
                        ),
                        evidence_steps=evidence,
                        confidence=1.0,
                    ),
                ),
                counterfactual_used=True,
                warnings=("all rollouts failed; no reward milestones emitted",),
            )

        milestones = []
        for milestone_id, proposition, description in _MILESTONE_CHAIN:
            supporting_actions = tuple(
                action for action in all_actions if proposition in action.propositions
            )
            if not supporting_actions:
                raise ValueError(
                    f"group lacks real evidence for required milestone {proposition!r}"
                )
            milestones.append(
                Milestone(
                    milestone_id=milestone_id,
                    proposition=proposition,
                    description=description,
                    evidence_steps=tuple(
                        _reference_for_proposition(action, proposition)
                        for action in supporting_actions
                    ),
                    confidence=1.0,
                )
            )

        bad_behaviors = []
        for tag, proposition in _BAD_BEHAVIOR_APS:
            supporting_actions = tuple(
                action for action in all_actions if proposition in action.propositions
            )
            if supporting_actions:
                bad_behaviors.append(
                    BadBehaviorRecord(
                        tag=tag,
                        description=f"Observed audit-only AP: {proposition}.",
                        evidence_steps=tuple(
                            _reference_for_proposition(action, proposition)
                            for action in supporting_actions
                        ),
                        confidence=1.0,
                    )
                )

        return CriticOutput(
            task_id=group_input.task_id,
            group_id=group_input.group_id,
            milestones=tuple(milestones),
            dependencies=tuple(
                MilestoneDependency(
                    prerequisite_id=_MILESTONE_CHAIN[index][0],
                    dependent_id=_MILESTONE_CHAIN[index + 1][0],
                )
                for index in range(len(_MILESTONE_CHAIN) - 1)
            ),
            bad_behaviors=tuple(bad_behaviors),
        )

    def critique(self, group_input: CriticGroupInput) -> CriticInvocationResult:
        try:
            output = self._output(group_input)
            output_text = json.dumps(
                output.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            return CriticInvocationResult(
                critic_kind=self.critic_kind,
                critic_version=self.critic_version,
                model_version=self.model_version,
                prompt_version=self.prompt_version,
                input_digest=group_input.digest,
                output=output,
                output_digest=output.digest,
                raw_output_digest=raw_text_digest(output_text),
                usage=CriticCallUsage(
                    call_count=1,
                    input_chars=len(group_input.model_dump_json()),
                    output_chars=len(output_text),
                    estimated_input_tokens=_estimated_tokens(
                        len(group_input.model_dump_json())
                    ),
                    estimated_output_tokens=_estimated_tokens(len(output_text)),
                    latency_ms=None,
                ),
            )
        except (ValueError, ValidationError) as exc:
            return CriticInvocationResult(
                critic_kind=self.critic_kind,
                critic_version=self.critic_version,
                model_version=self.model_version,
                prompt_version=self.prompt_version,
                input_digest=group_input.digest,
                error=f"mock_critic_error: {exc}",
                usage=CriticCallUsage(
                    call_count=1,
                    input_chars=len(group_input.model_dump_json()),
                    output_chars=0,
                    estimated_input_tokens=_estimated_tokens(
                        len(group_input.model_dump_json())
                    ),
                    estimated_output_tokens=0,
                    latency_ms=None,
                ),
            )


class LLMGroupCritic:
    """Strict JSON wrapper around an injected completion client.

    Merely constructing this class performs no network operation.  The package
    supplies no provider client and the M7 tests use deterministic fakes only.
    """

    critic_kind: Literal["llm"] = "llm"

    def __init__(
        self,
        client: InjectedCriticCompletionClient,
        *,
        critic_version: str,
        model_version: str,
        prompt_version: str = DEFAULT_CRITIC_PROMPT_VERSION,
    ) -> None:
        if not callable(getattr(client, "complete", None)):
            raise TypeError("client must implement complete(prompt=...)")
        for value in (critic_version, model_version, prompt_version):
            if not value.strip():
                raise ValueError("critic identity values must be non-blank")
        self.client = client
        self.critic_version = critic_version
        self.model_version = model_version
        self.prompt_version = prompt_version

    def _prompt(self, group_input: CriticGroupInput) -> str:
        schema = CriticOutput.model_json_schema()
        return (
            "You are a group-level memory logic critic. Return exactly one JSON "
            "object matching the supplied schema. Use only defined AP propositions "
            "and exact evidence action coordinates from the input. Bad behaviors "
            "are audit-only. Counterfactual suggestions are never reward eligible.\n"
            f"prompt_version={self.prompt_version}\n"
            "SCHEMA:\n"
            + json.dumps(
                schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\nGROUP_INPUT:\n"
            + json.dumps(
                group_input.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )

    def critique(self, group_input: CriticGroupInput) -> CriticInvocationResult:
        prompt = self._prompt(group_input)
        started = time.perf_counter()
        raw: Optional[str] = None
        try:
            raw = self.client.complete(prompt=prompt)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("completion must be a non-empty JSON string")
            output = CriticOutput.model_validate_json(raw)
            return CriticInvocationResult(
                critic_kind=self.critic_kind,
                critic_version=self.critic_version,
                model_version=self.model_version,
                prompt_version=self.prompt_version,
                input_digest=group_input.digest,
                output=output,
                output_digest=output.digest,
                raw_output_digest=raw_text_digest(raw),
                usage=CriticCallUsage(
                    call_count=1,
                    input_chars=len(prompt),
                    output_chars=len(raw),
                    estimated_input_tokens=_estimated_tokens(len(prompt)),
                    estimated_output_tokens=_estimated_tokens(len(raw)),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )
        except Exception as exc:  # injected clients are an untrusted boundary
            return CriticInvocationResult(
                critic_kind=self.critic_kind,
                critic_version=self.critic_version,
                model_version=self.model_version,
                prompt_version=self.prompt_version,
                input_digest=group_input.digest,
                raw_output_digest=(
                    raw_text_digest(raw) if isinstance(raw, str) else None
                ),
                error=f"llm_critic_error: {type(exc).__name__}: {exc}",
                usage=CriticCallUsage(
                    call_count=1,
                    input_chars=len(prompt),
                    output_chars=(len(raw) if isinstance(raw, str) else 0),
                    estimated_input_tokens=_estimated_tokens(len(prompt)),
                    estimated_output_tokens=(
                        _estimated_tokens(len(raw)) if isinstance(raw, str) else 0
                    ),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )


class GroupCriticCache:
    """Thread-safe cache keyed by the entire group and critic identity."""

    def __init__(self) -> None:
        self._entries: Dict[str, CriticInvocationResult] = {}
        self._keys: Dict[str, GroupCriticCacheKey] = {}
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
    def key_for(
        group_input: CriticGroupInput, critic: GroupCritic
    ) -> GroupCriticCacheKey:
        return GroupCriticCacheKey(
            task_id=group_input.task_id,
            group_id=group_input.group_id,
            ap_profile=group_input.ap_profile,
            group_input_digest=group_input.digest,
            prompt_version=critic.prompt_version,
            critic_kind=critic.critic_kind,
            critic_version=critic.critic_version,
            model_version=critic.model_version,
            source_report_digests=tuple(sorted(group_input.source_report_digests)),
        )

    @staticmethod
    def _cache_hit_copy(result: CriticInvocationResult) -> CriticInvocationResult:
        payload = result.model_dump(mode="python")
        usage = result.usage.model_dump(mode="python")
        # A hit performs no provider call.  Cached payload byte/token estimates
        # remain useful for audit, but provider-billed usage and latency must be
        # zero rather than copied from the original cold invocation.
        usage.update(
            call_count=0,
            cache_hit=True,
            provider_input_tokens=0,
            provider_output_tokens=0,
            provider_cost=0.0,
            latency_ms=None,
        )
        payload["usage"] = usage
        return CriticInvocationResult.model_validate(payload)

    def get_or_critique(
        self,
        group_input: CriticGroupInput,
        critic: GroupCritic,
    ) -> GroupCriticCacheLookup:
        key = self.key_for(group_input, critic)
        with self._lock:
            cached = self._entries.get(key.digest)
            if cached is not None:
                self._hits += 1
                return GroupCriticCacheLookup(
                    cache_key_digest=key.digest,
                    cache_hit=True,
                    result=self._cache_hit_copy(cached),
                )
            result = critic.critique(group_input)
            identity = (
                result.input_digest,
                result.critic_kind,
                result.critic_version,
                result.model_version,
                result.prompt_version,
            )
            expected = (
                group_input.digest,
                critic.critic_kind,
                critic.critic_version,
                critic.model_version,
                critic.prompt_version,
            )
            if identity != expected:
                raise ValueError("critic invocation provenance mismatches cache key")
            self._entries[key.digest] = result.model_copy(deep=True)
            self._keys[key.digest] = key
            self._misses += 1
            return GroupCriticCacheLookup(
                cache_key_digest=key.digest,
                cache_hit=False,
                result=result,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._keys.clear()
            self._hits = 0
            self._misses = 0


def select_critic_automaton(
    group_input: CriticGroupInput,
    invocation: CriticInvocationResult,
    *,
    state_cap: int = DEFAULT_AUTOMATON_STATE_CAP,
    fallback: Literal["hand_authored", "terminal_only"] = "hand_authored",
) -> CriticDecision:
    """Validate, compile, or explicitly select a fail-closed baseline."""

    if invocation.input_digest != group_input.digest:
        raise ValueError("invocation input digest does not match group input")
    validation = None
    reason: Optional[str] = invocation.error

    if invocation.output is not None:
        validation = validate_critic_output(
            group_input,
            invocation.output,
            state_cap=state_cap,
        )
        if validation.automaton_compilable:
            try:
                spec = compile_critic_output(
                    group_input,
                    invocation.output,
                    validation,
                    state_cap=state_cap,
                )
            except CriticCompilationError as exc:
                reason = f"critic_compilation_error: {exc}"
            else:
                return CriticDecision(
                    task_id=group_input.task_id,
                    group_id=group_input.group_id,
                    selected_source="critic",
                    invocation=invocation,
                    validation=validation,
                    automaton_spec=spec,
                )
        else:
            error_codes = tuple(
                item.code for item in validation.issues if item.severity == "error"
            )
            reason = "critic_validation_failed: " + (
                ",".join(error_codes) if error_codes else "not_reward_eligible"
            )

    # All-failure counterfactuals are shadow-only and must never receive a DFA.
    if group_input.all_failed:
        return CriticDecision(
            task_id=group_input.task_id,
            group_id=group_input.group_id,
            selected_source="terminal_only",
            invocation=invocation,
            validation=validation,
            fallback_reason=(
                reason or "all_failure_counterfactual_is_not_reward_eligible"
            ),
        )

    if fallback == "hand_authored":
        return CriticDecision(
            task_id=group_input.task_id,
            group_id=group_input.group_id,
            selected_source="hand_authored",
            invocation=invocation,
            validation=validation,
            automaton_spec=hand_authored_memory_dfa(),
            fallback_reason=(reason or "critic output unavailable"),
        )
    return CriticDecision(
        task_id=group_input.task_id,
        group_id=group_input.group_id,
        selected_source="terminal_only",
        invocation=invocation,
        validation=validation,
        fallback_reason=(reason or "critic output unavailable"),
    )


__all__ = [
    "DEFAULT_CRITIC_PROMPT_VERSION",
    "DEFAULT_MOCK_CRITIC_VERSION",
    "DEFAULT_MOCK_MODEL_VERSION",
    "GroupCritic",
    "GroupCriticCache",
    "InjectedCriticCompletionClient",
    "LLMGroupCritic",
    "MockGroupCritic",
    "select_critic_automaton",
]
