"""Online Oracle AP + hand-authored DFA credits for format-conditioned E3.

This module is not imported by the frozen M8b 318-count runtime gate.
It grounds HotpotQA supporting sentences from tool traces and the
question-retrieve environment path, then replays the M4 positive DFA.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from AgeMem_code_agentscope.action_schema import (
    ActionCreditRecord,
    ActionEvent,
    RewardBreakdownV2,
)
from AgeMem_code_agentscope.memory_oracle.automaton import DFARunner, hand_authored_memory_dfa
from AgeMem_code_agentscope.memory_oracle.models import AP_ORDER, OracleAPEvent
from trinity.common.action_event_contract import (
    join_action_events_to_credits,
    stable_action_id,
)


REWARD_VERSION = "agemem.reward.e3_oracle_dfa.v1"
DFA_SPEC_ID = "m4-memory-oracle-positive-v1"
LOGIC_BETA = 1.0
MILESTONE_WEIGHT = 0.25
VIOLATION_WEIGHT = 0.0
DEFAULT_MAX_STEPS = 48
FLAT_SPEC_ID = "agemem-flat-oracle-positive-v1"
FLAT_REWARD_VERSION = "agemem.reward.flat_oracle.v1"

# ``answered_correctly`` is deliberately excluded here: HotpotQA F1 already
# supplies the terminal answer reward.  Rewarding it again would both double
# count answer quality and create a synthetic credit with no ActionEvent join.
ACTION_MILESTONE_APS = (
    "stored_supporting_fact",
    "updated_stale_fact",
    "supporting_coverage_complete",
    "retrieved_supporting_fact",
)
ACTION_LOGIC_REWARD_CEILING = (
    LOGIC_BETA * MILESTONE_WEIGHT * len(ACTION_MILESTONE_APS)
)

_WHITESPACE = re.compile(r"\s+")


def normalize_sentence(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text or "").strip().lower())


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


@dataclass
class E3CreditReplay:
    """Deterministic trajectory credits plus the scalar DFA logic total."""

    credits: tuple[ActionCreditRecord, ...]
    env_total: float
    milestone_total: float
    violation_total: float
    logic_total: float
    training_total: float
    final_state: str
    final_status: str
    accepted: bool


@dataclass
class OracleRewardComparison:
    """Terminal, flat, and DFA rewards on one immutable action sequence."""

    terminal_total: float
    flat: E3CreditReplay
    dfa: E3CreditReplay
    action_count: int


@dataclass
class HotpotQAOracleGrounder:
    """Map HotpotQA memory effects onto M4 Oracle APs without an LLM."""

    task_id: str
    rollout_id: str
    seed: int
    supporting_sentences: tuple[str, ...]
    observed_sentences: tuple[str, ...]
    _gold_norm: dict[str, str] = field(init=False, repr=False)
    _observed_gold: set[str] = field(init=False, repr=False)
    _stored_gold: set[str] = field(default_factory=set, init=False, repr=False)
    _contents: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _timestep: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._gold_norm = {}
        for index, sentence in enumerate(self.supporting_sentences):
            key = normalize_sentence(sentence)
            if key:
                self._gold_norm[key] = f"gold:{index}"
        self._observed_gold = {
            self._gold_norm[normalize_sentence(sentence)]
            for sentence in self.observed_sentences
            if normalize_sentence(sentence) in self._gold_norm
        }

    def _match_gold(self, text: str) -> tuple[str, ...]:
        key = normalize_sentence(text)
        if not key:
            return ()
        fact_id = self._gold_norm.get(key)
        if fact_id:
            return (fact_id,)
        hits = []
        for gold_key, gold_id in self._gold_norm.items():
            if gold_key in key or key in gold_key:
                hits.append(gold_id)
        return _unique(hits)

    def _coverage_complete(self) -> bool:
        return bool(self._observed_gold) and self._observed_gold.issubset(self._stored_gold)

    def _event(
        self,
        *,
        stage: int,
        timestep: int,
        evidence: dict[str, tuple[str, ...]],
        answer_correct: bool = False,
    ) -> OracleAPEvent:
        if self._coverage_complete() and "supporting_coverage_complete" not in evidence:
            evidence["supporting_coverage_complete"] = tuple(sorted(self._observed_gold))
        if answer_correct:
            evidence["answered_correctly"] = ()
        propositions = tuple(ap for ap in AP_ORDER if ap in evidence)
        return OracleAPEvent(
            task_id=self.task_id,
            rollout_id=self.rollout_id,
            seed=self.seed,
            timestep=timestep,
            stage=stage,
            propositions=propositions,
            evidence_fact_ids=evidence,
        )

    def observe_stage1(self, stage: int = 1) -> OracleAPEvent:
        observed = tuple(sorted(self._observed_gold))
        evidence: dict[str, tuple[str, ...]] = {}
        if observed:
            evidence["observed_supporting_fact"] = observed
        event = self._event(stage=stage, timestep=self._timestep, evidence=evidence)
        self._timestep += 1
        return event

    def index_sentences(self, sentences: Sequence[str], *, stage: int = 3) -> OracleAPEvent:
        stored: list[str] = []
        irrelevant: list[str] = []
        for sentence in sentences:
            gold = self._match_gold(sentence)
            key = normalize_sentence(sentence)
            if not key:
                continue
            self._contents[f"env-index:{key}"] = sentence
            if gold:
                stored.extend(gold)
                self._stored_gold.update(gold)
            else:
                irrelevant.append(f"irr:{key[:24]}")
        evidence: dict[str, tuple[str, ...]] = {}
        if stored:
            evidence["stored_supporting_fact"] = _unique(stored)
        if irrelevant:
            evidence["stored_irrelevant_fact"] = _unique(irrelevant)
        event = self._event(stage=stage, timestep=self._timestep, evidence=evidence)
        self._timestep += 1
        return event

    def retrieve_contents(self, contents: Sequence[str], *, stage: int = 3) -> OracleAPEvent:
        supporting: list[str] = []
        irrelevant: list[str] = []
        for content in contents:
            gold = self._match_gold(content)
            if gold:
                supporting.extend(gold)
            elif normalize_sentence(content):
                irrelevant.append(f"irr:{normalize_sentence(content)[:24]}")
        evidence: dict[str, tuple[str, ...]] = {}
        if supporting:
            evidence["retrieved_supporting_fact"] = _unique(supporting)
        if irrelevant:
            evidence["retrieved_irrelevant_fact"] = _unique(irrelevant)
        event = self._event(stage=stage, timestep=self._timestep, evidence=evidence)
        self._timestep += 1
        return event

    def tool_event(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        result: Mapping[str, Any] | None,
        stage: int,
    ) -> OracleAPEvent:
        args = dict(arguments or {})
        payload = dict(result or {})
        evidence: dict[str, tuple[str, ...]] = {}
        if tool_name == "Add_memory":
            content = str(args.get("content") or "")
            memory_id = str(payload.get("memory_id") or "")
            add_applied = payload.get("outcome") == "added" and bool(memory_id)
            if add_applied:
                self._contents[memory_id] = content
                gold = self._match_gold(content)
                if gold:
                    self._stored_gold.update(gold)
                    evidence["stored_supporting_fact"] = gold
                elif normalize_sentence(content):
                    evidence["stored_irrelevant_fact"] = (memory_id,)
        elif tool_name == "Update_memory":
            memory_id = str(args.get("memory_id") or payload.get("memory_id") or "")
            previous = self._contents.get(memory_id, "")
            content = args.get("content")
            if content is None:
                content = previous
            content_text = str(content or "")
            if memory_id:
                self._contents[memory_id] = content_text
            previous_gold = self._match_gold(previous)
            new_gold = self._match_gold(content_text)
            if previous_gold and not new_gold:
                evidence["deleted_supporting_fact"] = previous_gold
                self._stored_gold.difference_update(previous_gold)
            if new_gold:
                evidence["stored_supporting_fact"] = new_gold
                self._stored_gold.update(new_gold)
            elif previous_gold:
                evidence["stored_irrelevant_fact"] = (memory_id or "irr:update",)
        elif tool_name == "Delete_memory":
            memory_id = str(args.get("memory_id") or payload.get("memory_id") or "")
            previous = self._contents.pop(memory_id, "")
            gold = self._match_gold(previous)
            if gold and payload.get("outcome") == "deleted":
                evidence["deleted_supporting_fact"] = gold
                self._stored_gold.difference_update(gold)
        elif tool_name == "Retrieve_memory":
            items = payload.get("items") or []
            supporting: list[str] = []
            irrelevant: list[str] = []
            if isinstance(items, list):
                for item in items:
                    content = ""
                    if isinstance(item, Mapping):
                        content = str(item.get("content") or "")
                    gold = self._match_gold(content)
                    if gold:
                        supporting.extend(gold)
                    elif normalize_sentence(content):
                        irrelevant.append(f"irr:{normalize_sentence(content)[:24]}")
            if supporting:
                evidence["retrieved_supporting_fact"] = _unique(supporting)
            if irrelevant:
                evidence["retrieved_irrelevant_fact"] = _unique(irrelevant)
        event = self._event(stage=max(1, int(stage or 1)), timestep=self._timestep, evidence=evidence)
        self._timestep += 1
        return event

    def answer_event(self, *, exact_match: float, stage: int = 3) -> OracleAPEvent:
        event = self._event(
            stage=stage,
            timestep=self._timestep,
            evidence={},
            answer_correct=float(exact_match) >= 1.0,
        )
        self._timestep += 1
        return event


def _credit_for_event(
    *,
    action_id: str,
    task_id: str,
    rollout_id: str,
    stage_id: int,
    timestep: int,
    event: OracleAPEvent,
    transition: Any,
    env_reward: float,
) -> ActionCreditRecord:
    milestone = MILESTONE_WEIGHT * len(transition.new_progress_edges)
    violation = VIOLATION_WEIGHT * len(transition.violations)
    total = env_reward + LOGIC_BETA * milestone + violation
    breakdown = RewardBreakdownV2(
        env=float(env_reward),
        milestone=float(milestone),
        violation=float(violation),
        trend=0.0,
        format=0.0,
        cost=0.0,
        total=float(total),
        automaton_state_before=transition.state_before,
        automaton_state_after=transition.state_after,
        automaton_status=transition.status,
        propositions=tuple(event.propositions),
        fired_edges=tuple(transition.fired_edges),
        newly_rewarded_edges=tuple(transition.new_progress_edges),
        violation_edges=tuple(transition.violations),
    )
    evidence = {
        str(key): tuple(str(item) for item in values)
        for key, values in event.evidence_fact_ids.items()
    }
    transition_ids = tuple(transition.fired_edges)
    return ActionCreditRecord(
        action_id=action_id,
        task_id=task_id,
        rollout_id=rollout_id,
        stage_id=stage_id,
        timestep=timestep,
        atomic_propositions=tuple(event.propositions),
        atomic_proposition_evidence=evidence,
        dfa_spec_id=DFA_SPEC_ID,
        transition_ids=transition_ids,
        transition_id=(transition_ids[0] if len(transition_ids) == 1 else None),
        dfa_state_before=transition.state_before,
        dfa_state_after=transition.state_after,
        reward_breakdown=breakdown,
        return_to_go=None,
        advantage=None,
        reward_version=REWARD_VERSION,
    )


def _flat_credit_for_event(
    *,
    action: ActionEvent,
    event: OracleAPEvent,
    seen_milestones: set[str],
) -> ActionCreditRecord:
    newly_rewarded = tuple(
        proposition
        for proposition in ACTION_MILESTONE_APS
        if proposition in event.propositions and proposition not in seen_milestones
    )
    seen_milestones.update(newly_rewarded)
    transition_ids = tuple(f"flat_{item}" for item in newly_rewarded)
    state_before = f"flat:{len(seen_milestones) - len(newly_rewarded)}"
    state_after = f"flat:{len(seen_milestones)}"
    milestone = MILESTONE_WEIGHT * len(newly_rewarded)
    breakdown = RewardBreakdownV2(
        env=0.0,
        milestone=float(milestone),
        violation=0.0,
        trend=0.0,
        format=0.0,
        cost=0.0,
        total=float(LOGIC_BETA * milestone),
        automaton_state_before=state_before,
        automaton_state_after=state_after,
        automaton_status="running",
        propositions=tuple(event.propositions),
        fired_edges=transition_ids,
        newly_rewarded_edges=transition_ids,
        violation_edges=(),
    )
    evidence = {
        str(key): tuple(str(item) for item in values)
        for key, values in event.evidence_fact_ids.items()
    }
    return ActionCreditRecord(
        action_id=action.action_id,
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        timestep=action.timestep,
        atomic_propositions=tuple(event.propositions),
        atomic_proposition_evidence=evidence,
        dfa_spec_id=FLAT_SPEC_ID,
        transition_ids=transition_ids,
        transition_id=(transition_ids[0] if len(transition_ids) == 1 else None),
        dfa_state_before=state_before,
        dfa_state_after=state_after,
        reward_breakdown=breakdown,
        return_to_go=None,
        advantage=None,
        reward_version=FLAT_REWARD_VERSION,
    )


def _validated_action_sequence(
    action_events: Sequence[ActionEvent | Mapping[str, Any]],
    *,
    task_id: str,
    rollout_id: str,
) -> tuple[ActionEvent, ...]:
    parsed = tuple(
        item
        if isinstance(item, ActionEvent)
        else ActionEvent.model_validate_json(
            json.dumps(item, ensure_ascii=False, allow_nan=False)
        )
        for item in action_events
    )
    action_ids: set[str] = set()
    previous_coordinate: tuple[int, int] | None = None
    for action in parsed:
        if action.task_id != task_id or action.rollout_id != rollout_id:
            raise ValueError("ActionEvent identity differs from replay identity")
        if action.action_id in action_ids:
            raise ValueError(f"duplicate replay action_id {action.action_id!r}")
        action_ids.add(action.action_id)
        coordinate = (action.assistant_turn_id, action.action_index_in_turn)
        if previous_coordinate is not None and coordinate <= previous_coordinate:
            raise ValueError("ActionEvents must be in assistant-turn/action-index order")
        previous_coordinate = coordinate
    return parsed


def replay_hotpotqa_oracle_comparison(
    *,
    task_id: str,
    rollout_id: str,
    seed: int,
    supporting_sentences: Sequence[str],
    observed_sentences: Sequence[str],
    action_events: Sequence[ActionEvent | Mapping[str, Any]],
    exact_match: float = 0.0,
    task_f1: float = 0.0,
    found_answer: bool = False,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> OracleRewardComparison:
    """Compare Flat-Oracle and ordered DFA on the exact same real actions.

    Only persisted LLM ``ActionEvent`` rows receive credits. Environment
    observation initializes the Oracle grounder and the final answer closes the
    DFA, but neither is fabricated into an action. Consequently every credit
    joins one real action exactly, and the sum of per-action rewards equals the
    trajectory logic reward.
    """

    actions = _validated_action_sequence(
        action_events,
        task_id=task_id,
        rollout_id=rollout_id,
    )
    grounder = HotpotQAOracleGrounder(
        task_id=task_id,
        rollout_id=rollout_id,
        seed=seed,
        supporting_sentences=tuple(supporting_sentences),
        observed_sentences=tuple(observed_sentences),
    )
    dfa_runner = DFARunner(hand_authored_memory_dfa(), max_steps=max_steps)

    # This establishes which gold facts were visible in Stage 1. It carries no
    # policy credit because it is an environment observation, not an action.
    dfa_runner.step(grounder.observe_stage1(), done=False)

    grounded_actions: list[tuple[ActionEvent, OracleAPEvent]] = []
    for action in actions:
        raw_output = action.result.get("output", action.result)
        result = raw_output if isinstance(raw_output, Mapping) else {}
        event = grounder.tool_event(
            tool_name=action.action_type,
            arguments=action.arguments,
            result=result,
            stage=action.stage_id,
        )
        grounded_actions.append((action, event))

    dfa_credits: list[ActionCreditRecord] = []
    for action, event in grounded_actions:
        transition = dfa_runner.step(event, done=False)
        dfa_credits.append(
            _credit_for_event(
                action_id=action.action_id,
                task_id=action.task_id,
                rollout_id=action.rollout_id,
                stage_id=action.stage_id,
                timestep=action.timestep,
                event=event,
                transition=transition,
                env_reward=0.0,
            )
        )

    # Answer correctness is already represented by terminal HotpotQA F1. It is
    # used here only to determine DFA acceptance, never as a second reward.
    dfa_runner.step(
        grounder.answer_event(exact_match=exact_match),
        done=True,
    )

    flat_seen: set[str] = set()
    flat_credits = tuple(
        _flat_credit_for_event(
            action=action,
            event=event,
            seen_milestones=flat_seen,
        )
        for action, event in grounded_actions
    )
    dfa_credit_tuple = tuple(dfa_credits)
    join_action_events_to_credits(actions, flat_credits)
    join_action_events_to_credits(actions, dfa_credit_tuple)

    env_total = float(task_f1) if found_answer else 0.0
    flat_milestone = sum(item.reward_breakdown.milestone for item in flat_credits)
    dfa_milestone = sum(
        item.reward_breakdown.milestone for item in dfa_credit_tuple
    )
    flat_logic = LOGIC_BETA * flat_milestone
    dfa_logic = LOGIC_BETA * dfa_milestone
    if flat_logic > ACTION_LOGIC_REWARD_CEILING + 1e-12:
        raise RuntimeError("Flat-Oracle action reward exceeded its frozen ceiling")
    if dfa_logic > ACTION_LOGIC_REWARD_CEILING + 1e-12:
        raise RuntimeError("Oracle DFA action reward exceeded its frozen ceiling")
    flat_status = "accepted" if float(exact_match) >= 1.0 else "rejected"
    flat = E3CreditReplay(
        credits=flat_credits,
        env_total=env_total,
        milestone_total=float(flat_milestone),
        violation_total=0.0,
        logic_total=float(flat_logic),
        training_total=float(env_total + flat_logic),
        final_state=f"flat:{len(flat_seen)}",
        final_status=flat_status,
        accepted=flat_status == "accepted",
    )
    dfa = E3CreditReplay(
        credits=dfa_credit_tuple,
        env_total=env_total,
        milestone_total=float(dfa_milestone),
        violation_total=0.0,
        logic_total=float(dfa_logic),
        training_total=float(env_total + dfa_logic),
        final_state=dfa_runner.state,
        final_status=str(dfa_runner.status),
        accepted=dfa_runner.status == "accepted",
    )
    if abs(
        sum(item.reward_breakdown.total for item in flat.credits)
        - flat.logic_total
    ) > 1e-12:
        raise RuntimeError("Flat-Oracle action credits do not conserve logic reward")
    if abs(
        sum(item.reward_breakdown.total for item in dfa.credits) - dfa.logic_total
    ) > 1e-12:
        raise RuntimeError("Oracle DFA action credits do not conserve logic reward")
    return OracleRewardComparison(
        terminal_total=env_total,
        flat=flat,
        dfa=dfa,
        action_count=len(actions),
    )


def replay_hotpotqa_oracle_dfa(
    *,
    task_id: str,
    rollout_id: str,
    seed: int,
    supporting_sentences: Sequence[str],
    observed_sentences: Sequence[str],
    indexed_sentences: Sequence[str] = (),
    retrieved_contents: Sequence[str] = (),
    tool_events: Sequence[Mapping[str, Any]] = (),
    exact_match: float = 0.0,
    task_f1: float = 0.0,
    found_answer: bool = False,
    max_steps: int = DEFAULT_MAX_STEPS,
    shadow: bool = False,
) -> E3CreditReplay:
    """Replay one HotpotQA rollout. Eval shadow keeps training_total = terminal F1."""

    grounder = HotpotQAOracleGrounder(
        task_id=task_id,
        rollout_id=rollout_id,
        seed=seed,
        supporting_sentences=tuple(supporting_sentences),
        observed_sentences=tuple(observed_sentences),
    )
    spec = hand_authored_memory_dfa()
    runner = DFARunner(spec, max_steps=max_steps)
    scheduled: list[tuple[str, int, int, OracleAPEvent, bool]] = []

    scheduled.append(
        (
            f"agemem-e3-env-{stable_action_id(rollout_id=rollout_id, stage_id=1, timestep=0, assistant_turn_id=0, action_index_in_turn=0)}",
            1,
            0,
            grounder.observe_stage1(),
            False,
        )
    )
    if indexed_sentences:
        scheduled.append(
            (
                f"agemem-e3-env-{stable_action_id(rollout_id=rollout_id, stage_id=3, timestep=0, assistant_turn_id=0, action_index_in_turn=1)}",
                3,
                0,
                grounder.index_sentences(indexed_sentences),
                False,
            )
        )
    if retrieved_contents:
        scheduled.append(
            (
                f"agemem-e3-env-{stable_action_id(rollout_id=rollout_id, stage_id=3, timestep=0, assistant_turn_id=0, action_index_in_turn=2)}",
                3,
                0,
                grounder.retrieve_contents(retrieved_contents),
                False,
            )
        )
    for index, raw in enumerate(tool_events):
        stage = int(raw.get("stage") or 1)
        timestep = int(raw.get("step") or index + 1)
        assistant_turn = int(raw.get("round") or 0)
        tool_index = int(raw.get("tool_index") or 0)
        action_id = stable_action_id(
            rollout_id=rollout_id,
            stage_id=stage,
            timestep=timestep,
            assistant_turn_id=max(0, assistant_turn),
            action_index_in_turn=max(0, tool_index),
        )
        scheduled.append(
            (
                action_id,
                stage,
                timestep,
                grounder.tool_event(
                    tool_name=str(raw.get("tool_name") or ""),
                    arguments=raw.get("arguments") if isinstance(raw.get("arguments"), Mapping) else {},
                    result=raw.get("result") if isinstance(raw.get("result"), Mapping) else {},
                    stage=stage,
                ),
                False,
            )
        )
    answer_timestep = (scheduled[-1][2] + 1) if scheduled else 1
    scheduled.append(
        (
            f"agemem-e3-env-{stable_action_id(rollout_id=rollout_id, stage_id=3, timestep=answer_timestep, assistant_turn_id=99, action_index_in_turn=0)}",
            3,
            answer_timestep,
            grounder.answer_event(exact_match=exact_match),
            True,
        )
    )

    env_total = float(task_f1) if found_answer else 0.0
    credits: list[ActionCreditRecord] = []
    for index, (action_id, stage_id, timestep, event, is_answer) in enumerate(scheduled):
        done = index == len(scheduled) - 1
        transition = runner.step(event, done=done)
        env_reward = env_total if is_answer else 0.0
        credits.append(
            _credit_for_event(
                action_id=action_id,
                task_id=task_id,
                rollout_id=rollout_id,
                stage_id=stage_id,
                timestep=timestep,
                event=event,
                transition=transition,
                env_reward=env_reward,
            )
        )

    milestone_total = sum(item.reward_breakdown.milestone for item in credits)
    violation_total = sum(item.reward_breakdown.violation for item in credits)
    logic_total = LOGIC_BETA * milestone_total + violation_total
    training_total = env_total if shadow else env_total + logic_total
    return E3CreditReplay(
        credits=tuple(credits),
        env_total=env_total,
        milestone_total=float(milestone_total),
        violation_total=float(violation_total),
        logic_total=float(logic_total),
        training_total=float(training_total),
        final_state=runner.state,
        final_status=str(runner.status),
        accepted=runner.status == "accepted",
    )


def replay_summary(replay: E3CreditReplay) -> dict[str, Any]:
    return {
        "e3_reward_version": REWARD_VERSION,
        "e3_dfa_spec_id": DFA_SPEC_ID,
        "e3_env_total": replay.env_total,
        "e3_milestone_total": replay.milestone_total,
        "e3_violation_total": replay.violation_total,
        "e3_logic_total": replay.logic_total,
        "e3_training_total": replay.training_total,
        "e3_dfa_final_state": replay.final_state,
        "e3_dfa_final_status": replay.final_status,
        "e3_dfa_accepted": replay.accepted,
        "e3_credit_count": len(replay.credits),
        "e3_oracle_credits": [item.canonical_dict() for item in replay.credits],
    }


def tool_events_from_trace(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for row in rows:
        if str(row.get("phase") or "") != "finish":
            continue
        name = str(row.get("tool_name") or "")
        if name not in {
            "Add_memory",
            "Update_memory",
            "Delete_memory",
            "Retrieve_memory",
        }:
            continue
        events.append(
            {
                "tool_name": name,
                "arguments": row.get("arguments") if isinstance(row.get("arguments"), Mapping) else {},
                "result": row.get("result") if isinstance(row.get("result"), Mapping) else {},
                "stage": row.get("stage"),
                "step": row.get("step"),
                "round": row.get("round"),
                "tool_index": row.get("tool_index"),
            }
        )
    return events
