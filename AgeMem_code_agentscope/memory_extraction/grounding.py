"""Action-grounded M6 atomic propositions without Oracle-label leakage.

This module consumes only the public action/result payload, memory snapshots,
validated triples, semantic-state deltas, and an explicitly supplied relevance
resolver.  It never reads ``oracle_labels`` or private memory metadata such as
``role``, ``stale``, or ``duplicate_of_fact_id``.
"""

from __future__ import annotations

from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..action_schema import ActionEvent, TrajectoryStepV2
from ..memory_oracle.models import APName, AP_ORDER, OracleAPEvent
from ..trajectory import MemorySnapshotItem
from .models import (
    APRecord,
    RelevanceDecision,
    SemanticRole,
    TripleRecord,
    text_digest,
)
from .state import StateDelta, StateFact


MEMORY_DELTA_SCHEMA_VERSION = "agemem.memory_delta.v1"
GROUNDED_ACTION_SCHEMA_VERSION = "agemem.grounded_action.v1"
EXTRACTED_AP_GROUNDER_VERSION = "agemem.extracted_ap_grounder.v2"


class APGroundingError(ValueError):
    """Raised when action provenance or memory snapshots are not trustworthy."""


class MemoryEvidence(BaseModel):
    """Role-free memory evidence used for exact semantic grounding."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    memory_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    status: Literal["active", "superseded", "discarded"]
    content: str = Field(min_length=1)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_digest(self) -> "MemoryEvidence":
        if self.content_digest != text_digest(self.content):
            raise ValueError("content_digest does not match memory content")
        return self

    @classmethod
    def from_snapshot(cls, item: MemorySnapshotItem) -> "MemoryEvidence":
        if not item.content:
            raise APGroundingError("empty memory content cannot ground a proposition")
        return cls(
            memory_id=item.memory_id,
            version=item.version,
            status=item.status,
            content=item.content,
            content_digest=text_digest(item.content),
        )


class MemoryDelta(BaseModel):
    """Version-aware change between the before/after snapshots of one action."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[MEMORY_DELTA_SCHEMA_VERSION] = MEMORY_DELTA_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    added_active: Tuple[MemoryEvidence, ...] = ()
    superseded: Tuple[MemoryEvidence, ...] = ()
    discarded: Tuple[MemoryEvidence, ...] = ()
    active_after: Tuple[MemoryEvidence, ...] = ()
    returned_memory_ids: Tuple[str, ...] = ()
    tool_text: str = ""
    tool_interrupted: bool = False

    @model_validator(mode="after")
    def validate_delta(self) -> "MemoryDelta":
        groups = (
            self.added_active,
            self.superseded,
            self.discarded,
            self.active_after,
        )
        for group in groups:
            keys = [(item.memory_id, item.version) for item in group]
            if len(keys) != len(set(keys)):
                raise ValueError("memory evidence keys must be unique in each group")
        if len(self.returned_memory_ids) != len(set(self.returned_memory_ids)):
            raise ValueError("returned_memory_ids must be unique")
        active_ids = {item.memory_id for item in self.active_after}
        if set(self.returned_memory_ids) - active_ids:
            raise ValueError("returned memories must be active after the action")
        return self


def _memory_sort_key(item: MemoryEvidence) -> Tuple[str, int, str]:
    return item.memory_id, item.version, item.content_digest


def _public_tool_result(action: ActionEvent) -> Tuple[str, bool]:
    """Read an allowlist of public result fields and ignore metadata entirely."""

    result = action.result
    content = result.get("content", ())
    if not isinstance(content, (list, tuple)):
        raise APGroundingError("tool result content must be a list")
    texts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise APGroundingError("tool result content blocks must be objects")
        value = block.get("text")
        if value is not None:
            if not isinstance(value, str):
                raise APGroundingError("tool result text must be a string")
            texts.append(value)
    interrupted = result.get("is_interrupted", False)
    if not isinstance(interrupted, bool):
        raise APGroundingError("tool result is_interrupted must be boolean")
    return "\n".join(texts), interrupted


def derive_memory_delta(
    step: TrajectoryStepV2,
    action: Optional[ActionEvent] = None,
) -> MemoryDelta:
    """Derive a research-mode, versioned delta without consulting metadata."""

    if action is None:
        if len(step.actions) != 1:
            raise APGroundingError(
                "step-level memory snapshots require exactly one action"
            )
        action = step.actions[0]
    if action not in step.actions:
        raise APGroundingError("action is not contained in its trajectory step")
    if len(step.actions) != 1:
        raise APGroundingError(
            "M6 grounding requires per-action snapshots for multi-action turns"
        )

    before = {(item.memory_id, item.version): item for item in step.memory_before}
    after = {(item.memory_id, item.version): item for item in step.memory_after}
    if len(before) != len(step.memory_before) or len(after) != len(step.memory_after):
        raise APGroundingError("memory snapshots contain duplicate ID/version keys")
    removed = set(before) - set(after)
    if removed:
        raise APGroundingError(
            "memory history was physically removed instead of versioned/soft-deleted"
        )

    superseded: List[MemoryEvidence] = []
    discarded: List[MemoryEvidence] = []
    added: List[MemoryEvidence] = []
    for key, after_item in after.items():
        before_item = before.get(key)
        if before_item is None:
            evidence = MemoryEvidence.from_snapshot(after_item)
            if after_item.status == "active":
                added.append(evidence)
            elif after_item.status == "discarded":
                discarded.append(evidence)
            continue
        if before_item.content != after_item.content:
            raise APGroundingError("one memory version changed content in place")
        if before_item.status == after_item.status:
            continue
        transition = before_item.status, after_item.status
        evidence = MemoryEvidence.from_snapshot(after_item)
        if transition == ("active", "superseded"):
            superseded.append(evidence)
        elif transition == ("active", "discarded"):
            discarded.append(evidence)
        else:
            raise APGroundingError(
                f"invalid in-place memory status transition {transition}"
            )

    active_after = sorted(
        (
            MemoryEvidence.from_snapshot(item)
            for item in step.memory_after
            if item.status == "active"
        ),
        key=_memory_sort_key,
    )
    tool_text, interrupted = _public_tool_result(action)
    kind = _action_kind(action)
    returned_ids = ()
    if "retrieve" in kind and not interrupted:
        returned_ids = tuple(
            sorted(
                item.memory_id
                for item in active_after
                if item.content
                and item.content in tool_text
                and item.memory_id in tool_text
            )
        )
    return MemoryDelta(
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        timestep=action.timestep,
        action_id=action.action_id,
        added_active=tuple(sorted(added, key=_memory_sort_key)),
        superseded=tuple(sorted(superseded, key=_memory_sort_key)),
        discarded=tuple(sorted(discarded, key=_memory_sort_key)),
        active_after=tuple(active_after),
        returned_memory_ids=returned_ids,
        tool_text=tool_text,
        tool_interrupted=interrupted,
    )


class RelevanceResolver(Protocol):
    """Resolve action-bound triples without changing cacheable candidates."""

    decision_version: str

    def resolve(self, triple: TripleRecord) -> RelevanceDecision:
        """Return relevant, irrelevant, or abstain with provenance."""


class EvidenceDigestRelevanceResolver:
    """Explicit evaluation resolver backed by human semantic-target digests.

    This is an Oracle-derived evaluation component, not a learned extractor.
    Its role map must never be put into the group candidate cache.
    """

    def __init__(
        self,
        roles_by_evidence_digest: Mapping[str, SemanticRole],
        *,
        decision_version: str = "agemem.human_oracle_target.v1",
    ) -> None:
        if not decision_version.strip():
            raise ValueError("decision_version must be non-blank")
        invalid = set(roles_by_evidence_digest.values()) - {
            "relevant",
            "irrelevant",
            "abstain",
        }
        if invalid:
            raise ValueError(f"invalid semantic roles: {sorted(invalid)}")
        self._roles = dict(roles_by_evidence_digest)
        self.decision_version = decision_version

    def resolve(self, triple: TripleRecord) -> RelevanceDecision:
        roles = {
            self._roles.get(text_digest(span.text), "abstain")
            for span in triple.evidence
        }
        role: SemanticRole
        if len(roles) == 1:
            role = roles.pop()
        else:
            role = "abstain"
        confidence = 1.0 if role != "abstain" else 0.0
        return RelevanceDecision.create(
            triple,
            role=role,
            confidence=confidence,
            decision_version=self.decision_version,
        )


class GroundedAction(BaseModel):
    """Complete, action-addressable semantic derivation before DFA replay."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[GROUNDED_ACTION_SCHEMA_VERSION] = (
        GROUNDED_ACTION_SCHEMA_VERSION
    )
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    memory_delta: MemoryDelta
    state_delta: StateDelta
    relevance_decisions: Tuple[RelevanceDecision, ...] = ()
    atomic_propositions: Tuple[APRecord, ...] = ()

    @model_validator(mode="after")
    def validate_provenance(self) -> "GroundedAction":
        identity = (
            self.task_id,
            self.rollout_id,
            self.stage_id,
            self.timestep,
            self.action_id,
        )
        memory_identity = (
            self.memory_delta.task_id,
            self.memory_delta.rollout_id,
            self.memory_delta.stage_id,
            self.memory_delta.timestep,
            self.memory_delta.action_id,
        )
        state_identity = (
            self.state_delta.task_id,
            self.state_delta.rollout_id,
            self.state_delta.stage_id,
            self.state_delta.position.timestep,
            self.state_delta.action_id,
        )
        if identity != memory_identity or identity != state_identity:
            raise ValueError("memory/state derivations must match the source action")
        triple_ids = set(self.state_delta.accepted_triple_ids)
        if any(
            decision.action_id != self.action_id or decision.triple_id not in triple_ids
            for decision in self.relevance_decisions
        ):
            raise ValueError(
                "relevance decision does not match an accepted action triple"
            )
        propositions = [item.proposition for item in self.atomic_propositions]
        if len(propositions) != len(set(propositions)):
            raise ValueError(
                "one action may contain at most one AP record per proposition"
            )
        expected_order = [ap for ap in AP_ORDER if ap in propositions]
        if propositions != expected_order:
            raise ValueError("atomic propositions must follow canonical AP_ORDER")
        if any(
            (
                item.task_id,
                item.rollout_id,
                item.stage_id,
                item.timestep,
                item.action_id,
            )
            != identity
            for item in self.atomic_propositions
        ):
            raise ValueError("AP provenance must match its source action")
        return self

    def to_oracle_event(self, *, seed: int) -> OracleAPEvent:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise APGroundingError("seed must be a non-negative integer")
        evidence = {
            item.proposition: (item.ap_id,) for item in self.atomic_propositions
        }
        return OracleAPEvent(
            task_id=self.task_id,
            rollout_id=self.rollout_id,
            seed=seed,
            timestep=self.timestep,
            stage=self.stage_id,
            propositions=tuple(item.proposition for item in self.atomic_propositions),
            evidence_fact_ids=evidence,
        )


def _action_kind(action: ActionEvent) -> str:
    return action.action_type.strip().casefold()


def _triple_evidence_digests(triple: TripleRecord) -> Set[str]:
    return {text_digest(item.text) for item in triple.evidence}


class ExtractedAPGrounder:
    """Ground M4 APs from validated content changes, never naked tool calls."""

    def __init__(
        self,
        *,
        relevance_resolver: RelevanceResolver,
        required_relevant_digests: Iterable[str],
        grounder_version: str = EXTRACTED_AP_GROUNDER_VERSION,
    ) -> None:
        required = frozenset(required_relevant_digests)
        if not required:
            raise ValueError(
                "at least one required relevant evidence digest is required"
            )
        if any(len(value) != 64 for value in required):
            raise ValueError("required relevance keys must be SHA-256 digests")
        if not grounder_version.strip():
            raise ValueError("grounder_version must be non-blank")
        self.relevance_resolver = relevance_resolver
        self.required_relevant_digests = required
        self.grounder_version = grounder_version

    @staticmethod
    def _state_facts_for_triples(
        active_facts: Iterable[StateFact], triple_ids: Set[str]
    ) -> Tuple[StateFact, ...]:
        return tuple(
            sorted(
                (
                    fact
                    for fact in active_facts
                    if set(fact.evidence_triple_ids) & triple_ids
                ),
                key=lambda item: item.state_fact_id,
            )
        )

    def ground(
        self,
        *,
        step: TrajectoryStepV2,
        action: ActionEvent,
        triples: Iterable[TripleRecord],
        state_delta: StateDelta,
        active_state_facts: Iterable[StateFact],
        state_triple_history: Optional[Mapping[str, TripleRecord]] = None,
        retrieved_relevant_digests_before: Iterable[str] = (),
    ) -> GroundedAction:
        records = tuple(triples)
        accepted = set(state_delta.accepted_triple_ids)
        usable = tuple(item for item in records if item.triple_id in accepted)
        if {item.triple_id for item in usable} != accepted:
            raise APGroundingError("state delta accepted triples are incomplete")
        decisions = tuple(
            sorted(
                (self.relevance_resolver.resolve(item) for item in usable),
                key=lambda item: item.triple_id,
            )
        )
        decision_by_triple = {item.triple_id: item for item in decisions}
        active_facts = tuple(active_state_facts)
        active_evidence_ids = {
            triple_id for fact in active_facts for triple_id in fact.evidence_triple_ids
        }
        history = dict(
            state_triple_history or {item.triple_id: item for item in usable}
        )
        history.update({item.triple_id: item for item in usable})
        if active_evidence_ids - set(history):
            raise APGroundingError(
                "state_triple_history is missing active StateFact evidence"
            )
        state_records = tuple(
            history[triple_id] for triple_id in sorted(active_evidence_ids)
        )
        if any(
            item.task_id != action.task_id or item.rollout_id != action.rollout_id
            for item in state_records
        ):
            raise APGroundingError(
                "state triple history crossed task/rollout boundaries"
            )
        role_by_triple = {
            item.triple_id: self.relevance_resolver.resolve(item).role
            for item in state_records
        }
        role_by_digest: Dict[str, Set[SemanticRole]] = {}
        triple_ids_by_digest: Dict[str, Set[str]] = {}
        for triple in state_records:
            role = role_by_triple[triple.triple_id]
            for digest in _triple_evidence_digests(triple):
                role_by_digest.setdefault(digest, set()).add(role)
                triple_ids_by_digest.setdefault(digest, set()).add(triple.triple_id)

        memory_delta = derive_memory_delta(step, action)
        active_fact_ids = {item.state_fact_id for item in active_facts}
        if set(state_delta.active_state_fact_ids) != active_fact_ids:
            raise APGroundingError("active state facts do not match StateDelta")

        evidence: Dict[APName, Dict[str, Set[str]]] = {}

        def add_evidence(
            proposition: APName,
            *,
            triple_ids: Iterable[str] = (),
            state_fact_ids: Iterable[str] = (),
            memory_ids: Iterable[str] = (),
        ) -> None:
            bucket = evidence.setdefault(
                proposition,
                {"triple": set(), "state": set(), "memory": set()},
            )
            bucket["triple"].update(triple_ids)
            bucket["state"].update(state_fact_ids)
            bucket["memory"].update(memory_ids)

        inserted_triples = {
            triple_id
            for fact in state_delta.inserted
            for triple_id in fact.evidence_triple_ids
        }
        observed = {
            item.triple_id
            for item in usable
            if item.triple_id in inserted_triples
            and decision_by_triple[item.triple_id].role == "relevant"
            and any(span.source == "observation" for span in item.evidence)
        }
        if observed:
            facts = self._state_facts_for_triples(active_facts, observed)
            add_evidence(
                "observed_supporting_fact",
                triple_ids=observed,
                state_fact_ids=(item.state_fact_id for item in facts),
            )

        for item in memory_delta.added_active:
            ids = triple_ids_by_digest.get(item.content_digest, set())
            roles = role_by_digest.get(item.content_digest, {"abstain"})
            facts = self._state_facts_for_triples(active_facts, ids)
            if roles == {"relevant"} and ids and facts:
                add_evidence(
                    "stored_supporting_fact",
                    triple_ids=ids,
                    state_fact_ids=(fact.state_fact_id for fact in facts),
                    memory_ids=(item.memory_id,),
                )
            elif roles == {"irrelevant"} and ids and facts:
                add_evidence(
                    "stored_irrelevant_fact",
                    triple_ids=ids,
                    state_fact_ids=(fact.state_fact_id for fact in facts),
                    memory_ids=(item.memory_id,),
                )

        kind = _action_kind(action)
        if "update" in kind and memory_delta.added_active and memory_delta.superseded:
            if state_delta.inserted and state_delta.superseded:
                ids = {
                    triple_id
                    for fact in state_delta.inserted
                    for triple_id in fact.evidence_triple_ids
                }
                add_evidence(
                    "updated_stale_fact",
                    triple_ids=ids,
                    state_fact_ids=(
                        item.state_fact_id for item in state_delta.inserted
                    ),
                    memory_ids=(item.memory_id for item in memory_delta.added_active),
                )

        for item in memory_delta.discarded:
            ids = triple_ids_by_digest.get(item.content_digest, set())
            roles = role_by_digest.get(item.content_digest, {"abstain"})
            if roles == {"relevant"} and ids:
                add_evidence(
                    "deleted_supporting_fact",
                    triple_ids=ids,
                    memory_ids=(item.memory_id,),
                )

        if "retrieve" in kind and not memory_delta.tool_interrupted:
            active_by_id = {item.memory_id: item for item in memory_delta.active_after}
            for memory_id in memory_delta.returned_memory_ids:
                item = active_by_id[memory_id]
                ids = triple_ids_by_digest.get(item.content_digest, set())
                roles = role_by_digest.get(item.content_digest, {"abstain"})
                facts = self._state_facts_for_triples(active_facts, ids)
                if roles == {"relevant"} and ids and facts:
                    add_evidence(
                        "retrieved_supporting_fact",
                        triple_ids=ids,
                        state_fact_ids=(fact.state_fact_id for fact in facts),
                        memory_ids=(memory_id,),
                    )
                elif roles == {"irrelevant"} and ids and facts:
                    add_evidence(
                        "retrieved_irrelevant_fact",
                        triple_ids=ids,
                        state_fact_ids=(fact.state_fact_id for fact in facts),
                        memory_ids=(memory_id,),
                    )

        active_memory_by_digest = {
            item.content_digest: item for item in memory_delta.active_after
        }
        retrieved_before = set(retrieved_relevant_digests_before)
        if retrieved_before - self.required_relevant_digests:
            raise APGroundingError(
                "retrieved relevance state contains an unknown required digest"
            )
        returned_relevant = {
            item.content_digest
            for item in memory_delta.active_after
            if item.memory_id in set(memory_delta.returned_memory_ids)
            and item.content_digest in self.required_relevant_digests
        }
        retrieved_after = retrieved_before | returned_relevant
        coverage_triples: Set[str] = set()
        coverage_facts: Set[str] = set()
        coverage_memory: Set[str] = set()
        coverage_complete = self.required_relevant_digests.issubset(retrieved_after)
        for digest in self.required_relevant_digests:
            memory = active_memory_by_digest.get(digest)
            ids = triple_ids_by_digest.get(digest, set())
            roles = role_by_digest.get(digest, {"abstain"})
            facts = self._state_facts_for_triples(active_facts, ids)
            if memory is None or not ids or roles != {"relevant"} or not facts:
                coverage_complete = False
                break
            coverage_triples.update(ids)
            coverage_facts.update(item.state_fact_id for item in facts)
            coverage_memory.add(memory.memory_id)
        # M4 defines coverage over accumulated successful retrievals.  Active
        # storage alone must not advance the coverage milestone.  The Oracle
        # emits this AP on the completing/subsequent RETRIEVE and on ANSWER.
        coverage_action = "retrieve" in kind or (step.done and "answer" in kind)
        if coverage_complete and coverage_action:
            add_evidence(
                "supporting_coverage_complete",
                triple_ids=coverage_triples,
                state_fact_ids=coverage_facts,
                memory_ids=coverage_memory,
            )

        if step.done and step.env_reward > 0.0 and "answer" in kind:
            add_evidence("answered_correctly")

        ap_records: List[APRecord] = []
        for proposition in AP_ORDER:
            item = evidence.get(proposition)
            if item is None:
                continue
            source_action_ids = {
                history[triple_id].action_id
                for triple_id in item["triple"]
                if triple_id in history
            }
            source_action_ids.add(action.action_id)
            ap_records.append(
                APRecord.create(
                    task_id=action.task_id,
                    rollout_id=action.rollout_id,
                    stage_id=action.stage_id,
                    timestep=action.timestep,
                    action_id=action.action_id,
                    proposition=proposition,
                    confidence=1.0,
                    evidence_triple_ids=tuple(sorted(item["triple"])),
                    evidence_state_fact_ids=tuple(sorted(item["state"])),
                    evidence_memory_ids=tuple(sorted(item["memory"])),
                    evidence_action_ids=tuple(sorted(source_action_ids)),
                    grounder_version=self.grounder_version,
                )
            )
        return GroundedAction(
            task_id=action.task_id,
            rollout_id=action.rollout_id,
            stage_id=action.stage_id,
            timestep=action.timestep,
            action_id=action.action_id,
            memory_delta=memory_delta,
            state_delta=state_delta,
            relevance_decisions=decisions,
            atomic_propositions=tuple(ap_records),
        )


__all__ = [
    "EXTRACTED_AP_GROUNDER_VERSION",
    "GROUNDED_ACTION_SCHEMA_VERSION",
    "MEMORY_DELTA_SCHEMA_VERSION",
    "APGroundingError",
    "EvidenceDigestRelevanceResolver",
    "ExtractedAPGrounder",
    "GroundedAction",
    "MemoryDelta",
    "MemoryEvidence",
    "RelevanceResolver",
    "derive_memory_delta",
]
