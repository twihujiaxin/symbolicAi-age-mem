"""Deterministic, rollout-isolated semantic state tracking for M6.

The tracker consumes action-bound :class:`TripleRecord` values.  It never
inspects Oracle labels, task answers, or memory-role metadata.  State changes
are append-only at the version-history level: a single-valued conflict closes
the old fact's half-open validity interval and appends a new version, while a
repeated value only reinforces provenance on the existing version.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..action_schema.models import ActionEvent
from .models import TripleRecord, canonical_digest


STATE_TRACKER_VERSION = "agemem.state_tracker.v1"
STATE_FACT_SCHEMA_VERSION = "agemem.state_fact.v1"
STATE_DELTA_SCHEMA_VERSION = "agemem.state_delta.v1"
STATE_SNAPSHOT_SCHEMA_VERSION = "agemem.state_snapshot.v1"
QUARANTINE_SCHEMA_VERSION = "agemem.state_quarantine.v1"

Cardinality = Literal["single", "multi"]
StateFactStatus = Literal["active", "superseded"]
StateQuarantineReason = Literal[
    "unknown_category",
    "unknown_subject",
    "unresolved_pronoun",
    "same_action_conflict",
]

_CATEGORY_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_PRONOUNS = {
    "he",
    "her",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "it",
    "its",
    "itself",
    "she",
    "that",
    "their",
    "theirs",
    "them",
    "themselves",
    "these",
    "they",
    "this",
    "those",
}


def _normalize(value: str) -> str:
    """Normalize semantic labels without fuzzy or substring matching."""

    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _non_blank(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _position_key(position: "ActionPosition") -> Tuple[int, int, int]:
    return (
        position.timestep,
        position.assistant_turn_id,
        position.action_index_in_turn,
    )


class StateTrackerError(ValueError):
    """Raised when an action stream or snapshot cannot be trusted."""


class ActionPosition(BaseModel):
    """Strict total order for actions, including multi-action assistant turns."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    timestep: int = Field(ge=0)
    assistant_turn_id: int = Field(ge=0)
    action_index_in_turn: int = Field(ge=0)

    @classmethod
    def from_action(cls, action: ActionEvent) -> "ActionPosition":
        return cls(
            timestep=action.timestep,
            assistant_turn_id=action.assistant_turn_id,
            action_index_in_turn=action.action_index_in_turn,
        )

    def key(self) -> Tuple[int, int, int]:
        return _position_key(self)


class CategorySpec(BaseModel):
    """Versioned relation policy controlling overwrite cardinality."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    cardinality: Cardinality

    @field_validator("name")
    @classmethod
    def name_must_be_canonical(cls, value: str) -> str:
        _non_blank(value, "category name")
        if _normalize(value) != value or not _CATEGORY_NAME.fullmatch(value):
            raise ValueError("category name must be a canonical lowercase slug")
        return value


class StateFact(BaseModel):
    """One immutable semantic-state version with retained provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[STATE_FACT_SCHEMA_VERSION] = STATE_FACT_SCHEMA_VERSION
    state_fact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    normalized_subject: str = Field(min_length=1)
    category: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    value: str = Field(min_length=1)
    normalized_value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    version: int = Field(ge=1)
    status: StateFactStatus
    source_step: int = Field(ge=0)
    source_action_id: str = Field(min_length=1)
    evidence_triple_ids: Tuple[str, ...] = Field(min_length=1)
    provenance_action_ids: Tuple[str, ...] = Field(min_length=1)
    valid_from: ActionPosition
    valid_to: Optional[ActionPosition] = None

    @field_validator("task_id", "rollout_id", "subject", "value", "source_action_id")
    @classmethod
    def text_must_not_be_blank(cls, value: str, info) -> str:
        return _non_blank(value, info.field_name)

    @model_validator(mode="after")
    def validate_version(self) -> "StateFact":
        if _normalize(self.subject) != self.normalized_subject:
            raise ValueError("normalized_subject does not match subject")
        if _normalize(self.value) != self.normalized_value:
            raise ValueError("normalized_value does not match value")
        if len(self.evidence_triple_ids) != len(set(self.evidence_triple_ids)):
            raise ValueError("evidence_triple_ids must be unique")
        if len(self.provenance_action_ids) != len(set(self.provenance_action_ids)):
            raise ValueError("provenance_action_ids must be unique")
        if self.provenance_action_ids[0] != self.source_action_id:
            raise ValueError("source_action_id must be the first provenance action")
        if self.source_step != self.valid_from.timestep:
            raise ValueError("source_step must equal valid_from.timestep")
        if self.status == "active" and self.valid_to is not None:
            raise ValueError("active state facts cannot have valid_to")
        if self.status == "superseded":
            if self.valid_to is None:
                raise ValueError("superseded state facts require valid_to")
            if self.valid_to.key() <= self.valid_from.key():
                raise ValueError("valid_to must be later than valid_from")
        if self.state_fact_id != self.expected_state_fact_id():
            raise ValueError("state_fact_id does not match semantic version identity")
        return self

    def expected_state_fact_id(self) -> str:
        return canonical_digest(
            {
                "namespace": STATE_FACT_SCHEMA_VERSION,
                "task_id": self.task_id,
                "rollout_id": self.rollout_id,
                "subject": self.normalized_subject,
                "category": self.category,
                "value": self.normalized_value,
                "version": self.version,
            }
        )

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        rollout_id: str,
        subject: str,
        category: str,
        value: str,
        confidence: float,
        version: int,
        source_action_id: str,
        evidence_triple_ids: Tuple[str, ...],
        valid_from: ActionPosition,
    ) -> "StateFact":
        normalized_subject = _normalize(subject)
        normalized_value = _normalize(value)
        identity = {
            "namespace": STATE_FACT_SCHEMA_VERSION,
            "task_id": task_id,
            "rollout_id": rollout_id,
            "subject": normalized_subject,
            "category": category,
            "value": normalized_value,
            "version": version,
        }
        return cls(
            state_fact_id=canonical_digest(identity),
            task_id=task_id,
            rollout_id=rollout_id,
            subject=subject,
            normalized_subject=normalized_subject,
            category=category,
            value=value,
            normalized_value=normalized_value,
            confidence=confidence,
            version=version,
            status="active",
            source_step=valid_from.timestep,
            source_action_id=source_action_id,
            evidence_triple_ids=evidence_triple_ids,
            provenance_action_ids=(source_action_id,),
            valid_from=valid_from,
        )


class QuarantineRecord(BaseModel):
    """Action-bound audit record for a triple excluded from semantic state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[QUARANTINE_SCHEMA_VERSION] = QUARANTINE_SCHEMA_VERSION
    quarantine_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    position: ActionPosition
    triple_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject: str = Field(min_length=1)
    category: str = Field(min_length=1)
    value: str = Field(min_length=1)
    reason: StateQuarantineReason
    message: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> "QuarantineRecord":
        if self.quarantine_id != self.expected_quarantine_id():
            raise ValueError("quarantine_id does not match rejected triple")
        return self

    def expected_quarantine_id(self) -> str:
        return canonical_digest(
            {
                "namespace": QUARANTINE_SCHEMA_VERSION,
                "task_id": self.task_id,
                "rollout_id": self.rollout_id,
                "action_id": self.action_id,
                "triple_id": self.triple_id,
                "reason": self.reason,
            }
        )

    @classmethod
    def from_triple(
        cls,
        triple: TripleRecord,
        *,
        reason: StateQuarantineReason,
        message: str,
    ) -> "QuarantineRecord":
        identity = {
            "namespace": QUARANTINE_SCHEMA_VERSION,
            "task_id": triple.task_id,
            "rollout_id": triple.rollout_id,
            "action_id": triple.action_id,
            "triple_id": triple.triple_id,
            "reason": reason,
        }
        return cls(
            quarantine_id=canonical_digest(identity),
            task_id=triple.task_id,
            rollout_id=triple.rollout_id,
            stage_id=triple.stage_id,
            action_id=triple.action_id,
            position=ActionPosition(
                timestep=triple.timestep,
                assistant_turn_id=triple.assistant_turn_id,
                action_index_in_turn=triple.action_index_in_turn,
            ),
            triple_id=triple.triple_id,
            subject=triple.subject,
            category=triple.category,
            value=triple.value,
            reason=reason,
            message=message,
        )


class StateDelta(BaseModel):
    """Auditable state mutations caused by exactly one original action."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[STATE_DELTA_SCHEMA_VERSION] = STATE_DELTA_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    position: ActionPosition
    input_triple_ids: Tuple[str, ...] = ()
    accepted_triple_ids: Tuple[str, ...] = ()
    quarantined_triple_ids: Tuple[str, ...] = ()
    inserted: Tuple[StateFact, ...] = ()
    superseded: Tuple[StateFact, ...] = ()
    reinforced: Tuple[StateFact, ...] = ()
    quarantine: Tuple[QuarantineRecord, ...] = ()
    active_state_fact_ids: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_delta(self) -> "StateDelta":
        id_groups = (
            self.input_triple_ids,
            self.accepted_triple_ids,
            self.quarantined_triple_ids,
            self.active_state_fact_ids,
        )
        if any(len(values) != len(set(values)) for values in id_groups):
            raise ValueError("delta identity lists must contain unique values")
        if set(self.accepted_triple_ids) & set(self.quarantined_triple_ids):
            raise ValueError("accepted and quarantined triples must be disjoint")
        if set(self.accepted_triple_ids) | set(self.quarantined_triple_ids) != set(
            self.input_triple_ids
        ):
            raise ValueError("every input triple must be accepted or quarantined")
        if (
            tuple(item.triple_id for item in self.quarantine)
            != self.quarantined_triple_ids
        ):
            raise ValueError(
                "quarantine rows must preserve quarantined_triple_ids order"
            )
        fact_id_groups = (
            tuple(item.state_fact_id for item in self.inserted),
            tuple(item.state_fact_id for item in self.superseded),
            tuple(item.state_fact_id for item in self.reinforced),
        )
        if any(len(values) != len(set(values)) for values in fact_id_groups):
            raise ValueError("state mutations must not repeat a state fact")
        if any(item.status != "active" for item in self.inserted + self.reinforced):
            raise ValueError("inserted and reinforced state facts must be active")
        if any(item.status != "superseded" for item in self.superseded):
            raise ValueError("superseded delta facts must be closed")
        for fact in self.inserted + self.superseded + self.reinforced:
            if fact.task_id != self.task_id or fact.rollout_id != self.rollout_id:
                raise ValueError("state mutation identity must match delta")
        for item in self.quarantine:
            if (
                item.task_id != self.task_id
                or item.rollout_id != self.rollout_id
                or item.action_id != self.action_id
                or item.position != self.position
            ):
                raise ValueError("quarantine identity must match delta")
        return self


class StateSnapshot(BaseModel):
    """Complete deterministic checkpoint for one isolated rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[STATE_SNAPSHOT_SCHEMA_VERSION] = (
        STATE_SNAPSHOT_SCHEMA_VERSION
    )
    tracker_version: Literal[STATE_TRACKER_VERSION] = STATE_TRACKER_VERSION
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    facts: Tuple[StateFact, ...]
    quarantine: Tuple[QuarantineRecord, ...]
    processed_action_ids: Tuple[str, ...] = Field(min_length=1)
    last_position: ActionPosition
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_snapshot(self) -> "StateSnapshot":
        if len(self.processed_action_ids) != len(set(self.processed_action_ids)):
            raise ValueError("processed_action_ids must be unique")
        fact_ids = [item.state_fact_id for item in self.facts]
        quarantine_ids = [item.quarantine_id for item in self.quarantine]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("snapshot state_fact_ids must be unique")
        if len(quarantine_ids) != len(set(quarantine_ids)):
            raise ValueError("snapshot quarantine_ids must be unique")
        for fact in self.facts:
            if fact.task_id != self.task_id or fact.rollout_id != self.rollout_id:
                raise ValueError("snapshot fact identity mismatch")
            if not set(fact.provenance_action_ids).issubset(self.processed_action_ids):
                raise ValueError("snapshot fact refers to an unprocessed action")
            if fact.valid_from.key() > self.last_position.key():
                raise ValueError("state fact begins after snapshot position")
            if (
                fact.valid_to is not None
                and fact.valid_to.key() > self.last_position.key()
            ):
                raise ValueError("state fact closes after snapshot position")
        for item in self.quarantine:
            if item.task_id != self.task_id or item.rollout_id != self.rollout_id:
                raise ValueError("snapshot quarantine identity mismatch")
            if item.action_id not in self.processed_action_ids:
                raise ValueError("quarantine refers to an unprocessed action")
        if self.digest != self.expected_digest():
            raise ValueError("snapshot digest does not match payload")
        return self

    def canonical_dict(self, *, include_digest: bool = True) -> Dict[str, object]:
        data = self.model_dump(mode="json")
        if not include_digest:
            data.pop("digest", None)
        return data

    def expected_digest(self) -> str:
        return canonical_digest(self.canonical_dict(include_digest=False))

    def to_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def create(
        cls,
        *,
        config_digest: str,
        task_id: str,
        rollout_id: str,
        facts: Tuple[StateFact, ...],
        quarantine: Tuple[QuarantineRecord, ...],
        processed_action_ids: Tuple[str, ...],
        last_position: ActionPosition,
    ) -> "StateSnapshot":
        digest_payload: Dict[str, object] = {
            "schema_version": STATE_SNAPSHOT_SCHEMA_VERSION,
            "tracker_version": STATE_TRACKER_VERSION,
            "config_digest": config_digest,
            "task_id": task_id,
            "rollout_id": rollout_id,
            "facts": [item.model_dump(mode="json") for item in facts],
            "quarantine": [item.model_dump(mode="json") for item in quarantine],
            "processed_action_ids": processed_action_ids,
            "last_position": last_position.model_dump(mode="json"),
        }
        return cls(
            config_digest=config_digest,
            task_id=task_id,
            rollout_id=rollout_id,
            facts=facts,
            quarantine=quarantine,
            processed_action_ids=processed_action_ids,
            last_position=last_position,
            digest=canonical_digest(digest_payload),
        )


@dataclass
class _RolloutState:
    task_id: str
    facts: Dict[str, StateFact] = field(default_factory=dict)
    quarantine: List[QuarantineRecord] = field(default_factory=list)
    processed_action_ids: List[str] = field(default_factory=list)
    last_position: Optional[ActionPosition] = None

    def clone(self) -> "_RolloutState":
        return _RolloutState(
            task_id=self.task_id,
            facts=dict(self.facts),
            quarantine=list(self.quarantine),
            processed_action_ids=list(self.processed_action_ids),
            last_position=self.last_position,
        )


class StateTracker:
    """Track semantic state independently for every rollout ID."""

    def __init__(
        self,
        *,
        categories: Iterable[CategorySpec],
        known_subjects: Iterable[str],
    ) -> None:
        category_items = tuple(categories)
        if not category_items:
            raise ValueError("at least one category specification is required")
        category_names = [item.name for item in category_items]
        if len(category_names) != len(set(category_names)):
            raise ValueError("category specification names must be unique")
        subject_items = tuple(known_subjects)
        if not subject_items or any(not item.strip() for item in subject_items):
            raise ValueError("known_subjects must contain non-blank values")
        normalized_subjects = tuple(_normalize(item) for item in subject_items)
        if len(normalized_subjects) != len(set(normalized_subjects)):
            raise ValueError("known_subjects must be unique after normalization")

        self._categories = {item.name: item for item in category_items}
        self._known_subjects = frozenset(normalized_subjects)
        self._config_digest = canonical_digest(
            {
                "tracker_version": STATE_TRACKER_VERSION,
                "categories": [
                    item.model_dump(mode="json")
                    for item in sorted(category_items, key=lambda value: value.name)
                ],
                "known_subjects": sorted(self._known_subjects),
            }
        )
        self._rollouts: Dict[str, _RolloutState] = {}

    @property
    def config_digest(self) -> str:
        return self._config_digest

    def _canonical_category(self, category: str) -> Optional[CategorySpec]:
        return self._categories.get(_normalize(category))

    def _subject_reason(self, subject: str) -> Optional[StateQuarantineReason]:
        normalized = _normalize(subject)
        if normalized in self._known_subjects:
            return None
        first = normalized.split(maxsplit=1)[0].strip("'\"()[]{}.,:;!?")
        if first in _PRONOUNS:
            return "unresolved_pronoun"
        return "unknown_subject"

    @staticmethod
    def _fact_sort_key(fact: StateFact) -> Tuple[object, ...]:
        return (
            fact.normalized_subject,
            fact.category,
            fact.version,
            fact.normalized_value,
            fact.state_fact_id,
        )

    @staticmethod
    def _quarantine_sort_key(item: QuarantineRecord) -> Tuple[object, ...]:
        return (*item.position.key(), item.triple_id, item.reason)

    @staticmethod
    def _triple_matches_action(triple: TripleRecord, action: ActionEvent) -> bool:
        return (
            triple.task_id,
            triple.rollout_id,
            triple.stage_id,
            triple.timestep,
            triple.action_id,
            triple.assistant_turn_id,
            triple.action_index_in_turn,
        ) == (
            action.task_id,
            action.rollout_id,
            action.stage_id,
            action.timestep,
            action.action_id,
            action.assistant_turn_id,
            action.action_index_in_turn,
        )

    @staticmethod
    def _facts_for_key(state: _RolloutState, key: Tuple[str, str]) -> List[StateFact]:
        return [
            fact
            for fact in state.facts.values()
            if (fact.normalized_subject, fact.category) == key
        ]

    @classmethod
    def _active_for_key(
        cls, state: _RolloutState, key: Tuple[str, str]
    ) -> List[StateFact]:
        return [
            fact for fact in cls._facts_for_key(state, key) if fact.status == "active"
        ]

    @staticmethod
    def _reinforce(
        fact: StateFact,
        records: Sequence[TripleRecord],
        action_id: str,
    ) -> StateFact:
        evidence = tuple(
            sorted(set(fact.evidence_triple_ids) | {item.triple_id for item in records})
        )
        provenance = fact.provenance_action_ids
        if action_id not in provenance:
            provenance = (*provenance, action_id)
        payload = fact.model_dump(mode="python")
        payload.update(
            confidence=max(fact.confidence, *(item.confidence for item in records)),
            evidence_triple_ids=evidence,
            provenance_action_ids=provenance,
        )
        return StateFact.model_validate(payload)

    @staticmethod
    def _close(fact: StateFact, position: ActionPosition) -> StateFact:
        payload = fact.model_dump(mode="python")
        payload.update(status="superseded", valid_to=position)
        return StateFact.model_validate(payload)

    def apply(
        self,
        action: ActionEvent,
        triples: Iterable[TripleRecord],
    ) -> StateDelta:
        """Apply one action atomically, failing closed on stream corruption."""

        records = tuple(triples)
        triple_ids = [item.triple_id for item in records]
        if len(triple_ids) != len(set(triple_ids)):
            raise StateTrackerError("input triple IDs must be unique")
        if any(not self._triple_matches_action(item, action) for item in records):
            raise StateTrackerError("triple identity does not match its source action")

        position = ActionPosition.from_action(action)
        existing = self._rollouts.get(action.rollout_id)
        if existing is None:
            working = _RolloutState(task_id=action.task_id)
        else:
            if existing.task_id != action.task_id:
                raise StateTrackerError(
                    "one rollout_id cannot contain multiple task IDs"
                )
            working = existing.clone()
        if action.action_id in working.processed_action_ids:
            raise StateTrackerError(f"duplicate action_id {action.action_id!r}")
        if (
            working.last_position is not None
            and position.key() <= working.last_position.key()
        ):
            raise StateTrackerError(
                "action positions must increase by "
                "(timestep, assistant_turn_id, action_index_in_turn)"
            )

        accepted: set[str] = set()
        quarantine: List[QuarantineRecord] = []
        groups: Dict[Tuple[str, str], List[TripleRecord]] = {}
        for triple in sorted(records, key=lambda item: item.triple_id):
            category = self._canonical_category(triple.category)
            if category is None:
                quarantine.append(
                    QuarantineRecord.from_triple(
                        triple,
                        reason="unknown_category",
                        message="category is not present in the versioned registry",
                    )
                )
                continue
            subject_reason = self._subject_reason(triple.subject)
            if subject_reason is not None:
                quarantine.append(
                    QuarantineRecord.from_triple(
                        triple,
                        reason=subject_reason,
                        message=(
                            "subject is an unresolved pronoun"
                            if subject_reason == "unresolved_pronoun"
                            else "subject is not present in the known-subject registry"
                        ),
                    )
                )
                continue
            groups.setdefault((_normalize(triple.subject), category.name), []).append(
                triple
            )

        inserted: List[StateFact] = []
        superseded: List[StateFact] = []
        reinforced: List[StateFact] = []
        for key in sorted(groups):
            category = self._categories[key[1]]
            by_value: Dict[str, List[TripleRecord]] = {}
            for record in groups[key]:
                by_value.setdefault(_normalize(record.value), []).append(record)
            if category.cardinality == "single" and len(by_value) > 1:
                for record in sorted(groups[key], key=lambda item: item.triple_id):
                    quarantine.append(
                        QuarantineRecord.from_triple(
                            record,
                            reason="same_action_conflict",
                            message=(
                                "one action proposed multiple values for a "
                                "single-valued state key"
                            ),
                        )
                    )
                continue

            for normalized_value in sorted(by_value):
                value_records = sorted(
                    by_value[normalized_value], key=lambda item: item.triple_id
                )
                accepted.update(item.triple_id for item in value_records)
                active = self._active_for_key(working, key)
                matching = [
                    fact for fact in active if fact.normalized_value == normalized_value
                ]
                if len(matching) > 1:
                    raise StateTrackerError(
                        "state contains duplicate active semantic values"
                    )
                if matching:
                    updated = self._reinforce(
                        matching[0], value_records, action.action_id
                    )
                    working.facts[updated.state_fact_id] = updated
                    reinforced.append(updated)
                    continue

                if category.cardinality == "single":
                    if len(active) > 1:
                        raise StateTrackerError(
                            "single-valued state key has multiple active values"
                        )
                    for old in active:
                        closed = self._close(old, position)
                        working.facts[closed.state_fact_id] = closed
                        superseded.append(closed)
                    prior_versions = [
                        fact.version for fact in self._facts_for_key(working, key)
                    ]
                    version = max(prior_versions, default=0) + 1
                else:
                    version = 1

                representative = value_records[0]
                created = StateFact.create(
                    task_id=action.task_id,
                    rollout_id=action.rollout_id,
                    subject=representative.subject,
                    category=category.name,
                    value=representative.value,
                    confidence=max(item.confidence for item in value_records),
                    version=version,
                    source_action_id=action.action_id,
                    evidence_triple_ids=tuple(
                        sorted(item.triple_id for item in value_records)
                    ),
                    valid_from=position,
                )
                if created.state_fact_id in working.facts:
                    raise StateTrackerError("state fact identity collision")
                working.facts[created.state_fact_id] = created
                inserted.append(created)

        quarantine.sort(key=self._quarantine_sort_key)
        quarantined_ids = tuple(item.triple_id for item in quarantine)
        working.quarantine.extend(quarantine)
        working.processed_action_ids.append(action.action_id)
        working.last_position = position
        self._rollouts[action.rollout_id] = working

        active_ids = tuple(
            item.state_fact_id
            for item in sorted(
                (fact for fact in working.facts.values() if fact.status == "active"),
                key=self._fact_sort_key,
            )
        )
        return StateDelta(
            task_id=action.task_id,
            rollout_id=action.rollout_id,
            stage_id=action.stage_id,
            action_id=action.action_id,
            position=position,
            input_triple_ids=tuple(sorted(triple_ids)),
            accepted_triple_ids=tuple(sorted(accepted)),
            quarantined_triple_ids=quarantined_ids,
            inserted=tuple(sorted(inserted, key=self._fact_sort_key)),
            superseded=tuple(sorted(superseded, key=self._fact_sort_key)),
            reinforced=tuple(sorted(reinforced, key=self._fact_sort_key)),
            quarantine=tuple(quarantine),
            active_state_fact_ids=active_ids,
        )

    def active_facts(self, rollout_id: str) -> Tuple[StateFact, ...]:
        state = self._rollouts.get(rollout_id)
        if state is None:
            return ()
        return tuple(
            sorted(
                (item for item in state.facts.values() if item.status == "active"),
                key=self._fact_sort_key,
            )
        )

    def history(self, rollout_id: str) -> Tuple[StateFact, ...]:
        state = self._rollouts.get(rollout_id)
        if state is None:
            return ()
        return tuple(sorted(state.facts.values(), key=self._fact_sort_key))

    def quarantine(self, rollout_id: str) -> Tuple[QuarantineRecord, ...]:
        state = self._rollouts.get(rollout_id)
        if state is None:
            return ()
        return tuple(sorted(state.quarantine, key=self._quarantine_sort_key))

    def snapshot(self, rollout_id: str) -> StateSnapshot:
        state = self._rollouts.get(rollout_id)
        if state is None or state.last_position is None:
            raise StateTrackerError(f"unknown or empty rollout {rollout_id!r}")
        return StateSnapshot.create(
            config_digest=self.config_digest,
            task_id=state.task_id,
            rollout_id=rollout_id,
            facts=tuple(sorted(state.facts.values(), key=self._fact_sort_key)),
            quarantine=tuple(sorted(state.quarantine, key=self._quarantine_sort_key)),
            processed_action_ids=tuple(state.processed_action_ids),
            last_position=state.last_position,
        )

    def restore(self, snapshot: StateSnapshot) -> None:
        if snapshot.config_digest != self.config_digest:
            raise StateTrackerError("snapshot tracker configuration does not match")
        for fact in snapshot.facts:
            if fact.category not in self._categories:
                raise StateTrackerError("snapshot contains an unknown category")
            if fact.normalized_subject not in self._known_subjects:
                raise StateTrackerError("snapshot contains an unknown subject")
        active_by_key: Dict[Tuple[str, str], List[StateFact]] = {}
        for fact in snapshot.facts:
            if fact.status == "active":
                active_by_key.setdefault(
                    (fact.normalized_subject, fact.category), []
                ).append(fact)
        for key, facts in active_by_key.items():
            category = self._categories[key[1]]
            if category.cardinality == "single" and len(facts) > 1:
                raise StateTrackerError(
                    "snapshot has multiple active values for a single-valued key"
                )
            values = [item.normalized_value for item in facts]
            if len(values) != len(set(values)):
                raise StateTrackerError("snapshot has duplicate active semantic values")

        self._rollouts[snapshot.rollout_id] = _RolloutState(
            task_id=snapshot.task_id,
            facts={item.state_fact_id: item for item in snapshot.facts},
            quarantine=list(snapshot.quarantine),
            processed_action_ids=list(snapshot.processed_action_ids),
            last_position=snapshot.last_position,
        )

    def reset(self, rollout_id: Optional[str] = None) -> None:
        if rollout_id is None:
            self._rollouts.clear()
        else:
            self._rollouts.pop(rollout_id, None)


__all__ = [
    "QUARANTINE_SCHEMA_VERSION",
    "STATE_DELTA_SCHEMA_VERSION",
    "STATE_FACT_SCHEMA_VERSION",
    "STATE_SNAPSHOT_SCHEMA_VERSION",
    "STATE_TRACKER_VERSION",
    "ActionPosition",
    "Cardinality",
    "CategorySpec",
    "QuarantineRecord",
    "StateDelta",
    "StateFact",
    "StateQuarantineReason",
    "StateSnapshot",
    "StateTracker",
    "StateTrackerError",
]
