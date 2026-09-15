"""Private, deterministic world solver for monotonic single-valued timelines.

This module accepts only private event/query records.  It has no dependency on
the public environment, memory store, policy action schema, or renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import DynamicQueryPrivate, PrivateEvent, WORLD_ORACLE_VERSION


class OracleError(ValueError):
    pass


@dataclass(frozen=True)
class ValidityInterval:
    event_id: str
    entity: str
    relation: str
    value: str
    valid_from: int
    valid_to: int | None
    observed_at: int

    def contains(self, time: int) -> bool:
        return self.valid_from <= time and (
            self.valid_to is None or time < self.valid_to
        )


@dataclass(frozen=True)
class OracleAnswer:
    query_id: str
    answers: tuple[str, ...]
    answerable: bool
    support_event_ids: tuple[str, ...]
    answer_available_at: int
    oracle_version: str = WORLD_ORACLE_VERSION


class WorldOracle:
    """Scan events directly; no rendered text is parsed or trusted."""

    version = WORLD_ORACLE_VERSION

    def __init__(self, events: Sequence[PrivateEvent]) -> None:
        if not events:
            raise OracleError("world requires at least one event")
        self.events = tuple(events)
        history_ids = {item.history_id for item in self.events}
        if len(history_ids) != 1:
            raise OracleError("one oracle instance cannot mix histories")
        observed = [item.observed_at for item in self.events]
        if observed != sorted(observed):
            raise OracleError("events must be monotonic by observed_at")
        by_key: dict[tuple[str, str], list[PrivateEvent]] = {}
        for event in self.events:
            by_key.setdefault((event.entity, event.relation), []).append(event)
        for values in by_key.values():
            effective = [item.effective_at for item in values]
            if effective != sorted(effective):
                raise OracleError("first-round effective time must be monotonic")
            if len(effective) != len(set(effective)):
                raise OracleError("single-valued updates cannot share effective_at")
        self._by_key = {key: tuple(value) for key, value in by_key.items()}

    def intervals(self, entity: str, relation: str) -> tuple[ValidityInterval, ...]:
        values = self._by_key.get((entity, relation), ())
        return tuple(
            ValidityInterval(
                event_id=event.event_id,
                entity=entity,
                relation=relation,
                value=event.value,
                valid_from=event.effective_at,
                valid_to=(
                    values[index + 1].effective_at if index + 1 < len(values) else None
                ),
                observed_at=event.observed_at,
            )
            for index, event in enumerate(values)
        )

    def _at(self, entity: str, relation: str, time: int) -> ValidityInterval | None:
        selected = None
        for event in self._by_key.get((entity, relation), ()):
            if event.effective_at <= time:
                selected = event
            else:
                break
        if selected is None:
            return None
        values = self._by_key[(entity, relation)]
        index = values.index(selected)
        next_event = values[index + 1] if index + 1 < len(values) else None
        return ValidityInterval(
            selected.event_id,
            entity,
            relation,
            selected.value,
            selected.effective_at,
            next_event.effective_at if next_event else None,
            selected.observed_at,
        )

    def _support_for_point(
        self, entity: str, relation: str, time: int
    ) -> tuple[tuple[str, ...], int]:
        values = self._by_key.get((entity, relation), ())
        selected_index = -1
        for index, event in enumerate(values):
            if event.effective_at <= time:
                selected_index = index
            else:
                break
        if selected_index < 0:
            return (), max((item.observed_at for item in values), default=0)
        selected = values[selected_index]
        following = (
            values[selected_index + 1] if selected_index + 1 < len(values) else None
        )
        support = (selected.event_id,) + ((following.event_id,) if following else ())
        available = (
            following.observed_at
            if following
            else max(item.observed_at for item in self.events)
        )
        return support, available

    def solve(self, query: DynamicQueryPrivate) -> OracleAnswer:
        if query.history_id != self.events[0].history_id:
            raise OracleError("query/history mismatch")
        if query.query_kind in {"state_at", "current_state"}:
            if query.query_kind == "current_state":
                values = self._by_key.get((query.entity, query.relation), ())
                if not values:
                    return OracleAnswer(query.query_id, ("unknown",), False, (), 0)
                point = values[-1].effective_at
                interval = self._at(query.entity, query.relation, point)
                support = (values[-1].event_id,)
                available = max(item.observed_at for item in self.events)
            else:
                point = int(query.query_time)
                interval = self._at(query.entity, query.relation, point)
                support, available = self._support_for_point(
                    query.entity, query.relation, point
                )
            if interval is None:
                return OracleAnswer(query.query_id, ("unknown",), False, (), available)
            return OracleAnswer(
                query.query_id, (interval.value,), True, support, available
            )

        if query.query_kind == "start_time":
            match = next(
                (
                    item
                    for item in self._by_key.get((query.entity, query.relation), ())
                    if item.value == query.target_value
                ),
                None,
            )
            if match is None:
                return OracleAnswer(query.query_id, ("unknown",), False, (), 0)
            return OracleAnswer(
                query.query_id,
                (str(match.effective_at),),
                True,
                (match.event_id,),
                match.observed_at,
            )

        if query.query_kind == "temporal_join":
            point = int(query.query_time)
            left = self._at(query.entity, query.relation, point)
            if left is None:
                return OracleAnswer(query.query_id, ("unknown",), False, (), 0)
            right = self._at(left.value, str(query.join_relation), point)
            if right is None:
                return OracleAnswer(query.query_id, ("unknown",), False, (), 0)
            left_support, left_available = self._support_for_point(
                query.entity, query.relation, point
            )
            right_support, right_available = self._support_for_point(
                left.value, str(query.join_relation), point
            )
            return OracleAnswer(
                query.query_id,
                (right.value,),
                True,
                tuple(dict.fromkeys(left_support + right_support)),
                max(left_available, right_available),
            )
        raise OracleError(f"unsupported query kind: {query.query_kind}")


def solve_world(
    events: Iterable[PrivateEvent], query: DynamicQueryPrivate
) -> OracleAnswer:
    return WorldOracle(tuple(events)).solve(query)


__all__ = [
    "OracleAnswer",
    "OracleError",
    "ValidityInterval",
    "WorldOracle",
    "solve_world",
]
