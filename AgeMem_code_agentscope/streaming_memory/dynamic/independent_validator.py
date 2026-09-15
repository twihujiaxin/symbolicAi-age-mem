"""Second-path interval validator for generated private labels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .schema import DynamicQueryPrivate, PrivateEvent


@dataclass(frozen=True)
class ValidationIssue:
    query_id: str
    code: str
    detail: str


def _interval_table(
    events: Sequence[PrivateEvent],
) -> dict[tuple[str, str], list[tuple[int, int | None, str, str, int]]]:
    grouped: dict[tuple[str, str], list[PrivateEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.entity, event.relation)].append(event)
    table = {}
    for key, values in grouped.items():
        values = sorted(values, key=lambda item: (item.effective_at, item.observed_at))
        table[key] = [
            (
                item.effective_at,
                values[index + 1].effective_at if index + 1 < len(values) else None,
                item.value,
                item.event_id,
                item.observed_at,
            )
            for index, item in enumerate(values)
        ]
    return table


def validate_private_labels(
    events: Sequence[PrivateEvent], queries: Sequence[DynamicQueryPrivate]
) -> tuple[ValidationIssue, ...]:
    """Recompute labels by interval lookup rather than calling WorldOracle."""

    table = _interval_table(events)
    issues: list[ValidationIssue] = []
    all_events = {item.event_id: item for item in events}
    final_observed = max((item.observed_at for item in events), default=0)

    def at(entity: str, relation: str, time: int):
        return next(
            (
                row
                for row in table.get((entity, relation), ())
                if row[0] <= time and (row[1] is None or time < row[1])
            ),
            None,
        )

    for query in queries:
        expected: tuple[str, ...]
        if query.query_kind == "current_state":
            rows = table.get((query.entity, query.relation), ())
            expected = (rows[-1][2],) if rows else ("unknown",)
            expected_available = final_observed
        elif query.query_kind == "state_at":
            row = at(query.entity, query.relation, int(query.query_time))
            expected = (row[2],) if row else ("unknown",)
            later = [
                item.observed_at
                for item in events
                if item.entity == query.entity
                and item.relation == query.relation
                and item.effective_at > int(query.query_time)
            ]
            expected_available = min(later) if later else final_observed
        elif query.query_kind == "start_time":
            rows = table.get((query.entity, query.relation), ())
            row = next((item for item in rows if item[2] == query.target_value), None)
            expected = (str(row[0]),) if row else ("unknown",)
            expected_available = row[4] if row else final_observed
        else:
            left = at(query.entity, query.relation, int(query.query_time))
            right = (
                at(left[2], str(query.join_relation), int(query.query_time))
                if left
                else None
            )
            expected = (right[2],) if right else ("unknown",)

            def point_available(entity, relation):
                later = [
                    item.observed_at
                    for item in events
                    if item.entity == entity
                    and item.relation == relation
                    and item.effective_at > int(query.query_time)
                ]
                return min(later) if later else final_observed

            expected_available = max(
                point_available(query.entity, query.relation),
                point_available(left[2], str(query.join_relation))
                if left
                else final_observed,
            )
        if expected != query.answers:
            issues.append(
                ValidationIssue(
                    query.query_id,
                    "answer_mismatch",
                    f"{expected!r} != {query.answers!r}",
                )
            )
        if query.answer_available_at != expected_available:
            issues.append(
                ValidationIssue(
                    query.query_id,
                    "availability_mismatch",
                    f"{expected_available} != {query.answer_available_at}",
                )
            )
        for alternative in query.support_alternatives:
            if not alternative or any(item not in all_events for item in alternative):
                issues.append(
                    ValidationIssue(
                        query.query_id, "invalid_support", repr(alternative)
                    )
                )
    return tuple(issues)


__all__ = ["ValidationIssue", "validate_private_labels"]
