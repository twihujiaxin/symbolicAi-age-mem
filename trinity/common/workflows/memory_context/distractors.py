"""Deterministic Stage-2 distractor selection for M8 experiments."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence


FIXED_DISTRACTOR_MESSAGES: tuple[str, ...] = (
    "What's the weather like today?",
    "Can you recommend a good recipe for chocolate cake?",
    "I'm thinking about learning a new programming language.",
    "Do you know any interesting facts about ancient civilizations?",
    "What are your thoughts on modern art movements?",
)
DISTRACTOR_SOURCES = frozenset({"fixed", "task", "provider"})


class DistractorContractError(ValueError):
    """Raised when an experiment does not define a reproducible distractor set."""


def _normalize_messages(
    values: Iterable[object],
    *,
    source: str,
) -> List[str]:
    messages: List[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise DistractorContractError(
                f"{source} distractor {index} must be a non-empty string"
            )
        messages.append(value.strip())
    return messages


def resolve_stage2_distractors(
    *,
    source: str,
    count: int,
    task_messages: Optional[Sequence[object]] = None,
    provider_generate: Optional[Callable[[int], Sequence[object]]] = None,
) -> List[str]:
    """Resolve exactly ``count`` messages without silently changing sources.

    ``fixed`` and ``task`` never invoke the provider.  ``provider`` is retained
    only for the E2 upstream-compatibility arm and must be passed explicitly.
    """

    if source not in DISTRACTOR_SOURCES:
        allowed = ", ".join(sorted(DISTRACTOR_SOURCES))
        raise DistractorContractError(
            f"stage2_distractor_source must be one of: {allowed}"
        )
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise DistractorContractError(
            "stage2_distractor_messages must be a non-negative integer"
        )
    if count == 0:
        return []

    if source == "fixed":
        messages = list(FIXED_DISTRACTOR_MESSAGES)
    elif source == "task":
        if task_messages is None:
            raise DistractorContractError(
                "task distractor source requires raw_task.distractor_messages"
            )
        messages = _normalize_messages(task_messages, source="task")
    else:
        if provider_generate is None:
            raise DistractorContractError(
                "provider distractor source requires provider_generate"
            )
        messages = _normalize_messages(
            provider_generate(count),
            source="provider",
        )

    if len(messages) < count:
        raise DistractorContractError(
            f"{source} distractor source returned {len(messages)} messages; "
            f"{count} required"
        )
    return messages[:count]


__all__ = [
    "DISTRACTOR_SOURCES",
    "FIXED_DISTRACTOR_MESSAGES",
    "DistractorContractError",
    "resolve_stage2_distractors",
]
