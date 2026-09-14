"""Immutable snapshot branching with a frozen lexical retriever and reader."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .environment import MemorySnapshot
from .schema import QueryRecord
from .token_budget import BudgetError, TokenAccounting, canonical_memory_payload


ANSWER_SYSTEM = "Answer using the supplied evidence. Return only <answer>...</answer>."


def _terms(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


class FrozenLexicalRetriever:
    """Deterministic BM25-like ranking over active memory payloads only."""

    version = "agemem-frozen-lexical-v1"

    def rank(self, question: str, memories: Sequence[dict]) -> list[dict]:
        query = set(_terms(question))
        scored = []
        for memory in memories:
            text_terms = _terms(str(memory.get("content", "")))
            overlap = sum(1 + math.log1p(text_terms.count(term)) for term in query if term in text_terms)
            scored.append((overlap, str(memory.get("memory_id", "")), memory))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored if item[0] > 0]


@dataclass(frozen=True)
class QueryBranchResult:
    query_branch_id: str
    query_id: str
    snapshot_id: str
    reader_version: str
    answer_text: str
    prompt_tokens: int
    retrieved_memory_ids: tuple[str, ...]
    retrieved_payloads: tuple[str, ...]
    tail_messages: tuple[dict[str, str], ...]
    actor_loss_masked: bool = True


class IndependentQueryRunner:
    def __init__(
        self,
        *,
        accounting: TokenAccounting,
        reader: Callable[[Sequence[dict[str, str]]], str],
        reader_version: str,
        context_total_tokens: int,
        answer_max_new_tokens: int,
        retrieval_payload_tokens: int,
        retrieval_top_k: int,
        retriever: FrozenLexicalRetriever | None = None,
    ) -> None:
        self.accounting = accounting
        self.reader = reader
        self.reader_version = reader_version
        self.context_total_tokens = context_total_tokens
        self.answer_max_new_tokens = answer_max_new_tokens
        self.retrieval_payload_tokens = retrieval_payload_tokens
        self.retrieval_top_k = retrieval_top_k
        self.retriever = retriever or FrozenLexicalRetriever()

    def run(self, snapshot: MemorySnapshot, query: QueryRecord, branch_index: int) -> QueryBranchResult:
        ranked = self.retriever.rank(query.question, snapshot.active_memories)
        base = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"QUESTION\n{query.question}"},
        ]
        selected: list[dict] = []
        payloads: list[str] = []
        payload_tokens = 0
        for memory in ranked:
            payload = canonical_memory_payload(memory)
            cost = self.accounting.count_text(payload)
            if len(selected) >= self.retrieval_top_k:
                break
            if payload_tokens + cost > self.retrieval_payload_tokens:
                continue
            proposed_payloads = payloads + [payload]
            proposed_messages = base + [{
                "role": "user",
                "content": "RETRIEVED MEMORY\n" + "\n".join(proposed_payloads),
            }]
            try:
                self.accounting.enforce_context(
                    proposed_messages, max_new_tokens=self.answer_max_new_tokens,
                    context_total_tokens=self.context_total_tokens,
                )
            except BudgetError:
                continue
            selected.append(memory)
            payloads.append(payload)
            payload_tokens += cost

        retrieval_message = {
            "role": "user",
            "content": "RETRIEVED MEMORY\n" + ("\n".join(payloads) if payloads else "(none)"),
        }
        # Tail is admitted newest-first by whole message, then restored to order.
        tail: list[dict[str, str]] = []
        for message in reversed(snapshot.context_tail):
            proposed_tail = [dict(message)] + tail
            proposed = base + proposed_tail + [retrieval_message]
            try:
                self.accounting.enforce_context(
                    proposed, max_new_tokens=self.answer_max_new_tokens,
                    context_total_tokens=self.context_total_tokens,
                )
            except BudgetError:
                continue
            tail = proposed_tail

        messages = base + tail + [retrieval_message]
        prompt_tokens = self.accounting.enforce_context(
            messages, max_new_tokens=self.answer_max_new_tokens,
            context_total_tokens=self.context_total_tokens,
        )
        before = (snapshot.memory_sha256, snapshot.tail_sha256)
        answer = self.reader(tuple(messages))
        after = (snapshot.memory_sha256, snapshot.tail_sha256)
        if before != after:
            raise RuntimeError("reader branch modified its immutable parent snapshot")
        return QueryBranchResult(
            query_branch_id=f"{snapshot.snapshot_id}:branch:{branch_index}",
            query_id=query.query_id, snapshot_id=snapshot.snapshot_id,
            reader_version=self.reader_version, answer_text=answer,
            prompt_tokens=prompt_tokens,
            retrieved_memory_ids=tuple(str(item["memory_id"]) for item in selected),
            retrieved_payloads=tuple(payloads), tail_messages=tuple(tail),
        )


__all__ = ["FrozenLexicalRetriever", "IndependentQueryRunner", "QueryBranchResult"]
