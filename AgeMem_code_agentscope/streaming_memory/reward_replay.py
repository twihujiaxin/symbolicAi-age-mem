"""Terminal/Flat-state/DFA replay for static streaming snapshots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from AgeMem_code_agentscope.hotpotqa_benchmark.metrics import answer_f1

from .environment import MemorySnapshot
from .query_runner import QueryBranchResult
from .schema import QueryGold


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


class ExtractiveSemanticGrounder:
    """Conservative CPU control: source pointer AND source sentence in body.

    This gate is useful for structural tests and extractive controls. It is not
    presented as a validated free-paraphrase semantic judge.
    """

    version = "extractive_source_and_content_v1"

    @staticmethod
    def supports(memory: Mapping, gold_ref: Mapping, source_sentence: str) -> bool:
        refs = memory.get("source_refs") or []
        pointer_ok = any(
            str(ref.get("document_key")) == str(gold_ref["document_key"])
            and int(ref.get("sentence_index", -1)) == int(gold_ref["sentence_index"])
            for ref in refs
        )
        content = _normalize(str(memory.get("content", "")))
        sentence = _normalize(source_sentence)
        return pointer_ok and bool(sentence) and sentence in content


@dataclass(frozen=True)
class QuerySemanticScore:
    query_id: str
    terminal_f1: float
    retained_coverage: float
    exposed_coverage: float
    flat_state: float
    dfa_state: float


@dataclass(frozen=True)
class RolloutReward:
    terminal: float
    flat_state: float
    dfa: float
    semantic_mean: float
    per_query: tuple[QuerySemanticScore, ...]
    flat_dfa_equivalent: bool


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def score_snapshot(
    *,
    snapshot: MemorySnapshot,
    branch_results: Sequence[QueryBranchResult],
    gold_by_query: Mapping[str, QueryGold],
    source_sentences: Mapping[tuple[str, int, str], str],
    semantic_lambda: float,
    grounder: ExtractiveSemanticGrounder | None = None,
) -> RolloutReward:
    if not branch_results:
        raise ValueError("at least one query branch is required")
    grounder = grounder or ExtractiveSemanticGrounder()
    query_scores = []
    for branch in branch_results:
        gold = gold_by_query[branch.query_id]
        retained, exposed = 0, 0
        for ref in gold.support_refs:
            key = (ref.document_key, ref.sentence_index, ref.sentence_sha256)
            sentence = source_sentences.get(key)
            if sentence is None:
                raise ValueError("gold source registry is incomplete")
            valid_ids = {
                str(memory["memory_id"])
                for memory in snapshot.active_memories
                if grounder.supports(memory, ref.model_dump(mode="json"), sentence)
            }
            if valid_ids:
                retained += 1
            if valid_ids.intersection(branch.retrieved_memory_ids):
                # Retrieval credit only applies to the payload that really entered C.
                exposed += 1
        fact_count = len(gold.support_refs)
        c_m, c_e = retained / fact_count, exposed / fact_count
        semantic = 0.5 * c_m + 0.5 * c_e
        # Static DFA terminal values absent=0, retained=.5, exposed=1.
        dfa_terminal = semantic
        query_scores.append(QuerySemanticScore(
            query_id=branch.query_id,
            terminal_f1=answer_f1(_extract_answer(branch.answer_text), gold.answer),
            retained_coverage=c_m, exposed_coverage=c_e,
            flat_state=semantic, dfa_state=dfa_terminal,
        ))
    task = sum(item.terminal_f1 for item in query_scores) / len(query_scores)
    semantic = sum(item.flat_state for item in query_scores) / len(query_scores)
    dfa_semantic = sum(item.dfa_state for item in query_scores) / len(query_scores)
    flat_total = task + semantic_lambda * semantic
    dfa_total = task + semantic_lambda * dfa_semantic
    return RolloutReward(
        terminal=task, flat_state=flat_total, dfa=dfa_total, semantic_mean=semantic,
        per_query=tuple(query_scores), flat_dfa_equivalent=abs(flat_total - dfa_total) <= 1e-12,
    )


def state_transition(previous: str, event: str, equivalent_representation_exists: bool = False) -> str:
    """Small replayable state machine used by rollback regression tests."""

    if previous not in {"absent", "retained", "exposed_valid"}:
        raise ValueError("unknown state")
    if event == "valid_store":
        return "retained"
    if event == "valid_exposure" and previous in {"retained", "exposed_valid"}:
        return "exposed_valid"
    if event in {"delete", "invalid_update"} and not equivalent_representation_exists:
        return "absent"
    return previous


__all__ = [
    "ExtractiveSemanticGrounder", "QuerySemanticScore", "RolloutReward",
    "score_snapshot", "state_transition",
]
