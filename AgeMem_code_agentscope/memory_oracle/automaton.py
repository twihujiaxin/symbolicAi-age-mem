"""Hand-authored positive memory DFA and deterministic finite-trace runner."""

from __future__ import annotations

from typing import List, Set

from .models import (
    AutomatonSpec,
    AutomatonTransition,
    DFAStepResult,
    DFAStatus,
    OracleAPEvent,
)


def hand_authored_memory_dfa() -> AutomatonSpec:
    """Return the M4 positive DFA described in PROJECT_HANDOFF.md.

    Coverage and retrieval can be emitted by the same M3 tool result. Global
    transition priorities therefore provide a deterministic within-step closure:
    q1 -> q2 on coverage, then q2 -> q3 on relevant retrieval.
    """

    active = ("q0", "q1", "q2", "q3")
    return AutomatonSpec(
        name="m4-memory-oracle-positive-v1",
        states=("q0", "q1", "q2", "q3", "q4", "q_reject", "q_timeout"),
        initial_state="q0",
        accepting_states=("q4",),
        rejecting_states=("q_reject",),
        timeout_state="q_timeout",
        transitions=(
            AutomatonTransition(
                edge_id="reject_deleted_support",
                proposition="deleted_supporting_fact",
                source_states=active,
                target_state="q_reject",
                priority=0,
                violation=True,
            ),
            AutomatonTransition(
                edge_id="progress_update_stale",
                proposition="updated_stale_fact",
                source_states=active,
                target_state=None,
                priority=5,
                progressive=True,
            ),
            AutomatonTransition(
                edge_id="progress_store_support",
                proposition="stored_supporting_fact",
                source_states=("q0",),
                target_state="q1",
                priority=10,
                progressive=True,
            ),
            AutomatonTransition(
                edge_id="progress_support_coverage",
                proposition="supporting_coverage_complete",
                source_states=("q1",),
                target_state="q2",
                priority=20,
                progressive=True,
            ),
            AutomatonTransition(
                edge_id="progress_retrieve_support",
                proposition="retrieved_supporting_fact",
                source_states=("q2",),
                target_state="q3",
                priority=30,
                progressive=True,
            ),
            AutomatonTransition(
                edge_id="progress_answer_correct",
                proposition="answered_correctly",
                source_states=("q3",),
                target_state="q4",
                priority=40,
                progressive=True,
            ),
            AutomatonTransition(
                edge_id="violation_store_irrelevant",
                proposition="stored_irrelevant_fact",
                source_states=active,
                target_state=None,
                priority=50,
                violation=True,
            ),
            AutomatonTransition(
                edge_id="violation_retrieve_irrelevant",
                proposition="retrieved_irrelevant_fact",
                source_states=active,
                target_state=None,
                priority=60,
                violation=True,
            ),
        ),
        source_milestones=(
            "stored_supporting_fact",
            "updated_stale_fact",
            "supporting_coverage_complete",
            "retrieved_supporting_fact",
            "answered_correctly",
        ),
    )


class DFARunner:
    """Deterministically replay AP events with once-only progress edges."""

    def __init__(self, spec: AutomatonSpec, *, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.spec = spec.model_copy(deep=True)
        self.max_steps = max_steps
        self.state = spec.initial_state
        self.status: DFAStatus = "running"
        self.steps_processed = 0
        self.visited_progress_edges: Set[str] = set()

    def reset(self) -> None:
        self.state = self.spec.initial_state
        self.status = "running"
        self.steps_processed = 0
        self.visited_progress_edges = set()

    def step(self, event: OracleAPEvent, *, done: bool = False) -> DFAStepResult:
        state_before = self.state
        if self.status != "running":
            return DFAStepResult(
                state_before=state_before,
                state_after=self.state,
                status=self.status,
            )

        fired: List[str] = []
        new_progress: List[str] = []
        repeated_progress: List[str] = []
        violations: List[str] = []
        propositions = set(event.propositions)
        for transition in sorted(self.spec.transitions, key=lambda item: item.priority):
            if transition.proposition not in propositions:
                continue
            if self.state not in transition.source_states:
                continue
            fired.append(transition.edge_id)
            if transition.progressive:
                if transition.edge_id in self.visited_progress_edges:
                    repeated_progress.append(transition.edge_id)
                else:
                    self.visited_progress_edges.add(transition.edge_id)
                    new_progress.append(transition.edge_id)
            if transition.violation:
                violations.append(transition.edge_id)
            if transition.target_state is not None:
                self.state = transition.target_state
            if self.state in self.spec.accepting_states:
                self.status = "accepted"
                break
            if self.state in self.spec.rejecting_states:
                self.status = "rejected"
                break

        self.steps_processed += 1
        if self.status == "running" and done:
            self.state = self.spec.rejecting_states[0]
            self.status = "rejected"
        elif self.status == "running" and self.steps_processed >= self.max_steps:
            self.state = self.spec.timeout_state
            self.status = "timed_out"

        return DFAStepResult(
            state_before=state_before,
            state_after=self.state,
            status=self.status,
            fired_edges=tuple(fired),
            new_progress_edges=tuple(new_progress),
            repeated_progress_edges=tuple(repeated_progress),
            violations=tuple(violations),
        )


__all__ = ["DFARunner", "hand_authored_memory_dfa"]
