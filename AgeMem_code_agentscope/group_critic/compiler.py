"""Deterministic compilation of validated milestone DAGs to positive DFAs."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..memory_oracle.models import AutomatonSpec, AutomatonTransition
from .models import (
    POSITIVE_MILESTONE_APS,
    CriticGroupInput,
    CriticOutput,
    CriticValidationReport,
)
from .validator import (
    DEFAULT_AUTOMATON_STATE_CAP,
    reachable_progress_masks,
    validate_critic_output,
)


class CriticCompilationError(ValueError):
    """Raised when unvalidated or non-compilable critic output is supplied."""


def _topological_milestone_ids(output: CriticOutput) -> Tuple[str, ...]:
    milestone_ids = {item.milestone_id for item in output.milestones}
    successors: Dict[str, Set[str]] = {item: set() for item in milestone_ids}
    indegree = {item: 0 for item in milestone_ids}
    for dependency in output.dependencies:
        prerequisite = dependency.prerequisite_id
        dependent = dependency.dependent_id
        if prerequisite not in milestone_ids or dependent not in milestone_ids:
            raise CriticCompilationError("dependency references an unknown milestone")
        if dependent not in successors[prerequisite]:
            successors[prerequisite].add(dependent)
            indegree[dependent] += 1
    ready = sorted(item for item, degree in indegree.items() if degree == 0)
    ordered: List[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for dependent in sorted(successors[current]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    if len(ordered) != len(milestone_ids):
        raise CriticCompilationError("milestone dependency graph is cyclic")
    return tuple(ordered)


def _state_name(mask: int, *, width: int) -> str:
    return f"q_m{mask:0{width}x}"


def compile_critic_output(
    group_input: CriticGroupInput,
    output: CriticOutput,
    report: Optional[CriticValidationReport] = None,
    *,
    state_cap: int = DEFAULT_AUTOMATON_STATE_CAP,
) -> AutomatonSpec:
    """Compile a valid DAG into a deterministic downward-closed bitset DFA.

    Bad-behavior records and counterfactual suggestions are intentionally not
    compiled: M7 keeps them audit-only and does not introduce a negative
    automaton or counterfactual reward.
    """

    # Keep the reward-sign and outcome-evidence boundary local to the compiler
    # as well as the validator.  This protects the executable boundary even if
    # a caller supplies a Pydantic object created through model_construct or a
    # future validator regression weakens the ordinary schema path.
    positive_aps = set(POSITIVE_MILESTONE_APS)
    successful_action_keys = {
        action.evidence.action_key
        for rollout in group_input.rollouts
        if rollout.terminal_outcome == "success"
        for action in rollout.actions
    }
    for milestone in output.milestones:
        if milestone.proposition not in positive_aps:
            raise CriticCompilationError(
                "reward milestone proposition is outside the positive AP allowlist"
            )
        if not any(
            reference.action_key in successful_action_keys
            for reference in milestone.evidence_steps
        ):
            raise CriticCompilationError(
                "reward milestone lacks evidence from a successful rollout"
            )

    fresh_report = validate_critic_output(
        group_input,
        output,
        state_cap=state_cap,
    )
    if report is not None and report != fresh_report:
        raise CriticCompilationError(
            "supplied validation report does not match fail-closed revalidation"
        )
    current = fresh_report
    if current.input_digest != group_input.digest:
        raise CriticCompilationError("validation report input digest mismatch")
    if current.output_digest != output.digest:
        raise CriticCompilationError("validation report output digest mismatch")
    if not current.valid or not current.automaton_compilable:
        codes = ", ".join(
            item.code for item in current.issues if item.severity == "error"
        )
        raise CriticCompilationError(
            f"critic output is not automaton-compilable: {codes or 'invalid'}"
        )

    milestone_ids = _topological_milestone_ids(output)
    dependencies = tuple(
        (item.prerequisite_id, item.dependent_id) for item in output.dependencies
    )
    masks = reachable_progress_masks(
        milestone_ids,
        dependencies,
        state_cap=state_cap,
    )
    if masks is None or not masks:
        raise CriticCompilationError(
            "validated DFA exceeds cap or has no accepting path"
        )
    if len(masks) != current.reachable_progress_state_count:
        raise CriticCompilationError("validation state count does not match compiler")

    milestone_by_id = {item.milestone_id: item for item in output.milestones}
    bit_index = {
        milestone_id: index for index, milestone_id in enumerate(milestone_ids)
    }
    prerequisite_masks: Dict[str, int] = defaultdict(int)
    for prerequisite, dependent in dependencies:
        prerequisite_masks[dependent] |= 1 << bit_index[prerequisite]

    ordered_masks = tuple(sorted(masks, key=lambda value: (value.bit_count(), value)))
    mask_set = set(ordered_masks)
    width = max(1, (len(milestone_ids) + 3) // 4)
    state_names = tuple(_state_name(mask, width=width) for mask in ordered_masks)
    full_mask = (1 << len(milestone_ids)) - 1

    transitions: List[AutomatonTransition] = []
    priority = 0
    for mask in ordered_masks:
        if mask == full_mask:
            continue
        source = _state_name(mask, width=width)
        for milestone_id in milestone_ids:
            bit = 1 << bit_index[milestone_id]
            if mask & bit or prerequisite_masks[milestone_id] & ~mask:
                continue
            target_mask = mask | bit
            if target_mask not in mask_set:
                raise CriticCompilationError(
                    "compiler reached undeclared progress state"
                )
            transitions.append(
                AutomatonTransition(
                    edge_id=f"progress_{source}_{milestone_id}",
                    proposition=milestone_by_id[milestone_id].proposition,
                    source_states=(source,),
                    target_state=_state_name(target_mask, width=width),
                    priority=priority,
                    progressive=True,
                )
            )
            priority += 1

    if not transitions:
        raise CriticCompilationError("compiled DFA has no transitions")
    states = (*state_names, "q_reject", "q_timeout")
    if len(states) > state_cap:
        raise CriticCompilationError("compiled automaton exceeds state cap")
    propositions = tuple(
        milestone_by_id[milestone_id].proposition for milestone_id in milestone_ids
    )
    return AutomatonSpec(
        name=f"m7-group-critic-positive-{output.digest[:16]}-v1",
        states=states,
        initial_state=_state_name(0, width=width),
        accepting_states=(_state_name(full_mask, width=width),),
        rejecting_states=("q_reject",),
        timeout_state="q_timeout",
        transitions=tuple(transitions),
        source_milestones=propositions,
    )


compile_validated_critic_output = compile_critic_output


__all__ = [
    "CriticCompilationError",
    "compile_critic_output",
    "compile_validated_critic_output",
]
