"""Fail-closed validation for M7 structured critic outputs."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from .models import (
    ActionAPTrace,
    CriticGroupInput,
    CriticOutput,
    CriticValidationIssue,
    CriticValidationReport,
    EvidenceStepRef,
    POSITIVE_MILESTONE_APS,
)


DEFAULT_AUTOMATON_STATE_CAP = 256
_BAD_BEHAVIOR_AP = {
    "stored_irrelevant_fact": "stored_irrelevant_fact",
    "retrieved_irrelevant_fact": "retrieved_irrelevant_fact",
    "deleted_supporting_fact": "deleted_supporting_fact",
}


def _issue(
    code: str,
    message: str,
    *,
    severity: str = "error",
    milestone_id: Optional[str] = None,
    action_id: Optional[str] = None,
) -> CriticValidationIssue:
    return CriticValidationIssue(
        severity=severity,
        code=code,
        message=message,
        milestone_id=milestone_id,
        action_id=action_id,
    )


def _action_index(
    group_input: CriticGroupInput,
) -> Dict[Tuple[str, str, int, int, str, int, int], ActionAPTrace]:
    return {
        action.evidence.action_key: action
        for rollout in group_input.rollouts
        for action in rollout.actions
    }


def _validate_generic_evidence(
    reference: EvidenceStepRef,
    action_index: Mapping[Tuple[str, str, int, int, str, int, int], ActionAPTrace],
    *,
    code_prefix: str,
) -> List[CriticValidationIssue]:
    action = action_index.get(reference.action_key)
    if action is None:
        return [
            _issue(
                f"{code_prefix}_unknown_action",
                "evidence reference does not match a complete group action coordinate",
                action_id=reference.action_id,
            )
        ]
    known_ids = {
        evidence_id
        for values in action.atomic_proposition_evidence.values()
        for evidence_id in values
    }
    unknown_ids = set(reference.ap_evidence_ids) - known_ids
    if unknown_ids:
        return [
            _issue(
                f"{code_prefix}_unknown_ap_evidence",
                "evidence reference contains AP evidence IDs absent from the action",
                action_id=reference.action_id,
            )
        ]
    return []


def _topological_order(
    milestone_ids: Tuple[str, ...], dependencies: Iterable[Tuple[str, str]]
) -> Optional[Tuple[str, ...]]:
    successors: Dict[str, Set[str]] = {item: set() for item in milestone_ids}
    indegree = {item: 0 for item in milestone_ids}
    for prerequisite, dependent in dependencies:
        if prerequisite not in successors or dependent not in successors:
            continue
        if dependent not in successors[prerequisite]:
            successors[prerequisite].add(dependent)
            indegree[dependent] += 1
    ready = sorted(item for item, degree in indegree.items() if degree == 0)
    ordered: List[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for child in sorted(successors[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    return tuple(ordered) if len(ordered) == len(milestone_ids) else None


def reachable_progress_masks(
    milestone_ids: Tuple[str, ...],
    dependencies: Iterable[Tuple[str, str]],
    *,
    state_cap: int,
) -> Optional[Tuple[int, ...]]:
    """Enumerate reachable downward-closed milestone bitsets deterministically.

    ``None`` indicates that the total compiled DFA size (progress states plus
    reject and timeout) exceeded ``state_cap``.
    """

    if state_cap < 4:
        return None
    index = {milestone_id: offset for offset, milestone_id in enumerate(milestone_ids)}
    prerequisites: Dict[str, int] = defaultdict(int)
    for prerequisite, dependent in dependencies:
        if prerequisite in index and dependent in index:
            prerequisites[dependent] |= 1 << index[prerequisite]

    full_mask = (1 << len(milestone_ids)) - 1
    seen = {0}
    queue = deque([0])
    while queue:
        mask = queue.popleft()
        for milestone_id in milestone_ids:
            bit = 1 << index[milestone_id]
            if mask & bit or prerequisites[milestone_id] & ~mask:
                continue
            target = mask | bit
            if target not in seen:
                seen.add(target)
                if len(seen) + 2 > state_cap:
                    return None
                queue.append(target)
    if full_mask not in seen:
        return ()
    return tuple(sorted(seen))


def validate_critic_output(
    group_input: CriticGroupInput,
    output: CriticOutput,
    *,
    state_cap: int = DEFAULT_AUTOMATON_STATE_CAP,
) -> CriticValidationReport:
    """Validate all provenance and graph constraints without compiling rewards."""

    if isinstance(state_cap, bool) or not isinstance(state_cap, int) or state_cap < 4:
        raise ValueError("state_cap must be an integer of at least 4")

    issues: List[CriticValidationIssue] = []
    if output.task_id != group_input.task_id or output.group_id != group_input.group_id:
        issues.append(
            _issue(
                "group_identity_mismatch",
                "critic output task/group identity does not match its complete input",
            )
        )

    action_index = _action_index(group_input)
    successful_rollout_ids = {
        rollout.rollout_id
        for rollout in group_input.rollouts
        if rollout.terminal_outcome == "success"
    }
    evidence_count = 0
    milestone_ids_seen = [item.milestone_id for item in output.milestones]
    proposition_values = [item.proposition for item in output.milestones]
    if len(milestone_ids_seen) != len(set(milestone_ids_seen)):
        issues.append(_issue("duplicate_milestone_id", "milestone IDs must be unique"))
    if len(proposition_values) != len(set(proposition_values)):
        issues.append(
            _issue(
                "duplicate_milestone_proposition",
                "milestone propositions must be unique",
            )
        )
    for milestone in output.milestones:
        evidence_count += len(milestone.evidence_steps)
        if milestone.proposition not in POSITIVE_MILESTONE_APS:
            issues.append(
                _issue(
                    "milestone_non_positive_ap",
                    "reward milestone proposition is not in the positive AP allowlist",
                    milestone_id=milestone.milestone_id,
                )
            )
        has_success_evidence = False
        for reference in milestone.evidence_steps:
            action = action_index.get(reference.action_key)
            if action is None:
                issues.append(
                    _issue(
                        "milestone_unknown_action",
                        "milestone evidence does not match a complete action coordinate",
                        milestone_id=milestone.milestone_id,
                        action_id=reference.action_id,
                    )
                )
                continue
            if milestone.proposition not in action.propositions:
                issues.append(
                    _issue(
                        "milestone_ap_not_supported",
                        "referenced action does not contain the milestone proposition",
                        milestone_id=milestone.milestone_id,
                        action_id=reference.action_id,
                    )
                )
                continue
            expected_ids = action.atomic_proposition_evidence[milestone.proposition]
            if reference.ap_evidence_ids != expected_ids:
                issues.append(
                    _issue(
                        "milestone_ap_evidence_mismatch",
                        "milestone AP evidence IDs do not exactly match the source action",
                        milestone_id=milestone.milestone_id,
                        action_id=reference.action_id,
                    )
                )
                continue
            if reference.rollout_id in successful_rollout_ids:
                has_success_evidence = True
        if not has_success_evidence:
            issues.append(
                _issue(
                    "milestone_missing_success_evidence",
                    "reward milestone needs exact AP evidence from a successful rollout",
                    milestone_id=milestone.milestone_id,
                )
            )

    for behavior in output.bad_behaviors:
        evidence_count += len(behavior.evidence_steps)
        for reference in behavior.evidence_steps:
            issues.extend(
                _validate_generic_evidence(
                    reference,
                    action_index,
                    code_prefix="bad_behavior",
                )
            )
            expected_ap = _BAD_BEHAVIOR_AP.get(behavior.tag)
            action = action_index.get(reference.action_key)
            if (
                expected_ap is not None
                and action is not None
                and expected_ap not in action.propositions
            ):
                issues.append(
                    _issue(
                        "bad_behavior_ap_not_supported",
                        "bad-behavior tag is not supported by the referenced action AP",
                        action_id=reference.action_id,
                    )
                )

    for suggestion in output.counterfactual_suggestions:
        evidence_count += len(suggestion.evidence_steps)
        for reference in suggestion.evidence_steps:
            issues.extend(
                _validate_generic_evidence(
                    reference,
                    action_index,
                    code_prefix="counterfactual",
                )
            )

    milestone_ids = tuple(item.milestone_id for item in output.milestones)
    declared_ids = set(milestone_ids)
    dependency_pairs = tuple(
        (item.prerequisite_id, item.dependent_id) for item in output.dependencies
    )
    if len(dependency_pairs) != len(set(dependency_pairs)):
        issues.append(_issue("duplicate_dependency", "dependencies must be unique"))
    for prerequisite, dependent in dependency_pairs:
        if prerequisite not in declared_ids or dependent not in declared_ids:
            issues.append(
                _issue(
                    "dependency_unknown_milestone",
                    "dependency endpoint is not a declared milestone",
                )
            )
        if prerequisite == dependent:
            issues.append(
                _issue("dependency_self_loop", "milestone cannot depend on itself")
            )

    topological = _topological_order(milestone_ids, dependency_pairs)
    if milestone_ids and topological is None:
        issues.append(
            _issue("dependency_cycle", "milestone dependency graph is cyclic")
        )

    if group_input.all_failed:
        if output.milestones or output.dependencies:
            issues.append(
                _issue(
                    "all_failure_reward_content",
                    "all-failure groups may record counterfactuals but no reward milestones",
                )
            )
        if not output.counterfactual_suggestions:
            issues.append(
                _issue(
                    "all_failure_missing_counterfactual",
                    "all-failure group must record a non-reward counterfactual suggestion",
                )
            )
    elif not output.milestones:
        issues.append(
            _issue(
                "initial_state_accepting",
                "a non-all-failure output needs at least one milestone",
            )
        )
    elif output.counterfactual_suggestions:
        issues.append(
            _issue(
                "counterfactual_with_observed_success",
                "counterfactual suggestions are reserved for all-failure groups",
            )
        )

    progress_masks: Optional[Tuple[int, ...]] = ()
    graph_has_error = any(
        item.code
        in {
            "dependency_unknown_milestone",
            "dependency_self_loop",
            "dependency_cycle",
            "initial_state_accepting",
            "all_failure_reward_content",
        }
        and item.severity == "error"
        for item in issues
    )
    if milestone_ids and not graph_has_error:
        progress_masks = reachable_progress_masks(
            milestone_ids,
            dependency_pairs,
            state_cap=state_cap,
        )
        if progress_masks is None:
            issues.append(
                _issue(
                    "state_cap_exceeded",
                    f"compiled automaton would exceed state cap {state_cap}",
                )
            )
            progress_masks = ()
        elif not progress_masks:
            issues.append(
                _issue(
                    "accepting_state_unreachable",
                    "full milestone state is not reachable from the initial state",
                )
            )

    valid = not any(item.severity == "error" for item in issues)
    compilable = valid and bool(milestone_ids) and bool(progress_masks)
    return CriticValidationReport(
        input_digest=group_input.digest,
        output_digest=output.digest,
        valid=valid,
        automaton_compilable=compilable,
        evidence_reference_count=evidence_count,
        reachable_progress_state_count=len(progress_masks or ()),
        issues=tuple(issues),
    )


__all__ = [
    "DEFAULT_AUTOMATON_STATE_CAP",
    "reachable_progress_masks",
    "validate_critic_output",
]
