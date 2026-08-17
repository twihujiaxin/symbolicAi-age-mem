"""Focused validator/compiler/replay tests for the M7 positive automaton."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from AgeMem_code_agentscope.group_critic import (
    CriticCompilationError,
    CriticValidationReport,
    CriticOutput,
    EvidenceStepRef,
    Milestone,
    MilestoneDependency,
    MockGroupCritic,
    compile_critic_output,
    select_critic_automaton,
    validate_critic_output,
)
from AgeMem_code_agentscope.memory_oracle import (
    DFARunner,
    OracleAPEvent,
    hand_authored_memory_dfa,
)

try:
    from .m7_group_critic_schema_test import make_group
except ImportError:  # unittest discovery imports modules without package context
    from m7_group_critic_schema_test import make_group


def _mock_output():
    group = make_group()
    invocation = MockGroupCritic().critique(group)
    if invocation.output is None:
        raise AssertionError(invocation.error)
    return group, invocation, invocation.output


def _event(timestep: int, proposition: str) -> OracleAPEvent:
    return OracleAPEvent(
        task_id="task-1",
        rollout_id="success",
        seed=0,
        timestep=timestep,
        stage=min(timestep + 1, 3),
        propositions=(proposition,),
    )


class M7GroupCriticAutomatonTest(unittest.TestCase):
    def test_chain_compiles_deterministically_and_matches_hand_acceptance(self) -> None:
        group, _, output = _mock_output()
        report = validate_critic_output(group, output)
        first = compile_critic_output(group, output, report)
        second = compile_critic_output(group, output)
        self.assertEqual(first, second)
        self.assertNotIn(first.initial_state, first.accepting_states)
        self.assertEqual(len(first.states), 7)
        self.assertEqual(
            first.source_milestones,
            (
                "stored_supporting_fact",
                "supporting_coverage_complete",
                "retrieved_supporting_fact",
                "answered_correctly",
            ),
        )

        compiled_runner = DFARunner(first, max_steps=12)
        hand_runner = DFARunner(hand_authored_memory_dfa(), max_steps=12)
        compiled = hand = None
        for timestep, proposition in enumerate(first.source_milestones):
            done = timestep == len(first.source_milestones) - 1
            compiled = compiled_runner.step(_event(timestep, proposition), done=done)
            hand = hand_runner.step(_event(timestep, proposition), done=done)
        assert compiled is not None and hand is not None
        self.assertEqual(compiled.status, "accepted")
        self.assertEqual(hand.status, "accepted")

    def test_repeated_and_looping_aps_cannot_farm_progress_reward(self) -> None:
        group, _, output = _mock_output()
        spec = compile_critic_output(group, output)
        runner = DFARunner(spec, max_steps=20)
        first = runner.step(_event(0, "stored_supporting_fact"))
        duplicate = runner.step(_event(1, "stored_supporting_fact"))
        loop = runner.step(_event(2, "retrieved_supporting_fact"))
        self.assertEqual(len(first.new_progress_edges), 1)
        self.assertEqual(duplicate.new_progress_edges, ())
        self.assertEqual(loop.new_progress_edges, ())

        total_new = len(first.new_progress_edges)
        result = None
        for timestep, proposition in enumerate(
            (
                "supporting_coverage_complete",
                "retrieved_supporting_fact",
                "answered_correctly",
            ),
            start=3,
        ):
            result = runner.step(
                _event(timestep, proposition),
                done=proposition == "answered_correctly",
            )
            total_new += len(result.new_progress_edges)
        assert result is not None
        self.assertEqual(result.status, "accepted")
        self.assertEqual(total_new, len(spec.source_milestones))

    def test_missing_required_milestone_rejects_at_terminal(self) -> None:
        group, _, output = _mock_output()
        runner = DFARunner(compile_critic_output(group, output), max_steps=10)
        runner.step(_event(0, "stored_supporting_fact"))
        runner.step(_event(1, "supporting_coverage_complete"))
        final = runner.step(_event(2, "answered_correctly"), done=True)
        self.assertEqual(final.status, "rejected")

    def test_validator_rejects_unknown_action_and_ap_evidence_mismatch(self) -> None:
        group, _, output = _mock_output()
        milestone = output.milestones[0]
        source = milestone.evidence_steps[0]
        unknown = source.model_copy(update={"action_id": "unknown-action"})
        invalid_output = output.model_copy(
            update={
                "milestones": (
                    milestone.model_copy(update={"evidence_steps": (unknown,)}),
                    *output.milestones[1:],
                )
            }
        )
        report = validate_critic_output(group, invalid_output)
        self.assertFalse(report.valid)
        self.assertIn("milestone_unknown_action", {item.code for item in report.issues})

        mismatch = source.model_copy(update={"ap_evidence_ids": ("fabricated",)})
        mismatch_output = output.model_copy(
            update={
                "milestones": (
                    milestone.model_copy(update={"evidence_steps": (mismatch,)}),
                    *output.milestones[1:],
                )
            }
        )
        mismatch_report = validate_critic_output(group, mismatch_output)
        self.assertFalse(mismatch_report.valid)
        self.assertIn(
            "milestone_ap_evidence_mismatch",
            {item.code for item in mismatch_report.issues},
        )

    def test_bad_behavior_ap_cannot_become_a_reward_milestone(self) -> None:
        group, _, output = _mock_output()
        positive = output.milestones[0]
        success_rollout = next(
            rollout
            for rollout in group.rollouts
            if rollout.terminal_outcome == "success"
        )
        source_action = success_rollout.actions[0]
        poisoned = positive.model_copy(
            update={
                "proposition": "stored_irrelevant_fact",
                "evidence_steps": (
                    source_action.evidence.model_copy(
                        update={
                            "ap_evidence_ids": source_action.evidence.ap_evidence_ids
                        }
                    ),
                ),
            }
        )
        poisoned_output = output.model_copy(
            update={"milestones": (poisoned, *output.milestones[1:])}
        )

        report = validate_critic_output(group, poisoned_output)
        self.assertFalse(report.valid)
        self.assertIn(
            "milestone_non_positive_ap",
            {item.code for item in report.issues},
        )
        with self.assertRaisesRegex(CriticCompilationError, "positive AP allowlist"):
            compile_critic_output(group, poisoned_output)

    def test_milestone_needs_supported_evidence_from_a_success_rollout(self) -> None:
        group, _, output = _mock_output()
        milestone = output.milestones[0]
        failure_reference = next(
            reference
            for reference in milestone.evidence_steps
            if reference.rollout_id.startswith("failure-")
        )
        failure_only = milestone.model_copy(
            update={"evidence_steps": (failure_reference,)}
        )
        failure_only_output = output.model_copy(
            update={"milestones": (failure_only, *output.milestones[1:])}
        )

        report = validate_critic_output(group, failure_only_output)
        self.assertFalse(report.valid)
        self.assertIn(
            "milestone_missing_success_evidence",
            {item.code for item in report.issues},
        )
        with self.assertRaisesRegex(
            CriticCompilationError,
            "evidence from a successful rollout",
        ):
            compile_critic_output(group, failure_only_output)

    def test_validator_rejects_cycle_and_state_cap_then_falls_back(self) -> None:
        group, invocation, output = _mock_output()
        cyclic = output.model_copy(
            update={
                "dependencies": (
                    *output.dependencies,
                    MilestoneDependency(
                        prerequisite_id="m_answer_correct",
                        dependent_id="m_store_support",
                    ),
                )
            }
        )
        cycle_report = validate_critic_output(group, cyclic)
        self.assertFalse(cycle_report.valid)
        self.assertIn("dependency_cycle", {item.code for item in cycle_report.issues})
        with self.assertRaises(CriticCompilationError):
            compile_critic_output(group, cyclic, cycle_report)

        cap_report = validate_critic_output(group, output, state_cap=6)
        self.assertFalse(cap_report.valid)
        self.assertIn("state_cap_exceeded", {item.code for item in cap_report.issues})
        fallback = select_critic_automaton(group, invocation, state_cap=6)
        self.assertEqual(fallback.selected_source, "hand_authored")
        assert fallback.fallback_reason is not None
        self.assertIn("state_cap_exceeded", fallback.fallback_reason)

    def test_compiler_revalidates_instead_of_trusting_a_stale_report(self) -> None:
        group, _, output = _mock_output()
        report = validate_critic_output(group, output)
        stale = report.model_copy(
            update={
                "reachable_progress_state_count": (
                    report.reachable_progress_state_count + 1
                )
            }
        )
        self.assertIsInstance(stale, CriticValidationReport)
        with self.assertRaisesRegex(
            CriticCompilationError,
            "does not match fail-closed revalidation",
        ):
            compile_critic_output(group, output, stale)

    def test_validator_rejects_initial_accepting_empty_output(self) -> None:
        group = make_group()
        empty = CriticOutput(task_id=group.task_id, group_id=group.group_id)
        report = validate_critic_output(group, empty)
        self.assertFalse(report.valid)
        self.assertIn("initial_state_accepting", {item.code for item in report.issues})

    def test_milestone_domain_is_a_strict_ap_literal(self) -> None:
        _, _, output = _mock_output()
        reference: EvidenceStepRef = output.milestones[0].evidence_steps[0]
        with self.assertRaises(ValidationError):
            Milestone(
                milestone_id="invalid",
                proposition="bare_ADD",
                description="A naked tool call must not become a milestone.",
                evidence_steps=(reference,),
                confidence=1.0,
            )
        with self.assertRaises(ValidationError):
            Milestone(
                milestone_id="negative-ap",
                proposition="stored_irrelevant_fact",
                description="Audit-only APs cannot become reward milestones.",
                evidence_steps=(reference,),
                confidence=1.0,
            )


if __name__ == "__main__":
    unittest.main()
