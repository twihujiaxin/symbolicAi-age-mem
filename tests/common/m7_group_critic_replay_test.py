"""Focused tests for M7 AP/credit-only automaton replay."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from AgeMem_code_agentscope.action_schema import (
    ActionCreditRecord,
    ActionEvent,
    RewardBreakdownV2,
    TrajectoryStepV2,
)
from AgeMem_code_agentscope.group_critic.replay import (
    GROUP_AUTOMATON_REPLAY_SCHEMA_VERSION,
    GroupAutomatonReplay,
    GroupAutomatonReplayError,
    audit_reward_farming,
)
from AgeMem_code_agentscope.memory_oracle import (
    AutomatonSpec,
    AutomatonTransition,
    RewardConfig,
    hand_authored_memory_dfa,
)
from AgeMem_code_agentscope.memory_oracle.replay import default_reward_config_path
from AgeMem_code_agentscope.trajectory import MemorySnapshotItem


TASK_ID = "m7-replay-task"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _source_credit(
    action: ActionEvent,
    propositions=(),
    *,
    evidence=None,
) -> ActionCreditRecord:
    evidence = (
        {ap: (f"evidence:{action.action_id}:{ap}",) for ap in propositions}
        if evidence is None
        else evidence
    )
    breakdown = RewardBreakdownV2(
        env=0.0,
        milestone=0.0,
        violation=0.0,
        trend=0.0,
        format=0.0,
        cost=0.0,
        total=0.0,
        automaton_state_before="source-before",
        automaton_state_after="source-after",
        automaton_status="running",
        propositions=tuple(propositions),
    )
    return ActionCreditRecord(
        action_id=action.action_id,
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        timestep=action.timestep,
        atomic_propositions=tuple(propositions),
        atomic_proposition_evidence=evidence,
        dfa_spec_id="source-dfa-v1",
        dfa_state_before="source-before",
        dfa_state_after="source-after",
        reward_breakdown=breakdown,
        return_to_go=3.0,
        advantage=0.5,
        reward_version="source-reward-v1",
    )


def _row(
    rollout_id: str,
    timestep: int,
    action_type: str,
    propositions=(),
    *,
    action_id=None,
    evidence=None,
    done=False,
    env_reward=0.0,
):
    action_id = action_id or f"{rollout_id}:action:{timestep}"
    action = ActionEvent(
        action_id=action_id,
        task_id=TASK_ID,
        rollout_id=rollout_id,
        stage_id=min(3, timestep // 2 + 1),
        timestep=timestep,
        assistant_turn_id=timestep,
        action_index_in_turn=0,
        source="rule",
        action_type=action_type,
        action_text="{}",
        arguments={},
        result={
            "content": (),
            "is_interrupted": False,
            "tool_call_id": action_id,
        },
    )
    step = TrajectoryStepV2(
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        timestep=action.timestep,
        observation="offline replay fixture",
        actions=(action,),
        memory_before=(),
        memory_after=(),
        env_reward=env_reward,
        done=done,
    )
    return step, _source_credit(action, propositions, evidence=evidence)


def _clean_rows(rollout_id="clean"):
    return (
        _row(rollout_id, 0, "Add_memory", ("stored_supporting_fact",)),
        _row(
            rollout_id,
            1,
            "Retrieve_memory",
            ("retrieved_supporting_fact", "supporting_coverage_complete"),
        ),
        _row(
            rollout_id,
            2,
            "Answer",
            ("answered_correctly",),
            done=True,
            env_reward=1.0,
        ),
    )


def _farming_rows(rollout_id="farming"):
    return (
        _row(rollout_id, 0, "Add_memory", ("stored_supporting_fact",)),
        _row(
            rollout_id,
            1,
            "Add_memory",
            ("stored_supporting_fact",),
            action_id=f"{rollout_id}:duplicate-add",
        ),
        _row(
            rollout_id,
            2,
            "Retrieve_memory",
            ("retrieved_supporting_fact", "supporting_coverage_complete"),
            action_id=f"{rollout_id}:action:1",
        ),
        _row(
            rollout_id,
            3,
            "Retrieve_memory",
            ("retrieved_supporting_fact",),
            action_id=f"{rollout_id}:duplicate-retrieve",
        ),
        _row(
            rollout_id,
            4,
            "Retrieve_memory",
            ("retrieved_supporting_fact",),
            action_id=f"{rollout_id}:loop-1",
        ),
        _row(
            rollout_id,
            5,
            "Retrieve_memory",
            ("retrieved_supporting_fact",),
            action_id=f"{rollout_id}:loop-2",
        ),
        _row(
            rollout_id,
            6,
            "Answer",
            ("answered_correctly",),
            action_id=f"{rollout_id}:action:2",
            done=True,
            env_reward=1.0,
        ),
    )


def _split(rows):
    return tuple(row[0] for row in rows), tuple(row[1] for row in rows)


class M7GroupCriticReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = RewardConfig.from_json(default_reward_config_path())
        cls.terminal_dfa = config.profile("terminal_dfa")
        cls.terminal_only = config.profile("terminal_only")
        cls.spec = hand_authored_memory_dfa()

    def _replay(self, rows, *, profile=None, spec=None, version="m7-test-reward-v1"):
        steps, credits = _split(rows)
        return GroupAutomatonReplay(
            profile or self.terminal_dfa,
            spec=spec or self.spec,
            reward_version=version,
        ).replay(steps, credits, seed=23)

    def test_replay_is_deterministic_preserves_ap_provenance_and_replaces_versions(
        self,
    ):
        rows = _clean_rows()
        first = self._replay(rows)
        second = self._replay(rows)

        self.assertEqual(first, second)
        self.assertEqual(first.to_json().encode(), second.to_json().encode())
        self.assertEqual(first.to_jsonl().encode(), second.to_jsonl().encode())
        self.assertEqual(first.schema_version, GROUP_AUTOMATON_REPLAY_SCHEMA_VERSION)
        self.assertTrue(first.accepted)
        self.assertEqual(first.final_state, "q4")
        self.assertEqual(first.env_total, 1.0)
        self.assertEqual(first.milestone_total, 1.0)
        self.assertEqual(first.logic_total, 1.0)
        self.assertEqual(first.total_reward, 2.0)

        source_credits = [credit for _, credit in rows]
        for source, replayed in zip(source_credits, first.credits):
            self.assertEqual(replayed.action_id, source.action_id)
            self.assertEqual(replayed.atomic_propositions, source.atomic_propositions)
            self.assertEqual(
                replayed.atomic_proposition_evidence,
                source.atomic_proposition_evidence,
            )
            self.assertEqual(replayed.dfa_spec_id, self.spec.name)
            self.assertEqual(replayed.reward_version, "m7-test-reward-v1")
            self.assertIsNone(replayed.return_to_go)
            self.assertIsNone(replayed.advantage)

    def test_terminal_only_reuses_profile_and_never_adds_logic_reward(self):
        result = self._replay(
            _clean_rows("terminal-only"),
            profile=self.terminal_only,
            version="m7-terminal-only-v1",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.milestone_total, 1.0)
        self.assertEqual(result.logic_total, 0.0)
        self.assertEqual(result.env_total, 1.0)
        self.assertEqual(result.total_reward, 1.0)

    def test_canonical_answer_ap_may_use_action_coordinate_as_empty_evidence(self):
        rows = list(_clean_rows("empty-answer-evidence"))
        answer_step, _ = rows[-1]
        rows[-1] = (
            answer_step,
            _source_credit(
                answer_step.actions[0],
                ("answered_correctly",),
                evidence={"answered_correctly": ()},
            ),
        )
        result = self._replay(tuple(rows))
        self.assertTrue(result.accepted)
        self.assertEqual(
            result.credits[-1].atomic_proposition_evidence,
            {"answered_correctly": ()},
        )

    def test_canonical_m6_oracle_gold_artifact_replays_without_recomputation(self):
        migration_root = REPOSITORY_ROOT / "runs" / "m6_schema_v2"
        manifest_path = migration_root / "migration_manifest.json"
        if not manifest_path.exists():
            self.skipTest("canonical M6 runtime artifacts are unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gold_file = next(item for item in manifest["files"] if item["policy"] == "gold")
        step_path = migration_root / gold_file["target_trajectory_path"]
        credit_path = migration_root / gold_file["target_credit_path"]
        steps = tuple(
            TrajectoryStepV2.model_validate_json(line)
            for line in step_path.read_text(encoding="utf-8").splitlines()
        )
        credits = tuple(
            ActionCreditRecord.model_validate_json(line)
            for line in credit_path.read_text(encoding="utf-8").splitlines()
        )

        result = GroupAutomatonReplay(
            self.terminal_dfa,
            spec=self.spec,
            reward_version="agemem.reward.m7.canonical_regression.v1",
        ).replay(steps, credits, seed=20260802)

        self.assertEqual(len(result.credits), gold_file["action_count"])
        self.assertTrue(result.accepted)
        self.assertEqual(result.final_state, "q4")
        self.assertEqual(
            tuple(item.action_id for item in result.credits),
            tuple(item.action_id for item in credits),
        )
        self.assertEqual(
            tuple(item.atomic_proposition_evidence for item in result.credits),
            tuple(item.atomic_proposition_evidence for item in credits),
        )
        answer = next(
            item
            for item in result.credits
            if "answered_correctly" in item.atomic_propositions
        )
        self.assertEqual(answer.atomic_proposition_evidence["answered_correctly"], ())

    def test_arbitrary_valid_spec_is_used_instead_of_hand_dfa(self):
        custom = AutomatonSpec(
            name="m7-custom-two-edge-v1",
            states=("s0", "s1", "accept", "reject", "timeout"),
            initial_state="s0",
            accepting_states=("accept",),
            rejecting_states=("reject",),
            timeout_state="timeout",
            transitions=(
                AutomatonTransition(
                    edge_id="store",
                    proposition="stored_supporting_fact",
                    source_states=("s0",),
                    target_state="s1",
                    priority=0,
                    progressive=True,
                ),
                AutomatonTransition(
                    edge_id="answer",
                    proposition="answered_correctly",
                    source_states=("s1",),
                    target_state="accept",
                    priority=1,
                    progressive=True,
                ),
            ),
            source_milestones=("stored_supporting_fact", "answered_correctly"),
        )
        result = self._replay(_clean_rows("custom"), spec=custom)
        self.assertTrue(result.accepted)
        self.assertEqual(result.dfa_spec_id, custom.name)
        self.assertEqual(result.milestone_total, 0.5)

    def test_duplicate_add_retrieve_and_loops_cannot_farm_once_only_reward(self):
        baseline = self._replay(_clean_rows("farming"))
        candidate = self._replay(_farming_rows())
        injected = (
            "farming:duplicate-add",
            "farming:duplicate-retrieve",
            "farming:loop-1",
            "farming:loop-2",
        )
        audit = audit_reward_farming(
            baseline=baseline,
            candidate=candidate,
            spec=self.spec,
            profile=self.terminal_dfa,
            injected_action_ids=injected,
        )

        self.assertTrue(baseline.accepted)
        self.assertTrue(candidate.accepted)
        self.assertEqual(candidate.milestone_total, baseline.milestone_total)
        self.assertEqual(candidate.logic_total, baseline.logic_total)
        self.assertTrue(audit.once_only)
        self.assertTrue(audit.within_progress_cap)
        self.assertTrue(audit.injected_actions_zero_milestone)
        self.assertTrue(audit.no_reward_gain)
        self.assertTrue(audit.passed)
        self.assertEqual(audit.violations, ())
        self.assertEqual(audit.injected_milestone_total, 0.0)
        self.assertEqual(audit.maximum_milestone_total, 1.25)
        candidate_by_id = {item.action_id: item for item in candidate.credits}
        self.assertTrue(
            all(
                candidate_by_id[action_id].reward_breakdown.milestone == 0.0
                for action_id in injected
            )
        )

    def test_farming_audit_rejects_undeclared_candidate_actions(self):
        baseline = self._replay(_clean_rows("farming"))
        candidate = self._replay(_farming_rows())

        with self.assertRaisesRegex(
            GroupAutomatonReplayError,
            "exactly preserve the baseline action_id sequence",
        ):
            audit_reward_farming(
                baseline=baseline,
                candidate=candidate,
                spec=self.spec,
                profile=self.terminal_dfa,
                injected_action_ids=(
                    "farming:duplicate-add",
                    "farming:duplicate-retrieve",
                    "farming:loop-1",
                ),
            )

    def test_join_order_duplicate_done_and_evidence_fail_closed(self):
        steps, credits = _split(_clean_rows("strict"))
        replay = GroupAutomatonReplay(
            self.terminal_dfa,
            spec=self.spec,
            reward_version="m7-strict-v1",
        )

        with self.subTest("identity"):
            mismatch = credits[0].model_copy(update={"action_id": "wrong-action"})
            with self.assertRaisesRegex(GroupAutomatonReplayError, "identity mismatch"):
                replay.replay(steps, (mismatch, *credits[1:]), seed=1)

        with self.subTest("order"):
            with self.assertRaisesRegex(
                GroupAutomatonReplayError, "strict action order"
            ):
                replay.replay(
                    (steps[1], steps[0], steps[2]),
                    (credits[1], credits[0], credits[2]),
                    seed=1,
                )

        with self.subTest("duplicate-action-id"):
            duplicate_row = _row(
                "strict",
                1,
                "Add_memory",
                ("stored_supporting_fact",),
                action_id=steps[0].actions[0].action_id,
            )
            with self.assertRaisesRegex(
                GroupAutomatonReplayError, "duplicate action_id"
            ):
                replay.replay(
                    (steps[0], duplicate_row[0], steps[2]),
                    (credits[0], duplicate_row[1], credits[2]),
                    seed=1,
                )

        with self.subTest("done-not-last"):
            early_done = steps[0].model_copy(update={"done": True})
            with self.assertRaisesRegex(GroupAutomatonReplayError, "done step"):
                replay.replay((early_done, *steps[1:]), credits, seed=1)

        with self.subTest("missing-ap-evidence-key"):
            action = steps[0].actions[0]
            missing_evidence = _source_credit(
                action,
                ("stored_supporting_fact",),
                evidence={},
            )
            with self.assertRaisesRegex(GroupAutomatonReplayError, "each source AP"):
                replay.replay(steps, (missing_evidence, *credits[1:]), seed=1)

        with self.subTest("one-action-per-step"):
            first_action = steps[0].actions[0]
            second_action = first_action.model_copy(
                update={
                    "action_id": "strict:second-action",
                    "action_index_in_turn": 1,
                }
            )
            multi = steps[0].model_copy(
                update={"actions": (first_action, second_action)}
            )
            with self.assertRaisesRegex(
                GroupAutomatonReplayError, "exactly one action"
            ):
                replay.replay((multi, *steps[1:]), credits, seed=1)

        with self.subTest("memory-continuity"):
            discontinuous = steps[1].model_copy(
                update={
                    "memory_before": steps[0].memory_after
                    + (MemorySnapshotItem(memory_id="unexpected-memory", content="x"),)
                }
            )
            with self.assertRaisesRegex(
                GroupAutomatonReplayError, "memory snapshots are not continuous"
            ):
                replay.replay((steps[0], discontinuous, steps[2]), credits, seed=1)

        with self.subTest("result-tool-call-id"):
            action = steps[0].actions[0]
            mismatched_action = action.model_copy(
                update={"result": {**action.result, "tool_call_id": "wrong-call"}}
            )
            mismatched_step = steps[0].model_copy(
                update={"actions": (mismatched_action,)}
            )
            with self.assertRaisesRegex(
                GroupAutomatonReplayError, "tool_call_id must match action_id"
            ):
                replay.replay((mismatched_step, *steps[1:]), credits, seed=1)


if __name__ == "__main__":
    unittest.main()
