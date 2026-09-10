"""Unit tests for E3 HotpotQA Oracle AP + DFA credits.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import unittest


try:
    from AgeMem_code_agentscope.action_schema import ActionEvent
    from trinity.common.action_event_contract import stable_action_id
    from trinity.common.e3_oracle_dfa import (
        replay_hotpotqa_oracle_comparison,
        replay_hotpotqa_oracle_dfa,
    )
except ModuleNotFoundError:
    ActionEvent = None
    stable_action_id = None
    replay_hotpotqa_oracle_comparison = None
    replay_hotpotqa_oracle_dfa = None


@unittest.skipUnless(replay_hotpotqa_oracle_dfa is not None, "agentscope not installed")
class E3OracleDfaTest(unittest.TestCase):
    @staticmethod
    def _action(
        *,
        rollout_id: str,
        turn: int,
        index: int,
        name: str,
        arguments: dict,
        output: dict,
    ):
        task_id = rollout_id.rsplit("/", 1)[0]
        action_id = stable_action_id(
            rollout_id=rollout_id,
            stage_id=1 if turn < 2 else 3,
            timestep=turn,
            assistant_turn_id=turn,
            action_index_in_turn=index,
        )
        return ActionEvent(
            action_id=action_id,
            task_id=task_id,
            rollout_id=rollout_id,
            stage_id=1 if turn < 2 else 3,
            timestep=turn,
            assistant_turn_id=turn,
            action_index_in_turn=index,
            source="llm",
            action_type=name,
            action_text=f"<{name}>",
            arguments=arguments,
            result={
                "trace_call_id": f"call-{turn}-{index}",
                "status": "ok",
                "output": output,
                "error": None,
            },
            response_token_ids=(1, 2),
            token_start=0,
            token_end=1,
            old_logprobs=(-0.1, -0.2),
            policy_version="model_version:0",
        )

    def test_progress_edges_are_rewarded_once_and_accept_on_correct_answer(self):
        support = ("Alice was born in Paris.", "Paris is the capital of France.")
        replay = replay_hotpotqa_oracle_dfa(
            task_id="t1",
            rollout_id="r1",
            seed=7,
            supporting_sentences=support,
            observed_sentences=support,
            indexed_sentences=support,
            retrieved_contents=support,
            tool_events=(),
            exact_match=1.0,
            task_f1=1.0,
            found_answer=True,
            shadow=False,
        )
        self.assertTrue(replay.accepted)
        self.assertEqual(replay.final_state, "q4")
        self.assertGreater(replay.milestone_total, 0.0)
        self.assertAlmostEqual(replay.env_total, 1.0)
        self.assertAlmostEqual(replay.training_total, replay.env_total + replay.logic_total)
        second = replay_hotpotqa_oracle_dfa(
            task_id="t1",
            rollout_id="r1",
            seed=7,
            supporting_sentences=support,
            observed_sentences=support,
            indexed_sentences=support,
            retrieved_contents=support,
            tool_events=(
                {
                    "tool_name": "Add_memory",
                    "arguments": {"content": support[0]},
                    "result": {"memory_id": "m1", "outcome": "added"},
                    "stage": 1,
                    "step": 1,
                    "round": 0,
                    "tool_index": 0,
                },
            ),
            exact_match=1.0,
            task_f1=1.0,
            found_answer=True,
        )
        self.assertLessEqual(second.milestone_total, replay.milestone_total + 0.25)

    def test_eval_shadow_keeps_training_total_equal_to_terminal_f1(self):
        support = ("The Thames flows through London.",)
        replay = replay_hotpotqa_oracle_dfa(
            task_id="t2",
            rollout_id="r2",
            seed=7,
            supporting_sentences=support,
            observed_sentences=support,
            indexed_sentences=support,
            retrieved_contents=support,
            exact_match=1.0,
            task_f1=0.8,
            found_answer=True,
            shadow=True,
        )
        self.assertAlmostEqual(replay.training_total, 0.8)
        self.assertGreater(replay.logic_total, 0.0)
        self.assertNotAlmostEqual(replay.logic_total + replay.env_total, replay.training_total)

    def test_missing_answer_is_zero_env_and_does_not_accept(self):
        replay = replay_hotpotqa_oracle_dfa(
            task_id="t3",
            rollout_id="r3",
            seed=7,
            supporting_sentences=("A supporting sentence.",),
            observed_sentences=("A supporting sentence.",),
            indexed_sentences=("unrelated stored text",),
            retrieved_contents=("unrelated stored text",),
            exact_match=0.0,
            task_f1=0.0,
            found_answer=False,
        )
        self.assertFalse(replay.accepted)
        self.assertEqual(replay.env_total, 0.0)
        self.assertEqual(replay.training_total, replay.logic_total)

    def test_real_action_comparison_has_exact_credit_join_and_no_answer_double_count(self):
        support = ("Alice was born in Paris.", "Paris is the capital of France.")
        rollout_id = "batch/task/0"
        actions = (
            self._action(
                rollout_id=rollout_id,
                turn=0,
                index=0,
                name="Add_memory",
                arguments={"content": support[0]},
                output={"memory_id": "m1", "outcome": "added"},
            ),
            self._action(
                rollout_id=rollout_id,
                turn=1,
                index=0,
                name="Add_memory",
                arguments={"content": support[1]},
                output={"memory_id": "m2", "outcome": "added"},
            ),
            self._action(
                rollout_id=rollout_id,
                turn=2,
                index=0,
                name="Retrieve_memory",
                arguments={"query": "Paris"},
                output={"items": [{"content": support[0]}, {"content": support[1]}]},
            ),
        )
        replay = replay_hotpotqa_oracle_comparison(
            task_id="batch/task",
            rollout_id=rollout_id,
            seed=7,
            supporting_sentences=support,
            observed_sentences=support,
            action_events=actions,
            exact_match=1.0,
            task_f1=1.0,
            found_answer=True,
        )
        self.assertEqual(len(replay.flat.credits), len(actions))
        self.assertEqual(len(replay.dfa.credits), len(actions))
        self.assertEqual(
            {item.action_id for item in replay.flat.credits},
            {item.action_id for item in actions},
        )
        self.assertAlmostEqual(replay.flat.logic_total, 0.75)
        self.assertAlmostEqual(replay.dfa.logic_total, 0.75)
        self.assertAlmostEqual(replay.flat.training_total, 1.75)
        self.assertAlmostEqual(replay.dfa.training_total, 1.75)
        self.assertTrue(replay.dfa.accepted)

    def test_dfa_distinguishes_wrong_order_while_flat_uses_same_aps(self):
        support = ("Alice was born in Paris.", "Paris is the capital of France.")
        rollout_id = "batch/task/1"
        actions = (
            self._action(
                rollout_id=rollout_id,
                turn=0,
                index=0,
                name="Retrieve_memory",
                arguments={"query": "Paris"},
                output={"items": [{"content": support[0]}]},
            ),
            self._action(
                rollout_id=rollout_id,
                turn=1,
                index=0,
                name="Add_memory",
                arguments={"content": support[0]},
                output={"memory_id": "m1", "outcome": "added"},
            ),
            self._action(
                rollout_id=rollout_id,
                turn=2,
                index=0,
                name="Add_memory",
                arguments={"content": support[1]},
                output={"memory_id": "m2", "outcome": "added"},
            ),
        )
        replay = replay_hotpotqa_oracle_comparison(
            task_id="batch/task",
            rollout_id=rollout_id,
            seed=7,
            supporting_sentences=support,
            observed_sentences=support,
            action_events=actions,
            exact_match=1.0,
            task_f1=1.0,
            found_answer=True,
        )
        self.assertAlmostEqual(replay.flat.logic_total, 0.75)
        self.assertAlmostEqual(replay.dfa.logic_total, 0.5)
        self.assertFalse(replay.dfa.accepted)

    def test_repeated_actions_do_not_farm_flat_or_dfa_reward(self):
        support = ("The Thames flows through London.",)
        rollout_id = "batch/task/2"
        actions = tuple(
            self._action(
                rollout_id=rollout_id,
                turn=turn,
                index=0,
                name="Add_memory",
                arguments={"content": support[0]},
                output={"memory_id": f"m{turn}", "outcome": "added"},
            )
            for turn in range(3)
        )
        replay = replay_hotpotqa_oracle_comparison(
            task_id="batch/task",
            rollout_id=rollout_id,
            seed=7,
            supporting_sentences=support,
            observed_sentences=support,
            action_events=actions,
        )
        self.assertAlmostEqual(replay.flat.logic_total, 0.5)
        self.assertAlmostEqual(replay.dfa.logic_total, 0.5)
        self.assertEqual(
            sum(credit.reward_breakdown.total for credit in replay.flat.credits),
            replay.flat.logic_total,
        )
        self.assertEqual(
            sum(credit.reward_breakdown.total for credit in replay.dfa.credits),
            replay.dfa.logic_total,
        )


if __name__ == "__main__":
    unittest.main()
