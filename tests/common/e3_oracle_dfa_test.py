"""Unit tests for E3 HotpotQA Oracle AP + DFA credits.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import unittest


try:
    from trinity.common.e3_oracle_dfa import replay_hotpotqa_oracle_dfa
except ModuleNotFoundError:
    replay_hotpotqa_oracle_dfa = None


@unittest.skipUnless(replay_hotpotqa_oracle_dfa is not None, "agentscope not installed")
class E3OracleDfaTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
