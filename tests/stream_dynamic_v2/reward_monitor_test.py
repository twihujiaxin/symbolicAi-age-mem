import unittest

from AgeMem_code_agentscope.streaming_memory.dynamic.lifecycle_monitor import (
    compare_evaluations,
    compile_monitor,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.reward_profiles import (
    DynamicRewardError,
    aggregate_dynamic_reward,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import DynamicQueryPrivate
from AgeMem_code_agentscope.streaming_memory.dynamic.state_evaluator import (
    CheckpointSemanticFrame,
    DirectStateEvaluator,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.temporal_grounder import (
    GroundingDecision,
    SemanticQueryState,
)


def query(qid, available=0):
    return DynamicQueryPrivate(
        query_id=qid,
        history_id="h",
        task_family="historical_state",
        query_kind="state_at",
        entity="e",
        relation="r",
        query_time=1,
        answers=("yes",),
        answer_available_at=available,
        support_alternatives=(("event",),),
        answer_type="entity",
        answerable=True,
    )


def state(qid, utility, conflict=False):
    return SemanticQueryState(
        query_id=qid,
        coverage=utility if not conflict else 1.0,
        conflict=conflict,
        utility=0.0 if conflict else utility,
        obligation_count=1,
        supported_obligation_count=int(utility == 1),
        decision=GroundingDecision(
            label="supported" if utility == 1 and not conflict else "unsupported",
            source_ok=True,
            content_ok=True,
            temporal_ok=True,
            evidence_revision_ids=(),
            reason_code="fixture",
        ),
    )


def evaluate(rows):
    frames = tuple(
        CheckpointSemanticFrame(
            f"c{index}", index, tuple(state(qid, value) for qid, value in values)
        )
        for index, values in enumerate(rows)
    )
    return frames, DirectStateEvaluator().evaluate(frames)


class DynamicRewardMonitorTest(unittest.TestCase):
    def test_section_11_6_numbers(self):
        queries = (query("q1"), query("q2"))
        answers = {"q1": "<answer>yes</answer>", "q2": "<answer>yes</answer>"}
        exposed = {"q1": 1.0, "q2": 1.0}
        _, direct_a = evaluate(((("q1", 1.0), ("q2", 1.0)), (("q1", 1.0), ("q2", 1.0))))
        _, direct_b = evaluate(((("q1", 0.0), ("q2", 1.0)), (("q1", 1.0), ("q2", 1.0))))
        a = aggregate_dynamic_reward(
            profile="V2_LIFE",
            queries=queries,
            answer_text_by_query=answers,
            exposure_utility_by_query=exposed,
            state_evaluation=direct_a,
        )
        b = aggregate_dynamic_reward(
            profile="V2_LIFE",
            queries=queries,
            answer_text_by_query=answers,
            exposure_utility_by_query=exposed,
            state_evaluation=direct_b,
        )
        self.assertEqual(
            (a.end_score, a.retention_score, a.life_score, a.total_reward),
            (1, 1, 1, 1.25),
        )
        self.assertEqual(
            (b.end_score, b.retention_score, b.life_score, b.total_reward),
            (1, 0.75, 0.875, 1.21875),
        )

    def test_direct_and_compiled_match_with_rollback(self):
        frames, direct = evaluate(((("q", 1.0),), (("q", 0.0),), (("q", 1.0),)))
        compiled = compile_monitor().evaluate(frames)
        comparison = compare_evaluations(direct, compiled)
        self.assertEqual(comparison["status"], "pass")
        self.assertEqual(comparison["max_abs_utility_diff"], 0)
        self.assertFalse(comparison["duplicate_gpu_experiment_required"])

    def test_profiles_share_task_and_do_not_reward_action_count(self):
        queries = (query("q"),)
        _, direct = evaluate(((("q", 1.0),), (("q", 1.0),), (("q", 1.0),)))
        kwargs = dict(
            queries=queries,
            answer_text_by_query={"q": "yes"},
            exposure_utility_by_query={"q": 1.0},
            state_evaluation=direct,
        )
        t = aggregate_dynamic_reward(profile="V2_T", **kwargs)
        end = aggregate_dynamic_reward(profile="V2_END", **kwargs)
        life = aggregate_dynamic_reward(profile="V2_LIFE", **kwargs)
        self.assertEqual(t.total_reward, 1)
        self.assertEqual(end.total_reward, life.total_reward)
        self.assertLessEqual(life.semantic_component, 1)

    def test_no_retention_opportunity_fails_closed(self):
        _, direct = evaluate(((("q", 1.0),),))
        with self.assertRaises(DynamicRewardError):
            aggregate_dynamic_reward(
                profile="V2_LIFE",
                queries=(query("q", available=2),),
                answer_text_by_query={"q": "yes"},
                exposure_utility_by_query={"q": 1},
                state_evaluation=direct,
            )

    def test_compiled_monitor_cannot_disable_rollback(self):
        with self.assertRaises(Exception):
            compile_monitor(
                {
                    "states": ["absent", "partial", "supported", "conflicting"],
                    "output": "coverage*(1-conflict)",
                    "checkpoint_schedule": "every_chunk_commit",
                    "rollback": False,
                }
            )


if __name__ == "__main__":
    unittest.main()
