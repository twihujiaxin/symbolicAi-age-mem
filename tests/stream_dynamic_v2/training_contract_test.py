import unittest
from pathlib import Path

from AgeMem_code_agentscope.streaming_memory.dynamic.config import load_dynamic_config

from trinity.common.dynamic_multiquery_contract import (
    DynamicBundleError,
    DynamicMemoryRolloutGroupBundle,
    DynamicQueryBranchScore,
    DynamicReadRollout,
)
from trinity.common.streaming_multiquery_contract import ReadActorSample


def rollout(index, task_scores, semantic, profile="V2_LIFE"):
    samples = (
        ReadActorSample(
            action_id=f"a{index}",
            shared_prefix_id=f"prefix{index}",
            token_ids=(1, 2),
            old_logprobs=(-0.1, -0.2),
            action_mask=(True, True),
            policy_version="p0",
        ),
    )
    branches = tuple(
        DynamicQueryBranchScore(f"b{index}-{i}", f"q{i}", value, "reader0", 5)
        for i, value in enumerate(task_scores)
    )
    task = sum(task_scores) / len(task_scores)
    return DynamicReadRollout(
        read_rollout_id=f"r{index}",
        snapshot_id=f"s{index}",
        shared_prefix_id=f"prefix{index}",
        policy_version="p0",
        samples=samples,
        branches=branches,
        task_reward=task,
        semantic_component=semantic,
        total_reward=task + 0.25 * semantic,
        reward_profile=profile,
    )


class DynamicTrainingContractTest(unittest.TestCase):
    def bundle(self, m=2):
        scores_a = (1, 0) * (m // 2)
        scores_b = (0, 0) * (m // 2)
        return DynamicMemoryRolloutGroupBundle(
            group_id="g",
            history_family_id="f",
            expected_rollouts=2,
            expected_queries_per_rollout=m,
            rollouts=(rollout(0, scores_a, 1), rollout(1, scores_b, 0)),
            reward_version="dynamic_reward_v2",
            reward_profile="V2_LIFE",
            lambda_semantic=0.25,
        )

    def test_k_not_k_times_m_and_prefix_once(self):
        bundle = self.bundle()
        self.assertEqual(len(bundle.advantages()), 2)
        self.assertEqual(len(bundle.actor_batch()), 2)
        self.assertAlmostEqual(bundle.advantages()[0], 1, places=6)
        self.assertAlmostEqual(bundle.advantages()[1], -1, places=6)

    def test_m_does_not_duplicate_read_tokens_or_scale_average(self):
        two, four = self.bundle(2), self.bundle(4)
        self.assertEqual(len(two.actor_batch()), len(four.actor_batch()))
        self.assertEqual(two.rollouts[0].task_reward, four.rollouts[0].task_reward)

    def test_monitor_alias_cannot_create_duplicate_gpu_arm(self):
        bad = self.bundle()
        object.__setattr__(bad, "reward_profile", "V2_LIFE_MONITOR")
        with self.assertRaises(DynamicBundleError):
            bad.actor_batch()

    def test_reader_tokens_are_masked(self):
        with self.assertRaises(DynamicBundleError):
            DynamicQueryBranchScore("b", "q", 1, "reader", 5, actor_loss_token_count=1)

    def test_main_profiles_change_only_reward_identity_and_run_name(self):
        root = Path(__file__).resolve().parents[2]
        values = []
        for name in ("pilot_terminal.yaml", "pilot_end.yaml", "pilot_life.yaml"):
            raw = load_dynamic_config(
                root / "configs/stream_dynamic_v2" / name
            ).model_dump(mode="json")
            raw["experiment"]["name"] = "PROFILE"
            raw["reward"]["profile"] = "PROFILE"
            raw["runtime"]["run_root"] = "PROFILE"
            values.append(raw)
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[1], values[2])


if __name__ == "__main__":
    unittest.main()
