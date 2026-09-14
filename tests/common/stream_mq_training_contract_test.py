import unittest

from trinity.common.streaming_multiquery_contract import (
    MemoryRolloutGroupBundle, QueryBranchScore, ReadActorSample, ReadRollout,
    StreamingBundleError,
)


def sample(run):
    return ReadActorSample(
        action_id=f"a{run}", shared_prefix_id=f"prefix{run}", token_ids=(1, 2),
        old_logprobs=(-.1, -.2), action_mask=(True, True), policy_version="model_version:0",
    )


def rollout(run, scores):
    branches = tuple(
        QueryBranchScore(f"b{run}-{i}", f"q{i}", value, "reader:0", 7)
        for i, value in enumerate(scores)
    )
    return ReadRollout(
        read_rollout_id=f"r{run}", snapshot_id=f"s{run}", shared_prefix_id=f"prefix{run}",
        policy_version="model_version:0", samples=(sample(run),), branches=branches,
        reward=sum(scores) / len(scores),
    )


class StreamingTrainingContractTest(unittest.TestCase):
    def bundle(self):
        return MemoryRolloutGroupBundle(
            group_id="g", history_family_id="h", expected_rollouts=2,
            expected_queries_per_rollout=2,
            rollouts=(rollout(0, (1, 0)), rollout(1, (0, 0))), reward_version="terminal-v1",
        )

    def test_k_not_k_times_m_defines_advantage(self):
        bundle = self.bundle()
        first, second = bundle.advantages()
        self.assertAlmostEqual(first, 1.0, places=6)
        self.assertAlmostEqual(second, -1.0, places=6)
        self.assertEqual(len(bundle.actor_batch()), 2)
        self.assertEqual(bundle.receipt()["query_branch_count"], 4)
        self.assertEqual(bundle.receipt()["actor_sample_count"], 2)

    def test_duplicating_query_vector_does_not_inflate_reward(self):
        doubled = MemoryRolloutGroupBundle(
            group_id="g", history_family_id="h", expected_rollouts=2,
            expected_queries_per_rollout=4,
            rollouts=(rollout(0, (1, 0, 1, 0)), rollout(1, (0, 0, 0, 0))),
            reward_version="terminal-v1",
        )
        self.assertEqual(self.bundle().rollouts[0].reward, doubled.rollouts[0].reward)
        self.assertEqual(self.bundle().advantages(), doubled.advantages())

    def test_incomplete_bundle_fails_closed(self):
        bad = MemoryRolloutGroupBundle(
            group_id="g", history_family_id="h", expected_rollouts=2,
            expected_queries_per_rollout=2, rollouts=(rollout(0, (1, 0)),),
            reward_version="terminal-v1",
        )
        with self.assertRaises(StreamingBundleError): bad.actor_batch()

    def test_reader_tokens_cannot_enter_actor_loss(self):
        with self.assertRaises(StreamingBundleError):
            QueryBranchScore("b", "q", 1, "reader", 4, actor_loss_token_count=1)

    def test_zero_variance_produces_zero_advantages(self):
        bundle = MemoryRolloutGroupBundle(
            group_id="g", history_family_id="h", expected_rollouts=2,
            expected_queries_per_rollout=2,
            rollouts=(rollout(0, (0, 0)), rollout(1, (0, 0))), reward_version="terminal-v1",
        )
        self.assertEqual(bundle.advantages(), (0.0, 0.0))

    def test_mixed_policy_version_fails_closed(self):
        other = rollout(1, (0, 0))
        object.__setattr__(other, "policy_version", "model_version:1")
        bundle = MemoryRolloutGroupBundle(
            group_id="g", history_family_id="h", expected_rollouts=2,
            expected_queries_per_rollout=2, rollouts=(rollout(0, (1, 0)), other),
            reward_version="terminal-v1",
        )
        with self.assertRaises(StreamingBundleError): bundle.validate_complete()


if __name__ == "__main__": unittest.main()
