"""M8a tests for strict HotpotQA reward and trajectory-credit profiles."""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
E1_CONFIG = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_dry_run.yaml"
REWARD_MODULE_PATH = (
    REPOSITORY_ROOT
    / "trinity"
    / "common"
    / "workflows"
    / "memory_reward"
    / "reward_profiles.py"
)
ALGORITHM_SOURCE = REPOSITORY_ROOT / "trinity" / "algorithm" / "algorithm.py"
CONFIG_SOURCE = REPOSITORY_ROOT / "trinity" / "common" / "config.py"
WORKFLOW_SOURCE = (
    REPOSITORY_ROOT
    / "trinity"
    / "common"
    / "workflows"
    / "memory_context"
    / "train_hotpotQA.py"
)

# Load the pure profile module directly so this standard-library unittest does
# not import Trinity's Ray/vLLM runtime package tree.
_spec = importlib.util.spec_from_file_location(
    "_agemem_reward_profiles_test", REWARD_MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
_reward_profiles = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _reward_profiles
_spec.loader.exec_module(_reward_profiles)

REWARD_PROFILE_SCHEMA_VERSION = _reward_profiles.REWARD_PROFILE_SCHEMA_VERSION
RewardProfileConfigError = _reward_profiles.RewardProfileConfigError
RewardProfileName = _reward_profiles.RewardProfileName
TerminalMetric = _reward_profiles.TerminalMetric
calculate_terminal_reward = _reward_profiles.calculate_terminal_reward
load_reward_profile = _reward_profiles.load_reward_profile
load_workflow_reward_profile = _reward_profiles.load_workflow_reward_profile
score_hotpot_answer = _reward_profiles.score_hotpot_answer
terminal_task_score = _reward_profiles.terminal_task_score
validate_e1_trajectory_credit_contract = (
    _reward_profiles.validate_e1_trajectory_credit_contract
)


def _yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\r\n]+)", text)
    if match is None:
        raise AssertionError(f"missing YAML key {key!r}")
    return match.group(1).strip().strip("'\"")


def _e1_workflow_args(**overrides):
    values = {
        "reward_profile": "terminal_only",
        "terminal_reward_metric": "hotpotqa_official",
        "milestone_reward_enabled": False,
    }
    values.update(overrides)
    return values


class HotpotAnswerMetricTest(unittest.TestCase):
    def test_official_normalization_and_overlap_metrics(self):
        exact = score_hotpot_answer("The Eiffel Tower!", "eiffel tower")
        self.assertEqual(exact.exact_match, 1.0)
        self.assertEqual(exact.f1, 1.0)

        partial = score_hotpot_answer("New York City", "York City")
        self.assertEqual(partial.exact_match, 0.0)
        self.assertAlmostEqual(partial.precision, 2.0 / 3.0)
        self.assertEqual(partial.recall, 1.0)
        self.assertAlmostEqual(partial.f1, 0.8)

    def test_yes_no_mismatch_is_zero_even_with_other_overlap(self):
        score = score_hotpot_answer("yes", "no")
        self.assertEqual(score.exact_match, 0.0)
        self.assertEqual(score.f1, 0.0)
        self.assertEqual(score.precision, 0.0)
        self.assertEqual(score.recall, 0.0)

    def test_non_string_answers_fail_closed(self):
        with self.assertRaisesRegex(TypeError, "must be strings"):
            score_hotpot_answer(42, "42")  # type: ignore[arg-type]


class RewardProfileValidationTest(unittest.TestCase):
    def test_e1_flat_workflow_profile_is_explicit_and_versioned(self):
        profile = load_workflow_reward_profile(_e1_workflow_args())

        self.assertEqual(profile.schema_version, REWARD_PROFILE_SCHEMA_VERSION)
        self.assertEqual(profile.name, RewardProfileName.E1_TERMINAL_ONLY)
        self.assertEqual(profile.terminal_metric, TerminalMetric.HOTPOTQA_OFFICIAL)
        self.assertTrue(profile.is_terminal_only)
        self.assertFalse(profile.uses_llm_judge)
        self.assertFalse(profile.milestone_reward_enabled)

    def test_missing_or_conflicting_e1_fields_are_rejected(self):
        missing_metric = _e1_workflow_args()
        del missing_metric["terminal_reward_metric"]
        with self.assertRaisesRegex(RewardProfileConfigError, "terminal_reward_metric"):
            load_workflow_reward_profile(missing_metric)

        with self.assertRaisesRegex(RewardProfileConfigError, "require.*false"):
            load_workflow_reward_profile(
                _e1_workflow_args(milestone_reward_enabled=True)
            )

        with self.assertRaisesRegex(RewardProfileConfigError, "must be a boolean"):
            load_workflow_reward_profile(_e1_workflow_args(milestone_reward_enabled=0))

    def test_e2_profile_preserves_frozen_legacy_weights(self):
        profile = load_workflow_reward_profile(
            {
                "reward_profile": "agemem_heuristic",
                "milestone_reward_enabled": False,
            }
        )

        self.assertEqual(profile.name, RewardProfileName.E2_AGEMEM_HEURISTIC)
        self.assertTrue(profile.uses_llm_judge)
        self.assertEqual(
            profile.heuristic_calculator_kwargs(),
            {
                "task_completion_weight": 0.5,
                "tool_efficiency_weight": 0.2,
                "context_management_weight": 0.15,
                "memory_management_weight": 0.15,
            },
        )
        with self.assertRaisesRegex(RewardProfileConfigError, "must not define"):
            load_workflow_reward_profile(
                {
                    "reward_profile": "agemem_heuristic",
                    "terminal_reward_metric": "hotpotqa_official",
                    "milestone_reward_enabled": False,
                }
            )

    def test_nested_profile_rejects_unknown_schema_and_fields(self):
        with self.assertRaisesRegex(RewardProfileConfigError, "schema_version"):
            load_reward_profile(
                {
                    "schema_version": "future",
                    "name": "e1_terminal_only",
                    "terminal_metric": "answer_f1",
                }
            )
        with self.assertRaisesRegex(RewardProfileConfigError, "unknown field"):
            load_reward_profile(
                {
                    "schema_version": REWARD_PROFILE_SCHEMA_VERSION,
                    "name": "e1_terminal_only",
                    "terminal_metric": "answer_f1",
                    "tool_bonus": 1.0,
                }
            )

    def test_terminal_reward_has_no_dense_or_timeout_component(self):
        profile = load_workflow_reward_profile(_e1_workflow_args())
        answer_score = score_hotpot_answer("New York City", "York City")
        task_score = terminal_task_score(profile, answer_score)
        outcome = calculate_terminal_reward(
            profile,
            task_score=task_score,
            found_answer=True,
        )

        self.assertAlmostEqual(outcome.total, 0.8)
        self.assertEqual(
            set(outcome.breakdown),
            {"terminal_hotpotqa_official", "total"},
        )
        self.assertFalse(
            any(
                dense_name in outcome.breakdown
                for dense_name in (
                    "tool_efficiency",
                    "context_management",
                    "memory_management",
                    "max_rounds_penalty",
                )
            )
        )

        missing = calculate_terminal_reward(
            profile,
            task_score=1.0,
            found_answer=False,
        )
        self.assertEqual(missing.total, 0.0)


class E1TrajectoryCreditContractTest(unittest.TestCase):
    def test_e1_config_selects_multi_step_trajectory_advantage(self):
        config_text = E1_CONFIG.read_text(encoding="utf-8")
        profile = load_workflow_reward_profile(
            {
                "reward_profile": _yaml_scalar(config_text, "reward_profile"),
                "terminal_reward_metric": _yaml_scalar(
                    config_text, "terminal_reward_metric"
                ),
                "milestone_reward_enabled": (
                    _yaml_scalar(config_text, "milestone_reward_enabled") == "true"
                ),
            }
        )
        algorithm_type = _yaml_scalar(config_text, "algorithm_type")
        advantage_fn = _yaml_scalar(config_text, "advantage_fn")
        repeat_times = int(_yaml_scalar(config_text, "repeat_times"))

        validate_e1_trajectory_credit_contract(
            profile,
            algorithm_type=algorithm_type,
            advantage_fn=advantage_fn,
            repeat_times=repeat_times,
        )
        algorithm_source = ALGORITHM_SOURCE.read_text(encoding="utf-8")
        multi_step_block = algorithm_source.split(
            '@ALGORITHM_TYPE.register_module("multi_step_grpo")', 1
        )[1]
        self.assertIn('"advantage_fn": "step_wise_grpo"', multi_step_block)
        self.assertIn("compute_advantage_in_trainer: bool = False", multi_step_block)
        self.assertIn(
            "multi_step_grpo requires advantage_fn='step_wise_grpo'",
            multi_step_block,
        )
        config_source = CONFIG_SOURCE.read_text(encoding="utf-8")
        self.assertIn("def _check_agemem_training_contract", config_source)
        self.assertIn("self._check_agemem_training_contract()", config_source)

    def test_wrong_algorithm_advantage_or_group_size_is_rejected(self):
        profile = load_workflow_reward_profile(_e1_workflow_args())
        invalid_cases = (
            ("grpo", "step_wise_grpo", 2, "algorithm_type"),
            ("multi_step_grpo", "grpo", 2, "advantage_fn"),
            ("multi_step_grpo", "step_wise_grpo", 1, "repeat_times"),
            ("multi_step_grpo", "step_wise_grpo", "2", "repeat_times"),
        )
        for algorithm_type, advantage_fn, repeat_times, pattern in invalid_cases:
            with self.subTest(
                algorithm_type=algorithm_type,
                advantage_fn=advantage_fn,
                repeat_times=repeat_times,
            ):
                with self.assertRaisesRegex(RewardProfileConfigError, pattern):
                    validate_e1_trajectory_credit_contract(
                        profile,
                        algorithm_type=algorithm_type,
                        advantage_fn=advantage_fn,
                        repeat_times=repeat_times,
                    )


class WorkflowAnswerRoutingTest(unittest.TestCase):
    def test_e1_branch_scores_before_the_only_llm_judge_await(self):
        source = WORKFLOW_SOURCE.read_text(encoding="utf-8")
        method = source.split("    async def _get_answer_score", 1)[1].split(
            "    async def _run_stage1_casual_chat", 1
        )[0]

        terminal_branch = method.index("if self.reward_profile.is_terminal_only")
        deterministic_score = method.index("score_hotpot_answer")
        terminal_return = method.index("terminal_task_score")
        llm_judge_await = method.index("await get_answer_llm_judge_score")
        self.assertLess(terminal_branch, deterministic_score)
        self.assertLess(deterministic_score, terminal_return)
        self.assertLess(terminal_return, llm_judge_await)


if __name__ == "__main__":
    unittest.main()
