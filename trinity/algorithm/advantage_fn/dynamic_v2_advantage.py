"""Population-std step-wise GRPO for dynamic multi-query V2."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from trinity.algorithm.advantage_fn.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.monitor import gather_metrics


class DynamicV2AdvantageError(ValueError):
    pass


@ADVANTAGE_FN.register_module("dynamic_v2_step_wise_grpo")
class DynamicV2StepWiseGRPOAdvantageFn(AdvantageFn, ExperienceOperator):
    """Normalize complete K rollout rewards with the frozen ddof=0 rule."""

    def __init__(self, epsilon: float = 1e-8, **_: object) -> None:
        self.epsilon = epsilon

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        if not exps:
            return [], {}
        output: list[Experience] = []
        metric_rows = []
        for task_exps in group_by(exps, "task").values():
            run_exps = group_by(task_exps, "run")
            group_ids = {
                str(exp.info.get("dynamic_group_id")) for exp in task_exps
            }
            if len(group_ids) != 1 or "None" in group_ids:
                raise DynamicV2AdvantageError("one task must contain one V2 group")
            expected_values = {
                int(exp.info.get("dynamic_expected_k", 0)) for exp in task_exps
            }
            if len(expected_values) != 1:
                raise DynamicV2AdvantageError("mixed expected K")
            expected_k = next(iter(expected_values))
            if expected_k < 2 or len(run_exps) != expected_k:
                raise DynamicV2AdvantageError("incomplete dynamic V2 K group")
            rewards = []
            ordered_runs = []
            for run_id, steps in run_exps.items():
                if not steps or any(exp.reward is None for exp in steps):
                    raise DynamicV2AdvantageError("missing rollout reward")
                reward_values = {float(exp.reward) for exp in steps}
                if len(reward_values) != 1:
                    raise DynamicV2AdvantageError("reward changed within one rollout")
                if any(exp.info.get("phase") != "ingest" for exp in steps):
                    raise DynamicV2AdvantageError("reader tokens entered actor batch")
                ordered_runs.append((run_id, steps))
                rewards.append(next(iter(reward_values)))
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(
                sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
            )
            scores = [
                0.0 if std == 0.0 else (reward - mean) / (std + self.epsilon)
                for reward in rewards
            ]
            for (_, steps), score in zip(ordered_runs, scores, strict=True):
                for exp in steps:
                    recorded = exp.info.get(
                        "dynamic_precomputed_advantage_ddof0"
                    )
                    if recorded is None or abs(float(recorded) - score) > 1e-6:
                        raise DynamicV2AdvantageError(
                            "runtime and trainer advantages differ"
                        )
                    exp.advantages = exp.action_mask * score
                    exp.returns = exp.advantages.clone()
                    output.append(exp)
            metric_rows.append(
                {
                    "reward_mean": mean,
                    "reward_std_ddof0": std,
                    "nonzero_advantage_rollouts": float(
                        sum(abs(score) > 0 for score in scores)
                    ),
                    "experience_count": float(len(task_exps)),
                }
            )
        try:
            metrics = gather_metrics(metric_rows, "dynamic_v2_group_advantages")
        except ValueError:
            metrics = {}
        return output, metrics

    def __call__(self, exps, **kwargs):
        return self.process(exps)

    @classmethod
    def compute_in_trainer(cls) -> bool:
        return False

    @classmethod
    def default_args(cls) -> Dict:
        return {"epsilon": 1e-8}


__all__ = [
    "DynamicV2AdvantageError",
    "DynamicV2StepWiseGRPOAdvantageFn",
]
