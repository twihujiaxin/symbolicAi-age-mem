"""多步骤场景的 Step-wise GRPO 优势计算。

核心思想：同一道题采样多条 run，用各 run 的终局奖励做组内相对评分；
随后把该评分广播到 run 中所有历史步骤，让早期记忆动作也获得学习信号。
"""
from typing import Dict, List, Optional, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.monitor import gather_metrics


@ADVANTAGE_FN.register_module("step_wise_grpo")
class StepWiseGRPOAdvantageFn(AdvantageFn, ExperienceOperator):
    """
    将最后一步代表的轨迹质量广播到此前所有步骤的优势函数。
    Inspired by rLLM (https://github.com/rllm-org/rllm).
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        enable_step_norm: bool = False,
        std_cal_level: str = "group",  # 'group' (task-level) or 'batch'
        **kwargs,
    ) -> None:
        """Initialize the Step-wise GRPO advantage function.

        Args:
            epsilon (float): A small value to avoid division by zero.
            enable_step_norm (bool): If True, normalize advantages by trajectory length.
            std_cal_level (str): The scope for calculating reward standard deviation.
                'group' (default): Std is calculated per task group.
                'batch': Std is calculated across all last-step rewards in the entire batch.
                The mean is always calculated per task group.
        """
        self.epsilon = epsilon
        self.enable_step_norm = enable_step_norm
        self.std_cal_level = std_cal_level
        if self.std_cal_level not in ["group", "batch"]:
            raise ValueError("std_cal_level must be either 'group' or 'batch'")

    def calculate_last_step_advantage(
        self,
        exps: Dict[str, Experience],
        precomputed_std: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """对同一 task 的多个 rollout 计算 GRPO 组内相对分数。

        Args:
            exps (Dict[str, Experience]): One experience per run, keyed by run ID.

        Returns:
            Dict[str, float]: A tuple containing the scores for each run.
            Dict[str, float]: Metrics for logging.
        """
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps.values()], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)
            scores = {}
            for rid, exp in exps.items():
                if self.std_cal_level == "batch" and precomputed_std is not None:
                    score = (exp.reward - group_reward_mean) / (precomputed_std + self.epsilon)
                else:
                    score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                scores[rid] = score.item()
            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }
        return scores, metrics

    def broadcast_advantages(
        self, run_exps: Dict[str, List[Experience]], scores: Dict[str, float]
    ) -> Dict[str, List[Experience]]:
        """把每条 run 的标量分数写入该 run 的所有有效 action token。

        Args:
            run_exps (Dict[str, List[Experience]]): Experiences grouped by run ID.
            scores (Dict[str, float]): Calculated scores for each run.

        Returns:
            Dict[str, List[Experience]]: Updated experiences with advantages broadcasted.
        """
        for run_id, exps in run_exps.items():
            score = scores[run_id]
            traj_length = len(exps)
            for exp in exps:
                # action_mask 保证只训练模型生成的动作 token，不训练 prompt token。
                exp.advantages = exp.action_mask * score  # type: ignore [operator]
                if self.enable_step_norm:
                    exp.advantages /= traj_length
                exp.returns = exp.advantages.clone()
        return run_exps

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        """按 task -> run -> step 的层级组织 Experience 并完成信用分配。"""
        if len(exps) == 0:
            return [], {}
        cnt = 0
        metric_list = []
        # 第 1 层按题目分组：GRPO 只能比较同一道题的不同采样结果。
        task_exps = group_by(exps, "task")

        # --- Pre-computation step for batch-level standard deviation ---
        precomputed_std = None
        if self.std_cal_level == "batch":
            all_laststep_rewards = []
            for task_exp in task_exps.values():
                # First, group all experiences by run to find the last step of each run
                task_run_exps = group_by(task_exp, "run")
                # Collect rewards from the last step of every run in the entire batch
                last_step_rewards = [
                    run_steps[-1].reward for run_steps in task_run_exps.values() if run_steps
                ]
                all_laststep_rewards.extend(last_step_rewards)

            if len(all_laststep_rewards) <= 1:
                precomputed_std = torch.tensor(1.0)
            else:
                precomputed_std = torch.std(torch.tensor(all_laststep_rewards, dtype=torch.float32))
        # --- End of pre-computation ---

        # 第 2 层按 rollout/run 分组：一条 run 内含三个阶段的多个 step。
        result_exps = []
        for task_exp in task_exps.values():
            run_exps = group_by(task_exp, "run")

            # 每条 run 取最后一个 Experience 的 reward 代表整条轨迹质量。
            last_step_exps = {run_id: step_exps[-1] for run_id, step_exps in run_exps.items()}
            scores, metrics = self.calculate_last_step_advantage(
                last_step_exps, precomputed_std=precomputed_std
            )
            metric_list.append(metrics)

            # 将终局相对分数广播到早期 ADD、SUMMARY、RETRIEVE 等动作。
            run_exps = self.broadcast_advantages(run_exps, scores)
            for exps in run_exps.values():
                cnt += len(exps)
                result_exps.extend(exps)

        try:
            metrics = gather_metrics(metric_list, "group_advantages")
            metrics["experience_count"] = cnt
        except ValueError:
            metrics = {}  # empty metric list causes ValueError, ignore it
        return result_exps, metrics

    def __call__(self, exps, **kwargs):
        return self.process(exps)

    @classmethod
    def compute_in_trainer(cls) -> bool:
        """Whether the advantage should be computed in the trainer loop."""
        return False

    @classmethod
    def default_args(cls) -> Dict:
        """Return the default configuration for this strategy."""
        return {
            "epsilon": 1e-6,
            "enable_step_norm": False,
        }
