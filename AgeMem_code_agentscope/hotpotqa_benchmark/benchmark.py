"""Deterministic M5 Oracle benchmark built on the M1-M4 offline pipeline."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..memory_oracle import OfflineRewardReplay
from ..toy_hotpotqa import ErrorMemoryPolicy, GoldMemoryPolicy, ToyEpisodeRunner
from ..trajectory import TrajectoryRecorder, TrajectoryReplay
from .adapter import HotpotQADataAdapter, manifest_digest, smoke_config_digest
from .metrics import (
    BenchmarkRecord,
    FailureAudit,
    OracleBenchmarkReport,
    aggregate_records,
    build_benchmark_record,
    report_digest,
    write_failures_jsonl,
    write_report_json,
)
from .models import (
    HotpotQAMemoryTask,
    HotpotQASmokeConfig,
    HotpotQASmokeManifest,
    PolicyName,
)


class OracleBenchmarkError(RuntimeError):
    """Raised when a model-free M5 benchmark invariant is violated."""


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Paths and validated report produced by one benchmark invocation."""

    runtime_root: Path
    report_path: Path
    failures_path: Path
    markdown_path: Path
    report: OracleBenchmarkReport


def _policy(name: PolicyName):
    if name == "gold":
        return GoldMemoryPolicy()
    if name in {"wrong_answer", "missing_support"}:
        return ErrorMemoryPolicy(name)
    raise OracleBenchmarkError(f"unsupported offline policy {name!r}")


def _rollout_id(task: HotpotQAMemoryTask, policy: PolicyName) -> str:
    return f"m5-{task.split}-{task.hotpot_id}-{policy}"


def _logical_path(kind: str, task: HotpotQAMemoryTask, policy: PolicyName) -> Path:
    safe_id = hashlib.sha256(task.hotpot_id.encode("utf-8")).hexdigest()[:24]
    return Path(kind) / task.split / f"{safe_id}.{policy}.jsonl"


async def _record_trajectory(
    *,
    task: HotpotQAMemoryTask,
    policy: PolicyName,
    seed: int,
    output: Path,
) -> str:
    """Record to a temporary file and atomically replace a generated trace."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        await ToyEpisodeRunner().run(
            task,
            _policy(policy),
            rollout_id=_rollout_id(task, policy),
            seed=seed,
            recorder=TrajectoryRecorder(temporary),
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return _rollout_id(task, policy)


def _write_reward(reward, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        reward.write_jsonl(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_expected_outcome(record: BenchmarkRecord) -> None:
    if record.policy == "gold":
        if not record.episode_success or not record.dfa_accepted:
            raise OracleBenchmarkError(
                f"gold Oracle pipeline failed for {record.task_id}: "
                f"episode={record.episode_success}, dfa={record.dfa_status}"
            )
        return
    if record.episode_success or record.dfa_accepted:
        raise OracleBenchmarkError(
            f"negative control {record.policy!r} unexpectedly passed "
            f"for {record.task_id}"
        )


def _markdown(report: OracleBenchmarkReport) -> str:
    lines = [
        "# M5 HotpotQA Oracle Benchmark",
        "",
        "本报告由本地 `hotpot_qa/fullwiki` smoke split、确定性规则策略、"
        "M1 轨迹重放和 M4 Oracle AP/DFA 离线生成。未调用 LLM，未执行模型训练。",
        "",
        "## 数据与协议",
        "",
        f"- Seed：`{report.seed}`",
        f"- Manifest digest：`{report.manifest_digest}`",
        f"- Smoke config digest：`{report.smoke_config_digest}`",
        f"- Reward profile：`{report.reward_profile.name}` "
        f"(`{report.reward_profile_digest}`)",
        f"- Report digest：`{report.digest}`",
        f"- Source sizes：train={report.source_split_sizes['train']}，"
        f"validation={report.source_split_sizes['validation']}，"
        f"official test={report.source_split_sizes['test']}",
        f"- 官方 test 标签不可见校验：{report.official_test_label_blind_count} 条",
        "- Smoke train 来自 source train；smoke dev/test 是 source validation 的"
        "固定互斥子集。官方 test 不进入 Oracle 指标。",
        "- `gold` 是 Oracle 上界；`wrong_answer` 与 `missing_support` 是确定性失败"
        "对照，不代表真实 base-model 表现。",
        "- `Retrieval recall@k` 是 Oracle-directed、整条 episode 中多次 top-1 "
        "检索结果并集对 supporting facts 的累计召回率；`k` 是唯一返回事实数，"
        "不是标准单查询模型 Recall@k。",
        "- `Context tokens` 是所有 timestep 已处理 observation 的、与 tokenizer "
        "无关的累计估算，因此会计入跨步骤重复上下文。",
        "- `Memory precision` 是最终 active memory records 中 supporting records "
        "所占比例。",
        "",
        "## 汇总指标",
        "",
        "| Split | Policy | N | Success | DFA accept | Answer EM | Answer F1 | "
        "Support coverage | Memory precision | Retrieval recall@k | Mean k | "
        "Context tokens | Tool calls | Reward |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.aggregates:
        lines.append(
            f"| {item.split} | {item.policy} | {item.rollout_count} | "
            f"{item.episode_success_rate:.3f} | {item.dfa_acceptance_rate:.3f} | "
            f"{item.answer_em:.3f} | {item.answer_f1:.3f} | "
            f"{item.supporting_fact_coverage:.3f} | {item.memory_precision:.3f} | "
            f"{item.retrieval_recall_at_k:.3f} | {item.mean_retrieval_k:.1f} | "
            f"{item.mean_context_tokens:.1f} | "
            f"{item.mean_tool_calls:.1f} | {item.mean_total_reward:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 失败审计",
            "",
            f"失败/拒绝轨迹：{len(report.failures)} / {len(report.records)}。",
            "每条失败记录保存 supporting fact 指针与 ID、最终 memory 版本历史、"
            "检索覆盖和自动机状态；不复制完整上下文或答案文本。",
            "",
            "## 范围限制",
            "",
            "该结果仅验证真实数据适配、Oracle AP 上界、轨迹确定性和离线奖励链路。"
            "自然语言 AP 抽取、真实 base model、Critic、GRPO 与训练均不属于 M5。",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: OracleBenchmarkReport, path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_markdown(report), encoding="utf-8", newline="\n")
    return output


class HotpotQAOracleBenchmark:
    """Collect, replay, reward, and report deterministic real-data episodes."""

    def __init__(self, adapter: HotpotQADataAdapter) -> None:
        self.adapter = adapter
        self.reward_replay = OfflineRewardReplay.from_config("terminal_dfa")

    async def run(
        self,
        *,
        config: HotpotQASmokeConfig,
        manifest: HotpotQASmokeManifest,
        runtime_root: str | Path,
        report_root: Optional[str | Path] = None,
    ) -> BenchmarkArtifacts:
        runtime = Path(runtime_root).expanduser().resolve()
        summaries = Path(report_root or runtime).expanduser().resolve()
        runtime.mkdir(parents=True, exist_ok=True)
        summaries.mkdir(parents=True, exist_ok=True)

        maximum_gold_steps = 2 * config.max_supporting_facts + 3
        if maximum_gold_steps > self.reward_replay.profile.max_steps:
            raise OracleBenchmarkError(
                "smoke config can exceed the DFA replay budget: "
                f"gold requires up to {maximum_gold_steps} steps, profile allows "
                f"{self.reward_replay.profile.max_steps}"
            )
        self.adapter.verify_manifest(manifest, config)
        label_blind_count = self.adapter.validate_official_test_is_label_blind()
        records = []
        failures = []

        for selection in manifest.selections:
            task = self.adapter.adapt(selection, config)
            for policy in config.policies:
                trajectory_relative = _logical_path("trajectories", task, policy)
                reward_relative = _logical_path("rewards", task, policy)
                trajectory_path = runtime / trajectory_relative
                reward_path = runtime / reward_relative
                if not trajectory_path.resolve().is_relative_to(runtime):
                    raise OracleBenchmarkError("trajectory path escaped runtime root")
                if not reward_path.resolve().is_relative_to(runtime):
                    raise OracleBenchmarkError("reward path escaped runtime root")
                rollout_id = await _record_trajectory(
                    task=task,
                    policy=policy,
                    seed=config.seed,
                    output=trajectory_path,
                )

                trajectory = TrajectoryReplay.from_jsonl(trajectory_path)
                replay = trajectory.replay(
                    task_id=task.task_id,
                    rollout_id=rollout_id,
                    require_complete=True,
                )
                repeated_replay = trajectory.replay(
                    task_id=task.task_id,
                    rollout_id=rollout_id,
                    require_complete=True,
                )
                if replay != repeated_replay:
                    raise OracleBenchmarkError(
                        f"trajectory replay is not deterministic for {rollout_id}"
                    )

                reward = self.reward_replay.replay_jsonl(
                    trajectory_path,
                    task=task,
                    rollout_id=rollout_id,
                )
                repeated_reward = self.reward_replay.replay_jsonl(
                    trajectory_path,
                    task=task,
                    rollout_id=rollout_id,
                )
                if reward != repeated_reward:
                    raise OracleBenchmarkError(
                        f"reward replay is not deterministic for {rollout_id}"
                    )
                _write_reward(reward, reward_path)

                record, failure = build_benchmark_record(
                    task=task,
                    policy=policy,
                    seed=config.seed,
                    replay=replay,
                    reward=reward,
                    trajectory_path=trajectory_relative.as_posix(),
                    reward_path=reward_relative.as_posix(),
                )
                _assert_expected_outcome(record)
                records.append(record)
                if failure is not None:
                    failures.append(failure)

        aggregates = aggregate_records(records)
        base_payload: Dict[str, object] = {
            "source_split_sizes": {
                split: self.adapter.split_size(split)
                for split in ("train", "validation", "test")
            },
            "source_fingerprints": self.adapter.fingerprints(),
            "official_test_label_blind_count": label_blind_count,
            "manifest_digest": manifest_digest(manifest),
            "smoke_config": config,
            "smoke_config_digest": smoke_config_digest(config),
            "reward_profile": self.reward_replay.profile,
            "reward_profile_digest": report_digest(
                self.reward_replay.profile.model_dump(mode="json")
            ),
            "seed": config.seed,
            "records": tuple(records),
            "aggregates": aggregates,
            "failures": tuple(failures),
        }
        canonical_payload = {
            key: (
                [item.model_dump(mode="json") for item in value]
                if key in {"records", "aggregates", "failures"}
                else (
                    value.model_dump(mode="json")
                    if hasattr(value, "model_dump")
                    else value
                )
            )
            for key, value in base_payload.items()
        }
        canonical_payload.update(
            {
                "schema_version": 1,
                "benchmark_name": "m5-hotpotqa-fullwiki-oracle-smoke",
                "dataset_name": "hotpot_qa",
                "dataset_config": "fullwiki",
            }
        )
        report = OracleBenchmarkReport(
            **base_payload,
            digest=report_digest(canonical_payload),
        )
        report_path = write_report_json(report, summaries / "oracle_benchmark.json")
        failures_path = write_failures_jsonl(
            report, summaries / "failures.jsonl"
        )
        markdown_path = write_markdown_report(
            report, summaries / "oracle_benchmark.md"
        )
        return BenchmarkArtifacts(
            runtime_root=runtime,
            report_path=report_path,
            failures_path=failures_path,
            markdown_path=markdown_path,
            report=report,
        )


__all__ = [
    "BenchmarkArtifacts",
    "HotpotQAOracleBenchmark",
    "OracleBenchmarkError",
    "write_markdown_report",
]
