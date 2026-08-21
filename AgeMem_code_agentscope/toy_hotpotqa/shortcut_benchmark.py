"""Deterministic pre-training audit for Stage-1/Stage-2 reward shortcuts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dataset import ToyTaskDataset
from .stage2_challenge import (
    Stage2BenchmarkReport,
    Stage2ChallengeDataset,
    run_stage2_challenge_benchmark,
)
from .storage_baselines import (
    LEXICAL_TOKEN_COUNTER_NAME,
    OracleSafeStorePolicy,
    Stage1StorageBenchmark,
    Stage1StorageRunResult,
    StoragePolicyName,
    StoreAllPolicy,
    StoreNonePolicy,
    count_ltm_tokens,
)


SHORTCUT_REPORT_SCHEMA_VERSION = "agemem.anti_shortcut_benchmark.v2"
DEFAULT_STAGE1_TASK_ID = "toy-train-005"
DEFAULT_SEED = 7
EXPECTED_STAGE1_POLICIES = (
    "store-all",
    "store-none",
    "oracle-safe-store",
)
EXPECTED_STAGE2_POLICIES = (
    "always_keep",
    "always_clear",
    "opaque_id_control",
    "oracle_safe_compress",
)
EXPECTED_GATE_NAMES = (
    "store_all_is_not_an_oracle_equivalent",
    "store_none_loses_future_support",
    "oracle_safe_store_is_feasible",
    "always_keep_exceeds_context_budget",
    "always_clear_loses_delayed_support",
    "opaque_id_min_control_is_not_oracle_equivalent",
    "oracle_safe_compress_is_feasible",
)


class Stage1ShortcutMetrics(BaseModel):
    """One Stage-1 policy result reduced to anti-shortcut metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: StoragePolicyName
    uses_oracle_labels: bool
    selected_fact_count: int = Field(ge=0)
    admitted_write_count: int = Field(ge=0)
    rejected_budget_count: int = Field(ge=0)
    stored_token_count: int = Field(ge=0)
    budget_tokens: int = Field(ge=1)
    supporting_recall: float = Field(ge=0.0, le=1.0)
    memory_precision: float = Field(ge=0.0, le=1.0)
    average_admitted_write_tokens: float = Field(ge=0.0)
    supporting_complete: bool
    budget_compliant: bool

    @model_validator(mode="after")
    def validate_derived_metrics(self) -> "Stage1ShortcutMetrics":
        if self.uses_oracle_labels != (self.policy == "oracle-safe-store"):
            raise ValueError("Stage-1 Oracle declaration does not match policy")
        if self.admitted_write_count + self.rejected_budget_count != (
            self.selected_fact_count
        ):
            raise ValueError("Stage-1 action counts are inconsistent")
        if self.supporting_complete != (self.supporting_recall == 1.0):
            raise ValueError("supporting_complete does not match supporting_recall")
        if self.budget_compliant != (self.stored_token_count <= self.budget_tokens):
            raise ValueError("budget_compliant does not match token counts")
        return self


class ShortcutGate(BaseModel):
    """A derived, fail-closed acceptance condition for the benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    passed: bool
    evidence: str = Field(min_length=1)


class AntiShortcutBenchmarkReport(BaseModel):
    """Versioned report combining the two independent sidecar benchmarks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.anti_shortcut_benchmark.v2"] = (
        SHORTCUT_REPORT_SCHEMA_VERSION
    )
    seed: int = Field(ge=0)
    real_llm_call_count: Literal[0] = 0
    stage1_task_id: str = Field(min_length=1)
    stage1_task_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage1_token_counter: Literal["unicode-lexical-v1"]
    stage1_budget_tokens: int = Field(ge=1)
    stage1: Dict[StoragePolicyName, Stage1ShortcutMetrics]
    stage2: Stage2BenchmarkReport
    gates: Tuple[ShortcutGate, ...]
    passed: bool
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> "AntiShortcutBenchmarkReport":
        if set(self.stage1) != set(EXPECTED_STAGE1_POLICIES):
            raise ValueError("Stage-1 report must contain exactly three baselines")
        for key, metrics in self.stage1.items():
            if metrics.policy != key:
                raise ValueError("Stage-1 metric policy must match its map key")
            if metrics.budget_tokens != self.stage1_budget_tokens:
                raise ValueError("Stage-1 metric budget does not match report")
        if self.stage2.seed != self.seed:
            raise ValueError("Stage-2 seed must match the combined report")
        if set(self.stage2.aggregates) != set(EXPECTED_STAGE2_POLICIES):
            raise ValueError("Stage-2 report must contain exactly four controls")
        expected_gates = _derive_gates(self.stage1, self.stage2)
        if tuple(gate.name for gate in self.gates) != EXPECTED_GATE_NAMES:
            raise ValueError("shortcut report has an unexpected gate set or order")
        if self.gates != expected_gates:
            raise ValueError("shortcut gates do not match derived metrics")
        if self.passed != all(gate.passed for gate in expected_gates):
            raise ValueError("report passed flag does not match derived gates")
        expected_digest = shortcut_report_digest(self, include_digest=False)
        if self.digest != expected_digest:
            raise ValueError("shortcut report digest does not match its payload")
        return self


def _stage1_metrics(
    result: Stage1StorageRunResult,
    *,
    supporting_fact_count: int,
) -> Stage1ShortcutMetrics:
    active_count = len(result.active_memories)
    supporting_count = len(result.stored_supporting_fact_ids)
    admitted = [decision for decision in result.decisions if decision.admitted]
    admitted_tokens = [
        result.audit_events[decision.audit_event_index].attempted_content_tokens or 0
        for decision in admitted
    ]
    return Stage1ShortcutMetrics(
        policy=result.policy,
        uses_oracle_labels=result.uses_oracle_labels,
        selected_fact_count=len(result.selected_fact_ids),
        admitted_write_count=len(admitted),
        rejected_budget_count=sum(
            decision.reason == "budget_exceeded" for decision in result.decisions
        ),
        stored_token_count=result.active_tokens,
        budget_tokens=result.budget_tokens,
        supporting_recall=supporting_count / supporting_fact_count,
        memory_precision=(supporting_count / active_count if active_count else 0.0),
        average_admitted_write_tokens=(
            sum(admitted_tokens) / len(admitted_tokens) if admitted_tokens else 0.0
        ),
        supporting_complete=supporting_count == supporting_fact_count,
        budget_compliant=result.active_tokens <= result.budget_tokens,
    )


def _canonical_payload_digest(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _derive_gates(
    stage1: Mapping[StoragePolicyName, Stage1ShortcutMetrics],
    stage2: Stage2BenchmarkReport,
) -> Tuple[ShortcutGate, ...]:
    store_all = stage1["store-all"]
    store_none = stage1["store-none"]
    oracle_store = stage1["oracle-safe-store"]
    keep = stage2.aggregates["always_keep"]
    clear = stage2.aggregates["always_clear"]
    id_only = stage2.aggregates["opaque_id_control"]
    oracle_compress = stage2.aggregates["oracle_safe_compress"]
    return (
        ShortcutGate(
            name="store_all_is_not_an_oracle_equivalent",
            passed=(
                store_all.rejected_budget_count > 0
                and not store_all.supporting_complete
                and store_all.memory_precision < 1.0
            ),
            evidence=(
                f"support_recall={store_all.supporting_recall:.3f}, "
                f"memory_precision={store_all.memory_precision:.3f}, "
                f"budget_rejections={store_all.rejected_budget_count}"
            ),
        ),
        ShortcutGate(
            name="store_none_loses_future_support",
            passed=(
                store_none.supporting_recall == 0.0
                and store_none.stored_token_count == 0
            ),
            evidence=(
                f"support_recall={store_none.supporting_recall:.3f}, "
                f"stored_tokens={store_none.stored_token_count}"
            ),
        ),
        ShortcutGate(
            name="oracle_safe_store_is_feasible",
            passed=(
                oracle_store.supporting_complete
                and oracle_store.memory_precision == 1.0
                and oracle_store.budget_compliant
            ),
            evidence=(
                f"support_recall={oracle_store.supporting_recall:.3f}, "
                f"memory_precision={oracle_store.memory_precision:.3f}"
            ),
        ),
        ShortcutGate(
            name="always_keep_exceeds_context_budget",
            passed=(
                keep.future_support_recall == 1.0
                and keep.budget_compliance_rate == 0.0
                and keep.safe_success_rate == 0.0
            ),
            evidence=(
                f"support_recall={keep.future_support_recall:.3f}, "
                f"budget_rate={keep.budget_compliance_rate:.3f}"
            ),
        ),
        ShortcutGate(
            name="always_clear_loses_delayed_support",
            passed=(
                clear.future_support_recall == 0.0
                and clear.budget_compliance_rate == 1.0
                and clear.safe_success_rate == 0.0
            ),
            evidence=(
                f"support_recall={clear.future_support_recall:.3f}, "
                f"budget_rate={clear.budget_compliance_rate:.3f}"
            ),
        ),
        ShortcutGate(
            name="opaque_id_min_control_is_not_oracle_equivalent",
            passed=(
                id_only.future_support_recall < 1.0 and id_only.safe_success_rate < 1.0
            ),
            evidence=(
                f"support_recall={id_only.future_support_recall:.3f}, "
                f"safe_success={id_only.safe_success_rate:.3f}"
            ),
        ),
        ShortcutGate(
            name="oracle_safe_compress_is_feasible",
            passed=(
                oracle_compress.future_support_recall == 1.0
                and oracle_compress.distractor_removal_recall == 1.0
                and oracle_compress.budget_compliance_rate == 1.0
                and oracle_compress.safe_success_rate == 1.0
            ),
            evidence=(
                f"support_recall={oracle_compress.future_support_recall:.3f}, "
                f"distractor_removal={oracle_compress.distractor_removal_recall:.3f}, "
                f"budget_rate={oracle_compress.budget_compliance_rate:.3f}"
            ),
        ),
    )


def shortcut_report_digest(
    report: AntiShortcutBenchmarkReport | Dict[str, object],
    *,
    include_digest: bool = False,
) -> str:
    """Return a repeatability checksum, not an authenticity signature."""

    if isinstance(report, AntiShortcutBenchmarkReport):
        payload = report.model_dump(mode="json")
    else:
        payload = dict(report)
    if not include_digest:
        payload.pop("digest", None)
    return _canonical_payload_digest(payload)


def run_anti_shortcut_benchmark(
    *,
    seed: int = DEFAULT_SEED,
    stage1_task_id: str = DEFAULT_STAGE1_TASK_ID,
    task_dataset: Optional[ToyTaskDataset] = None,
    stage2_dataset: Optional[Stage2ChallengeDataset] = None,
) -> AntiShortcutBenchmarkReport:
    """Run both sidecars and derive gates without an Agent, LLM, or network."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    tasks = task_dataset or ToyTaskDataset.from_json()
    task = tasks.get(stage1_task_id)
    if not task.distractor_fact_ids:
        raise ValueError("Stage-1 shortcut task must contain a distractor")
    support_budget = sum(
        count_ltm_tokens(task.fact(fact_id).sentence)
        for fact_id in task.supporting_fact_ids
    )
    stage1_runner = Stage1StorageBenchmark(token_budget=support_budget)
    stage1_results = {}
    for policy in (StoreAllPolicy(), StoreNonePolicy(), OracleSafeStorePolicy()):
        result = stage1_runner.run(
            task,
            policy,
            rollout_id=f"shortcut-{task.task_id}-{policy.name}-seed-{seed}",
            seed=seed,
        )
        stage1_results[policy.name] = _stage1_metrics(
            result,
            supporting_fact_count=len(task.supporting_fact_ids),
        )

    stage2 = run_stage2_challenge_benchmark(stage2_dataset, seed=seed)
    gates = _derive_gates(stage1_results, stage2)
    payload = {
        "schema_version": SHORTCUT_REPORT_SCHEMA_VERSION,
        "seed": seed,
        "real_llm_call_count": 0,
        "stage1_task_id": task.task_id,
        "stage1_task_digest": _canonical_payload_digest(task.model_dump(mode="json")),
        "stage1_token_counter": LEXICAL_TOKEN_COUNTER_NAME,
        "stage1_budget_tokens": support_budget,
        "stage1": {
            name: metrics.model_dump(mode="json")
            for name, metrics in stage1_results.items()
        },
        "stage2": stage2.model_dump(mode="json"),
        "gates": [gate.model_dump(mode="json") for gate in gates],
        "passed": all(gate.passed for gate in gates),
    }
    payload["digest"] = shortcut_report_digest(payload)
    return AntiShortcutBenchmarkReport.model_validate(payload)


def anti_shortcut_markdown(report: AntiShortcutBenchmarkReport) -> str:
    """Render a compact evidence report without reproducing private text."""

    stage1_rows = "\n".join(
        "| {policy} | {recall:.3f} | {precision:.3f} | {tokens} | {rejected} |".format(
            policy=name,
            recall=row.supporting_recall,
            precision=row.memory_precision,
            tokens=row.stored_token_count,
            rejected=row.rejected_budget_count,
        )
        for name in EXPECTED_STAGE1_POLICIES
        for row in (report.stage1[name],)
    )
    stage2_rows = "\n".join(
        "| {policy} | {support:.3f} | {removed:.3f} | {budget:.3f} | {safe:.3f} |".format(
            policy=name,
            support=row.future_support_recall,
            removed=row.distractor_removal_recall,
            budget=row.budget_compliance_rate,
            safe=row.safe_success_rate,
        )
        for name in EXPECTED_STAGE2_POLICIES
        for row in (report.stage2.aggregates[name],)
    )
    gate_rows = "\n".join(
        f"| {gate.name} | {'PASS' if gate.passed else 'FAIL'} | {gate.evidence} |"
        for gate in report.gates
    )
    return f"""# Stage 1/2 Anti-Shortcut Benchmark

本报告由确定性 Toy/Oracle sidecar 生成，不调用 LLM、embedding 服务或网络，也不进入训练 buffer。

- Schema: `{report.schema_version}`
- Seed: `{report.seed}`
- Stage 1 task: `{report.stage1_task_id}`
- Stage 1 task digest: `{report.stage1_task_digest}`
- Stage 1 token counter: `{report.stage1_token_counter}`
- Stage 1 LTM budget: `{report.stage1_budget_tokens}` lexical tokens
- Stage 2 cases: `{report.stage2.case_count}`
- Stage 2 private dataset digest: `{report.stage2.dataset_digest}`
- Stage 2 token counter: `{report.stage2.token_counter}`
- Stage 2 budget scope: `{report.stage2.budget_scope}`
- Real LLM calls: `{report.real_llm_call_count}`
- Overall gate: `{"PASS" if report.passed else "FAIL"}`
- Repeatability checksum (not an authenticity signature): `{report.digest}`

## Stage 1

| Policy | Support recall | Memory precision | Stored tokens | Budget rejects |
|---|---:|---:|---:|---:|
{stage1_rows}

`oracle-safe-store` 是使用私有 supporting labels 的离线上界，不是可部署策略。

## Stage 2

| Policy | Future-support recall | Distractor removal | Budget compliance | Safe success |
|---|---:|---:|---:|---:|
{stage2_rows}

`oracle_safe_compress` 是使用私有 segment labels 的离线上界。公开 Stage 2 输入不包含 `task_id`、split、原始消息/segment ID、`future_query` / `future_answer`、场景类型或 Oracle role；每个 seed 使用与角色无关的不透明句柄。`opaque_id_control` 仅测试“保留字典序最小句柄”这一条固定 ID-only 规则，不能代表穷尽所有 ID-only 策略。Supporting message 正文仍可能包含未来答案事实，但当时没有查询可用于判断其相关性。

## Gates

| Gate | Result | Evidence |
|---|---|---|
{gate_rows}

## Evidence Boundary

该结果只证明当前构造能暴露 Store-All、Always-Keep、Always-Clear 和当前 min-ID 控制，并证明 Oracle 可行解存在。SHA-256 只用于确定性重复与输入绑定，不提供来源认证。该结果不代表已训练模型表现，也不证明真实 LLM 能达到 Oracle 上界。现有 E1 terminal-only 配置和 M3-M7 artifact 均未改写。
"""


def write_anti_shortcut_report(
    report: AntiShortcutBenchmarkReport,
    *,
    output_dir: str | Path,
    docs_path: Optional[str | Path] = None,
) -> Tuple[Path, Path]:
    """Persist canonical JSON and Markdown without non-deterministic metadata."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "anti_shortcut_benchmark.json"
    markdown_path = output / "anti_shortcut_benchmark.md"
    json_path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = anti_shortcut_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    if docs_path is not None:
        docs = Path(docs_path)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(markdown, encoding="utf-8", newline="\n")
    return json_path, markdown_path


__all__ = [
    "AntiShortcutBenchmarkReport",
    "DEFAULT_SEED",
    "DEFAULT_STAGE1_TASK_ID",
    "SHORTCUT_REPORT_SCHEMA_VERSION",
    "ShortcutGate",
    "Stage1ShortcutMetrics",
    "anti_shortcut_markdown",
    "run_anti_shortcut_benchmark",
    "shortcut_report_digest",
    "write_anti_shortcut_report",
]
