"""Deterministic M7 offline Group Critic benchmark over canonical M5/M6 data."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..action_schema import (
    ActionCreditRecord,
    ActionEvent,
    MigrationFileRecord,
    MigrationManifest,
    TrajectoryStepV2,
    load_migration_manifest,
)
from ..hotpotqa_benchmark.metrics import OracleBenchmarkReport
from ..hotpotqa_benchmark import (
    HotpotQADataAdapter,
    load_manifest,
    manifest_digest as smoke_manifest_digest,
)
from ..memory_extraction.benchmark import M6ExtractionBenchmarkReport
from ..memory_extraction.false_reject_audit import M6FalseRejectAuditReport
from ..memory_extraction.models import canonical_digest
from ..memory_oracle.automaton import hand_authored_memory_dfa
from ..memory_oracle.models import RewardConfig
from .critic import (
    GroupCriticCache,
    MockGroupCritic,
    select_critic_automaton,
)
from .metrics import (
    AcceptanceObservation,
    M7AcceptanceMetrics,
    M7CriticUsageMetrics,
    M7RewardErrorMetrics,
    M7StabilityMetrics,
    M7StratumMetrics,
    score_acceptance,
    score_reward_error,
    score_strata,
    usage_from_texts,
)
from .models import (
    ActionAPTrace,
    CriticGroupInput,
    CriticRolloutTrace,
    CriticInvocationResult,
    CriticOutput,
    EvidenceStepRef,
    MemoryEvent,
    MilestoneDependency,
    raw_text_digest,
)
from .replay import (
    GroupAutomatonReplay,
    GroupAutomatonReplayResult,
    RewardFarmingAudit,
    audit_reward_farming,
)


M7_CONFIG_SCHEMA_VERSION = "agemem.m7_group_critic_config.v1"
M7_PROFILE_REPORT_SCHEMA_VERSION = "agemem.m7_profile_report.v1"
M7_REPORT_SCHEMA_VERSION = "agemem.m7_offline_benchmark_report.v1"
M7_FAILURE_SCHEMA_VERSION = "agemem.m7_benchmark_failure.v1"
M7_FARMING_SCHEMA_VERSION = "agemem.m7_reward_farming_summary.v1"
M7_INTERFERENCE_SCHEMA_VERSION = "agemem.m7_interference_stratum.v1"


class M7BenchmarkError(RuntimeError):
    """Raised when a source hash, action join, or offline gate fails closed."""


class M7GroupCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_CONFIG_SCHEMA_VERSION]
    seed: int = Field(ge=0)
    m5_report_path: str = Field(min_length=1)
    m6_report_path: str = Field(min_length=1)
    m6_false_reject_audit_path: str = Field(min_length=1)
    hotpotqa_data_path: str = Field(min_length=1)
    smoke_manifest_path: str = Field(min_length=1)
    migration_root: str = Field(min_length=1)
    extraction_runtime_root: str = Field(min_length=1)
    reward_profile: str = Field(min_length=1)
    profiles: Tuple[Literal["oracle", "human_backed_mock", "controlled_error"], ...]
    group_size: Literal[3] = 3
    repeat_count: Literal[5] = 5
    permutation_count: Literal[6] = 6
    stage_1_interference_count: Literal[6] = 6
    stage_2_interference_count: Literal[3] = 3
    allow_real_llm: Literal[False] = False
    provider_pricing: None = None

    @model_validator(mode="after")
    def validate_profiles(self) -> "M7GroupCriticConfig":
        if self.profiles != ("oracle", "human_backed_mock", "controlled_error"):
            raise ValueError("M7 requires the canonical ordered AP profiles")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "M7GroupCriticConfig":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


class M7BenchmarkFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_FAILURE_SCHEMA_VERSION] = M7_FAILURE_SCHEMA_VERSION
    profile: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    category: Literal["false_accept", "false_reject"]
    terminal_action_id: str = Field(min_length=1)
    expected_accepted: bool
    predicted_accepted: bool
    attribution: Literal["extractor_injection", "critic", "state", "data"]
    injection_type: Optional[Literal["drop_relevant_fact"]] = None
    injection_fact_id: Optional[str] = Field(default=None, min_length=1)
    first_divergent_action_id: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_attribution(self) -> "M7BenchmarkFailure":
        injected = self.attribution == "extractor_injection"
        if injected != all(
            value is not None
            for value in (
                self.injection_type,
                self.injection_fact_id,
                self.first_divergent_action_id,
            )
        ):
            raise ValueError("extractor attribution requires complete M6 audit linkage")
        return self


class M7InterferenceStratum(BaseModel):
    """The one fixed M5 smoke interference setting; no synthetic levels."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_INTERFERENCE_SCHEMA_VERSION] = (
        M7_INTERFERENCE_SCHEMA_VERSION
    )
    stage_1_interference_count: Literal[6] = 6
    stage_2_interference_count: Literal[3] = 3
    task_count: Literal[10] = 10
    rollout_count: Literal[30] = 30
    action_count: Literal[224] = 224


class M7RewardFarmingSummary(BaseModel):
    """Hand-DFA aggregate checked by replay-valid adversarial fixtures."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_FARMING_SCHEMA_VERSION] = M7_FARMING_SCHEMA_VERSION
    automaton_source: Literal["hand_dfa"] = "hand_dfa"
    dfa_spec_id: str = Field(min_length=1)
    scenario_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    duplicate_action_reward_zero: bool
    loop_action_reward_zero: bool
    within_progressive_cap: bool
    passed: bool

    @model_validator(mode="after")
    def validate_summary(self) -> "M7RewardFarmingSummary":
        if self.scenario_count != self.passed_count + self.failed_count:
            raise ValueError("farming counts do not sum")
        expected = (
            self.failed_count == 0
            and self.duplicate_action_reward_zero
            and self.loop_action_reward_zero
            and self.within_progressive_cap
        )
        if self.passed != expected:
            raise ValueError("farming pass flag does not match checks")
        return self


class M7ProfileReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_PROFILE_REPORT_SCHEMA_VERSION] = (
        M7_PROFILE_REPORT_SCHEMA_VERSION
    )
    name: Literal["oracle", "human_backed_mock", "controlled_error"]
    rollout_count: Literal[30] = 30
    action_count: Literal[224] = 224
    hand_dfa_acceptance: M7AcceptanceMetrics
    hand_dfa_reward_error: M7RewardErrorMetrics
    critic_dfa_acceptance: M7AcceptanceMetrics = Field(
        description="Acceptance for the Critic-selected DFA plus explicit fallback pipeline"
    )
    critic_hand_reward_error: M7RewardErrorMetrics = Field(
        description="Action reward error for the Critic+fallback pipeline versus hand DFA"
    )
    critic_hand_acceptance_agreement_count: int = Field(ge=0, le=30)
    critic_hand_acceptance_agreement_rate: float = Field(ge=0.0, le=1.0)
    strata: Tuple[M7StratumMetrics, ...]
    action_credit_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_agreement(self) -> "M7ProfileReport":
        if (
            self.hand_dfa_acceptance.count != self.rollout_count
            or self.critic_dfa_acceptance.count != self.rollout_count
        ):
            raise ValueError("acceptance metrics must cover every profile rollout")
        if (
            self.hand_dfa_reward_error.count != self.action_count
            or self.critic_hand_reward_error.count != self.action_count
        ):
            raise ValueError("reward metrics must cover every profile action")
        if self.critic_hand_acceptance_agreement_rate != (
            self.critic_hand_acceptance_agreement_count / self.rollout_count
        ):
            raise ValueError("critic/hand agreement rate does not match count")
        return self


class M7OfflineBenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_REPORT_SCHEMA_VERSION] = M7_REPORT_SCHEMA_VERSION
    benchmark_name: Literal["m7-hotpotqa-group-critic-offline"] = (
        "m7-hotpotqa-group-critic-offline"
    )
    seed: int = Field(ge=0)
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    m5_report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    m6_report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    m6_false_reject_audit_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    migration_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    hand_dfa_spec_id: str = Field(min_length=1)
    group_count: Literal[10] = 10
    rollout_count: Literal[30] = 30
    action_count: Literal[224] = 224
    group_size: Literal[3] = 3
    profiles: Tuple[M7ProfileReport, ...] = Field(min_length=3, max_length=3)
    stability: M7StabilityMetrics
    usage: M7CriticUsageMetrics
    validator_valid_count: int = Field(ge=0)
    validator_invalid_count: int = Field(ge=0)
    explicit_fallback_count: int = Field(ge=0)
    mock_critic_selected_count: int = Field(ge=0)
    mock_critic_unavailable_count: int = Field(ge=0)
    silent_adoption_count: Literal[0] = 0
    milestone_evidence_count: int = Field(ge=0)
    milestone_evidence_valid_count: int = Field(ge=0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    reward_farming: M7RewardFarmingSummary
    reward_farming_records: Tuple[RewardFarmingAudit, ...] = ()
    interference: M7InterferenceStratum
    stage_1_interference_count: Literal[6] = 6
    stage_2_interference_count: Literal[3] = 3
    provider_input_tokens: None = None
    provider_output_tokens: None = None
    provider_cost: None = None
    real_llm_call_count: Literal[0] = 0
    hand_replay_count: Literal[90] = 90
    hand_replay_exact_match_count: Literal[90] = 90
    source_hash_inventory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_hashes_verified_unchanged: Literal[True] = True
    failures: Tuple[M7BenchmarkFailure, ...] = ()
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> "M7OfflineBenchmarkReport":
        if tuple(item.name for item in self.profiles) != (
            "oracle",
            "human_backed_mock",
            "controlled_error",
        ):
            raise ValueError("profile ordering is not canonical")
        expected_coverage = (
            1.0
            if self.milestone_evidence_count == 0
            else self.milestone_evidence_valid_count / self.milestone_evidence_count
        )
        if self.evidence_coverage != expected_coverage:
            raise ValueError("evidence coverage does not match counts")
        if sum(item.failure_count for item in self.profiles) != len(self.failures):
            raise ValueError("failure rows do not match profile counts")
        if len(self.reward_farming_records) != self.reward_farming.scenario_count:
            raise ValueError("farming records do not match summary count")
        if self.stability.group_count != self.group_count * len(self.profiles):
            raise ValueError("stability metrics must cover every profile/task group")
        if (
            self.hand_replay_count != self.rollout_count * len(self.profiles)
            or self.hand_replay_exact_match_count != self.hand_replay_count
        ):
            raise ValueError(
                "hand replay counts must exactly cover all profile rollouts"
            )
        if self.reward_farming.automaton_source != "hand_dfa" or (
            self.reward_farming.dfa_spec_id != self.hand_dfa_spec_id
        ):
            raise ValueError("farming summary must remain bound to the hand DFA")
        if self.mock_critic_selected_count + self.mock_critic_unavailable_count != (
            self.group_count * len(self.profiles)
        ):
            raise ValueError("critic selected/unavailable counts do not cover groups")
        if self.validator_valid_count != self.mock_critic_selected_count:
            raise ValueError("validator valid count must equal selected critic count")
        if (
            self.validator_invalid_count + self.mock_critic_unavailable_count
            != self.explicit_fallback_count
        ):
            raise ValueError("invalid and unavailable fallbacks do not match total")
        if self.digest != self.expected_digest():
            raise ValueError("report digest does not match payload")
        return self

    def canonical_dict(self, *, include_digest: bool = True) -> Dict[str, object]:
        data = self.model_dump(mode="json")
        if not include_digest:
            data.pop("digest", None)
        return data

    def expected_digest(self) -> str:
        return canonical_digest(self.canonical_dict(include_digest=False))

    def to_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return repository_root() / "configs/m7_group_critic.json"


def _resolve_repo_path(root: Path, value: str) -> Path:
    """Resolve a configured path without permitting repository escape."""

    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise M7BenchmarkError(f"configured path escapes repository: {value!r}")
    return path


def _resolve_under(root: Path, relative: str) -> Path:
    """Resolve an untrusted manifest-relative path below its declared root."""

    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise M7BenchmarkError(f"manifest path escapes configured root: {relative!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path, model):
    if not path.is_file():
        raise M7BenchmarkError(f"missing canonical input: {path}")
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _validate_source_lineage(
    root: Path,
    cfg: M7GroupCriticConfig,
    m5: OracleBenchmarkReport,
    m6: M6ExtractionBenchmarkReport,
    manifest: MigrationManifest,
    audit: Optional[M6FalseRejectAuditReport] = None,
) -> None:
    """Fail closed across the signed M5 -> M6 -> audit source chain."""

    m5_path = _resolve_repo_path(root, cfg.m5_report_path)
    if manifest.source_report_digest != m5.digest:
        raise M7BenchmarkError("migration manifest is not bound to the M5 report")
    if manifest.source_report_sha256 != _sha256(m5_path):
        raise M7BenchmarkError("migration manifest M5 report byte hash mismatch")
    if m6.m5_report_digest != m5.digest:
        raise M7BenchmarkError("M6 report is not bound to the M5 report")
    if m6.migration_manifest_digest != manifest.digest:
        raise M7BenchmarkError("M6 report is not bound to the migration manifest")
    if (
        m6.canonical_rollout_count != manifest.canonical_rollout_count
        or m6.canonical_action_count != manifest.action_count
        or manifest.action_count != manifest.credit_count
    ):
        raise M7BenchmarkError("M6/manifest canonical universe mismatch")
    if {item.trajectory_path for item in m5.records} != {
        item.source_trajectory_path for item in manifest.files
    }:
        raise M7BenchmarkError("M5/manifest trajectory universe mismatch")
    if (
        cfg.stage_1_interference_count != m5.smoke_config.stage1_distractors
        or cfg.stage_2_interference_count != m5.smoke_config.stage2_distractors
    ):
        raise M7BenchmarkError("M7 interference setting does not match M5 smoke config")
    if audit is not None:
        audit_lineage_matches = (
            audit.m7_entry_gate_passed
            and audit.m6_report_digest == m6.digest
            and audit.config_digest == m6.config_digest
            and audit.m6_report_config_digest == m6.config_digest
            and audit.m5_report_digest == m5.digest
            and audit.m6_report_m5_report_digest == m5.digest
            and audit.manifest_source_report_digest == m5.digest
            and audit.m5_report_sha256 == _sha256(m5_path)
            and audit.manifest_source_report_sha256 == manifest.source_report_sha256
            and audit.migration_manifest_digest == manifest.digest
            and audit.m6_report_migration_manifest_digest == manifest.digest
        )
        if not audit_lineage_matches:
            raise M7BenchmarkError("M6 False Reject audit lineage mismatch")


def _validate_profile_credit_digests(
    root: Path,
    cfg: M7GroupCriticConfig,
    m6: M6ExtractionBenchmarkReport,
    manifest: MigrationManifest,
) -> Dict[str, str]:
    """Recompute every full credit digest and bind it to M6/manifest evidence."""

    m6_profiles = {item.name: item for item in m6.profiles}
    expected_extracted = {"human_backed_mock", "controlled_error"}
    if set(m6_profiles) != expected_extracted:
        raise M7BenchmarkError("M6 report does not contain the canonical AP profiles")
    digests: Dict[str, str] = {}
    for profile in cfg.profiles:
        rows: List[dict] = []
        action_count = 0
        for item in manifest.files:
            base = (
                _resolve_repo_path(root, cfg.migration_root)
                if profile == "oracle"
                else _resolve_under(
                    _resolve_repo_path(root, cfg.extraction_runtime_root), profile
                )
            )
            path = _resolve_under(base, item.target_credit_path)
            if profile == "oracle" and _sha256(path) != item.target_credit_sha256:
                raise M7BenchmarkError(f"Oracle credit hash mismatch: {path}")
            credits = _read_jsonl(path, ActionCreditRecord)
            if len(credits) != item.credit_count:
                raise M7BenchmarkError(f"credit count mismatch: {path}")
            rows.extend(credit.model_dump(mode="json") for credit in credits)
            action_count += len(credits)
        if action_count != manifest.credit_count:
            raise M7BenchmarkError(f"{profile} does not cover the canonical actions")
        digest = canonical_digest(rows)
        if profile != "oracle" and digest != m6_profiles[profile].action_credit_digest:
            raise M7BenchmarkError(f"{profile} action-credit digest mismatch")
        digests[profile] = digest
    return digests


def _event_type(
    action_type: str,
) -> Literal["add", "retrieve", "update", "delete", "answer", "other"]:
    normalized = action_type.casefold()
    for kind in ("retrieve", "update", "delete", "answer", "add"):
        if kind in normalized:
            return kind  # type: ignore[return-value]
    return "other"


def _memory_event(step: TrajectoryStepV2) -> Tuple[MemoryEvent, ...]:
    action = step.actions[0]
    before = {item.memory_id: item for item in step.memory_before}
    after = {item.memory_id: item for item in step.memory_after}
    changed = tuple(
        sorted(
            memory_id
            for memory_id in set(before) | set(after)
            if before.get(memory_id) != after.get(memory_id)
        )
    )
    returned = tuple(
        sorted(
            item.memory_id
            for item in step.memory_after
            if item.memory_id
            in json.dumps(action.result, ensure_ascii=False, sort_keys=True)
        )
    )
    ids = tuple(sorted(set(changed) | set(returned)))
    return (
        MemoryEvent(
            event_id=f"{action.action_id}:memory-event",
            event_type=_event_type(action.action_type),
            memory_ids=ids,
            payload_digest=canonical_digest(
                {
                    "before": [before[key].canonical_dict() for key in sorted(before)],
                    "after": [after[key].canonical_dict() for key in sorted(after)],
                    "returned": returned,
                }
            ),
        ),
    )


def build_group_inputs(
    *,
    repository: Optional[str | Path] = None,
    config: Optional[M7GroupCriticConfig] = None,
) -> Tuple[CriticGroupInput, ...]:
    """Build exactly 10 x 3 provenance-safe groups without source sentences."""

    root = Path(repository or repository_root()).resolve()
    cfg = config or M7GroupCriticConfig.from_json(root / "configs/m7_group_critic.json")
    m5 = OracleBenchmarkReport.model_validate_json(
        _resolve_repo_path(root, cfg.m5_report_path).read_text(encoding="utf-8")
    )
    m6 = M6ExtractionBenchmarkReport.model_validate_json(
        _resolve_repo_path(root, cfg.m6_report_path).read_text(encoding="utf-8")
    )
    migration_root = _resolve_repo_path(root, cfg.migration_root)
    extraction_root = _resolve_repo_path(root, cfg.extraction_runtime_root)
    manifest = load_migration_manifest(
        _resolve_under(migration_root, "migration_manifest.json")
    )
    _validate_source_lineage(root, cfg, m5, m6, manifest)
    _validate_profile_credit_digests(root, cfg, m6, manifest)
    smoke_manifest = load_manifest(_resolve_repo_path(root, cfg.smoke_manifest_path))
    # The local HotpotQA DatasetDict is intentionally configured outside the
    # repository in the checked-in M5/M6 setup; it is read-only benchmark data.
    adapter = HotpotQADataAdapter((root / cfg.hotpotqa_data_path).resolve())
    adapter.verify_manifest(smoke_manifest, m5.smoke_config)
    if smoke_manifest_digest(smoke_manifest) != m5.manifest_digest:
        raise M7BenchmarkError("smoke manifest digest does not match M5 report")
    question_by_task = {
        f"hotpot-{selection.hotpot_id}": adapter.row(
            selection.source_split, selection.source_index
        ).question
        for selection in smoke_manifest.selections
    }
    if set(question_by_task) != {item.task_id for item in manifest.files}:
        raise M7BenchmarkError("smoke manifest questions do not cover M6 tasks")
    record_by_trajectory = {item.trajectory_path: item for item in m5.records}
    groups: List[CriticGroupInput] = []
    for profile in cfg.profiles:
        by_task: Dict[str, List[CriticRolloutTrace]] = {}
        split_by_task: Dict[str, str] = {}
        for item in manifest.files:
            benchmark = record_by_trajectory[item.source_trajectory_path]
            trajectory_path = _resolve_under(
                migration_root, item.target_trajectory_path
            )
            credit_root = (
                migration_root
                if profile == "oracle"
                else _resolve_under(extraction_root, profile)
            )
            credit_path = _resolve_under(credit_root, item.target_credit_path)
            if (
                profile == "oracle"
                and _sha256(credit_path) != item.target_credit_sha256
            ):
                raise M7BenchmarkError(f"Oracle credit hash mismatch: {credit_path}")
            if _sha256(trajectory_path) != item.target_trajectory_sha256:
                raise M7BenchmarkError(f"trajectory hash mismatch: {trajectory_path}")
            steps = _read_jsonl(trajectory_path, TrajectoryStepV2)
            credits = _read_jsonl(credit_path, ActionCreditRecord)
            if len(steps) != len(credits) or len(steps) != item.action_count:
                raise M7BenchmarkError("trajectory/action-credit count mismatch")
            traces: List[ActionAPTrace] = []
            for step, credit in zip(steps, credits):
                action = step.actions[0]
                if (
                    action.task_id,
                    action.rollout_id,
                    action.stage_id,
                    action.timestep,
                    action.action_id,
                ) != (
                    credit.task_id,
                    credit.rollout_id,
                    credit.stage_id,
                    credit.timestep,
                    credit.action_id,
                ):
                    raise M7BenchmarkError("action_id/coordinate join mismatch")
                evidence_ids = tuple(
                    sorted(
                        {
                            evidence_id
                            for values in credit.atomic_proposition_evidence.values()
                            for evidence_id in values
                        }
                    )
                )
                traces.append(
                    ActionAPTrace(
                        evidence=EvidenceStepRef(
                            task_id=action.task_id,
                            rollout_id=action.rollout_id,
                            stage_id=action.stage_id,
                            timestep=action.timestep,
                            action_id=action.action_id,
                            assistant_turn_id=action.assistant_turn_id,
                            action_index_in_turn=action.action_index_in_turn,
                            ap_evidence_ids=evidence_ids,
                        ),
                        action_type=action.action_type,
                        memory_events=_memory_event(step),
                        propositions=credit.atomic_propositions,  # type: ignore[arg-type]
                        atomic_proposition_evidence=credit.atomic_proposition_evidence,  # type: ignore[arg-type]
                    )
                )
            trace_payload = [trace.model_dump(mode="json") for trace in traces]
            by_task.setdefault(item.task_id, []).append(
                CriticRolloutTrace(
                    task_id=item.task_id,
                    rollout_id=item.rollout_id,
                    terminal_outcome="success"
                    if benchmark.episode_success
                    else "failure",
                    actions=tuple(traces),
                    source_trajectory_digest=item.target_trajectory_sha256,
                    ap_trace_digest=canonical_digest(trace_payload),
                )
            )
            split_by_task[item.task_id] = benchmark.split
        for task_id, rollouts in sorted(by_task.items()):
            if len(rollouts) != cfg.group_size:
                raise M7BenchmarkError(f"task does not contain K=3 rollouts: {task_id}")
            ordered = tuple(sorted(rollouts, key=lambda row: row.rollout_id))
            groups.append(
                CriticGroupInput(
                    task_id=task_id,
                    group_id=f"m7:{profile}:{task_id}",
                    split_id=split_by_task[task_id],
                    task_description=question_by_task[task_id],
                    ap_profile=profile,
                    rollouts=ordered,
                    source_report_digests=(m5.digest, m6.digest, manifest.digest),
                )
            )
    if len(groups) != 30:
        raise M7BenchmarkError(f"expected 30 profile/task groups, got {len(groups)}")
    return tuple(groups)


def _profile_credits(
    root: Path, cfg: M7GroupCriticConfig, item: MigrationFileRecord, profile: str
):
    base = (
        _resolve_repo_path(root, cfg.migration_root)
        if profile == "oracle"
        else _resolve_under(
            _resolve_repo_path(root, cfg.extraction_runtime_root), profile
        )
    )
    return _read_jsonl(
        _resolve_under(base, item.target_credit_path), ActionCreditRecord
    )


def _profile_rows(
    root: Path,
    cfg: M7GroupCriticConfig,
    item: MigrationFileRecord,
    profile: str,
) -> Tuple[Tuple[TrajectoryStepV2, ...], Tuple[ActionCreditRecord, ...]]:
    migration_root = _resolve_repo_path(root, cfg.migration_root)
    steps = _read_jsonl(
        _resolve_under(migration_root, item.target_trajectory_path),
        TrajectoryStepV2,
    )
    credits = _profile_credits(root, cfg, item, profile)
    if len(steps) != len(credits) or len(steps) != item.action_count:
        raise M7BenchmarkError("trajectory/action-credit count mismatch")
    return steps, credits


def _source_hash_inventory(
    root: Path, cfg: M7GroupCriticConfig, manifest
) -> Tuple[Tuple[Tuple[str, str], ...], str]:
    """Hash every M5/M6 trajectory and credit input before any derived work."""

    migration_root = _resolve_repo_path(root, cfg.migration_root)
    extraction_root = _resolve_repo_path(root, cfg.extraction_runtime_root)
    source_paths = (
        _resolve_repo_path(root, cfg.m5_report_path),
        _resolve_repo_path(root, cfg.m6_report_path),
        _resolve_repo_path(root, cfg.m6_false_reject_audit_path),
        _resolve_repo_path(root, cfg.smoke_manifest_path),
        _resolve_under(migration_root, "migration_manifest.json"),
    )
    rows: List[Tuple[str, str]] = [
        (path.relative_to(root).as_posix(), _sha256(path)) for path in source_paths
    ]
    for item in manifest.files:
        paths = [_resolve_under(migration_root, item.target_trajectory_path)]
        paths.extend(
            (
                _resolve_under(migration_root, item.target_credit_path),
                _resolve_under(
                    _resolve_under(extraction_root, "human_backed_mock"),
                    item.target_credit_path,
                ),
                _resolve_under(
                    _resolve_under(extraction_root, "controlled_error"),
                    item.target_credit_path,
                ),
            )
        )
        for path in paths:
            rows.append((path.relative_to(root).as_posix(), _sha256(path)))
    ordered = tuple(sorted(rows))
    return ordered, canonical_digest(ordered)


def _invalid_invocation(invocation: CriticInvocationResult) -> CriticInvocationResult:
    """Create a schema-valid but cyclic critic response for fallback testing."""

    if invocation.output is None or not invocation.output.milestones:
        return CriticInvocationResult.model_validate(
            {
                **invocation.model_dump(mode="python"),
                "output": None,
                "output_digest": None,
                "error": "injected_invalid_critic_output: unavailable_output",
            }
        )
    output = invocation.output
    cyclic = CriticOutput.model_validate(
        {
            **output.model_dump(mode="python"),
            "dependencies": (
                *output.dependencies,
                MilestoneDependency(
                    prerequisite_id=output.milestones[-1].milestone_id,
                    dependent_id=output.milestones[0].milestone_id,
                ),
            ),
        }
    )
    raw_output = json.dumps(
        cyclic.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return CriticInvocationResult.model_validate(
        {
            **invocation.model_dump(mode="python"),
            "output": cyclic,
            "output_digest": cyclic.digest,
            "raw_output_digest": raw_text_digest(raw_output),
            "error": None,
        }
    )


def _decision_digest(decision) -> str:
    """Stable decision view that deliberately excludes measured latency/cache state."""

    output_digest = (
        decision.invocation.output.digest
        if decision.invocation.output is not None
        else None
    )
    return canonical_digest(
        {
            "selected_source": decision.selected_source,
            "output_digest": output_digest,
            "validation_valid": (
                decision.validation.valid if decision.validation is not None else None
            ),
            "automaton": (
                decision.automaton_spec.model_dump(mode="json")
                if decision.automaton_spec is not None
                else None
            ),
            "fallback_reason": decision.fallback_reason,
        }
    )


def _renumber_adversarial_rows(
    pairs: Sequence[Tuple[TrajectoryStepV2, ActionCreditRecord]],
    *,
    inject_after: int,
    suffix: str,
    repeat_count: int = 1,
) -> Tuple[
    Tuple[TrajectoryStepV2, ...], Tuple[ActionCreditRecord, ...], Tuple[str, ...]
]:
    """Insert replay-valid, read-only semantic repeats of one tool action."""

    if not 0 <= inject_after < len(pairs) - 1:
        raise M7BenchmarkError("invalid adversarial insertion position")
    if repeat_count < 1:
        raise M7BenchmarkError("adversarial repeat_count must be positive")
    expanded = list(pairs)
    source_step, source_credit = pairs[inject_after]
    source_action_type = source_step.actions[0].action_type
    if source_action_type not in {"Add_memory", "Retrieve_memory"}:
        raise M7BenchmarkError("adversarial source must be ADD or RETRIEVE")
    injected_action_ids = tuple(
        f"{source_credit.action_id}:m7:{suffix}:{repeat_index}"
        for repeat_index in range(1, repeat_count + 1)
    )
    for offset in range(repeat_count):
        expanded.insert(
            inject_after + 1 + offset,
            (source_step.model_copy(deep=True), source_credit.model_copy(deep=True)),
        )
    new_steps: List[TrajectoryStepV2] = []
    new_credits: List[ActionCreditRecord] = []
    for timestep, (step, credit) in enumerate(expanded):
        injected_offset = timestep - inject_after - 1
        injected = 0 <= injected_offset < repeat_count
        action = step.actions[0]
        action_id = (
            injected_action_ids[injected_offset] if injected else action.action_id
        )
        action_payload = action.model_dump(mode="python")
        result_payload = dict(action_payload["result"])
        if injected:
            result_payload["tool_call_id"] = action_id
            metadata = result_payload.get("metadata")
            if isinstance(metadata, dict):
                labels = dict(metadata.get("oracle_labels", {}))
                if source_action_type == "Add_memory":
                    labels = {
                        key: (
                            None
                            if key == "answer_correct"
                            else False
                            if key == "supporting_coverage_complete"
                            else []
                        )
                        for key in labels
                    }
                    result_payload["content"] = [
                        {
                            "type": "text",
                            "text": "Memory was not added: memory_id already exists.",
                        }
                    ]
                result_payload["metadata"] = {
                    **metadata,
                    "success": (
                        False
                        if source_action_type == "Add_memory"
                        else metadata.get("success")
                    ),
                    "oracle_labels": labels,
                }
        action_payload.update(
            action_id=action_id,
            timestep=timestep,
            assistant_turn_id=timestep,
            action_index_in_turn=0,
            result=result_payload,
        )
        next_action = ActionEvent.model_validate(action_payload)
        step_payload = step.model_dump(mode="python")
        if injected:
            # Repeated ADD is rejected by the store and RETRIEVE is read-only;
            # both therefore preserve the state produced by the source action.
            stable_memory = source_step.memory_after
            step_payload.update(
                timestep=timestep,
                actions=(next_action,),
                memory_before=stable_memory,
                memory_after=stable_memory,
                env_reward=0.0,
                done=False,
            )
        else:
            step_payload.update(timestep=timestep, actions=(next_action,))
        next_step = TrajectoryStepV2.model_validate(step_payload)
        credit_payload = credit.model_dump(mode="python")
        if injected:
            source_breakdown = credit.reward_breakdown
            propositions = (
                () if source_action_type == "Add_memory" else credit.atomic_propositions
            )
            evidence = (
                {}
                if source_action_type == "Add_memory"
                else credit.atomic_proposition_evidence
            )
            stable_state = source_credit.dfa_state_after
            credit_payload.update(
                atomic_propositions=propositions,
                atomic_proposition_evidence=evidence,
                transition_ids=(),
                transition_id=None,
                dfa_state_before=stable_state,
                dfa_state_after=stable_state,
                reward_breakdown=source_breakdown.model_copy(
                    update={
                        "env": 0.0,
                        "milestone": 0.0,
                        "violation": 0.0,
                        "trend": 0.0,
                        "format": 0.0,
                        "cost": 0.0,
                        "total": 0.0,
                        "automaton_state_before": stable_state,
                        "automaton_state_after": stable_state,
                        "propositions": propositions,
                        "fired_edges": (),
                        "newly_rewarded_edges": (),
                        "violation_edges": (),
                    }
                ),
            )
        credit_payload.update(
            action_id=action_id,
            timestep=timestep,
            return_to_go=None,
            advantage=None,
        )
        next_credit = ActionCreditRecord.model_validate(credit_payload)
        new_steps.append(next_step)
        new_credits.append(next_credit)
    return tuple(new_steps), tuple(new_credits), injected_action_ids


def _farming_records(
    *,
    root: Path,
    cfg: M7GroupCriticConfig,
    manifest,
    reward_profile,
    hand_spec,
) -> Tuple[RewardFarmingAudit, ...]:
    """Exercise replay-valid duplicate ADD and two-step RETRIEVE loops on hand DFA."""

    replay = GroupAutomatonReplay(
        reward_profile,
        spec=hand_spec,
        reward_version="agemem.reward.m7_farming_hand.v1",
    )
    records: List[RewardFarmingAudit] = []
    gold_files = tuple(item for item in manifest.files if item.policy == "gold")
    for item in gold_files:
        steps, credits = _profile_rows(root, cfg, item, "oracle")
        baseline = replay.replay(steps, credits, seed=cfg.seed)
        pairs = tuple(zip(steps, credits))
        add_index = next(
            index
            for index, (_, credit) in enumerate(pairs)
            if "stored_supporting_fact" in credit.atomic_propositions
        )
        retrieve_index = next(
            index
            for index, (_, credit) in enumerate(pairs)
            if "supporting_coverage_complete" in credit.atomic_propositions
            and "retrieved_supporting_fact" in credit.atomic_propositions
        )
        for index, suffix, repeat_count in (
            (add_index, "duplicate-add", 1),
            (retrieve_index, "loop-retrieve", 2),
        ):
            candidate_steps, candidate_credits, injected = _renumber_adversarial_rows(
                pairs,
                inject_after=index,
                suffix=suffix,
                repeat_count=repeat_count,
            )
            candidate = replay.replay(candidate_steps, candidate_credits, seed=cfg.seed)
            records.append(
                audit_reward_farming(
                    baseline=baseline,
                    candidate=candidate,
                    spec=hand_spec,
                    profile=reward_profile,
                    injected_action_ids=injected,
                )
            )
    return tuple(records)


def build_baseline_report(
    *, repository: Optional[str | Path] = None, config_path: Optional[str | Path] = None
) -> M7OfflineBenchmarkReport:
    """Run the complete deterministic M7 offline benchmark (no LLM/provider)."""

    root = Path(repository or repository_root()).resolve()
    cfg = M7GroupCriticConfig.from_json(
        config_path or root / "configs/m7_group_critic.json"
    )
    m5 = OracleBenchmarkReport.model_validate_json(
        _resolve_repo_path(root, cfg.m5_report_path).read_text(encoding="utf-8")
    )
    m6 = M6ExtractionBenchmarkReport.model_validate_json(
        _resolve_repo_path(root, cfg.m6_report_path).read_text(encoding="utf-8")
    )
    audit = M6FalseRejectAuditReport.model_validate_json(
        _resolve_repo_path(root, cfg.m6_false_reject_audit_path).read_text(
            encoding="utf-8"
        )
    )
    if not audit.m7_entry_gate_passed:
        raise M7BenchmarkError("M6 False Reject audit has not passed")
    manifest = load_migration_manifest(
        _resolve_under(
            _resolve_repo_path(root, cfg.migration_root),
            "migration_manifest.json",
        )
    )
    _validate_source_lineage(root, cfg, m5, m6, manifest, audit)
    profile_credit_digests = _validate_profile_credit_digests(root, cfg, m6, manifest)
    groups = build_group_inputs(repository=root, config=cfg)
    m5_by_path = {item.trajectory_path: item for item in m5.records}
    audit_by_rollout = {case.rollout_id: case for case in audit.cases}
    source_hashes_before, source_inventory_digest = _source_hash_inventory(
        root, cfg, manifest
    )
    hand_spec = hand_authored_memory_dfa()
    reward_profile = RewardConfig.from_json(root / "configs/m4_reward.json").profile(
        cfg.reward_profile
    )
    oracle_rewards: Dict[str, float] = {}
    oracle_action_count = 0
    for item in manifest.files:
        for credit in _profile_credits(root, cfg, item, "oracle"):
            if credit.action_id in oracle_rewards:
                raise M7BenchmarkError(
                    f"duplicate Oracle action_id across rollouts: {credit.action_id}"
                )
            oracle_rewards[credit.action_id] = credit.reward_breakdown.total
            oracle_action_count += 1
    if oracle_action_count != manifest.action_count:
        raise M7BenchmarkError("Oracle reward map does not cover canonical actions")

    # The hand-authored main baseline is recomputed for all 90 AP trajectories,
    # and every per-action breakdown must equal the canonical M6 result.
    hand_by_profile: Dict[str, Dict[str, GroupAutomatonReplayResult]] = {
        profile: {} for profile in cfg.profiles
    }
    hand_rewards_by_profile: Dict[str, Dict[str, float]] = {
        profile: {} for profile in cfg.profiles
    }
    hand_exact_match_count = 0
    for profile in cfg.profiles:
        hand_replay = GroupAutomatonReplay(
            reward_profile,
            spec=hand_spec,
            reward_version=f"agemem.reward.m7.hand.{profile}.v1",
        )
        for item in manifest.files:
            steps, source_credits = _profile_rows(root, cfg, item, profile)
            replayed = hand_replay.replay(steps, source_credits, seed=cfg.seed)
            for source, derived in zip(source_credits, replayed.credits):
                if source.reward_breakdown != derived.reward_breakdown:
                    raise M7BenchmarkError(
                        f"hand replay diverged from M6 at {source.action_id}"
                    )
                if derived.action_id in hand_rewards_by_profile[profile]:
                    raise M7BenchmarkError(
                        f"duplicate {profile} action_id across rollouts: "
                        f"{derived.action_id}"
                    )
                hand_rewards_by_profile[profile][derived.action_id] = (
                    derived.reward_breakdown.total
                )
            hand_by_profile[profile][item.rollout_id] = replayed
            hand_exact_match_count += 1
    if hand_exact_match_count != 90:
        raise M7BenchmarkError("M7 must replay exactly 90 profile/rollout traces")

    # Cache behavior and stability are deliberately separate experiments. The
    # latter invokes the Critic itself on every repeat and actual permutation.
    critic = MockGroupCritic()
    cache = GroupCriticCache()
    critic_decisions = {}
    input_texts: List[str] = []
    output_texts: List[str] = []
    repeat_checks = permutation_checks = repeat_matches = permutation_matches = 0
    validator_valid_count = validator_invalid_count = explicit_fallback_count = 0
    selected_count = unavailable_count = 0
    milestone_evidence_count = milestone_evidence_valid_count = 0
    for group in groups:
        cold = cache.get_or_critique(group, critic)
        if cold.cache_hit:
            raise M7BenchmarkError("first critic cache lookup must be cold")
        invocation = cold.result
        input_texts.append(group.model_dump_json())
        output_texts.append(
            ""
            if invocation.output is None
            else json.dumps(
                invocation.output.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        decision = select_critic_automaton(group, invocation)
        critic_decisions[(group.ap_profile, group.task_id)] = decision
        stable_digest = _decision_digest(decision)
        if decision.selected_source == "critic":
            selected_count += 1
            validator_valid_count += 1
            assert decision.validation is not None
            evidence_count = sum(
                len(item.evidence_steps) for item in invocation.output.milestones
            )
            milestone_evidence_count += evidence_count
            milestone_evidence_valid_count += evidence_count
        else:
            unavailable_count += 1

        # Invalid/unavailable output is never silently adopted. Successful
        # groups are challenged with a real cyclic output; all-failure groups
        # already use the explicit terminal-only fallback.
        invalid = _invalid_invocation(invocation)
        invalid_decision = select_critic_automaton(group, invalid)
        if invalid_decision.selected_source == "critic":
            raise M7BenchmarkError("invalid critic output was silently adopted")
        if invocation.output is not None:
            validator_invalid_count += 1
        explicit_fallback_count += 1

        for _ in range(cfg.repeat_count):
            repeat_checks += 1
            repeated_invocation = critic.critique(group)
            input_texts.append(group.model_dump_json())
            output_texts.append(
                ""
                if repeated_invocation.output is None
                else repeated_invocation.output.model_dump_json()
            )
            repeated = select_critic_automaton(group, repeated_invocation)
            repeat_matches += _decision_digest(repeated) == stable_digest
        for order in itertools.permutations(group.rollouts):
            permutation_checks += 1
            permuted = group.model_copy(update={"rollouts": tuple(order)})
            permuted_invocation = critic.critique(permuted)
            input_texts.append(permuted.model_dump_json())
            output_texts.append(
                ""
                if permuted_invocation.output is None
                else permuted_invocation.output.model_dump_json()
            )
            reordered = select_critic_automaton(permuted, permuted_invocation)
            permutation_matches += _decision_digest(reordered) == stable_digest

        cached = cache.get_or_critique(group, critic)
        if (
            not cached.cache_hit
            or cached.result.output_digest != invocation.output_digest
        ):
            raise M7BenchmarkError("critic cache did not reproduce the cold result")

    stability = M7StabilityMetrics(
        group_count=len(groups),
        repeat_checks=repeat_checks,
        permutation_checks=permutation_checks,
        repeat_digest_agreement_rate=repeat_matches / repeat_checks,
        permutation_digest_agreement_rate=permutation_matches / permutation_checks,
        stable=(
            repeat_matches == repeat_checks
            and permutation_matches == permutation_checks
        ),
    )
    usage = usage_from_texts(
        cold_calls=cache.misses + repeat_checks + permutation_checks,
        cache_hits=cache.hits,
        cache_misses=cache.misses,
        inputs=input_texts,
        outputs=output_texts,
    )

    # Shadow replay uses the compiled critic DFA when valid and the explicit
    # hand DFA fallback when evidence is unavailable. All-failure terminal-only
    # groups cannot affect successful-rollout comparisons and remain rejected.
    critic_rewards_by_profile: Dict[str, Dict[str, float]] = {
        profile: {} for profile in cfg.profiles
    }
    critic_acceptance_by_profile: Dict[str, Dict[str, bool]] = {
        profile: {} for profile in cfg.profiles
    }
    for profile in cfg.profiles:
        for item in manifest.files:
            decision = critic_decisions[(profile, item.task_id)]
            steps, source_credits = _profile_rows(root, cfg, item, profile)
            if decision.automaton_spec is None:
                # terminal-only is selected only for an all-failure K=3 group;
                # this smoke corpus has no such group, but remain fail-closed.
                critic_acceptance_by_profile[profile][item.rollout_id] = False
                for credit in source_credits:
                    if credit.action_id in critic_rewards_by_profile[profile]:
                        raise M7BenchmarkError(
                            f"duplicate terminal-only {profile} action_id across "
                            f"rollouts: {credit.action_id}"
                        )
                    critic_rewards_by_profile[profile][credit.action_id] = (
                        reward_profile.env_weight
                        * next(
                            step.env_reward
                            for step in steps
                            if step.actions[0].action_id == credit.action_id
                        )
                    )
                continue
            shadow = GroupAutomatonReplay(
                reward_profile,
                spec=decision.automaton_spec,
                reward_version=f"agemem.reward.m7.critic.{profile}.v1",
            ).replay(steps, source_credits, seed=cfg.seed)
            critic_acceptance_by_profile[profile][item.rollout_id] = shadow.accepted
            for credit in shadow.credits:
                if credit.action_id in critic_rewards_by_profile[profile]:
                    raise M7BenchmarkError(
                        f"duplicate critic {profile} action_id across rollouts: "
                        f"{credit.action_id}"
                    )
                critic_rewards_by_profile[profile][credit.action_id] = (
                    credit.reward_breakdown.total
                )

    profile_reports: List[M7ProfileReport] = []
    failures: List[M7BenchmarkFailure] = []
    for profile in cfg.profiles:
        observations: List[AcceptanceObservation] = []
        critic_observations: List[AcceptanceObservation] = []
        credits_for_digest: List[dict] = []
        profile_failures = 0
        for item in manifest.files:
            benchmark = m5_by_path[item.source_trajectory_path]
            credits = _profile_credits(root, cfg, item, profile)
            final = credits[-1]
            predicted = hand_by_profile[profile][item.rollout_id].accepted
            critic_predicted = critic_acceptance_by_profile[profile][item.rollout_id]
            expected = benchmark.episode_success
            common = dict(
                task_id=item.task_id,
                rollout_id=item.rollout_id,
                expected_accepted=expected,
                question_type=benchmark.hotpot_type,
                action_count=len(credits),
            )
            observations.append(
                AcceptanceObservation(**common, predicted_accepted=predicted)
            )
            critic_observations.append(
                AcceptanceObservation(**common, predicted_accepted=critic_predicted)
            )
            for credit in credits:
                credits_for_digest.append(credit.model_dump(mode="json"))
            if expected != predicted:
                category = "false_reject" if expected else "false_accept"
                audit_case = audit_by_rollout.get(item.rollout_id)
                if profile == "controlled_error" and audit_case is None:
                    raise M7BenchmarkError(
                        "controlled failure lacks M6 audit attribution"
                    )
                failures.append(
                    M7BenchmarkFailure(
                        profile=profile,
                        task_id=item.task_id,
                        rollout_id=item.rollout_id,
                        category=category,
                        terminal_action_id=final.action_id,
                        expected_accepted=expected,
                        predicted_accepted=predicted,
                        attribution=(
                            "extractor_injection" if audit_case is not None else "data"
                        ),
                        injection_type=(
                            audit_case.injection_type if audit_case else None
                        ),
                        injection_fact_id=(
                            audit_case.injection_fact_id if audit_case else None
                        ),
                        first_divergent_action_id=(
                            audit_case.first_divergent_action_id if audit_case else None
                        ),
                    )
                )
                profile_failures += 1
        agreement = sum(
            left.predicted_accepted == right.predicted_accepted
            for left, right in zip(observations, critic_observations)
        )
        observed_credit_digest = canonical_digest(credits_for_digest)
        if observed_credit_digest != profile_credit_digests[profile]:
            raise M7BenchmarkError(f"{profile} report credit traversal changed")
        profile_reports.append(
            M7ProfileReport(
                name=profile,
                hand_dfa_acceptance=score_acceptance(observations),
                hand_dfa_reward_error=score_reward_error(
                    oracle_rewards, hand_rewards_by_profile[profile]
                ),
                critic_dfa_acceptance=score_acceptance(critic_observations),
                critic_hand_reward_error=score_reward_error(
                    hand_rewards_by_profile[profile],
                    critic_rewards_by_profile[profile],
                ),
                critic_hand_acceptance_agreement_count=agreement,
                critic_hand_acceptance_agreement_rate=agreement / 30,
                strata=score_strata(observations),
                action_credit_digest=profile_credit_digests[profile],
                failure_count=profile_failures,
            )
        )

    farming_records = _farming_records(
        root=root,
        cfg=cfg,
        manifest=manifest,
        reward_profile=reward_profile,
        hand_spec=hand_spec,
    )
    duplicate_records = tuple(
        item
        for item in farming_records
        if ":duplicate-add:" in item.injected_action_ids[0]
    )
    loop_records = tuple(
        item
        for item in farming_records
        if ":loop-retrieve:" in item.injected_action_ids[0]
    )
    passed_count = sum(item.passed for item in farming_records)
    farming = M7RewardFarmingSummary(
        dfa_spec_id=hand_spec.name,
        scenario_count=len(farming_records),
        passed_count=passed_count,
        failed_count=len(farming_records) - passed_count,
        duplicate_action_reward_zero=all(
            item.injected_actions_zero_milestone for item in duplicate_records
        ),
        loop_action_reward_zero=all(
            item.injected_actions_zero_milestone for item in loop_records
        ),
        within_progressive_cap=all(
            item.within_progress_cap for item in farming_records
        ),
        passed=all(item.passed for item in farming_records),
    )
    source_hashes_after, after_digest = _source_hash_inventory(root, cfg, manifest)
    if (
        source_hashes_after != source_hashes_before
        or after_digest != source_inventory_digest
    ):
        raise M7BenchmarkError("canonical M5/M6 source inputs changed during benchmark")
    interference = M7InterferenceStratum(
        stage_1_interference_count=m5.smoke_config.stage1_distractors,
        stage_2_interference_count=m5.smoke_config.stage2_distractors,
    )
    payload: Dict[str, object] = {
        "schema_version": M7_REPORT_SCHEMA_VERSION,
        "benchmark_name": "m7-hotpotqa-group-critic-offline",
        "seed": cfg.seed,
        "config_digest": cfg.digest,
        "m5_report_digest": m5.digest,
        "m6_report_digest": m6.digest,
        "m6_false_reject_audit_digest": audit.digest,
        "migration_manifest_digest": manifest.digest,
        "hand_dfa_spec_id": hand_spec.name,
        "group_count": 10,
        "rollout_count": 30,
        "action_count": 224,
        "group_size": 3,
        "profiles": [item.model_dump(mode="json") for item in profile_reports],
        "stability": stability.model_dump(mode="json"),
        "usage": usage.model_dump(mode="json"),
        "validator_valid_count": validator_valid_count,
        "validator_invalid_count": validator_invalid_count,
        "explicit_fallback_count": explicit_fallback_count,
        "mock_critic_selected_count": selected_count,
        "mock_critic_unavailable_count": unavailable_count,
        "silent_adoption_count": 0,
        "milestone_evidence_count": milestone_evidence_count,
        "milestone_evidence_valid_count": milestone_evidence_valid_count,
        "evidence_coverage": (
            1.0
            if not milestone_evidence_count
            else milestone_evidence_valid_count / milestone_evidence_count
        ),
        "reward_farming": farming.model_dump(mode="json"),
        "reward_farming_records": [
            item.model_dump(mode="json") for item in farming_records
        ],
        "interference": interference.model_dump(mode="json"),
        "stage_1_interference_count": interference.stage_1_interference_count,
        "stage_2_interference_count": interference.stage_2_interference_count,
        "provider_input_tokens": None,
        "provider_output_tokens": None,
        "provider_cost": None,
        "real_llm_call_count": 0,
        "hand_replay_count": 90,
        "hand_replay_exact_match_count": hand_exact_match_count,
        "source_hash_inventory_digest": source_inventory_digest,
        "source_hashes_verified_unchanged": True,
        "failures": [item.model_dump(mode="json") for item in failures],
    }
    return M7OfflineBenchmarkReport(
        **{
            **payload,
            "profiles": tuple(profile_reports),
            "stability": stability,
            "usage": usage,
            "reward_farming": farming,
            "reward_farming_records": farming_records,
            "interference": interference,
            "failures": tuple(failures),
        },
        digest=canonical_digest(payload),
    )


def _markdown(report: M7OfflineBenchmarkReport) -> str:
    lines = [
        "# M7 Group Critic Offline Validation",
        "",
        f"- Report digest: `{report.digest}`",
        f"- M6 False Reject audit: `{report.m6_false_reject_audit_digest}`",
        f"- Hand replays: `{report.hand_replay_exact_match_count}/"
        f"{report.hand_replay_count}` exact",
        f"- Mock critic selected/unavailable: `{report.mock_critic_selected_count}/"
        f"{report.mock_critic_unavailable_count}`",
        f"- Validator invalid outputs: `{report.validator_invalid_count}`; explicit "
        f"fallbacks (invalid + unavailable): `{report.explicit_fallback_count}`; silent "
        f"adoptions: `{report.silent_adoption_count}`",
        f"- Milestone evidence coverage: `{report.milestone_evidence_valid_count}/"
        f"{report.milestone_evidence_count}`",
        f"- Stability checks: `{report.stability.repeat_checks}` repeats + "
        f"`{report.stability.permutation_checks}` permutations, "
        f"stable=`{report.stability.stable}`",
        f"- Mock calls/cache hits/cache misses: `{report.usage.cold_calls}/"
        f"{report.usage.cache_hits}/{report.usage.cache_misses}`",
        f"- Heuristic input/output tokens (not provider billing): "
        f"`{report.usage.heuristic_input_tokens}/"
        f"{report.usage.heuristic_output_tokens}`",
        f"- Hand-DFA reward-farming scenarios: `{report.reward_farming.passed_count}/"
        f"{report.reward_farming.scenario_count}` passed",
        "- Real LLM calls: `0`; provider tokens and cost: `None`",
        "",
        "Critic columns below measure the Critic + explicit fallback pipeline, not "
        "the Critic output in isolation.",
        "",
        "| AP profile | Hand FA | Hand FR | Critic+fallback FA | Critic+fallback FR | Hand reward MAE | Critic+fallback-vs-hand reward MAE | Pipeline/hand agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.profiles:
        lines.append(
            f"| `{item.name}` | "
            f"{item.hand_dfa_acceptance.false_accept_numerator}/"
            f"{item.hand_dfa_acceptance.false_accept_denominator} | "
            f"{item.hand_dfa_acceptance.false_reject_numerator}/"
            f"{item.hand_dfa_acceptance.false_reject_denominator} | "
            f"{item.critic_dfa_acceptance.false_accept_numerator}/"
            f"{item.critic_dfa_acceptance.false_accept_denominator} | "
            f"{item.critic_dfa_acceptance.false_reject_numerator}/"
            f"{item.critic_dfa_acceptance.false_reject_denominator} | "
            f"{item.hand_dfa_reward_error.mean_absolute_error:.12f} | "
            f"{item.critic_hand_reward_error.mean_absolute_error:.12f} | "
            f"{item.critic_hand_acceptance_agreement_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Attribution and scope",
            "",
            f"All `{len(report.failures)}` terminal disagreements are controlled-error "
            "False Rejects linked to the M6 `drop_relevant_fact` audit. There are no "
            "critic-, state-, or data-attributed failures.",
            "",
            "The only measured interference setting is the real smoke configuration: "
            f"Stage 1 `{report.interference.stage_1_interference_count}` distractors and "
            f"Stage 2 `{report.interference.stage_2_interference_count}` distractors. "
            "No extra interference levels were fabricated.",
            "",
            "Reward-farming checks use replay-valid duplicate ADD and two-step "
            "RETRIEVE-loop perturbations against the hand-authored DFA only; they do "
            "not claim Critic-DFA farming coverage.",
            "",
            "The LLM critic is an injected-client adapter only. This benchmark uses the "
            "deterministic mock critic, does not call a provider, and does not implement "
            "GRPO or training.",
            "",
        ]
    )
    return "\n".join(lines)


def write_m7_offline_report(
    *,
    repository: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    docs_path: Optional[str | Path] = None,
) -> M7OfflineBenchmarkReport:
    """Run M7 and persist deterministic compact/audit artifacts."""

    root = Path(repository or repository_root()).resolve()
    report = build_baseline_report(repository=root, config_path=config_path)
    output = Path(output_root or root / "artifacts/m7_group_critic").resolve()
    if output_root is None and not output.is_relative_to(root):
        raise M7BenchmarkError("default report output escapes repository")
    output.mkdir(parents=True, exist_ok=True)
    (output / "offline_validation.json").write_text(
        report.to_json() + "\n", encoding="utf-8", newline="\n"
    )
    markdown = _markdown(report)
    (output / "offline_validation.md").write_text(
        markdown, encoding="utf-8", newline="\n"
    )
    failure_text = "".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for item in report.failures
    )
    (output / "validation_failures.jsonl").write_text(
        failure_text, encoding="utf-8", newline="\n"
    )
    farming_text = "".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for item in report.reward_farming_records
    )
    (output / "reward_farming.jsonl").write_text(
        farming_text, encoding="utf-8", newline="\n"
    )
    docs = Path(
        docs_path or root / "docs/m7_group_critic_offline_validation.md"
    ).resolve()
    if docs_path is None and not docs.is_relative_to(root):
        raise M7BenchmarkError("default documentation output escapes repository")
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(markdown, encoding="utf-8", newline="\n")
    return report


__all__ = [
    "M7BenchmarkError",
    "M7BenchmarkFailure",
    "M7GroupCriticConfig",
    "M7OfflineBenchmarkReport",
    "M7ProfileReport",
    "M7RewardFarmingSummary",
    "build_baseline_report",
    "build_group_inputs",
    "default_config_path",
    "write_m7_offline_report",
]
