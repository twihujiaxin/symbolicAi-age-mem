"""Deterministic M6 controlled-error False Reject audit.

This module is deliberately downstream of the immutable M5/M6 artifacts.  It
does not rerun extraction or invent missing provenance; it joins the checked-in
configuration/annotations with the canonical action-level AP and reward files.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..action_schema import (
    ActionCreditRecord,
    TrajectoryStepV2,
    load_migration_manifest,
)
from ..hotpotqa_benchmark import OracleBenchmarkReport
from ..memory_oracle import (
    DFARunner,
    OracleAPEvent,
    RewardConfig,
    RewardProfile,
    hand_authored_memory_dfa,
)
from .annotations import load_annotation_corpus
from .benchmark import M6BenchmarkConfig, M6ExtractionBenchmarkReport, repository_root
from .models import APRecord, canonical_digest
from .state import STATE_FACT_SCHEMA_VERSION


M6_FALSE_REJECT_AUDIT_SCHEMA_VERSION = "agemem.m6_false_reject_audit.v2"
M6_FALSE_REJECT_CASE_SCHEMA_VERSION = "agemem.m6_false_reject_case.v2"
M6_FALSE_REJECT_STEP_SCHEMA_VERSION = "agemem.m6_false_reject_step.v1"


class FalseRejectAuditError(RuntimeError):
    """Raised when existing M6 artifacts cannot support a complete audit."""


class RewardPropagationStep(BaseModel):
    """One action where Oracle and controlled-error AP/reward traces differ."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M6_FALSE_REJECT_STEP_SCHEMA_VERSION] = (
        M6_FALSE_REJECT_STEP_SCHEMA_VERSION
    )
    action_id: str = Field(min_length=1)
    timestep: int = Field(ge=0)
    oracle_aps: Tuple[str, ...]
    extracted_aps: Tuple[str, ...]
    oracle_only_aps: Tuple[str, ...]
    extracted_only_aps: Tuple[str, ...]
    oracle_state_before: str = Field(min_length=1)
    oracle_state_after: str = Field(min_length=1)
    extracted_state_before: str = Field(min_length=1)
    extracted_state_after: str = Field(min_length=1)
    oracle_edges: Tuple[str, ...]
    extracted_edges: Tuple[str, ...]
    reward_error: float


class FalseRejectCase(BaseModel):
    """Complete explanation for one controlled-error False Reject rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M6_FALSE_REJECT_CASE_SCHEMA_VERSION] = (
        M6_FALSE_REJECT_CASE_SCHEMA_VERSION
    )
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    injection_type: Literal["drop_relevant_fact"]
    injection_fact_id: str = Field(min_length=1)
    missing_triples: Tuple[Tuple[str, str, str], ...] = Field(min_length=1)
    first_divergent_action_id: str = Field(min_length=1)
    first_divergent_timestep: int = Field(ge=0)
    first_divergent_assistant_turn_id: int = Field(ge=0)
    first_divergent_action_index_in_turn: int = Field(ge=0)
    oracle_ap_trace: Tuple[Tuple[str, ...], ...] = Field(min_length=1)
    extracted_ap_trace: Tuple[Tuple[str, ...], ...] = Field(min_length=1)
    propagation: Tuple[RewardPropagationStep, ...] = Field(min_length=1)
    missing_state_fact_ids: Tuple[str, ...] = Field(min_length=1)
    expected_state_fact_ids: Tuple[str, ...] = Field(min_length=1)
    grounding_evidence_ap_ids: Tuple[str, ...] = Field(min_length=1)
    dfa_checked_action_count: int = Field(ge=1)
    state_tracker_correct: bool
    ap_grounding_correct: bool
    action_id_alignment_correct: bool
    dfa_definition_too_strict: bool
    dfa_implementation_correct: bool
    oracle_final_status: Literal["accepted"] = "accepted"
    extracted_final_status: Literal["rejected"] = "rejected"
    oracle_total_reward: float
    extracted_total_reward: float
    total_reward_error: float
    classification: Literal["expected_extractor_omission"] = (
        "expected_extractor_omission"
    )
    causal_chain_complete: bool

    @model_validator(mode="after")
    def validate_rewards(self) -> "FalseRejectCase":
        if not math.isclose(
            self.extracted_total_reward - self.oracle_total_reward,
            self.total_reward_error,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("trajectory reward error does not match totals")
        if self.propagation[0].action_id != self.first_divergent_action_id:
            raise ValueError("first divergent action must lead propagation trace")
        if self.missing_state_fact_ids != self.expected_state_fact_ids:
            raise ValueError(
                "missing StateFacts must exactly match canonical identities"
            )
        expected_chain = (
            self.state_tracker_correct
            and self.ap_grounding_correct
            and self.action_id_alignment_correct
            and self.dfa_implementation_correct
            and not self.dfa_definition_too_strict
            and self.oracle_final_status == "accepted"
            and self.extracted_final_status == "rejected"
            and bool(self.grounding_evidence_ap_ids)
            and self.dfa_checked_action_count == 2 * len(self.oracle_ap_trace)
            and len(self.oracle_ap_trace) == len(self.extracted_ap_trace)
        )
        if self.causal_chain_complete != expected_chain:
            raise ValueError("causal_chain_complete does not match computed evidence")
        return self


class M6FalseRejectAuditReport(BaseModel):
    """Source-text-free gate proving all M6 controlled FRs are explained."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M6_FALSE_REJECT_AUDIT_SCHEMA_VERSION] = (
        M6_FALSE_REJECT_AUDIT_SCHEMA_VERSION
    )
    benchmark_name: Literal["m6-controlled-error-false-reject-audit"] = (
        "m6-controlled-error-false-reject-audit"
    )
    m6_report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    m6_report_config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    m5_report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    m6_report_m5_report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    m5_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    migration_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    m6_report_migration_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_source_report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_false_accept_numerator: Literal[0] = 0
    human_false_reject_numerator: Literal[0] = 0
    controlled_false_accept_numerator: Literal[0] = 0
    controlled_false_reject_numerator: Literal[5] = 5
    controlled_false_reject_denominator: Literal[10] = 10
    relevant_drop_fact_ids: Tuple[str, ...] = Field(min_length=5, max_length=5)
    irrelevant_corrupt_fact_ids: Tuple[str, ...]
    cases: Tuple[FalseRejectCase, ...] = Field(min_length=5, max_length=5)
    unexplained_count: int = Field(ge=0)
    state_tracker_error_count: int = Field(ge=0)
    ap_grounding_error_count: int = Field(ge=0)
    action_alignment_error_count: int = Field(ge=0)
    dfa_implementation_error_count: int = Field(ge=0)
    dfa_checked_action_count: int = Field(ge=1)
    m7_entry_gate_passed: bool
    real_llm_call_count: Literal[0] = 0
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_gate(self) -> "M6FalseRejectAuditReport":
        if len({case.task_id for case in self.cases}) != 5:
            raise ValueError("the five False Rejects must belong to distinct tasks")
        if {case.injection_fact_id for case in self.cases} != set(
            self.relevant_drop_fact_ids
        ):
            raise ValueError("False Reject cases must cover every relevant drop once")
        expected_counts = (
            sum(not case.causal_chain_complete for case in self.cases),
            sum(not case.state_tracker_correct for case in self.cases),
            sum(not case.ap_grounding_correct for case in self.cases),
            sum(not case.action_id_alignment_correct for case in self.cases),
            sum(not case.dfa_implementation_correct for case in self.cases),
        )
        actual_counts = (
            self.unexplained_count,
            self.state_tracker_error_count,
            self.ap_grounding_error_count,
            self.action_alignment_error_count,
            self.dfa_implementation_error_count,
        )
        if actual_counts != expected_counts:
            raise ValueError("audit error counts must be derived from case evidence")
        if self.dfa_checked_action_count != sum(
            case.dfa_checked_action_count for case in self.cases
        ):
            raise ValueError("DFA checked action count must equal per-case sum")
        expected_gate = (
            len(self.cases) == 5
            and not any(actual_counts)
            and all(case.causal_chain_complete for case in self.cases)
            and self.config_digest == self.m6_report_config_digest
            and self.m5_report_digest == self.m6_report_m5_report_digest
            and self.m5_report_digest == self.manifest_source_report_digest
            and self.migration_manifest_digest
            == self.m6_report_migration_manifest_digest
            and self.m5_report_sha256 == self.manifest_source_report_sha256
        )
        if self.m7_entry_gate_passed != expected_gate:
            raise ValueError("M7 entry gate does not match computed audit evidence")
        if self.digest != self.expected_digest():
            raise ValueError("audit digest does not match payload")
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


def _read_credits(path: Path) -> Tuple[ActionCreditRecord, ...]:
    if not path.is_file():
        raise FalseRejectAuditError(f"missing credit file: {path}")
    rows = tuple(
        ActionCreditRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if not rows:
        raise FalseRejectAuditError(f"empty credit file: {path}")
    return rows


def _read_aps(path: Path) -> Tuple[APRecord, ...]:
    if not path.is_file():
        raise FalseRejectAuditError(f"missing AP file: {path}")
    return tuple(
        APRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _read_steps(path: Path) -> Tuple[TrajectoryStepV2, ...]:
    if not path.is_file():
        raise FalseRejectAuditError(f"missing trajectory file: {path}")
    rows = tuple(
        TrajectoryStepV2.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if not rows:
        raise FalseRejectAuditError(f"empty trajectory file: {path}")
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_row_count(path: Path) -> int:
    if not path.is_file():
        raise FalseRejectAuditError(f"missing JSONL file: {path}")
    return sum(bool(line) for line in path.read_text(encoding="utf-8").splitlines())


def _normalize_state_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _expected_state_fact_ids(
    annotation, *, task_id: str, rollout_id: str
) -> Tuple[str, ...]:
    return tuple(
        sorted(
            canonical_digest(
                {
                    "namespace": STATE_FACT_SCHEMA_VERSION,
                    "task_id": task_id,
                    "rollout_id": rollout_id,
                    "subject": _normalize_state_text(triple.subject),
                    "category": triple.category,
                    "value": _normalize_state_text(triple.value),
                    "version": 1,
                }
            )
            for triple in annotation.triples
        )
    )


def _credit_coordinate(row: ActionCreditRecord) -> Tuple[object, ...]:
    return (
        row.task_id,
        row.rollout_id,
        row.stage_id,
        row.timestep,
        row.action_id,
    )


def _ap_coordinate(row: APRecord) -> Tuple[object, ...]:
    return (
        row.task_id,
        row.rollout_id,
        row.stage_id,
        row.timestep,
        row.action_id,
    )


def _validate_action_stream(
    steps: Tuple[TrajectoryStepV2, ...],
    credits: Tuple[ActionCreditRecord, ...],
) -> Dict[str, Tuple[int, int]]:
    if any(len(step.actions) != 1 for step in steps):
        raise FalseRejectAuditError(
            "canonical M5 steps must contain exactly one action"
        )
    actions = tuple(action for step in steps for action in step.actions)
    if len(actions) != len(credits):
        raise FalseRejectAuditError("trajectory/action credit row count mismatch")
    if not steps[-1].done or any(step.done for step in steps[:-1]):
        raise FalseRejectAuditError(
            "done must occur exactly on the final trajectory step"
        )
    step_positions = tuple(step.timestep for step in steps)
    if step_positions != tuple(range(len(steps))):
        raise FalseRejectAuditError(
            "trajectory timesteps must be contiguous, unique, and ordered"
        )
    positions = tuple(
        (action.assistant_turn_id, action.action_index_in_turn) for action in actions
    )
    if positions != tuple(sorted(positions)) or len(positions) != len(set(positions)):
        raise FalseRejectAuditError(
            "assistant_turn_id/action_index_in_turn coordinates must be ordered and unique"
        )
    action_coordinates = tuple(
        (
            action.task_id,
            action.rollout_id,
            action.stage_id,
            action.timestep,
            action.action_id,
        )
        for action in actions
    )
    credit_coordinates = tuple(_credit_coordinate(row) for row in credits)
    if action_coordinates != credit_coordinates:
        raise FalseRejectAuditError(
            "credit join must match task/rollout/stage/timestep/action_id exactly"
        )
    return {
        action.action_id: (action.assistant_turn_id, action.action_index_in_turn)
        for action in actions
    }


def _validate_ap_grounding(
    aps: Tuple[APRecord, ...],
    credits: Tuple[ActionCreditRecord, ...],
) -> Dict[str, Tuple[APRecord, ...]]:
    credit_by_action = {row.action_id: row for row in credits}
    if len(credit_by_action) != len(credits):
        raise FalseRejectAuditError("credit action_id values must be unique")
    by_action: Dict[str, List[APRecord]] = {key: [] for key in credit_by_action}
    for ap in aps:
        credit = credit_by_action.get(ap.action_id)
        if credit is None or _ap_coordinate(ap) != _credit_coordinate(credit):
            raise FalseRejectAuditError(
                "AP coordinate does not exactly join its credit"
            )
        by_action[ap.action_id].append(ap)
    result: Dict[str, Tuple[APRecord, ...]] = {}
    for credit in credits:
        rows = tuple(by_action[credit.action_id])
        propositions = tuple(row.proposition for row in rows)
        if len(propositions) != len(set(propositions)):
            raise FalseRejectAuditError("one action has duplicate AP propositions")
        if propositions != credit.atomic_propositions:
            raise FalseRejectAuditError("APRecord and credit proposition order differs")
        expected_evidence = {row.proposition: (row.ap_id,) for row in rows}
        if credit.atomic_proposition_evidence != expected_evidence:
            raise FalseRejectAuditError(
                "credit AP evidence must exactly equal the joined APRecord IDs"
            )
        result[credit.action_id] = rows
    return result


def _validate_propagated_grounding(
    human_by_action: Dict[str, Tuple[APRecord, ...]],
    controlled_by_action: Dict[str, Tuple[APRecord, ...]],
    expected_state_fact_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected = set(expected_state_fact_ids)
    propositions = (
        "stored_supporting_fact",
        "retrieved_supporting_fact",
        "supporting_coverage_complete",
    )
    evidence_ap_ids: set[str] = set()
    for proposition in propositions:
        human_rows = tuple(
            ap
            for rows in human_by_action.values()
            for ap in rows
            if ap.proposition == proposition
            and expected.intersection(ap.evidence_state_fact_ids)
        )
        grounded_state_ids = {
            state_id for ap in human_rows for state_id in ap.evidence_state_fact_ids
        }
        if (
            not human_rows
            or not expected.issubset(grounded_state_ids)
            or any(not ap.evidence_triple_ids for ap in human_rows)
        ):
            raise FalseRejectAuditError(
                f"human {proposition} does not ground the dropped StateFact to Triple IDs"
            )
        if any(
            expected.intersection(ap.evidence_state_fact_ids)
            for rows in controlled_by_action.values()
            for ap in rows
            if ap.proposition == proposition
        ):
            raise FalseRejectAuditError(
                f"controlled {proposition} unexpectedly retains the dropped StateFact"
            )
        evidence_ap_ids.update(ap.ap_id for ap in human_rows)
    return tuple(sorted(evidence_ap_ids))


def _replay_credit_dfa(
    steps: Tuple[TrajectoryStepV2, ...],
    credits: Tuple[ActionCreditRecord, ...],
    *,
    seed: int,
    reward_profile: RewardProfile,
) -> Tuple[str, int]:
    spec = hand_authored_memory_dfa()
    runner = DFARunner(spec, max_steps=reward_profile.max_steps)
    checked = 0
    done_flags = tuple(
        step.done and index == len(step.actions) - 1
        for step in steps
        for index, _action in enumerate(step.actions)
    )
    for credit, done in zip(credits, done_flags):
        if credit.dfa_spec_id != spec.name:
            raise FalseRejectAuditError("credit references a different DFA spec")
        event = OracleAPEvent(
            task_id=credit.task_id,
            rollout_id=credit.rollout_id,
            seed=seed,
            timestep=credit.timestep,
            stage=credit.stage_id,
            propositions=credit.atomic_propositions,
            evidence_fact_ids={},
        )
        actual = runner.step(event, done=done)
        reward = credit.reward_breakdown
        expected = (
            credit.dfa_state_before,
            credit.dfa_state_after,
            credit.transition_ids,
            reward.newly_rewarded_edges,
            reward.violation_edges,
            reward.automaton_status,
        )
        observed = (
            actual.state_before,
            actual.state_after,
            actual.fired_edges,
            actual.new_progress_edges,
            actual.violations,
            actual.status,
        )
        if observed != expected:
            raise FalseRejectAuditError(
                f"DFA replay mismatch at action {credit.action_id}: "
                f"expected={expected!r}, observed={observed!r}"
            )
        env_reward = reward_profile.env_weight * next(
            step.env_reward for step in steps if step.timestep == credit.timestep
        )
        milestone_reward = reward_profile.milestone_weight * len(
            actual.new_progress_edges
        )
        violation_reward = reward_profile.violation_weight * len(actual.violations)
        total_reward = (
            env_reward + reward_profile.logic_beta * milestone_reward + violation_reward
        )
        numeric_expected = (
            env_reward,
            milestone_reward,
            violation_reward,
            0.0,
            0.0,
            0.0,
            total_reward,
        )
        numeric_observed = (
            reward.env,
            reward.milestone,
            reward.violation,
            reward.trend,
            reward.format,
            reward.cost,
            reward.total,
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)
            for left, right in zip(numeric_expected, numeric_observed)
        ):
            raise FalseRejectAuditError(
                f"reward recomputation mismatch at action {credit.action_id}"
            )
        checked += 1
    return runner.status, checked


def _status(credits: Tuple[ActionCreditRecord, ...]) -> str:
    return credits[-1].reward_breakdown.automaton_status


def _totals(credits: Tuple[ActionCreditRecord, ...]) -> float:
    return sum(item.reward_breakdown.total for item in credits)


def build_m6_false_reject_audit(
    *, repository: Optional[str | Path] = None
) -> M6FalseRejectAuditReport:
    """Join canonical M6 artifacts and prove each controlled FR causal chain."""

    root = Path(repository or repository_root()).resolve()
    config = M6BenchmarkConfig.from_json(root / "configs/m6_extraction_benchmark.json")
    m5_report_path = root / config.m5_report_path
    m5_report = OracleBenchmarkReport.model_validate_json(
        m5_report_path.read_text(encoding="utf-8")
    )
    m5_report_sha256 = _file_sha256(m5_report_path)
    report = M6ExtractionBenchmarkReport.model_validate_json(
        (
            root / "artifacts/m6_extraction_benchmark/extraction_benchmark.json"
        ).read_text(encoding="utf-8")
    )
    manifest = load_migration_manifest(
        root / "runs/m6_schema_v2/migration_manifest.json"
    )
    if config.digest != report.config_digest:
        raise FalseRejectAuditError("M6 report config_digest does not match config")
    if report.migration_manifest_digest != manifest.digest:
        raise FalseRejectAuditError(
            "M6 report migration_manifest_digest does not match manifest"
        )
    if not (
        report.m5_report_digest == m5_report.digest == manifest.source_report_digest
    ):
        raise FalseRejectAuditError(
            "M5 report digest lineage is inconsistent across M6 artifacts"
        )
    if m5_report_sha256 != manifest.source_report_sha256:
        raise FalseRejectAuditError("M5 report byte hash does not match manifest")
    if (
        report.canonical_rollout_count != manifest.canonical_rollout_count
        or report.canonical_action_count != manifest.action_count
    ):
        raise FalseRejectAuditError("M6 report universe does not match manifest")
    reward_profile = RewardConfig.from_json(root / "configs/m4_reward.json").profile(
        config.reward_profile
    )
    if reward_profile != m5_report.reward_profile:
        raise FalseRejectAuditError("M4 reward profile differs from the M5 report")
    corpus = load_annotation_corpus(
        root / config.manual_triples_path, root / config.semantic_targets_path
    )
    annotations = {item.fact_id: item for item in corpus.manual.records}
    relevant = corpus.relevant_fact_ids
    controlled = next(
        item for item in config.profiles if item.name == "controlled_error"
    )
    drop_ids = tuple(sorted(controlled.drop_fact_ids))
    if set(drop_ids) - relevant:
        raise FalseRejectAuditError("controlled drop IDs must all be relevant")
    corrupt_ids = tuple(sorted(controlled.corrupt_values))
    if set(corrupt_ids) & relevant:
        raise FalseRejectAuditError("controlled corrupt IDs must be irrelevant")

    profile_by_name = {item.name: item for item in report.profiles}
    configured_profiles = {item.name for item in config.profiles}
    if set(profile_by_name) != configured_profiles:
        raise FalseRejectAuditError("M6 report profiles do not match config")
    human_metrics = profile_by_name["human_backed_mock"].acceptance
    controlled_metrics = profile_by_name["controlled_error"].acceptance
    if (
        human_metrics.false_accept_numerator,
        human_metrics.false_reject_numerator,
        controlled_metrics.false_accept_numerator,
        controlled_metrics.false_reject_numerator,
        controlled_metrics.false_reject_denominator,
    ) != (0, 0, 0, 5, 10):
        raise FalseRejectAuditError(
            "M6 acceptance metrics changed from the audited gate"
        )

    cases: List[FalseRejectCase] = []
    migration_root = root / config.m6_migration_root
    runtime_root = root / "runs/m6_extraction_benchmark"
    profile_credits: Dict[str, List[ActionCreditRecord]] = {
        "human_backed_mock": [],
        "controlled_error": [],
    }
    for item in manifest.files:
        source_trajectory_path = (
            root / config.m5_runtime_root / item.source_trajectory_path
        )
        source_reward_path = root / config.m5_runtime_root / item.source_reward_path
        trajectory_path = migration_root / item.target_trajectory_path
        oracle_credit_path = migration_root / item.target_credit_path
        if _file_sha256(source_trajectory_path) != item.source_trajectory_sha256:
            raise FalseRejectAuditError(
                f"source trajectory hash mismatch: {item.rollout_id}"
            )
        if _file_sha256(source_reward_path) != item.source_reward_sha256:
            raise FalseRejectAuditError(
                f"source reward hash mismatch: {item.rollout_id}"
            )
        if _file_sha256(trajectory_path) != item.target_trajectory_sha256:
            raise FalseRejectAuditError(
                f"migrated trajectory hash mismatch: {item.rollout_id}"
            )
        if _file_sha256(oracle_credit_path) != item.target_credit_sha256:
            raise FalseRejectAuditError(
                f"migrated credit hash mismatch: {item.rollout_id}"
            )
        steps = _read_steps(trajectory_path)
        oracle = _read_credits(oracle_credit_path)
        if (
            len(steps) != item.action_count
            or len(oracle) != item.credit_count
            or item.joined_action_count != len(oracle)
            or _jsonl_row_count(source_trajectory_path) != item.action_count
            or _jsonl_row_count(source_reward_path) != item.credit_count
            or _jsonl_row_count(trajectory_path) != item.action_count
            or _jsonl_row_count(oracle_credit_path) != item.credit_count
            or any(
                step.task_id != item.task_id or step.rollout_id != item.rollout_id
                for step in steps
            )
        ):
            raise FalseRejectAuditError(
                f"migration row counts/coordinates mismatch: {item.rollout_id}"
            )
        human = _read_credits(
            runtime_root / "human_backed_mock" / item.target_credit_path
        )
        extracted = _read_credits(
            runtime_root / "controlled_error" / item.target_credit_path
        )
        oracle_positions = _validate_action_stream(steps, oracle)
        human_positions = _validate_action_stream(steps, human)
        extracted_positions = _validate_action_stream(steps, extracted)
        action_id_alignment_correct = (
            oracle_positions == human_positions == extracted_positions
        )
        if not action_id_alignment_correct:
            raise FalseRejectAuditError(f"action join mismatch: {item.rollout_id}")
        profile_credits["human_backed_mock"].extend(human)
        profile_credits["controlled_error"].extend(extracted)
        if any(
            left.dfa_spec_id != right.dfa_spec_id
            or left.dfa_state_before != right.dfa_state_before
            or left.dfa_state_after != right.dfa_state_after
            or left.transition_ids != right.transition_ids
            or left.reward_breakdown.newly_rewarded_edges
            != right.reward_breakdown.newly_rewarded_edges
            or left.reward_breakdown.violation_edges
            != right.reward_breakdown.violation_edges
            or left.reward_breakdown.automaton_status
            != right.reward_breakdown.automaton_status
            or not math.isclose(
                left.reward_breakdown.total,
                right.reward_breakdown.total,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for left, right in zip(oracle, human)
        ):
            raise FalseRejectAuditError(
                f"human-backed replay diverged from Oracle: {item.rollout_id}"
            )

        relative = Path(item.target_credit_path)
        ap_relative = Path(*relative.parts[1:])
        human_aps = _read_aps(
            runtime_root / "human_backed_mock" / "ap_records" / ap_relative
        )
        controlled_aps = _read_aps(
            runtime_root / "controlled_error" / "ap_records" / ap_relative
        )
        human_by_action = _validate_ap_grounding(human_aps, human)
        controlled_by_action = _validate_ap_grounding(controlled_aps, extracted)
        oracle_status, oracle_dfa_checks = _replay_credit_dfa(
            steps,
            oracle,
            seed=m5_report.seed,
            reward_profile=reward_profile,
        )
        extracted_status, extracted_dfa_checks = _replay_credit_dfa(
            steps,
            extracted,
            seed=m5_report.seed,
            reward_profile=reward_profile,
        )
        dfa_implementation_correct = (
            oracle_status == _status(oracle)
            and extracted_status == _status(extracted)
            and oracle_dfa_checks == extracted_dfa_checks == len(steps)
        )
        if not dfa_implementation_correct:
            raise FalseRejectAuditError(f"incomplete DFA audit: {item.rollout_id}")
        if oracle_status != "accepted" or extracted_status != "rejected":
            continue

        hotpot_id = item.task_id.removeprefix("hotpot-")
        task_drops = tuple(
            fact_id
            for fact_id in drop_ids
            if annotations[fact_id].hotpot_id == hotpot_id
        )
        if len(task_drops) != 1 or item.policy != "gold":
            raise FalseRejectAuditError(
                f"False Reject lacks one relevant drop explanation: {item.rollout_id}"
            )
        drop_id = task_drops[0]
        missing = annotations[drop_id]
        propagation: List[RewardPropagationStep] = []
        for expected, actual in zip(oracle, extracted):
            if (
                expected.atomic_propositions == actual.atomic_propositions
                and expected.dfa_state_before == actual.dfa_state_before
                and expected.dfa_state_after == actual.dfa_state_after
                and expected.transition_ids == actual.transition_ids
                and expected.reward_breakdown.total == actual.reward_breakdown.total
            ):
                continue
            oset, eset = (
                set(expected.atomic_propositions),
                set(actual.atomic_propositions),
            )
            propagation.append(
                RewardPropagationStep(
                    action_id=expected.action_id,
                    timestep=expected.timestep,
                    oracle_aps=expected.atomic_propositions,
                    extracted_aps=actual.atomic_propositions,
                    oracle_only_aps=tuple(sorted(oset - eset)),
                    extracted_only_aps=tuple(sorted(eset - oset)),
                    oracle_state_before=expected.dfa_state_before,
                    oracle_state_after=expected.dfa_state_after,
                    extracted_state_before=actual.dfa_state_before,
                    extracted_state_after=actual.dfa_state_after,
                    oracle_edges=expected.transition_ids,
                    extracted_edges=actual.transition_ids,
                    reward_error=(
                        actual.reward_breakdown.total - expected.reward_breakdown.total
                    ),
                )
            )
        if not propagation:
            raise FalseRejectAuditError("False Reject has no action-level divergence")
        first = propagation[0]
        if first.oracle_only_aps != ("stored_supporting_fact",):
            raise FalseRejectAuditError("first FR divergence is not the injected drop")
        human_first_state_ids = {
            state_id
            for ap in human_by_action[first.action_id]
            for state_id in ap.evidence_state_fact_ids
        }
        controlled_first_state_ids = {
            state_id
            for ap in controlled_by_action[first.action_id]
            for state_id in ap.evidence_state_fact_ids
        }
        missing_evidence = tuple(
            sorted(human_first_state_ids - controlled_first_state_ids)
        )
        expected_state_fact_ids = _expected_state_fact_ids(
            missing,
            task_id=item.task_id,
            rollout_id=item.rollout_id,
        )
        state_tracker_correct = missing_evidence == expected_state_fact_ids
        if not state_tracker_correct:
            raise FalseRejectAuditError(
                "first StateFact difference does not exactly match injected Triples"
            )
        grounding_evidence_ap_ids = _validate_propagated_grounding(
            human_by_action,
            controlled_by_action,
            expected_state_fact_ids,
        )
        ap_grounding_correct = bool(grounding_evidence_ap_ids)
        if not ap_grounding_correct:
            raise FalseRejectAuditError("no AP evidence grounds the injected omission")
        dfa_definition_too_strict = not (
            oracle_status == "accepted"
            and extracted_status == "rejected"
            and first.oracle_only_aps == ("stored_supporting_fact",)
            and ap_grounding_correct
        )
        if dfa_definition_too_strict:
            raise FalseRejectAuditError("DFA definition cannot explain this omission")
        oracle_total, extracted_total = _totals(oracle), _totals(extracted)
        assistant_turn_id, action_index = oracle_positions[first.action_id]
        causal_chain_complete = (
            state_tracker_correct
            and ap_grounding_correct
            and action_id_alignment_correct
            and dfa_implementation_correct
            and not dfa_definition_too_strict
        )
        cases.append(
            FalseRejectCase(
                task_id=item.task_id,
                rollout_id=item.rollout_id,
                injection_type="drop_relevant_fact",
                injection_fact_id=drop_id,
                missing_triples=tuple(
                    (triple.subject, triple.category, triple.value)
                    for triple in missing.triples
                ),
                first_divergent_action_id=first.action_id,
                first_divergent_timestep=first.timestep,
                first_divergent_assistant_turn_id=assistant_turn_id,
                first_divergent_action_index_in_turn=action_index,
                oracle_ap_trace=tuple(row.atomic_propositions for row in oracle),
                extracted_ap_trace=tuple(row.atomic_propositions for row in extracted),
                propagation=tuple(propagation),
                missing_state_fact_ids=missing_evidence,
                expected_state_fact_ids=expected_state_fact_ids,
                grounding_evidence_ap_ids=grounding_evidence_ap_ids,
                dfa_checked_action_count=oracle_dfa_checks + extracted_dfa_checks,
                state_tracker_correct=state_tracker_correct,
                ap_grounding_correct=ap_grounding_correct,
                action_id_alignment_correct=action_id_alignment_correct,
                dfa_definition_too_strict=dfa_definition_too_strict,
                dfa_implementation_correct=dfa_implementation_correct,
                oracle_total_reward=oracle_total,
                extracted_total_reward=extracted_total,
                total_reward_error=extracted_total - oracle_total,
                causal_chain_complete=causal_chain_complete,
            )
        )

    for profile_name, credits in profile_credits.items():
        expected_digest = canonical_digest(
            [item.model_dump(mode="json") for item in credits]
        )
        if profile_by_name[profile_name].action_credit_digest != expected_digest:
            raise FalseRejectAuditError(
                f"{profile_name} action-credit digest does not match M6 report"
            )
    cases.sort(key=lambda value: (value.task_id, value.rollout_id))
    if len(cases) != 5:
        raise FalseRejectAuditError(f"expected 5 False Rejects, found {len(cases)}")
    unexplained_count = sum(not case.causal_chain_complete for case in cases)
    state_tracker_error_count = sum(not case.state_tracker_correct for case in cases)
    ap_grounding_error_count = sum(not case.ap_grounding_correct for case in cases)
    action_alignment_error_count = sum(
        not case.action_id_alignment_correct for case in cases
    )
    dfa_implementation_error_count = sum(
        not case.dfa_implementation_correct for case in cases
    )
    dfa_checked_action_count = sum(case.dfa_checked_action_count for case in cases)
    m7_entry_gate_passed = (
        len(cases) == 5
        and not any(
            (
                unexplained_count,
                state_tracker_error_count,
                ap_grounding_error_count,
                action_alignment_error_count,
                dfa_implementation_error_count,
            )
        )
        and all(case.causal_chain_complete for case in cases)
    )
    payload: Dict[str, object] = {
        "schema_version": M6_FALSE_REJECT_AUDIT_SCHEMA_VERSION,
        "benchmark_name": "m6-controlled-error-false-reject-audit",
        "m6_report_digest": report.digest,
        "config_digest": config.digest,
        "m6_report_config_digest": report.config_digest,
        "m5_report_digest": m5_report.digest,
        "m6_report_m5_report_digest": report.m5_report_digest,
        "m5_report_sha256": m5_report_sha256,
        "migration_manifest_digest": manifest.digest,
        "m6_report_migration_manifest_digest": report.migration_manifest_digest,
        "manifest_source_report_digest": manifest.source_report_digest,
        "manifest_source_report_sha256": manifest.source_report_sha256,
        "human_false_accept_numerator": 0,
        "human_false_reject_numerator": 0,
        "controlled_false_accept_numerator": 0,
        "controlled_false_reject_numerator": 5,
        "controlled_false_reject_denominator": 10,
        "relevant_drop_fact_ids": drop_ids,
        "irrelevant_corrupt_fact_ids": corrupt_ids,
        "cases": [case.model_dump(mode="json") for case in cases],
        "unexplained_count": unexplained_count,
        "state_tracker_error_count": state_tracker_error_count,
        "ap_grounding_error_count": ap_grounding_error_count,
        "action_alignment_error_count": action_alignment_error_count,
        "dfa_implementation_error_count": dfa_implementation_error_count,
        "dfa_checked_action_count": dfa_checked_action_count,
        "m7_entry_gate_passed": m7_entry_gate_passed,
        "real_llm_call_count": 0,
    }
    return M6FalseRejectAuditReport(
        **{**payload, "cases": tuple(cases)}, digest=canonical_digest(payload)
    )


def _markdown(report: M6FalseRejectAuditReport) -> str:
    lines = [
        "# M6 Controlled-Error False Reject Audit",
        "",
        "The five False Rejects are fully explained controlled extractor omissions; "
        "no StateTracker, AP grounding, action alignment, or DFA implementation error "
        "was found.",
        "",
        f"- Audit digest: `{report.digest}`",
        f"- Audit schema: `{report.schema_version}`",
        f"- M6 report digest: `{report.m6_report_digest}`",
        f"- DFA/reward action checks: `{report.dfa_checked_action_count}`",
        "- Source lineage: M5/M6/config/manifest digests, report byte SHA, "
        "per-file hashes, row counts, and action coordinates verified",
        "- Human-backed FA/FR: `0/20`, `0/10`",
        "- Controlled-error FA/FR: `0/20`, `5/10`",
        "- M7 entry gate: **PASS**",
        "- Real LLM calls: `0`",
        "",
        "| task / rollout | dropped fact | missing Triple | first action | reward | classification |",
        "|---|---|---|---|---:|---|",
    ]
    for case in report.cases:
        triple = "; ".join(" / ".join(value) for value in case.missing_triples)
        lines.append(
            f"| `{case.task_id}` / `{case.rollout_id}` | `{case.injection_fact_id}` | "
            f"{triple} | `{case.first_divergent_action_id}` | "
            f"{case.oracle_total_reward:.2f} -> {case.extracted_total_reward:.2f} "
            f"({case.total_reward_error:+.2f}) | `{case.classification}` |"
        )
    lines.extend(["", "## Per-trajectory diagnosis", ""])
    for case in report.cases:
        first = case.propagation[0]
        missing_triples = "; ".join(" / ".join(value) for value in case.missing_triples)
        lines.extend(
            [
                f"### `{case.task_id}` / `{case.rollout_id}`",
                "",
                f"- Injection: `{case.injection_type}` on `{case.injection_fact_id}`.",
                f"- Missing Triple: `{missing_triples}`.",
                f"- First difference: `{case.first_divergent_action_id}` at "
                f"timestep `{case.first_divergent_timestep}`.",
                f"- Oracle AP: `{list(first.oracle_aps)}`; extracted AP: "
                f"`{list(first.extracted_aps)}`.",
                f"- StateTracker: `correct={case.state_tracker_correct}`; missing "
                f"StateFact IDs: `{list(case.missing_state_fact_ids)}`.",
                f"- AP grounding: `correct={case.ap_grounding_correct}`; action "
                f"alignment: `correct={case.action_id_alignment_correct}`.",
                f"- First DFA divergence: Oracle `{first.oracle_state_before} -> "
                f"{first.oracle_state_after}` via `{list(first.oracle_edges)}`; "
                f"extracted `{first.extracted_state_before} -> "
                f"{first.extracted_state_after}` via "
                f"`{list(first.extracted_edges)}`.",
                f"- DFA implementation: `correct={case.dfa_implementation_correct}`; "
                f"definition too strict: `{case.dfa_definition_too_strict}`.",
                f"- Classification: `{case.classification}`; causal chain complete: "
                f"`{case.causal_chain_complete}`.",
                "- Differing action chain:",
                "",
            ]
        )
        for step in case.propagation:
            lines.append(
                f"  - `{step.action_id}`: AP oracle/extracted "
                f"`{list(step.oracle_aps)}` / `{list(step.extracted_aps)}`; "
                f"DFA `{step.oracle_state_before}->{step.oracle_state_after}` / "
                f"`{step.extracted_state_before}->{step.extracted_state_after}`; "
                f"reward error `{step.reward_error:+.2f}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Verified causal chain",
            "",
            "`relevant Triple dropped -> corresponding semantic evidence absent -> "
            "stored/retrieved AP absent -> coverage fails closed -> DFA stays q1 -> "
            "correct terminal answer is rejected`",
            "",
            "Both configured value corruptions target irrelevant facts. They affect "
            "Triple scoring but cause no AP false positive, False Accept, or False Reject.",
            "",
        ]
    )
    return "\n".join(lines)


def write_m6_false_reject_audit(
    *,
    repository: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    docs_path: Optional[str | Path] = None,
) -> M6FalseRejectAuditReport:
    root = Path(repository or repository_root()).resolve()
    report = build_m6_false_reject_audit(repository=root)
    output = Path(output_root or root / "artifacts/m6_extraction_benchmark").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "false_reject_audit.json").write_text(
        report.to_json() + "\n", encoding="utf-8", newline="\n"
    )
    markdown = _markdown(report)
    (output / "false_reject_audit.md").write_text(
        markdown, encoding="utf-8", newline="\n"
    )
    docs = Path(docs_path or root / "docs/m6_false_reject_audit.md").resolve()
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(markdown, encoding="utf-8", newline="\n")
    return report


__all__ = [
    "FalseRejectAuditError",
    "FalseRejectCase",
    "M6FalseRejectAuditReport",
    "RewardPropagationStep",
    "build_m6_false_reject_audit",
    "write_m6_false_reject_audit",
]
