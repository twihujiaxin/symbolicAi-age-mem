"""Fail-closed, non-destructive migration of canonical M5 artifacts.

Only rollout paths named by the validated M5 Oracle report are considered.
Extra files under ``runs/`` are intentionally ignored.  The M1 trajectories
and M4 reward JSONL files are read-only inputs; migrated views are written to
separate ``trajectories_v2`` and ``action_credits`` trees.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Tuple

from pydantic import ValidationError

from ..hotpotqa_benchmark.metrics import BenchmarkRecord, OracleBenchmarkReport
from ..memory_oracle.automaton import hand_authored_memory_dfa
from ..memory_oracle.models import RewardedTrajectoryStep
from ..trajectory import TrajectoryReplay, TrajectoryStep, TrajectoryValidationError
from .models import (
    M5_ORACLE_REWARD_VERSION,
    M5_TO_M6_MIGRATION_VERSION,
    ActionCreditRecord,
    ActionEvent,
    ActionSource,
    MigrationFileRecord,
    MigrationManifest,
    RewardBreakdownV2,
    TrajectoryStepV2,
)


class SchemaMigrationError(RuntimeError):
    """Raised when an M5 artifact cannot be migrated without inventing data."""


@dataclass(frozen=True)
class MigrationResult:
    """Validated result and deterministic files produced by a migration."""

    output_root: Path
    manifest_path: Path
    manifest: MigrationManifest


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_path(raw: str, *, expected_root: str) -> PurePosixPath:
    if not raw or "\\" in raw:
        raise SchemaMigrationError(f"invalid logical artifact path: {raw!r}")
    logical = PurePosixPath(raw)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise SchemaMigrationError(f"unsafe logical artifact path: {raw!r}")
    if not logical.parts or logical.parts[0] != expected_root:
        raise SchemaMigrationError(
            f"expected {expected_root!r} artifact path, got {raw!r}"
        )
    return logical


def _resolve_logical(root: Path, logical: PurePosixPath) -> Path:
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*logical.parts).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise SchemaMigrationError(f"artifact path escaped root: {logical.as_posix()}")
    return resolved


def _read_reward_jsonl(path: Path) -> Tuple[RewardedTrajectoryStep, ...]:
    if not path.is_file():
        raise SchemaMigrationError(f"reward JSONL does not exist: {path}")
    rows: List[RewardedTrajectoryStep] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise SchemaMigrationError(
                    f"blank reward JSONL row at {path}:{line_number}"
                )
            try:
                raw = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SchemaMigrationError(
                    f"invalid reward JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            try:
                rows.append(RewardedTrajectoryStep.model_validate(raw))
            except ValidationError as exc:
                raise SchemaMigrationError(
                    f"invalid reward schema at {path}:{line_number}: {exc}"
                ) from exc
    if not rows:
        raise SchemaMigrationError(f"reward JSONL is empty: {path}")
    return tuple(rows)


def _source_for_policy(policy: str) -> ActionSource:
    if policy == "gold":
        return "oracle"
    if policy in {"wrong_answer", "missing_support"}:
        return "error_injector"
    raise SchemaMigrationError(f"unsupported M5 policy: {policy!r}")


def migrate_m5_step(
    step: TrajectoryStep,
    rewarded_step: RewardedTrajectoryStep,
    *,
    source: ActionSource,
    dfa_spec_id: str,
) -> Tuple[TrajectoryStepV2, ActionCreditRecord]:
    """Migrate one proven M5 single-action step without token fabrication."""

    if step.schema_version != 1:
        raise SchemaMigrationError("only M1 schema_version=1 is supported")
    if step.old_logprob is not None:
        raise SchemaMigrationError(
            "legacy scalar old_logprob cannot be converted to token old_logprobs"
        )
    if step.stage not in {1, 2, 3}:
        raise SchemaMigrationError("M5 migration requires stage in {1, 2, 3}")
    if len(step.tool_calls) != 1 or len(step.tool_results) != 1:
        raise SchemaMigrationError(
            "M5 assistant_turn_id inference requires exactly one tool call and result"
        )
    call = step.tool_calls[0]
    result = step.tool_results[0]
    if result.tool_call_id != call.id or result.name != call.name:
        raise SchemaMigrationError("tool result does not uniquely match the tool call")

    event = rewarded_step.event
    reward = rewarded_step.reward
    expected_identity = (step.task_id, step.rollout_id, step.timestep, step.stage)
    if (
        event.task_id,
        event.rollout_id,
        event.timestep,
        event.stage,
    ) != expected_identity:
        raise SchemaMigrationError(
            "reward event identity does not match trajectory step"
        )
    if (
        reward.task_id,
        reward.rollout_id,
        reward.timestep,
    ) != expected_identity[:3]:
        raise SchemaMigrationError(
            "reward breakdown identity does not match trajectory step"
        )

    # M5 is deterministic rule/oracle data.  It contains no model response token
    # sequence or policy logprobs, so all model-only fields deliberately remain None.
    action = ActionEvent(
        action_id=call.id,
        task_id=step.task_id,
        rollout_id=step.rollout_id,
        stage_id=step.stage,
        timestep=step.timestep,
        assistant_turn_id=step.timestep,
        action_index_in_turn=0,
        source=source,
        action_type=call.name,
        action_text=step.action_text,
        arguments=call.input,
        result=result.model_dump(mode="json"),
        response_token_ids=None,
        token_start=None,
        token_end=None,
        old_logprobs=None,
        policy_version=None,
    )
    trajectory_step = TrajectoryStepV2(
        task_id=step.task_id,
        rollout_id=step.rollout_id,
        stage_id=step.stage,
        timestep=step.timestep,
        observation=step.observation,
        actions=(action,),
        memory_before=tuple(item.model_copy(deep=True) for item in step.memory_before),
        memory_after=tuple(item.model_copy(deep=True) for item in step.memory_after),
        env_reward=step.env_reward,
        done=step.done,
    )
    transition_ids = tuple(reward.fired_edges)
    breakdown = RewardBreakdownV2(
        env=reward.env,
        milestone=reward.milestone,
        violation=reward.violation,
        trend=reward.trend,
        format=reward.format,
        cost=0.0,
        total=reward.total,
        automaton_state_before=reward.automaton_state_before,
        automaton_state_after=reward.automaton_state_after,
        automaton_status=reward.automaton_status,
        propositions=tuple(reward.propositions),
        fired_edges=transition_ids,
        newly_rewarded_edges=tuple(reward.newly_rewarded_edges),
        violation_edges=tuple(reward.violation_edges),
    )
    credit = ActionCreditRecord(
        action_id=call.id,
        task_id=step.task_id,
        rollout_id=step.rollout_id,
        stage_id=step.stage,
        timestep=step.timestep,
        atomic_propositions=tuple(event.propositions),
        atomic_proposition_evidence={
            proposition: tuple(fact_ids)
            for proposition, fact_ids in event.evidence_fact_ids.items()
        },
        dfa_spec_id=dfa_spec_id,
        transition_ids=transition_ids,
        transition_id=transition_ids[0] if len(transition_ids) == 1 else None,
        dfa_state_before=reward.automaton_state_before,
        dfa_state_after=reward.automaton_state_after,
        reward_breakdown=breakdown,
        return_to_go=None,
        advantage=None,
        reward_version=M5_ORACLE_REWARD_VERSION,
    )
    return trajectory_step, credit


def _canonical_payload(
    models: Iterable[TrajectoryStepV2 | ActionCreditRecord],
) -> bytes:
    return "".join(item.to_json_line() + "\n" for item in models).encode("utf-8")


def _atomic_write(path: Path, payload: bytes, *, immutable_sources: set[Path]) -> None:
    resolved = path.resolve()
    if resolved in immutable_sources:
        raise SchemaMigrationError(f"refusing to overwrite source artifact: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.m6-migration.tmp")
    if temporary.resolve() in immutable_sources:
        raise SchemaMigrationError("temporary output collides with a source artifact")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _migrate_record(
    record: BenchmarkRecord,
    *,
    runtime_root: Path,
    output_root: Path,
    immutable_sources: set[Path],
    global_action_ids: set[str],
) -> MigrationFileRecord:
    trajectory_logical = _logical_path(
        record.trajectory_path, expected_root="trajectories"
    )
    reward_logical = _logical_path(record.reward_path, expected_root="rewards")
    trajectory_path = _resolve_logical(runtime_root, trajectory_logical)
    reward_path = _resolve_logical(runtime_root, reward_logical)
    try:
        trajectory = TrajectoryReplay.from_jsonl(trajectory_path)
    except TrajectoryValidationError as exc:
        raise SchemaMigrationError(
            f"invalid source trajectory {trajectory_path}: {exc}"
        ) from exc
    if not trajectory.steps:
        raise SchemaMigrationError(f"source trajectory is empty: {trajectory_path}")
    rollout_ids = {step.rollout_id for step in trajectory.steps}
    task_ids = {step.task_id for step in trajectory.steps}
    if task_ids != {record.task_id} or len(rollout_ids) != 1:
        raise SchemaMigrationError(
            "report record does not identify exactly one rollout"
        )
    rollout_id = next(iter(rollout_ids))
    expected_rollout_id = f"m5-{record.split}-{record.hotpot_id}-{record.policy}"
    if rollout_id != expected_rollout_id:
        raise SchemaMigrationError(
            "trajectory is not the deterministic rollout named by the M5 report"
        )
    replay = trajectory.replay(
        task_id=record.task_id,
        rollout_id=rollout_id,
        require_complete=True,
    )
    if replay.digest != record.trajectory_digest:
        raise SchemaMigrationError("source trajectory digest does not match M5 report")

    rewarded_steps = _read_reward_jsonl(reward_path)
    if len(rewarded_steps) != len(replay.steps):
        raise SchemaMigrationError("trajectory and reward row counts differ")
    source = _source_for_policy(record.policy)
    dfa_spec_id = hand_authored_memory_dfa().name
    migrated_steps: List[TrajectoryStepV2] = []
    credits: List[ActionCreditRecord] = []
    for step, rewarded_step in zip(replay.steps, rewarded_steps):
        expected_action_id = f"{rollout_id}:call:{step.timestep}"
        if len(step.tool_calls) != 1 or step.tool_calls[0].id != expected_action_id:
            raise SchemaMigrationError(
                "M5 action_id must be the deterministic unique tool_call.id"
            )
        migrated_step, credit = migrate_m5_step(
            step,
            rewarded_step,
            source=source,
            dfa_spec_id=dfa_spec_id,
        )
        if credit.action_id in global_action_ids:
            raise SchemaMigrationError(
                f"duplicate global action_id: {credit.action_id}"
            )
        global_action_ids.add(credit.action_id)
        migrated_steps.append(migrated_step)
        credits.append(credit)

    action_ids = tuple(item.actions[0].action_id for item in migrated_steps)
    credit_ids = tuple(item.action_id for item in credits)
    if action_ids != credit_ids or len(action_ids) != len(set(action_ids)):
        raise SchemaMigrationError(
            "ActionEvent to ActionCreditRecord join is not one-to-one"
        )

    relative_tail = trajectory_logical.parts[1:]
    target_trajectory_logical = PurePosixPath("trajectories_v2", *relative_tail)
    target_credit_logical = PurePosixPath("action_credits", *relative_tail)
    target_trajectory_path = _resolve_logical(output_root, target_trajectory_logical)
    target_credit_path = _resolve_logical(output_root, target_credit_logical)
    trajectory_payload = _canonical_payload(migrated_steps)
    credit_payload = _canonical_payload(credits)
    _atomic_write(
        target_trajectory_path,
        trajectory_payload,
        immutable_sources=immutable_sources,
    )
    _atomic_write(
        target_credit_path,
        credit_payload,
        immutable_sources=immutable_sources,
    )
    return MigrationFileRecord(
        task_id=record.task_id,
        rollout_id=rollout_id,
        policy=record.policy,
        source=source,
        source_trajectory_path=trajectory_logical.as_posix(),
        source_reward_path=reward_logical.as_posix(),
        target_trajectory_path=target_trajectory_logical.as_posix(),
        target_credit_path=target_credit_logical.as_posix(),
        source_trajectory_sha256=_sha256_file(trajectory_path),
        source_reward_sha256=_sha256_file(reward_path),
        target_trajectory_sha256=_sha256_bytes(trajectory_payload),
        target_credit_sha256=_sha256_bytes(credit_payload),
        action_count=len(action_ids),
        credit_count=len(credit_ids),
        joined_action_count=len(set(action_ids) & set(credit_ids)),
    )


def migrate_m5_canonical_report(
    report_path: str | Path,
    *,
    runtime_root: str | Path,
    output_root: str | Path,
) -> MigrationResult:
    """Migrate exactly the rollout universe named by one canonical M5 report.

    The source report, M1 trajectory files, and M4 reward files are hashed before
    and after all writes.  A mismatch aborts the migration.  No timestamp or
    absolute path enters output bytes, so identical sources yield byte-identical
    output in different directories.
    """

    report_file = Path(report_path).expanduser().resolve()
    runtime = Path(runtime_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not report_file.is_file():
        raise SchemaMigrationError(f"M5 report does not exist: {report_file}")
    try:
        report = OracleBenchmarkReport.model_validate_json(
            report_file.read_text(encoding="utf-8")
        )
    except (ValidationError, ValueError) as exc:
        raise SchemaMigrationError(f"invalid canonical M5 report: {exc}") from exc
    if report.schema_version != 1:
        raise SchemaMigrationError("only M5 report schema_version=1 is supported")

    records = tuple(
        sorted(
            report.records,
            key=lambda item: (item.trajectory_path, item.reward_path),
        )
    )
    source_paths: List[Path] = [report_file]
    for record in records:
        trajectory_logical = _logical_path(
            record.trajectory_path, expected_root="trajectories"
        )
        reward_logical = _logical_path(record.reward_path, expected_root="rewards")
        source_paths.extend(
            (
                _resolve_logical(runtime, trajectory_logical),
                _resolve_logical(runtime, reward_logical),
            )
        )
    if len(source_paths) != len(set(source_paths)):
        raise SchemaMigrationError("canonical report contains duplicate source paths")
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise SchemaMigrationError(
            f"canonical source artifact is missing: {missing[0]}"
        )
    immutable_sources = {path.resolve() for path in source_paths}
    source_hashes_before = {path: _sha256_file(path) for path in source_paths}

    global_action_ids: set[str] = set()
    file_records = tuple(
        _migrate_record(
            record,
            runtime_root=runtime,
            output_root=output,
            immutable_sources=immutable_sources,
            global_action_ids=global_action_ids,
        )
        for record in records
    )
    action_count = sum(item.action_count for item in file_records)
    credit_count = sum(item.credit_count for item in file_records)
    joined_count = sum(item.joined_action_count for item in file_records)
    if len(global_action_ids) != action_count:
        raise SchemaMigrationError("global action_id uniqueness check failed")

    source_hashes_after = {path: _sha256_file(path) for path in source_paths}
    if source_hashes_before != source_hashes_after:
        raise SchemaMigrationError("a source artifact changed during migration")

    base_payload = {
        "migration_version": M5_TO_M6_MIGRATION_VERSION,
        "source_benchmark_name": report.benchmark_name,
        "source_report_digest": report.digest,
        "source_report_sha256": source_hashes_before[report_file],
        "canonical_rollout_count": len(file_records),
        "action_count": action_count,
        "credit_count": credit_count,
        "joined_action_count": joined_count,
        "source_hashes_verified_unchanged": True,
        "files": tuple(file_records),
    }
    digest_payload = {
        "schema_version": "agemem.migration_manifest.v1",
        **{
            key: (
                [item.model_dump(mode="json") for item in value]
                if key == "files"
                else value
            )
            for key, value in base_payload.items()
        },
    }
    manifest = MigrationManifest(
        **base_payload,
        digest=MigrationManifest.digest_payload(digest_payload),
    )
    manifest_path = output / "migration_manifest.json"
    manifest_payload = (manifest.to_json() + "\n").encode("utf-8")
    _atomic_write(
        manifest_path,
        manifest_payload,
        immutable_sources=immutable_sources,
    )
    # The manifest write is also forbidden from changing an input.  Recheck so
    # the guarantee covers the complete public operation rather than data files only.
    if source_hashes_before != {path: _sha256_file(path) for path in source_paths}:
        raise SchemaMigrationError(
            "a source artifact changed while writing the manifest"
        )
    return MigrationResult(
        output_root=output,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_migration_manifest(path: str | Path) -> MigrationManifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        return MigrationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise SchemaMigrationError(f"invalid migration manifest: {exc}") from exc


__all__ = [
    "MigrationResult",
    "SchemaMigrationError",
    "load_migration_manifest",
    "migrate_m5_canonical_report",
    "migrate_m5_step",
]
