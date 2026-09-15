"""Deterministic CPU fixture collection and V2 reward replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..token_budget import TokenAccounting
from .environment import (
    DynamicCheckpoint,
    DynamicMemoryEnvironment,
    canonical_dynamic_payload,
)
from .lifecycle_monitor import compare_evaluations, compile_monitor
from .reward_profiles import aggregate_dynamic_reward
from .schema import (
    DynamicHistoryPublic,
    DynamicQueryPrivate,
    PrivateEvent,
    SourceRegistryRecord,
)
from .state_evaluator import DirectStateEvaluator, build_semantic_frames
from .temporal_grounder import exposed_payloads_as_memories, validate_memory_semantics
from .world_generator import validate_dynamic_manifest
from .world_oracle import WorldOracle


class ReplayError(ValueError):
    pass


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _memory_digest(memories: Sequence[Mapping[str, Any]]) -> str:
    raw = json.dumps(
        list(memories), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _claim_for_event(event: PrivateEvent, oracle: WorldOracle) -> dict[str, Any]:
    interval = next(
        item
        for item in oracle.intervals(event.entity, event.relation)
        if item.event_id == event.event_id
    )
    return {
        "entity": event.entity,
        "relation": event.relation,
        "value": event.value,
        "effective_at": event.effective_at,
        "valid_from": interval.valid_from,
        "valid_to": interval.valid_to,
    }


def _timeline_action(
    *,
    events: Sequence[PrivateEvent],
    registry: Mapping[str, SourceRegistryRecord],
    oracle: WorldOracle,
    update: bool,
) -> dict[str, Any]:
    return {
        "type": "UPDATE" if update else "ADD",
        "memory_id": "timeline",
        "title": "动态事实时间线",
        "tags": ["timeline"],
        "content": "\n".join(registry[item.source_ref].text for item in events),
        "source_refs": [item.source_ref for item in events],
        "claims": [_claim_for_event(item, oracle) for item in events],
        "custom": {"representation": "ordered_event_timeline"},
    }


def collect_fixture_run(
    *,
    manifest_path: Path,
    output_root: Path,
    accounting: TokenAccounting,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect four deterministic non-model trajectories from one D1 history."""

    validate_dynamic_manifest(manifest_path)
    data_root = manifest_path.parent
    history = DynamicHistoryPublic.model_validate(
        _load_jsonl(data_root / "histories.public.jsonl")[0]
    )
    all_events = [
        PrivateEvent.model_validate(row)
        for row in _load_jsonl(data_root / "events.private.jsonl")
    ]
    events = [item for item in all_events if item.history_id == history.history_id]
    all_queries = [
        DynamicQueryPrivate.model_validate(row)
        for row in _load_jsonl(data_root / "queries.private.jsonl")
    ]
    public_query_by_id = {
        row["query_id"]: row for row in _load_jsonl(data_root / "queries.public.jsonl")
    }
    candidates = [item for item in all_queries if item.history_id == history.history_id]
    preferred = ["current_state", "historical_state", "multi_update", "temporal_join"]
    queries = []
    for family in preferred:
        item = next(value for value in candidates if value.task_family == family)
        queries.append(item)
    registry_all = [
        SourceRegistryRecord.model_validate(row)
        for row in _load_jsonl(data_root / "source_registry.private.jsonl")
    ]
    registry = {
        item.source_ref: item
        for item in registry_all
        if item.history_id == history.history_id
    }
    events_by_ref = {item.source_ref: item for item in events}
    relevant_ids = {
        event_id
        for query in queries
        for alt in query.support_alternatives
        for event_id in alt
    }
    relevant = [item for item in events if item.event_id in relevant_ids]
    oracle = WorldOracle(events)
    trajectories = []
    strategies = ("eager_timeline", "late_timeline", "current_only", "wrong_body")
    budget = config["budget"]
    for strategy in strategies:
        env = DynamicMemoryEnvironment(
            history,
            accounting=accounting,
            read_rollout_id=f"fixture-{strategy}",
            policy_version="fixture-policy:0",
            ingest_max_new_tokens=int(budget["ingest_max_new_tokens"]),
            max_decisions_per_chunk=int(budget["max_decisions_per_chunk"]),
            answer_tail_tokens=int(budget["answer_tail_tokens"]),
            retrieval_payload_tokens=int(budget["retrieved_payload_tokens"]),
        )
        seen_relevant: list[PrivateEvent] = []
        observed_refs: set[str] = set()
        action_checkpoints: list[DynamicCheckpoint] = []

        def capture_action() -> None:
            active = env.policy_memory()
            action_checkpoints.append(
                DynamicCheckpoint(
                    checkpoint_id=f"action-{strategy}-{len(action_checkpoints):04d}",
                    checkpoint_index=env.policy_observation()["chunk"]["observed_at"],
                    memory_revision=env.memory_revision,
                    active_memories=active,
                    observed_source_refs=tuple(sorted(observed_refs)),
                    memory_sha256=_memory_digest(active),
                )
            )

        for chunk in history.chunks:
            messages = env.admit_next_chunk()
            accounting.enforce_context(
                messages,
                max_new_tokens=int(budget["ingest_max_new_tokens"]),
                context_total_tokens=int(budget["context_total_tokens"]),
            )
            observed_refs.update(chunk.source_refs)
            seen_relevant.extend(
                events_by_ref[ref]
                for ref in chunk.source_refs
                if ref in events_by_ref and events_by_ref[ref] in relevant
            )
            final_chunk = chunk.observed_at == len(history.chunks) - 1
            if strategy == "eager_timeline" and seen_relevant:
                result = env.execute(
                    _timeline_action(
                        events=seen_relevant,
                        registry=registry,
                        oracle=oracle,
                        update=env.memory_revision > 0,
                    )
                )
                if not result.admitted:
                    raise ReplayError(f"eager fixture failed: {result.code}")
                capture_action()
            elif strategy == "late_timeline" and final_chunk:
                result = env.execute(
                    _timeline_action(
                        events=relevant,
                        registry=registry,
                        oracle=oracle,
                        update=False,
                    )
                )
                if not result.admitted:
                    raise ReplayError(f"late fixture failed: {result.code}")
                capture_action()
            elif strategy == "current_only" and final_chunk:
                last_by_key: dict[tuple[str, str], PrivateEvent] = {}
                for event in relevant:
                    last_by_key[(event.entity, event.relation)] = event
                result = env.execute(
                    _timeline_action(
                        events=list(last_by_key.values()),
                        registry=registry,
                        oracle=oracle,
                        update=False,
                    )
                )
                if not result.admitted:
                    raise ReplayError(f"current-only fixture failed: {result.code}")
                capture_action()
            elif strategy == "wrong_body" and final_chunk:
                action = _timeline_action(
                    events=relevant,
                    registry=registry,
                    oracle=oracle,
                    update=False,
                )
                action["content"] = (
                    "这些来源只是一份泛化概述，不包含任何具体人物、关系或时间。"
                )
                result = env.execute(action)
                if not result.admitted:
                    raise ReplayError(f"wrong-body fixture failed: {result.code}")
                capture_action()
            env.execute({"type": "NEXT"})
            capture_action()
        snapshot = env.finalize()
        payloads = tuple(
            canonical_dynamic_payload(item) for item in snapshot.active_memories
        )
        exposed_by_query = {}
        for query in queries:
            selected = []
            used = 0
            for payload in payloads:
                cost = accounting.count_text(payload)
                if used + cost <= int(budget["retrieved_payload_tokens"]):
                    selected.append(payload)
                    used += cost
            messages = [
                {"role": "system", "content": "Answer from retrieved memory only."},
                {
                    "role": "user",
                    "content": public_query_by_id[query.query_id]["question"],
                },
                {"role": "user", "content": "RETRIEVED MEMORY\n" + "\n".join(selected)},
            ]
            try:
                accounting.enforce_context(
                    messages,
                    max_new_tokens=int(budget["answer_max_new_tokens"]),
                    context_total_tokens=int(budget["context_total_tokens"]),
                )
            except Exception:
                selected = []
            exposed_by_query[query.query_id] = selected
        trajectories.append(
            {
                "strategy": strategy,
                "checkpoints": [asdict(item) for item in env.checkpoints],
                "action_checkpoints": [asdict(item) for item in action_checkpoints],
                "snapshot": asdict(snapshot),
                "exposed_payloads_by_query": exposed_by_query,
                "answer_text_by_query": {
                    query.query_id: (
                        "<answer>wrong</answer>"
                        if strategy == "wrong_body"
                        else f"<answer>{query.answers[0]}</answer>"
                    )
                    for query in queries
                },
                "audit_revision_count": len(env.private_audit_ledger()),
                "max_memory_tokens": max(
                    (
                        sum(
                            accounting.count_text(canonical_dynamic_payload(memory))
                            for memory in cp.active_memories
                        )
                        for cp in env.checkpoints
                    ),
                    default=0,
                ),
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "agemem.dynamic.fixture_run.v2",
        "identity": {
            "manifest_sha256": _digest_file(manifest_path),
            "tokenizer_name": accounting.name,
            "tokenizer_revision": accounting.revision,
            "policy_version": "fixture-policy:0",
            "reader_version": "fixture-reader:oracle-answer",
            "model_used": False,
            "gpu_used": False,
        },
        "history": history.model_dump(mode="json"),
        "events": [item.model_dump(mode="json") for item in events],
        "queries": [item.model_dump(mode="json") for item in queries],
        "source_registry": [item.model_dump(mode="json") for item in registry.values()],
        "trajectories": trajectories,
    }
    path = output_root / "fixture_run.private.json"
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": "pass",
        "path": str(path),
        "trajectories": len(trajectories),
        "checkpoints_per_trajectory": len(history.chunks),
    }


def replay_fixture_run(
    run_dir: Path, *, profiles: Sequence[str], lambda_semantic: float = 0.25
) -> dict[str, Any]:
    path = run_dir / "fixture_run.private.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = [PrivateEvent.model_validate(item) for item in raw["events"]]
    queries = [DynamicQueryPrivate.model_validate(item) for item in raw["queries"]]
    registry = {
        item.source_ref: item
        for item in (
            SourceRegistryRecord.model_validate(row) for row in raw["source_registry"]
        )
    }
    events_by_id = {item.event_id: item for item in events}
    direct_backend = DirectStateEvaluator()
    trajectory_reports = []
    total_state_rows = 0
    total_action_state_rows = 0
    max_diff = 0.0
    for trajectory in raw["trajectories"]:
        checkpoints = tuple(
            DynamicCheckpoint(**item) for item in trajectory["checkpoints"]
        )
        frames = build_semantic_frames(
            checkpoints,
            queries=queries,
            source_registry=registry,
            events_by_id=events_by_id,
        )
        direct = direct_backend.evaluate(frames)
        compiled = compile_monitor().evaluate(frames)
        comparison = compare_evaluations(direct, compiled)
        if comparison["status"] != "pass":
            raise ReplayError("direct/compiled monitor mismatch")
        total_state_rows += comparison["compared_state_rows"]
        max_diff = max(max_diff, comparison["max_abs_utility_diff"])
        action_frames = build_semantic_frames(
            tuple(
                DynamicCheckpoint(**item)
                for item in trajectory.get("action_checkpoints", [])
            ),
            queries=queries,
            source_registry=registry,
            events_by_id=events_by_id,
        )
        action_comparison = compare_evaluations(
            direct_backend.evaluate(action_frames),
            compile_monitor().evaluate(action_frames),
        )
        if action_comparison["status"] != "pass":
            raise ReplayError("direct/compiled action-state mismatch")
        total_action_state_rows += action_comparison["compared_state_rows"]
        max_diff = max(max_diff, action_comparison["max_abs_utility_diff"])
        exposure = {
            query.query_id: validate_memory_semantics(
                exposed_payloads_as_memories(
                    trajectory["exposed_payloads_by_query"].get(query.query_id, [])
                ),
                observed_source_refs=checkpoints[-1].observed_source_refs,
                source_registry=registry,
                events_by_id=events_by_id,
                query=query,
            ).utility
            for query in queries
        }
        rewards = {}
        for profile in profiles:
            evaluation = compiled if profile == "V2_LIFE_MONITOR" else direct
            rewards[profile] = asdict(
                aggregate_dynamic_reward(
                    profile=profile,
                    queries=queries,
                    answer_text_by_query=trajectory["answer_text_by_query"],
                    exposure_utility_by_query=exposure,
                    state_evaluation=evaluation,
                    lambda_semantic=lambda_semantic,
                )
            )
        trajectory_reports.append(
            {
                "strategy": trajectory["strategy"],
                "rewards": rewards,
                "backend_equivalence": comparison,
                "action_backend_equivalence": action_comparison,
            }
        )
    profile_means = {
        profile: sum(
            item["rewards"][profile]["total_reward"] for item in trajectory_reports
        )
        / len(trajectory_reports)
        for profile in profiles
    }
    end_life_differences = (
        sum(
            abs(
                item["rewards"].get("V2_END", {}).get("total_reward", 0.0)
                - item["rewards"].get("V2_LIFE", {}).get("total_reward", 0.0)
            )
            > 1e-12
            for item in trajectory_reports
        )
        if {"V2_END", "V2_LIFE"}.issubset(profiles)
        else None
    )
    report = {
        "schema_version": "agemem.dynamic.reward_replay.v2",
        "status": "pass",
        "identity": raw["identity"],
        "profiles": list(profiles),
        "lambda_semantic": lambda_semantic,
        "trajectory_count": len(trajectory_reports),
        "profile_mean_total_reward": profile_means,
        "end_life_different_trajectories": end_life_differences,
        "direct_compiled": {
            "status": "pass",
            "compared_state_rows": total_state_rows,
            "compared_action_state_rows": total_action_state_rows,
            "max_abs_utility_diff": max_diff,
            "duplicate_gpu_experiment_required": False,
        },
        "trajectories": trajectory_reports,
        "research_boundary": "deterministic CPU fixture only; no model, optimizer, or learning claim",
    }
    (run_dir / "reward_replay.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["ReplayError", "collect_fixture_run", "replay_fixture_run"]
