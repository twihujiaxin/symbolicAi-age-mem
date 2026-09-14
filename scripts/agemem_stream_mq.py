#!/usr/bin/env python3
"""CLI for the isolated streaming multi-query protocol (S0--S4)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AgeMem_code_agentscope.streaming_memory.data_builder import (  # noqa: E402
    StreamingHotpotBuilder,
    validate_manifest,
)
from AgeMem_code_agentscope.streaming_memory.environment import (  # noqa: E402
    MemorySnapshot,
    StreamingMemoryEnvironment,
)
from AgeMem_code_agentscope.streaming_memory.query_runner import (  # noqa: E402
    IndependentQueryRunner,
    QueryBranchResult,
)
from AgeMem_code_agentscope.streaming_memory.reward_replay import score_snapshot  # noqa: E402
from AgeMem_code_agentscope.streaming_memory.schema import (  # noqa: E402
    QueryGold,
    QueryRecord,
    StreamEpisodePublic,
    StreamingBuildConfig,
)
from AgeMem_code_agentscope.streaming_memory.token_budget import (  # noqa: E402
    TokenAccounting,
    load_tokenizer,
)


EXIT_CONFIG = 2
EXIT_DATA = 3
EXIT_INFRASTRUCTURE = 4
EXIT_MODEL_BEHAVIOR = 5


def _load_config(path: str | Path) -> tuple[StreamingBuildConfig, Path]:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    text = config_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("non-JSON YAML requires PyYAML") from exc
        payload = yaml.safe_load(text)
    return StreamingBuildConfig.model_validate(payload), config_path.resolve()


def _accounting(config: StreamingBuildConfig) -> TokenAccounting:
    configured = os.getenv("AGEMEM_STREAM_MQ_TOKENIZER_PATH") or config.model.get("tokenizer_path")
    revision = os.getenv("AGEMEM_STREAM_MQ_TOKENIZER_REVISION") or config.model.get("tokenizer_revision")
    tokenizer = load_tokenizer(configured, revision)
    return TokenAccounting.from_tokenizer(tokenizer, name=configured, revision=revision)


def _output_root(config: StreamingBuildConfig) -> Path:
    value = Path(str(config.data["output_root"]))
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def build_data(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config)
    accounting = _accounting(config)
    builder = StreamingHotpotBuilder.from_local_dataset(
        config=config, accounting=accounting, repository_root=REPOSITORY_ROOT
    )
    manifest = builder.build(_output_root(config))
    print(json.dumps({
        "status": "pass", "build_id": manifest.build_id,
        "output_root": str(_output_root(config)), "split_counts": manifest.split_counts,
        "query_counts": manifest.query_counts, "statistics": manifest.statistics,
        "tokenizer": manifest.tokenizer_name,
    }, ensure_ascii=False, indent=2))
    return 0


def validate_data(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    print(json.dumps(validate_manifest(path), ensure_ascii=False, indent=2))
    return 0


def _load_first_episode(root: Path) -> StreamEpisodePublic:
    line = (root / "episodes.public.jsonl").read_text(encoding="utf-8").splitlines()[0]
    return StreamEpisodePublic.model_validate_json(line)


def env_smoke(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config)
    root = _output_root(config)
    validate_manifest(root / "manifest.json")
    accounting = _accounting(config)
    episode = _load_first_episode(root)
    env = StreamingMemoryEnvironment(
        episode, accounting=accounting, read_rollout_id="cpu-scripted-0",
        policy_version="scripted-off-policy-v1",
        ingest_max_new_tokens=int(config.environment["ingest_max_new_tokens"]),
        decisions_per_chunk=int(config.environment["decisions_per_chunk"]),
        answer_tail_tokens=int(config.environment["answer_tail_tokens"]),
    )
    for index, chunk in enumerate(episode.chunks):
        messages = env.admit_next_chunk()
        if any('"question":' in item["content"] or "QUESTION\n" in item["content"] for item in messages):
            raise RuntimeError("ingest observation leaked a question field")
        refs = [
            {"document_key": source.document_key, "sentence_index": sentence_index}
            for source in chunk.sources
            for sentence_index in range(source.sentence_start, source.sentence_end + 1)
        ]
        result = env.execute({
            "type": "ADD", "memory_id": f"mem-{index:04d}",
            "title": chunk.sources[0].title, "content": chunk.text,
            "source_refs": refs, "tags": ["scripted-extractive"],
        })
        if not result.admitted:
            env.execute({"type": "NEXT"})
    snapshot = env.finalize()
    queries = [
        QueryRecord.model_validate_json(line)
        for line in (root / "episodes.questions.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["episode_id"] == episode.episode_id
    ]
    runner = IndependentQueryRunner(
        accounting=accounting, reader=lambda messages: "<answer>fixture</answer>",
        reader_version="fixed-scripted-reader-v1",
        context_total_tokens=episode.context_budget_tokens,
        answer_max_new_tokens=int(config.environment["answer_max_new_tokens"]),
        retrieval_payload_tokens=int(config.query["retrieval_payload_tokens"]),
        retrieval_top_k=int(config.query["retrieval_top_k"]),
    )
    branches = [runner.run(snapshot, query, index) for index, query in enumerate(queries)]
    run_root = Path(str(config.runtime["run_root"]))
    if not run_root.is_absolute():
        run_root = REPOSITORY_ROOT / run_root
    run_root.mkdir(parents=True, exist_ok=True)
    replay_payload = {
        "protocol": "streaming_multiquery_v1", "data_root": str(root),
        "semantic_lambda": 0.25,
        "rollouts": [{"snapshot": asdict(snapshot), "branches": [asdict(item) for item in branches]}],
    }
    replay_text = json.dumps(replay_payload, ensure_ascii=False, indent=2) + "\n"
    replay_input = run_root / "replay_input.json"
    if replay_input.exists() and replay_input.read_text(encoding="utf-8") != replay_text:
        raise RuntimeError(f"refusing to overwrite conflicting replay input: {replay_input}")
    replay_input.write_text(replay_text, encoding="utf-8")
    print(json.dumps({
        "status": "pass", "policy": args.policy, "chunks_observed": len(episode.chunks),
        "snapshot_id": snapshot.snapshot_id, "active_memories": len(snapshot.active_memories),
        "memory_tokens": env.memory_tokens(), "fifo_evictions": sum(
            event.kind == "fifo_context_eviction" for event in env.events
        ), "branch_count": len(branches),
        "branch_parent_equal": len({item.snapshot_id for item in branches}) == 1,
        "reader_actor_loss_masked": all(item.actor_loss_masked for item in branches),
        "replay_input": str(replay_input),
    }, ensure_ascii=False, indent=2))
    return 0


def gate(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config)
    root = _output_root(config)
    validation = validate_manifest(root / "manifest.json")
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests/common", "-p", "stream_mq_*_test.py"]
    result = subprocess.run(command, cwd=REPOSITORY_ROOT)
    if result.returncode:
        return result.returncode
    print(json.dumps({"status": "pass", "scope": args.scope, "data": validation}, indent=2))
    return 0


def replay(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    input_path = run_dir / "replay_input.json"
    if not input_path.is_file():
        raise FileNotFoundError(
            "new-protocol replay_input.json is absent; legacy trajectories cannot be coerced"
        )
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    profiles = tuple(args.profiles.split(","))
    if payload.get("protocol") != "streaming_multiquery_v1":
        raise ValueError("replay input is not streaming_multiquery_v1")
    allowed_profiles = {"terminal", "flat_state", "dfa"}
    if not profiles or not set(profiles).issubset(allowed_profiles):
        raise ValueError("profiles must be terminal,flat_state,dfa")
    data_root = Path(payload["data_root"])
    if not data_root.is_absolute():
        data_root = (run_dir / data_root).resolve()
    validate_manifest(data_root / "manifest.json")
    gold_by_query = {
        row.query_id: row
        for row in (
            QueryGold.model_validate_json(line)
            for line in (data_root / "episodes.gold.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    registry = {}
    for line in (data_root / "source_registry.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        registry[(row["document_key"], int(row["sentence_index"]), row["sentence_sha256"])] = row["sentence"]
    results = []
    flat_dfa_max_delta = 0.0
    for item in payload.get("rollouts") or []:
        snapshot_row = item["snapshot"]
        snapshot = MemorySnapshot(
            snapshot_id=snapshot_row["snapshot_id"], episode_id=snapshot_row["episode_id"],
            read_rollout_id=snapshot_row["read_rollout_id"],
            active_memories=tuple(snapshot_row["active_memories"]),
            context_tail=tuple(snapshot_row.get("context_tail") or ()),
            observed_sources=tuple(tuple(value) for value in snapshot_row["observed_sources"]),
            memory_sha256=snapshot_row["memory_sha256"], tail_sha256=snapshot_row["tail_sha256"],
            policy_version=snapshot_row["policy_version"],
        )
        branches = [
            QueryBranchResult(
                query_branch_id=row["query_branch_id"], query_id=row["query_id"],
                snapshot_id=row["snapshot_id"], reader_version=row["reader_version"],
                answer_text=row["answer_text"], prompt_tokens=int(row["prompt_tokens"]),
                retrieved_memory_ids=tuple(row.get("retrieved_memory_ids") or ()),
                retrieved_payloads=tuple(row.get("retrieved_payloads") or ()),
                tail_messages=tuple(row.get("tail_messages") or ()),
                actor_loss_masked=bool(row.get("actor_loss_masked", True)),
            )
            for row in item["branches"]
        ]
        digest = lambda value: hashlib.sha256(  # noqa: E731
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if digest(snapshot.active_memories) != snapshot.memory_sha256 or digest(snapshot.context_tail) != snapshot.tail_sha256:
            raise ValueError("snapshot payload digest mismatch")
        if any(branch.snapshot_id != snapshot.snapshot_id for branch in branches):
            raise ValueError("query branch does not join its immutable snapshot")
        if any(not branch.actor_loss_masked for branch in branches):
            raise ValueError("reader branch is marked trainable in actor loss")
        reward = score_snapshot(
            snapshot=snapshot, branch_results=branches, gold_by_query=gold_by_query,
            source_sentences=registry, semantic_lambda=float(payload.get("semantic_lambda", 0.25)),
        )
        flat_dfa_max_delta = max(flat_dfa_max_delta, abs(reward.flat_state - reward.dfa))
        results.append({
            "read_rollout_id": snapshot.read_rollout_id,
            "terminal": reward.terminal, "flat_state": reward.flat_state, "dfa": reward.dfa,
            "semantic_mean": reward.semantic_mean,
            "flat_dfa_equivalent": reward.flat_dfa_equivalent,
        })
    if not results:
        raise ValueError("replay input contains no rollouts")
    identity_payload = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((data_root / "manifest.json").read_bytes()).hexdigest(),
        "profiles": profiles,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report_payload = {
        "schema_version": "agemem.stream_mq.reward_replay.v1",
        "protocol": "streaming_multiquery_v1", "status": "pass",
        "replay_identity_sha256": identity, "profiles": profiles, "rollouts": results,
        "flat_dfa_max_abs_delta": flat_dfa_max_delta,
        "flat_dfa_equivalent": flat_dfa_max_delta <= 1e-12,
        "duplicate_dfa_gpu_experiment_required": flat_dfa_max_delta > 1e-12,
    }
    report_path = run_dir / "reward_replay.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("replay_identity_sha256") != identity:
            raise ValueError("refusing to overwrite reward replay with a different identity")
        report_payload = existing
    else:
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(report_path)
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
    return 0


def blocked_model_command(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config)
    unresolved = [
        key for key in ("policy_path", "policy_revision", "reader_path", "reader_revision")
        if not config.model.get(key)
    ]
    print(json.dumps({
        "status": "blocked_s5_model_runtime", "command": args.command,
        "unresolved_model_fields": unresolved,
        "reason": "S0-S4 do not authorize fabricating a model run or optimizer profile",
    }, indent=2))
    return EXIT_INFRASTRUCTURE


def report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    receipts = sorted(run_dir.glob("**/*receipt*.json")) if run_dir.exists() else []
    print(json.dumps({
        "status": "pass" if receipts else "no_runtime_receipts",
        "protocol": "streaming_multiquery_v1", "receipt_files": [str(p) for p in receipts],
        "learning_effective": None,
    }, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-data"); build.add_argument("--config", required=True); build.set_defaults(func=build_data)
    validate = sub.add_parser("validate-data"); validate.add_argument("--manifest", required=True); validate.set_defaults(func=validate_data)
    smoke = sub.add_parser("env-smoke"); smoke.add_argument("--config", required=True); smoke.add_argument("--policy", choices=("scripted",), default="scripted"); smoke.set_defaults(func=env_smoke)
    replay_parser = sub.add_parser("replay"); replay_parser.add_argument("--run-dir", required=True); replay_parser.add_argument("--profiles", default="terminal,flat_state,dfa"); replay_parser.set_defaults(func=replay)
    gate_parser = sub.add_parser("gate"); gate_parser.add_argument("--scope", choices=("cpu",), default="cpu"); gate_parser.add_argument("--config", required=True); gate_parser.set_defaults(func=gate)
    for name in ("preflight", "diagnose", "train", "evaluate"):
        item = sub.add_parser(name); item.add_argument("--config", required=True); item.set_defaults(func=blocked_model_command)
    report_parser = sub.add_parser("report"); report_parser.add_argument("--run-dir", required=True); report_parser.set_defaults(func=report)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.func(args))
    except (ValueError, KeyError) as exc:
        print(f"CONFIG_OR_DATA_ERROR: {exc}", file=sys.stderr)
        return EXIT_DATA
    except (OSError, RuntimeError) as exc:
        print(f"INFRASTRUCTURE_ERROR: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE


if __name__ == "__main__":
    raise SystemExit(main())
