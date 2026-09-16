#!/usr/bin/env python
"""CLI for dynamic streaming-memory v2 CPU delivery and gated model phases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AgeMem_code_agentscope.streaming_memory.dynamic.config import (  # noqa: E402
    load_dynamic_config,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.replay import (  # noqa: E402
    collect_fixture_run,
    replay_fixture_run,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.world_generator import (  # noqa: E402
    DynamicWorldBuilder,
    validate_dynamic_manifest,
)
from AgeMem_code_agentscope.streaming_memory.token_budget import (  # noqa: E402
    TokenAccounting,
    load_tokenizer,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def cmd_audit_current(_: argparse.Namespace) -> int:
    agents = list(ROOT.glob("AGENTS.md")) + list(ROOT.parent.glob("AGENTS.md"))
    result = {
        "status": "pass",
        "repository_root": str(ROOT),
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short").splitlines(),
        "agents_files": [str(item) for item in agents],
        "v1_report": str(ROOT / "docs/streaming_multiquery_implementation_report.md"),
        "v2_audit": str(ROOT / "docs/v2_current_implementation_audit.md"),
    }
    _print(result)
    return 0


def _config_accounting(config_path: Path):
    config = load_dynamic_config(config_path)
    tokenizer = load_tokenizer(
        config.model.get("tokenizer_path"), config.model.get("tokenizer_revision")
    )
    accounting = TokenAccounting.from_tokenizer(
        tokenizer, revision=config.model.get("tokenizer_revision")
    )
    return config, accounting


def cmd_build_data(args: argparse.Namespace) -> int:
    config, accounting = _config_accounting(_path(args.config))
    output = _path(config.data["output_root"])
    manifest = DynamicWorldBuilder(
        config=config, accounting=accounting, repository_root=ROOT
    ).build(output)
    _print(
        {
            "status": "pass",
            "manifest": str(output / "manifest.json"),
            "build_id": manifest.build_id,
            "statistics": manifest.statistics,
            "validation": manifest.validation,
        }
    )
    return 0


def cmd_validate_data(args: argparse.Namespace) -> int:
    _print(validate_dynamic_manifest(_path(args.manifest)))
    return 0


def cmd_env_smoke(args: argparse.Namespace) -> int:
    config, accounting = _config_accounting(_path(args.config))
    manifest = _path(
        args.manifest or str(Path(config.data["output_root"]) / "manifest.json")
    )
    output = _path(args.output or config.runtime["run_root"])
    _print(
        collect_fixture_run(
            manifest_path=manifest,
            output_root=output,
            accounting=accounting,
            config=config.model_dump(mode="json"),
        )
    )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    profiles = tuple(item.strip() for item in args.profiles.split(",") if item.strip())
    report = replay_fixture_run(
        _path(args.run_dir), profiles=profiles, lambda_semantic=args.lambda_semantic
    )
    _print(
        {
            key: report[key]
            for key in (
                "status",
                "trajectory_count",
                "profile_mean_total_reward",
                "end_life_different_trajectories",
                "direct_compiled",
                "research_boundary",
            )
        }
    )
    return 0


def cmd_compare_backends(args: argparse.Namespace) -> int:
    report_path = _path(args.run_dir) / "reward_replay.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparison = report["direct_compiled"]
    _print(comparison)
    return 0 if comparison["status"] == "pass" else 1


def _valid_gpu_ids(value) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(item) is int and item >= 0 for item in value)
        and len(value) == len(set(value))
    )


def preflight_report(config_path: Path) -> dict:
    config = load_dynamic_config(config_path)
    missing = []
    requirements = {
        "inherit.v1_runtime_lock": config.inherit.get("v1_runtime_lock"),
        "inherit.initialization_checkpoint": config.inherit.get(
            "initialization_checkpoint"
        ),
        "model.policy_path": config.model.get("policy_path"),
        "model.policy_revision": config.model.get("policy_revision"),
        "model.tokenizer_path": config.model.get("tokenizer_path"),
        "model.tokenizer_revision": config.model.get("tokenizer_revision"),
        "model.reader_path": config.model.get("reader_path"),
        "model.reader_revision": config.model.get("reader_revision"),
        "memory.retriever_revision": config.memory.get("retriever_revision"),
        "grounding.validator_revision": config.grounding.get("validator_revision"),
        "training.optimizer_lock": config.training.get("optimizer_lock"),
        "data.manifest": config.data.get("manifest"),
        "evaluation.frozen_test_manifest": config.evaluation.get(
            "frozen_test_manifest"
        ),
        "runtime.gpu_ids": config.runtime.get("gpu_ids"),
    }
    for name, value in requirements.items():
        if value is None or value == "":
            missing.append(name)
    gpu_ids = requirements["runtime.gpu_ids"]
    if not _valid_gpu_ids(gpu_ids) and "runtime.gpu_ids" not in missing:
        missing.append("runtime.gpu_ids")
    if config.model.get("tokenizer_path") == "debug-lexical":
        missing.append("production_frozen_tokenizer_required")
    return {
        "status": "pass" if not missing else "blocked",
        "config": str(config_path),
        "reward_profile": config.reward["profile"],
        "protocol_version": config.protocol_version,
        "missing_or_unresolved": missing,
        "model_runtime_producer": "implemented_gpu_unverified",
        "note": "The Trinity producer/workflow exists, but preflight does not launch Ray/vLLM. P4-P6 still require a production lock, launcher config, and real GPU validation.",
    }


def cmd_preflight(args: argparse.Namespace) -> int:
    result = preflight_report(_path(args.config))
    _print(result)
    return 0 if result["status"] == "pass" else 2


def cmd_model_phase(args: argparse.Namespace) -> int:
    result = preflight_report(_path(args.config))
    if result["status"] == "pass":
        result["status"] = "blocked"
        result["missing_or_unresolved"] = [
            "dynamic_v2_trinity_launcher_config_and_explicit_gpu_run_required"
        ]
    result["requested_phase"] = args.command
    _print(result)
    return 2


def cmd_evaluate(args: argparse.Namespace) -> int:
    path = _path(args.run_dir) / "reward_replay.json"
    if not path.is_file():
        _print(
            {
                "status": "blocked",
                "reason": "reward_replay.json_not_found",
                "path": str(path),
            }
        )
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    _print(
        {
            "status": report["status"],
            "profile_mean_total_reward": report["profile_mean_total_reward"],
            "research_boundary": report["research_boundary"],
        }
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = ROOT / "docs/v2_dynamic_memory_implementation_report.md"
    _print(
        {
            "status": "pass" if report.is_file() else "blocked",
            "implementation_report": str(report),
            "experiment_root": str(_path(args.experiment_root)),
        }
    )
    return 0 if report.is_file() else 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Dynamic streaming-memory v2: CPU data/replay and gated model phases."
    )
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "audit-current", help="Print live Git and audit document identities."
    ).set_defaults(func=cmd_audit_current)
    build = commands.add_parser(
        "build-data", help="Build separated D0/D1 public/private data and manifest."
    )
    build.add_argument("--config", required=True)
    build.set_defaults(func=cmd_build_data)
    validate = commands.add_parser(
        "validate-data",
        help="Verify digests, joins, split isolation, Oracle labels, and leakage.",
    )
    validate.add_argument("--manifest", required=True)
    validate.set_defaults(func=cmd_validate_data)
    smoke = commands.add_parser(
        "env-smoke",
        help="Collect deterministic non-model versioned-memory fixture trajectories.",
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--manifest")
    smoke.add_argument("--output")
    smoke.set_defaults(func=cmd_env_smoke)
    replay = commands.add_parser(
        "replay", help="Replay V2 rewards over an existing private fixture run."
    )
    replay.add_argument("--run-dir", required=True)
    replay.add_argument("--profiles", default="V2_T,V2_END,V2_LIFE,V2_LIFE_MONITOR")
    replay.add_argument("--lambda-semantic", type=float, default=0.25)
    replay.set_defaults(func=cmd_replay)
    compare = commands.add_parser(
        "compare-backends", help="Read the exact direct/compiled differential result."
    )
    compare.add_argument("--run-dir", required=True)
    compare.set_defaults(func=cmd_compare_backends)
    preflight = commands.add_parser(
        "preflight", help="Fail closed on unresolved production identities/resources."
    )
    preflight.add_argument("--config", required=True)
    preflight.set_defaults(func=cmd_preflight)
    for name in ("diagnose", "train"):
        command = commands.add_parser(
            name,
            help=f"Gated {name} entry; never substitutes CPU fixtures for a model run.",
        )
        command.add_argument("--config", required=True)
        command.set_defaults(func=cmd_model_phase)
    evaluate = commands.add_parser(
        "evaluate", help="Summarize an existing run without changing it."
    )
    evaluate.add_argument("--run-dir", required=True)
    evaluate.set_defaults(func=cmd_evaluate)
    report = commands.add_parser(
        "report", help="Locate the implementation report and experiment root."
    )
    report.add_argument("--experiment-root", required=True)
    report.set_defaults(func=cmd_report)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        _print({"status": "fail", "error_type": type(exc).__name__, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
