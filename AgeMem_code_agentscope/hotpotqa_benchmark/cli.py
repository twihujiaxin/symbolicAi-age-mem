"""Command-line entry point for the model-free M5 Oracle benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional, Sequence

from .adapter import (
    HotpotQADataAdapter,
    default_hotpotqa_path,
    default_smoke_config_path,
    load_manifest,
    load_smoke_config,
    repository_root,
    write_manifest,
)
from .benchmark import HotpotQAOracleBenchmark, write_markdown_report


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Run the deterministic, LLM-free M5 HotpotQA Oracle benchmark."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_hotpotqa_path(),
        help="Local Hugging Face DatasetDict saved with save_to_disk.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_smoke_config_path(),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data" / "splits" / "hotpotqa_smoke_manifest.json",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=root / "runs" / "m5_hotpotqa_smoke",
        help="Ignored directory for full source and reward JSONL files.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=root / "artifacts" / "m5_hotpotqa_smoke",
        help="Directory for compact benchmark and failure-audit artifacts.",
    )
    parser.add_argument(
        "--documentation-path",
        type=Path,
        default=root / "docs" / "m5_hotpotqa_oracle_benchmark.md",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Deterministically rebuild the smoke manifest from local data.",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict:
    config = load_smoke_config(args.config)
    adapter = HotpotQADataAdapter(args.data_path)
    manifest_path = args.manifest.expanduser().resolve()
    if args.rebuild_manifest or not manifest_path.is_file():
        manifest = adapter.build_smoke_manifest(config)
        write_manifest(manifest, manifest_path)
    else:
        manifest = load_manifest(manifest_path)

    artifacts = await HotpotQAOracleBenchmark(adapter).run(
        config=config,
        manifest=manifest,
        runtime_root=args.runtime_root,
        report_root=args.report_root,
    )
    documentation_path = write_markdown_report(
        artifacts.report, args.documentation_path
    )
    return {
        "digest": artifacts.report.digest,
        "records": len(artifacts.report.records),
        "failures": len(artifacts.report.failures),
        "manifest": str(manifest_path),
        "report": str(artifacts.report_path),
        "failure_audit": str(artifacts.failures_path),
        "documentation": str(documentation_path),
        "runtime_root": str(artifacts.runtime_root),
        "llm_calls": 0,
        "training_started": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
