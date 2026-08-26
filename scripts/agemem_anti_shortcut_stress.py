#!/usr/bin/env python3
"""Run the expanded Stage 1/2 anti-shortcut stress experiment offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AgeMem_code_agentscope.toy_hotpotqa.shortcut_stress import (  # noqa: E402
    AntiShortcutStressReport,
    TokenCounterSpec,
    frozen_hf_token_counter,
    lexical_token_counter,
    run_anti_shortcut_stress,
    write_anti_shortcut_stress_report,
)


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "anti_shortcut_stress.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "anti_shortcut_stress"
DEFAULT_DOCS = REPOSITORY_ROOT / "docs" / "anti_shortcut_stress.md"


class StressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.anti_shortcut_stress.config.v1"]
    seeds: Tuple[int, ...] = Field(min_length=50)
    stage1_budgets: Tuple[int, ...] = Field(min_length=3)
    token_counter: Literal["unicode-lexical-v1"]


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_config(path: str | Path) -> StressConfig:
    selected = _repository_path(path)
    try:
        raw = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read stress config: {error}") from error
    config = StressConfig.model_validate(raw)
    if len(config.seeds) != len(set(config.seeds)) or any(
        seed < 0 for seed in config.seeds
    ):
        raise ValueError("config seeds must be unique and non-negative")
    if len(config.stage1_budgets) != len(set(config.stage1_budgets)) or any(
        budget <= 0 for budget in config.stage1_budgets
    ):
        raise ValueError("config budgets must be unique and positive")
    return config


def _counter_from_args(
    arguments: argparse.Namespace,
) -> Tuple[Callable[[str], int], TokenCounterSpec]:
    if arguments.tokenizer_path is None:
        if arguments.tokenizer_revision is not None:
            raise ValueError("--tokenizer-revision requires --tokenizer-path")
        return lexical_token_counter()
    if arguments.tokenizer_revision is None:
        raise ValueError("--tokenizer-path requires --tokenizer-revision")
    return frozen_hf_token_counter(
        _repository_path(arguments.tokenizer_path),
        repository_id=arguments.tokenizer_repository_id,
        revision=arguments.tokenizer_revision,
    )


def _canonical_report_bytes(report: AntiShortcutStressReport) -> bytes:
    return (
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic multi-task/multi-seed Stage 1 stress set and "
            "paired-counterfactual Stage 2 stress set."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--docs-path", default=str(DEFAULT_DOCS))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument(
        "--tokenizer-repository-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    arguments = parser.parse_args(argv)
    if arguments.no_write and arguments.verify_existing:
        parser.error("--no-write and --verify-existing are mutually exclusive")
    try:
        config = load_config(arguments.config)
        counter, counter_spec = _counter_from_args(arguments)
        report = run_anti_shortcut_stress(
            seeds=config.seeds,
            stage1_budgets=config.stage1_budgets,
            token_counter=counter,
            token_counter_spec=counter_spec,
        )
        output_dir = _repository_path(arguments.output_dir)
        docs_path = _repository_path(arguments.docs_path)
        if arguments.verify_existing:
            existing = output_dir / "anti_shortcut_stress.json"
            if not existing.is_file():
                raise ValueError(f"existing report is missing: {existing}")
            if existing.read_bytes() != _canonical_report_bytes(report):
                raise ValueError("existing stress report does not match this run")
        elif not arguments.no_write:
            write_anti_shortcut_stress_report(
                report,
                output_dir=output_dir,
                docs_path=docs_path,
            )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "status": "pass" if report.passed else "fail",
                "digest": report.digest,
                "token_counter": report.token_counter.name,
                "stage1_arms_per_policy": report.stage1.arm_count_per_policy,
                "stage2_arms_per_policy": report.stage2.arm_count_per_policy,
                "gate_count": len(report.gates),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
