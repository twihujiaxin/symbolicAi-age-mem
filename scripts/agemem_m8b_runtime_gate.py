#!/usr/bin/env python3
"""Run the locked M8b regression suite and fail closed on drift."""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from typing import Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trinity.common.m8b_preflight import write_report  # noqa: E402


RUNTIME_GATE_SCHEMA_VERSION = "agemem.m8b_runtime_gate.v2"
DEFAULT_LOCK_PATH = REPOSITORY_ROOT / "configs/m8b_autodl_preflight.json"

M8A_MODULES = (
    "tests.buffer.task_file_reader_dataset_dict_test",
    "tests.common.hotpotqa_reward_profile_test",
    "tests.common.m8_action_event_contract_test",
    "tests.common.m8a_distractor_contract_test",
    "tests.common.m8a_lightweight_store_import_test",
    "tests.common.m8a_memory_isolation_test",
    "tests.common.m8a_packaging_contract_test",
    "tests.common.m8b_provider_usage_test",
    "tests.common.m8b_preflight_test",
    "tests.common.m8b_model_manifest_test",
    "tests.common.m8b_runtime_gate_test",
    "tests.common.m8b_runtime_fail_closed_test",
    "tests.common.m8b_postflight_test",
)

M1_TO_M7_MODULES = (
    "tests.common.trajectory_test",
    "tests.common.memory_store_test",
    "tests.common.toy_hotpotqa_environment_test",
    "tests.common.memory_oracle_reward_test",
    "tests.common.hotpotqa_oracle_benchmark_test",
    "tests.common.m6_schema_migration_test",
    "tests.common.m6_extractor_test",
    "tests.common.m6_state_tracker_test",
    "tests.common.m6_grounding_reward_test",
    "tests.common.m6_extraction_metrics_test",
    "tests.common.m6_extraction_benchmark_test",
    "tests.common.m6_false_reject_audit_test",
    "tests.common.m7_group_critic_schema_test",
    "tests.common.m7_group_critic_automaton_test",
    "tests.common.m7_group_critic_replay_test",
    "tests.common.m7_group_critic_benchmark_test",
)

TOOL_TRACE_MODULES = ("tests.common.tool_trace_test",)


def build_suite(scope: str) -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    names = list(M8A_MODULES)
    if scope == "all":
        names.extend(M1_TO_M7_MODULES)
        names.extend(TOOL_TRACE_MODULES)
    return loader.loadTestsFromNames(names)


def load_expected_test_counts(lock_path: Path) -> Mapping[str, int]:
    """Load the exact test counts covered by the committed runtime lock."""

    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read runtime gate lock: {error}") from error
    if not isinstance(lock, dict):
        raise ValueError("runtime gate lock must be a JSON object")
    runtime = lock.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("runtime gate lock is missing runtime")
    counts = runtime.get("expected_test_counts")
    if not isinstance(counts, dict) or set(counts) != {"m8a", "all"}:
        raise ValueError(
            "runtime.expected_test_counts must contain exactly m8a and all"
        )
    for scope, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"runtime.expected_test_counts.{scope} must be a positive integer"
            )
    if counts["all"] < counts["m8a"]:
        raise ValueError(
            "runtime.expected_test_counts.all must be at least m8a"
        )
    return counts


def _test_ids(entries) -> list[str]:
    return sorted(test.id() for test, *_unused in entries)


def result_report(
    result: unittest.TestResult,
    *,
    scope: str,
    expected_tests: Optional[int] = None,
    suite_tests: Optional[int] = None,
) -> dict:
    failure_ids = _test_ids(result.failures)
    error_ids = _test_ids(result.errors)
    skipped = sorted(
        (
            {"test_id": test.id(), "reason": reason}
            for test, reason in result.skipped
        ),
        key=lambda item: item["test_id"],
    )
    unexpected = sorted(test.id() for test in result.unexpectedSuccesses)
    count_match = (
        expected_tests is None
        or (
            suite_tests == expected_tests
            and result.testsRun == expected_tests
        )
    )
    passed = (
        not failure_ids
        and not error_ids
        and not skipped
        and not unexpected
        and count_match
    )
    return {
        "schema_version": RUNTIME_GATE_SCHEMA_VERSION,
        "scope": scope,
        "status": "pass" if passed else "fail",
        "expected_tests": expected_tests,
        "suite_tests": suite_tests,
        "tests_run": result.testsRun,
        "count_match": count_match,
        "failure_count": len(failure_ids),
        "error_count": len(error_ids),
        "skip_count": len(skipped),
        "unexpected_success_count": len(unexpected),
        "failures": failure_ids,
        "errors": error_ids,
        "skipped": skipped,
        "unexpected_successes": unexpected,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run M8b tests and reject failures, errors, and skips."
    )
    parser.add_argument("--scope", choices=("m8a", "all"), default="m8a")
    parser.add_argument("--lock", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument(
        "--output", default="runs/m8b_preflight/runtime_gate_report.json"
    )
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=2)
    arguments = parser.parse_args(argv)

    lock_path = Path(arguments.lock)
    if not lock_path.is_absolute():
        lock_path = REPOSITORY_ROOT / lock_path
    try:
        expected_tests = load_expected_test_counts(lock_path)[arguments.scope]
    except ValueError as error:
        parser.error(str(error))

    suite = build_suite(arguments.scope)
    suite_tests = suite.countTestCases()
    runner = unittest.TextTestRunner(verbosity=arguments.verbosity)
    result = runner.run(suite)
    report = result_report(
        result,
        scope=arguments.scope,
        expected_tests=expected_tests,
        suite_tests=suite_tests,
    )
    output = Path(arguments.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    write_report(report, output)
    print(
        "M8b runtime gate "
        f"{report['status'].upper()}: run={report['tests_run']} "
        f"expected={report['expected_tests']} suite={report['suite_tests']} "
        f"fail={report['failure_count']} error={report['error_count']} "
        f"skip={report['skip_count']}; report={output.resolve()}"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
