#!/usr/bin/env python3
"""CPU diagnosis report for format-conditioned 4B frozen benches.

Reads Stage-3 JSONL, tool traces, and receipts. Not a main-score table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trinity.common.action_event_contract import (  # noqa: E402
    TRUNCATED_TOOL_CALL_SPAN_ERROR,
)
from trinity.common.e1_4b_format_conditioned import (  # noqa: E402
    HELDOUT_JOB,
    MEM_GOLD_JOB,
    MEM_NO_RETRIEVE_JOB,
    MEM_NORMAL_JOB,
    SIGNAL_JOB,
    load_lock,
    resolve_job_alias,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _job_dir(checkpoint_root: Path, job: str) -> Path:
    return checkpoint_root / "Trinity-RFT-AgeMem-M8" / job


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _pstdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _last_turn_by_execution(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("task_id") or ""), str(row.get("execution_id") or ""))
        previous = latest.get(key)
        if previous is None or int(row.get("round") or 0) >= int(previous.get("round") or 0):
            latest[key] = dict(row)
        if "task_score" in row:
            latest[key]["task_score"] = row["task_score"]
            latest[key]["found_answer"] = row.get("found_answer", latest[key].get("found_answer"))
            latest[key]["repaired"] = row.get("repaired", latest[key].get("repaired"))
    return list(latest.values())


def _signal_groups(last_turns: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in last_turns:
        task_id = str(row.get("task_id") or "")
        if "task_score" in row and row["task_score"] is not None:
            score = float(row["task_score"])
        else:
            score = 1.0 if row.get("found_answer") else 0.0
        groups[task_id].append(score)
    return groups


def _receipt_metrics(job_dir: Path) -> dict[str, Any]:
    receipts = sorted((job_dir / "receipts").glob("bench_step_*.json"))
    if not receipts:
        return {}
    payload = _load_json(receipts[0])
    metrics = payload.get("metrics") or {}
    summaries = payload.get("task_summaries") or []
    score_keys = [key for key in metrics if "task_score" in key]
    return {
        "receipt": receipts[0].name,
        "task_summaries": summaries,
        "task_score_metrics": {key: metrics[key] for key in score_keys},
        "failed_count": sum(int(item.get("failed_count") or 0) for item in summaries),
    }


def _retrieve_stats(trace_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = 0
    used = 0
    disabled = 0
    for row in trace_rows:
        if row.get("tool_name") != "Retrieve_memory":
            continue
        if row.get("event") == "usage" or row.get("kind") == "usage":
            continue
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        if row.get("status") in {None, "success", "cancelled", "error"} or "result" in row:
            if result or row.get("status"):
                attempted += 1
        if result.get("outcome") == "disabled":
            disabled += 1
        if (
            result.get("used_by_following_response") is True
            or usage.get("used_by_following_response") is True
        ):
            used += 1
    # usage events also mark used_by_following_response
    for row in trace_rows:
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        if usage.get("used_by_following_response") is True:
            used += 1
    return {
        "retrieve_attempted": attempted,
        "retrieve_used_by_following_response": used,
        "retrieve_disabled": disabled,
    }


def _truncated_rate(trace_rows: Sequence[Mapping[str, Any]], last_turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truncated_traces = 0
    for row in trace_rows:
        error = str(row.get("error") or "")
        if TRUNCATED_TOOL_CALL_SPAN_ERROR in error:
            truncated_traces += 1
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        if TRUNCATED_TOOL_CALL_SPAN_ERROR in str(result.get("validation_error") or ""):
            truncated_traces += 1
    truncated_turns = sum(
        1
        for row in last_turns
        if "truncated tool-call" in str(row.get("response_preview") or "").lower()
    )
    return {
        "truncated_tool_call_trace_events": truncated_traces,
        "truncated_tool_call_last_turns": truncated_turns,
        "last_turn_count": len(last_turns),
    }


def _join_failures(trace_rows: Sequence[Mapping[str, Any]]) -> int:
    failures = 0
    for row in trace_rows:
        error = str(row.get("error") or "")
        if "ActionContractError" in error:
            failures += 1
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        if result.get("validation_error") and "action" in str(result.get("validation_error")).lower():
            failures += 1
    return failures


def summarize_job(checkpoint_root: Path, job: str) -> dict[str, Any]:
    job_dir = _job_dir(checkpoint_root, job)
    trajectories = job_dir / "trajectories"
    turns = _read_jsonl(trajectories / "stage3_final_turn.jsonl")
    traces = _read_jsonl(trajectories / "tool_calls.jsonl")
    last_turns = _last_turn_by_execution(turns)
    groups = _signal_groups(last_turns) if job == SIGNAL_JOB else {}
    group_stds = {task_id: _pstdev(scores) for task_id, scores in groups.items()}
    nonzero = sum(1 for std in group_stds.values() if std > 0)
    format_turns = [row for row in turns if "task_score" not in row]
    if not format_turns:
        format_turns = turns
    summary = {
        "job": job,
        "present": job_dir.is_dir(),
        "stage3_rows": len(turns),
        "last_turns": len(last_turns),
        "found_answer_last": sum(1 for row in last_turns if row.get("found_answer")),
        "repaired_rows": sum(1 for row in turns if row.get("repaired")),
        "has_answer_tag_last": sum(1 for row in last_turns if row.get("has_answer_tag")),
        "found_answer_format_rows": sum(1 for row in format_turns if row.get("found_answer")),
        "format_row_count": len(format_turns),
        **_retrieve_stats(traces),
        **_truncated_rate(traces, last_turns),
        "action_contract_join_failures": _join_failures(traces),
        "receipts": _receipt_metrics(job_dir),
    }
    if job == SIGNAL_JOB:
        per_task = []
        for task_id, scores in sorted(groups.items()):
            per_task.append(
                {
                    "task_id": task_id,
                    "k": len(scores),
                    "last_step_f1": scores,
                    "mean": _mean(scores),
                    "group_std": group_stds[task_id],
                }
            )
        summary["per_task"] = per_task
        summary["tasks_with_group_std_gt_0"] = nonzero
        summary["task_count"] = len(groups)
        summary["fraction_group_std_gt_0"] = (
            (nonzero / len(groups)) if groups else None
        )
    last_scores = [
        float(row["task_score"])
        for row in last_turns
        if row.get("task_score") is not None
    ]
    summary["last_step_f1_mean"] = _mean(last_scores)
    return summary


def _print_value(label: str, value: Any) -> None:
    if isinstance(value, float):
        print(f"{label}: {value:.6f}")
        return
    print(f"{label}: {value}")


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"== {summary['job']} ==")
    if not summary.get("present"):
        print("MISSING_JOB_DIR")
        return
    for key in (
        "stage3_rows",
        "last_turns",
        "found_answer_last",
        "has_answer_tag_last",
        "repaired_rows",
        "retrieve_attempted",
        "retrieve_used_by_following_response",
        "retrieve_disabled",
        "truncated_tool_call_trace_events",
        "action_contract_join_failures",
        "last_step_f1_mean",
    ):
        _print_value(key, summary.get(key))
    receipts = summary.get("receipts") or {}
    if receipts:
        print("receipt", receipts.get("receipt"))
        print("failed_count", receipts.get("failed_count"))
        for key, value in (receipts.get("task_score_metrics") or {}).items():
            _print_value(key, value)
    if summary["job"] == SIGNAL_JOB:
        _print_value("task_count", summary.get("task_count"))
        _print_value("tasks_with_group_std_gt_0", summary.get("tasks_with_group_std_gt_0"))
        _print_value("fraction_group_std_gt_0", summary.get("fraction_group_std_gt_0"))
        for item in summary.get("per_task") or []:
            print(
                "task",
                item["task_id"],
                "k",
                item["k"],
                "mean",
                None if item["mean"] is None else round(item["mean"], 6),
                "group_std",
                round(item["group_std"], 6),
                "f1",
                item["last_step_f1"],
            )
    if summary.get("action_contract_join_failures"):
        print(
            "JOIN_FAILURES_MUST_BE_ZERO_TO_CONTINUE",
            summary["action_contract_join_failures"],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print format-conditioned 4B diagnosis metrics from a checkpoint root."
    )
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--job", default="")
    arguments = parser.parse_args(argv)
    checkpoint_root = Path(arguments.checkpoint_root).expanduser().resolve()
    lock = load_lock()
    jobs: Iterable[str]
    if arguments.job:
        jobs = [resolve_job_alias(arguments.job)]
    else:
        jobs = [lock["jobs"][key] for key in ("signal", "heldout", "mem_normal", "mem_no_retrieve", "mem_gold_support")]
    print("schema", lock["schema_version"])
    print("selection_status", lock.get("selection_status"))
    print(
        "interpretation: if gold-support is still mostly wrong, inspect answer/context first; "
        "if no-retrieve ≈ normal, Stage-3 memory may be unused (STM leftover, parametric knowledge, or task construction)."
    )
    summaries = [summarize_job(checkpoint_root, job) for job in jobs]
    for summary in summaries:
        print_summary(summary)
    present = {item["job"]: item for item in summaries if item.get("present")}
    if MEM_NORMAL_JOB in present and MEM_NO_RETRIEVE_JOB in present and MEM_GOLD_JOB in present:
        print("== 32-dev memory-necessity ==")
        for job in (MEM_NORMAL_JOB, MEM_NO_RETRIEVE_JOB, MEM_GOLD_JOB):
            _print_value(f"{job}/last_step_f1_mean", present[job].get("last_step_f1_mean"))
    if HELDOUT_JOB in present:
        print("== 2-row held-out regression ==")
        _print_value("heldout/last_step_f1_mean", present[HELDOUT_JOB].get("last_step_f1_mean"))
        print("closed format-group pair was 0.5 / 1.0 / 0.0; this job checks that pair still exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
