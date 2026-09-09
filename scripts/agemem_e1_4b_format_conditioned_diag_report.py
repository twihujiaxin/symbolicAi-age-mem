#!/usr/bin/env python3
"""Complete post-hoc statistics for the format-conditioned 4B diagnosis.

Gold supporting facts are read only by this report. They never enter policy
observations or rewards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trinity.common.e1_4b_format_conditioned import (  # noqa: E402
    HELDOUT_JOB,
    MEM_GOLD_JOB,
    MEM_NO_RETRIEVE_JOB,
    MEM_NORMAL_JOB,
    SIGNAL_JOB,
    load_lock,
    resolve_job_alias,
)
from trinity.common.m8b_preflight import _canonical_json_sha256  # noqa: E402


# Keep the report CPU-only. Importing action_event_contract pulls the online
# Pydantic action schema even though the report only needs frozen wire values.
ACTION_CONTRACT_VERSION = "agemem.online_action_contract.v1"
TRUNCATED_TOOL_CALL_SPAN_ERROR = (
    "truncated tool-call JSON has no exact character span"
)
ACTION_EVENT_SCHEMA_VERSION = "agemem.action_event.v2"
DIAGNOSTIC_EXPERIENCE_SCHEMA_VERSION = "agemem.bench_experience_audit.v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def _job_dir(checkpoint_root: Path, job: str) -> Path:
    return checkpoint_root / "Trinity-RFT-AgeMem-M8" / job


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _pstdev(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _execution_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("task_id") or ""), str(row.get("execution_id") or "")


def _collapse_stage3_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge the before-score and after-score copies of one model turn."""

    merged: dict[tuple[str, str, int, bool], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = (
            *_execution_key(row),
            int(row.get("round") or 0),
            bool(row.get("repaired")),
        )
        previous = merged.setdefault(key, {})
        for field, value in row.items():
            if value is not None or field not in previous:
                previous[field] = value
    return list(merged.values())


def _last_turn_by_execution(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _collapse_stage3_rows(rows):
        key = _execution_key(row)
        previous = latest.get(key)
        coordinate = int(row.get("round") or 0), bool(row.get("repaired"))
        old_coordinate = (
            (int(previous.get("round") or 0), bool(previous.get("repaired")))
            if previous is not None
            else (-1, False)
        )
        if previous is None or coordinate >= old_coordinate:
            latest[key] = dict(row)
    return list(latest.values())


def _group_key(row: Mapping[str, Any]) -> str:
    return str(row.get("hotpot_id") or row.get("task_id") or "").strip()


def _signal_groups(
    last_turns: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in last_turns:
        key = _group_key(row)
        groups[key]
        if row.get("task_score") is not None:
            groups[key].append(float(row["task_score"]))
    return groups


def _format_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_execution: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in _collapse_stage3_rows(rows):
        by_execution[_execution_key(row)].append(row)

    native = nudged = repaired = repair_success = final_tagged = 0
    for execution_rows in by_execution.values():
        ordered = sorted(
            execution_rows,
            key=lambda row: (int(row.get("round") or 0), bool(row.get("repaired"))),
        )
        native += int(
            any(
                row.get("has_answer_tag")
                and not row.get("nudged")
                and not row.get("repaired")
                for row in ordered
            )
        )
        nudged += int(any(row.get("nudged") for row in ordered))
        repair_rows = [row for row in ordered if row.get("repaired")]
        repaired += int(bool(repair_rows))
        repair_success += int(
            bool(repair_rows) and any(row.get("has_answer_tag") for row in repair_rows)
        )
        final_tagged += int(bool(ordered[-1].get("has_answer_tag")))

    count = len(by_execution)
    return {
        "execution_count": count,
        "native_answer_tag_executions": native,
        "native_answer_tag_rate": _ratio(native, count),
        "nudge_triggered_executions": nudged,
        "nudge_trigger_rate": _ratio(nudged, count),
        "repair_triggered_executions": repaired,
        "repair_trigger_rate": _ratio(repaired, count),
        "repair_success_executions": repair_success,
        "repair_success_rate": _ratio(repair_success, repaired),
        "final_answer_tag_executions": final_tagged,
        "final_answer_tag_rate": _ratio(final_tagged, count),
    }


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


def _finish_rows(
    trace_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in trace_rows
        if row.get("phase") == "finish"
        or (
            row.get("phase") is None
            and (row.get("status") is not None or "result" in row)
        )
    ]


def _used_retrieval_call_ids(
    trace_rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    used = {
        str(row.get("call_id"))
        for row in trace_rows
        if row.get("phase") == "usage"
        and isinstance(row.get("usage"), dict)
        and row["usage"].get("used_by_following_response") is True
    }
    for row in _finish_rows(trace_rows):
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        if result.get("used_by_following_response") is True:
            used.add(str(row.get("call_id")))
    return used


def _retrieve_stats(trace_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finishes = [
        row
        for row in _finish_rows(trace_rows)
        if row.get("tool_name") == "Retrieve_memory"
    ]
    disabled = sum(
        1
        for row in finishes
        if isinstance(row.get("result"), dict)
        and row["result"].get("outcome") == "disabled"
    )
    return {
        "retrieve_attempted": len(finishes),
        "retrieve_used_by_following_response": len(
            _used_retrieval_call_ids(trace_rows)
        ),
        "retrieve_disabled": disabled,
    }


def _tool_call_truncation_stats(
    trace_rows: Sequence[Mapping[str, Any]],
    stage3_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trace_ids = set()
    for row in trace_rows:
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        if TRUNCATED_TOOL_CALL_SPAN_ERROR in str(row.get("error") or "") or (
            TRUNCATED_TOOL_CALL_SPAN_ERROR
            in str(result.get("validation_error") or "")
        ):
            trace_ids.add(str(row.get("call_id") or row.get("record_id") or ""))
    suspected_turns = sum(
        1
        for row in _collapse_stage3_rows(stage3_rows)
        if "<tool_call>" in str(row.get("response_preview") or "")
        and "</tool_call>" not in str(row.get("response_preview") or "")
    )
    return {
        "truncated_tool_call_trace_events": len(trace_ids),
        "suspected_truncated_tool_call_turns": suspected_turns,
    }


def _join_failures(trace_rows: Sequence[Mapping[str, Any]]) -> int:
    failures = 0
    for row in trace_rows:
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        validation_error = str(result.get("validation_error") or "")
        failures += int("ActionContractError" in str(row.get("error") or ""))
        failures += int(bool(validation_error) and "action" in validation_error.lower())
    return failures


def _max_response_tokens(job_dir: Path, fallback: int = 1024) -> int:
    config_path = job_dir / "config.json"
    if not config_path.is_file():
        return fallback


def _stable_action_id(event: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "rollout_id": event.get("rollout_id"),
            "stage_id": event.get("stage_id"),
            "timestep": event.get("timestep"),
            "assistant_turn_id": event.get("assistant_turn_id"),
            "action_index_in_turn": event.get("action_index_in_turn"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"agemem-act-{hashlib.sha256(payload).hexdigest()[:24]}"
    try:
        value = (_load_json(config_path).get("model") or {}).get(
            "max_response_tokens"
        )
        return int(value) if int(value) > 0 else fallback
    except (TypeError, ValueError, RuntimeError):
        return fallback


def _audit_action_contract(
    experience_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    max_response_tokens: int,
) -> dict[str, Any]:
    """Audit the serialized action/token/logprob/policy linkage."""

    errors: list[str] = []
    action_ids: set[str] = set()
    event_trace_ids: set[str] = set()
    policy_versions: set[str] = set()
    policy_versions_by_task: dict[str, set[str]] = defaultdict(set)
    rollout_ids: set[str] = set()
    execution_ids: set[str] = set()
    action_count = contract_count = max_hits = 0
    stage3_count = stage3_max_hits = 0
    failure_count = 0

    def fail(index: int, message: str) -> None:
        nonlocal failure_count
        failure_count += 1
        if len(errors) < 50:
            errors.append(f"experience[{index}]: {message}")

    def fail_global(message: str) -> None:
        nonlocal failure_count
        failure_count += 1
        if len(errors) < 50:
            errors.append(message)

    for index, row in enumerate(experience_rows):
        if row.get("diagnostic_schema_version") != DIAGNOSTIC_EXPERIENCE_SCHEMA_VERSION:
            fail(index, "missing or wrong diagnostic record schema version")
        response_length = row.get("response_length")
        if (
            isinstance(response_length, bool)
            or not isinstance(response_length, int)
            or response_length < 0
        ):
            fail(index, "missing or invalid response_length")
        hit_limit = (
            isinstance(response_length, int)
            and not isinstance(response_length, bool)
            and response_length >= max_response_tokens
        )
        max_hits += int(hit_limit)
        response_ids = row.get("response_token_ids")
        old_logprobs = row.get("old_logprobs")
        action_mask = row.get("action_mask")
        response_text = row.get("response_text")
        if (
            not isinstance(response_ids, list)
            or not response_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in response_ids
            )
        ):
            fail(index, "persisted response_token_ids are missing or invalid")
            response_ids = []
        if (
            not isinstance(old_logprobs, list)
            or len(old_logprobs) != len(response_ids)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in old_logprobs
            )
        ):
            fail(index, "persisted response token/logprob lengths differ or are invalid")
            old_logprobs = []
        if isinstance(response_length, int) and len(response_ids) != response_length:
            fail(index, "persisted response tokens differ from response_length")
        if not isinstance(action_mask, list) or len(action_mask) != len(response_ids):
            fail(index, "persisted action_mask length differs from response tokens")
        if not isinstance(response_text, str):
            fail(index, "persisted response_text is missing")
            response_text = ""
        eid = row.get("eid") if isinstance(row.get("eid"), dict) else {}
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        if any(field not in eid for field in ("batch", "task", "run", "step")):
            fail(index, "EID is incomplete")
        if (
            isinstance(info.get("trace_stage"), bool)
            or not isinstance(info.get("trace_stage"), int)
            or isinstance(info.get("trace_step"), bool)
            or not isinstance(info.get("trace_step"), int)
            or info.get("trace_step") != eid.get("step")
        ):
            fail(index, "trace stage/step is missing")
        if info.get("trace_stage") == 3:
            stage3_count += 1
            stage3_max_hits += int(hit_limit)
        if info.get("agemem_action_contract") != ACTION_CONTRACT_VERSION:
            fail(index, "missing or wrong action contract version")
            continue
        contract_count += 1
        if info.get("agemem_on_policy_eligible") is not True:
            fail(index, "not marked on-policy eligible")
        if info.get("agemem_trajectory_source") != "llm":
            fail(index, "trajectory source is not llm")
        if "agemem_action_event_drafts" in info:
            fail(index, "unfinished action drafts remain")
        policy_version = info.get("policy_version")
        if not isinstance(policy_version, str) or not policy_version:
            fail(index, "missing policy_version")
        else:
            policy_versions.add(policy_version)
            if info.get("model_version") is not None and policy_version != (
                f"model_version:{info['model_version']}"
            ):
                fail(index, "policy_version differs from model_version")

        task_id = f"{eid.get('batch', '')}/{eid.get('task', '')}"
        rollout_id = f"{task_id}/{eid.get('run', '')}"
        rollout_ids.add(rollout_id)
        execution_id = str(info.get("trace_execution_id") or "")
        if execution_id:
            execution_ids.add(execution_id)
        else:
            fail(index, "trace_execution_id is missing")
        events = info.get("agemem_action_events")
        if not isinstance(events, list):
            fail(index, "action events are not a list")
            continue
        spans = info.get("agemem_action_character_spans")
        if not isinstance(spans, list) or len(spans) != len(events):
            fail(index, "character span count differs from action count")
            spans = []
        raw_offsets = info.get("agemem_response_token_char_offsets")
        offsets: list[tuple[int, int]] = []
        previous_offset_end = 0
        if not isinstance(raw_offsets, list) or len(raw_offsets) != len(response_ids):
            fail(index, "response token offset count differs from response tokens")
        else:
            for raw_offset in raw_offsets:
                if (
                    not isinstance(raw_offset, (list, tuple))
                    or len(raw_offset) != 2
                    or isinstance(raw_offset[0], bool)
                    or isinstance(raw_offset[1], bool)
                    or not isinstance(raw_offset[0], int)
                    or not isinstance(raw_offset[1], int)
                    or raw_offset[0] != previous_offset_end
                    or not raw_offset[0] <= raw_offset[1] <= len(response_text)
                ):
                    fail(index, "response token offsets are invalid")
                    offsets = []
                    break
                offsets.append((raw_offset[0], raw_offset[1]))
                previous_offset_end = raw_offset[1]
            if offsets and previous_offset_end != len(response_text):
                fail(index, "response token offsets do not cover response_text")
                offsets = []

        tool_call_ids = info.get("tool_call_ids")
        if not isinstance(tool_call_ids, list) or any(
            not isinstance(call_id, str) or not call_id for call_id in tool_call_ids
        ):
            fail(index, "tool_call_ids are missing or invalid")
            tool_call_ids = []
        event_call_ids: list[str] = []
        previous_token_end_by_turn: dict[int, int] = {}
        previous_char_end = -1

        for event_index, event in enumerate(events):
            action_count += 1
            if not isinstance(event, dict):
                fail(index, f"action[{event_index}] is not an object")
                continue
            action_id = event.get("action_id")
            if not isinstance(action_id, str) or not action_id:
                fail(index, f"action[{event_index}] has no action_id")
            elif action_id in action_ids:
                fail(index, f"duplicate action_id {action_id}")
            else:
                action_ids.add(action_id)
            if event.get("schema_version") != ACTION_EVENT_SCHEMA_VERSION:
                fail(index, f"action[{event_index}] has wrong schema_version")
            if event.get("source") != "llm":
                fail(index, f"action[{event_index}] source is not llm")
            if event.get("action_index_in_turn") != event_index:
                fail(index, f"action[{event_index}] index is not contiguous")
            if action_id != _stable_action_id(event):
                fail(index, f"action[{event_index}] action_id is not deterministic")
            if event.get("task_id") != task_id or event.get("rollout_id") != rollout_id:
                fail(index, f"action[{event_index}] identity differs from EID")
            if event.get("stage_id") != info.get("trace_stage") or event.get(
                "timestep"
            ) != info.get("trace_step"):
                fail(index, f"action[{event_index}] coordinate differs from Experience")
            token_ids = event.get("response_token_ids")
            logprobs = event.get("old_logprobs")
            if token_ids != response_ids:
                fail(index, f"action[{event_index}] tokens differ from Experience")
            if (
                not isinstance(logprobs, list)
                or len(logprobs) != len(old_logprobs)
                or any(
                    not isinstance(actual, (int, float))
                    or abs(float(actual) - float(expected)) > 1e-6
                    for actual, expected in zip(logprobs, old_logprobs)
                )
            ):
                fail(index, f"action[{event_index}] logprobs differ from Experience")
            start, end = event.get("token_start"), event.get("token_end")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(response_ids)
            ):
                fail(index, f"action[{event_index}] has an invalid token span")
            assistant_turn_id = event.get("assistant_turn_id")
            if isinstance(assistant_turn_id, bool) or not isinstance(
                assistant_turn_id, int
            ):
                fail(index, f"action[{event_index}] assistant_turn_id is invalid")
            elif isinstance(start, int) and isinstance(end, int):
                previous_token_end = previous_token_end_by_turn.get(
                    assistant_turn_id, -1
                )
                if start < previous_token_end:
                    fail(index, f"action[{event_index}] token span overlaps")
                previous_token_end_by_turn[assistant_turn_id] = end
            if event.get("policy_version") != policy_version:
                fail(index, f"action[{event_index}] policy_version differs")
            result = event.get("result") if isinstance(event.get("result"), dict) else {}
            call_id = result.get("trace_call_id")
            if not isinstance(call_id, str) or not call_id:
                fail(index, f"action[{event_index}] has no trace_call_id")
            else:
                event_trace_ids.add(call_id)
                event_call_ids.append(call_id)
            if event_index < len(spans):
                span = spans[event_index]
                if not isinstance(span, dict) or span.get("action_id") != action_id:
                    fail(index, f"action[{event_index}] does not join character span")
                else:
                    char_start, char_end = span.get("char_start"), span.get("char_end")
                    if (
                        isinstance(char_start, bool)
                        or isinstance(char_end, bool)
                        or not isinstance(char_start, int)
                        or not isinstance(char_end, int)
                        or not 0 <= char_start < char_end <= len(response_text)
                        or char_start < previous_char_end
                        or response_text[char_start:char_end]
                        != event.get("action_text")
                    ):
                        fail(index, f"action[{event_index}] character span is invalid")
                    else:
                        covered = [
                            token_index
                            for token_index, (offset_start, offset_end) in enumerate(
                                offsets
                            )
                            if offset_end > char_start and offset_start < char_end
                        ]
                        expected_span = (
                            (covered[0], covered[-1] + 1) if covered else None
                        )
                        if expected_span != (start, end):
                            fail(
                                index,
                                f"action[{event_index}] token/character spans differ",
                            )
                        previous_char_end = char_end

        if tool_call_ids != event_call_ids:
            fail(index, "tool_call_ids do not join ActionEvents one-to-one")

    # Build this after EID parsing; every task/K-group must be sampled from one
    # frozen policy version.
    for row in experience_rows:
        eid = row.get("eid") if isinstance(row.get("eid"), dict) else {}
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        policy_version = info.get("policy_version")
        if isinstance(policy_version, str) and policy_version:
            task_id = f"{eid.get('batch', '')}/{eid.get('task', '')}"
            policy_versions_by_task[task_id].add(policy_version)
    mixed_policy_tasks = sorted(
        task_id
        for task_id, versions in policy_versions_by_task.items()
        if len(versions) > 1
    )
    if mixed_policy_tasks:
        fail_global(
            f"{len(mixed_policy_tasks)} task groups contain mixed policy versions"
        )

    finish_ids = {
        str(row.get("call_id"))
        for row in _finish_rows(trace_rows)
        if row.get("call_id")
    }
    missing_events = finish_ids - event_trace_ids
    missing_traces = event_trace_ids - finish_ids
    if missing_events:
        fail_global(f"{len(missing_events)} finished tool calls have no ActionEvent")
    if missing_traces:
        fail_global(f"{len(missing_traces)} ActionEvents have no finished tool trace")
    return {
        "experience_file_present": bool(experience_rows),
        "experience_count": len(experience_rows),
        "contract_experience_count": contract_count,
        "rollout_count_in_experiences": len(rollout_ids),
        "execution_count_in_experiences": len(execution_ids),
        "action_event_count": action_count,
        "unique_action_id_count": len(action_ids),
        "policy_versions": sorted(policy_versions),
        "mixed_policy_task_count": len(mixed_policy_tasks),
        "mixed_policy_tasks": mixed_policy_tasks,
        "max_response_tokens": max_response_tokens,
        "max_token_hit_experiences": max_hits,
        "max_token_hit_rate": _ratio(max_hits, len(experience_rows)),
        "stage3_experiences": stage3_count,
        "stage3_max_token_hit_experiences": stage3_max_hits,
        "stage3_max_token_hit_rate": _ratio(stage3_max_hits, stage3_count),
        "finished_tool_call_count": len(finish_ids),
        "unjoined_finished_tool_call_count": len(missing_events),
        "unjoined_action_event_count": len(missing_traces),
        "action_contract_failure_count": failure_count,
        "action_contract_failures": errors[:50],
    }


def _normalise_evidence(value: Any) -> str:
    return " ".join(value.casefold().split()) if isinstance(value, str) else ""


def _contains_fact(container: Any, fact: str) -> bool:
    normalised = _normalise_evidence(container)
    return bool(fact and normalised and fact in normalised)


def _extract_supporting_sentences(row: Mapping[str, Any]) -> list[str]:
    """Resolve HotpotQA supporting pointers without importing the GPU runtime."""

    supporting = row.get("supporting_facts")
    context = row.get("context")
    if not isinstance(supporting, dict) or not isinstance(context, dict):
        return []
    support_titles = supporting.get("title")
    support_indices = supporting.get("sent_id")
    titles = context.get("title")
    sentence_groups = context.get("sentences")
    if not all(
        isinstance(value, (list, tuple))
        for value in (support_titles, support_indices, titles, sentence_groups)
    ):
        return []
    if len(support_titles) != len(support_indices):
        return []
    title_to_index = {str(title).strip(): index for index, title in enumerate(titles)}
    sentences = []
    for title, raw_index in zip(support_titles, support_indices):
        try:
            sentence_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        title_index = title_to_index.get(str(title).strip())
        if title_index is None or not 0 <= title_index < len(sentence_groups):
            continue
        group = sentence_groups[title_index]
        if isinstance(group, (list, tuple)) and 0 <= sentence_index < len(group):
            sentences.append(str(group[sentence_index]))
    return sentences


def _load_support_by_hotpot_id(
    hotpotqa_path: Path,
    lock: Mapping[str, Any],
    job: str,
) -> dict[str, list[str]]:
    from datasets import load_from_disk

    if job == SIGNAL_JOB:
        split, locked_rows = "train", lock["fixed_train_rows"]
    elif job == HELDOUT_JOB:
        split, locked_rows = "validation", lock["held_out_rows"]
    else:
        split, locked_rows = "validation", lock["fixed_dev_rows"]
    dataset = load_from_disk(str(hotpotqa_path))
    support = {}
    for locked in locked_rows:
        row = dataset[split][int(locked["source_index"])]
        hotpot_id = str(row.get("id") or "")
        if hotpot_id != str(locked["hotpot_id"]):
            raise RuntimeError(f"HotpotQA ID drift for {locked['hotpot_id']}")
        if _canonical_json_sha256(row) != locked["content_sha256"]:
            raise RuntimeError(f"HotpotQA content drift for {hotpot_id}")
        support[hotpot_id] = _extract_supporting_sentences(row)
    return support


def _support_trace_stats(
    trace_rows: Sequence[Mapping[str, Any]],
    stage3_rows: Sequence[Mapping[str, Any]],
    support_by_hotpot_id: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    execution_to_hotpot = {
        str(row.get("execution_id")): str(row.get("hotpot_id"))
        for row in stage3_rows
        if row.get("execution_id") and row.get("hotpot_id")
    }
    traces_by_execution: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        if row.get("execution_id"):
            traces_by_execution[str(row["execution_id"])].append(row)
    used_call_ids = _used_retrieval_call_ids(trace_rows)

    fact_total = saved_total = retrieved_total = exposed_total = 0
    any_saved = all_saved = any_retrieved = all_retrieved = 0
    any_exposed = all_exposed = mapped = 0
    per_execution = []
    for execution_id, hotpot_id in sorted(execution_to_hotpot.items()):
        raw_facts = support_by_hotpot_id.get(hotpot_id)
        if raw_facts is None:
            continue
        mapped += 1
        facts = [_normalise_evidence(item) for item in raw_facts]
        facts = [item for item in facts if item]
        saved_contents: list[str] = []
        retrieved_by_call: dict[str, list[str]] = defaultdict(list)
        for row in _finish_rows(traces_by_execution.get(execution_id, [])):
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            arguments = (
                row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
            )
            if row.get("tool_name") in {"Add_memory", "Update_memory"} and result.get(
                "effect_applied"
            ) is True:
                if isinstance(arguments.get("content"), str):
                    saved_contents.append(arguments["content"])
            if row.get("tool_name") == "Retrieve_memory":
                call_id = str(row.get("call_id") or "")
                items = result.get("items") if isinstance(result.get("items"), list) else []
                retrieved_by_call[call_id].extend(
                    str(item["content"])
                    for item in items
                    if isinstance(item, dict) and isinstance(item.get("content"), str)
                )
        retrieved_contents = [item for values in retrieved_by_call.values() for item in values]
        exposed_contents = [
            item
            for call_id, values in retrieved_by_call.items()
            if call_id in used_call_ids
            for item in values
        ]
        saved = [any(_contains_fact(item, fact) for item in saved_contents) for fact in facts]
        retrieved = [
            any(_contains_fact(item, fact) for item in retrieved_contents) for fact in facts
        ]
        exposed = [
            any(_contains_fact(item, fact) for item in exposed_contents) for fact in facts
        ]
        fact_total += len(facts)
        saved_total += sum(saved)
        retrieved_total += sum(retrieved)
        exposed_total += sum(exposed)
        any_saved += int(any(saved))
        all_saved += int(bool(facts) and all(saved))
        any_retrieved += int(any(retrieved))
        all_retrieved += int(bool(facts) and all(retrieved))
        any_exposed += int(any(exposed))
        all_exposed += int(bool(facts) and all(exposed))
        per_execution.append(
            {
                "execution_id": execution_id,
                "hotpot_id": hotpot_id,
                "support_fact_count": len(facts),
                "saved_support_fact_count": sum(saved),
                "retrieved_support_fact_count": sum(retrieved),
                "exposed_support_fact_count": sum(exposed),
            }
        )
    rollout_count = len(execution_to_hotpot)
    return {
        "available": True,
        "rollout_count": rollout_count,
        "mapped_rollout_count": mapped,
        "unmapped_rollout_count": rollout_count - mapped,
        "support_fact_instances": fact_total,
        "saved_support_fact_instances": saved_total,
        "saved_support_fact_recall": _ratio(saved_total, fact_total),
        "retrieved_support_fact_instances": retrieved_total,
        "retrieved_support_fact_recall": _ratio(retrieved_total, fact_total),
        "exposed_support_fact_instances": exposed_total,
        "exposed_support_fact_recall": _ratio(exposed_total, fact_total),
        "rollouts_with_any_support_saved": any_saved,
        "rollouts_with_all_support_saved": all_saved,
        "rollouts_with_any_support_retrieved": any_retrieved,
        "rollouts_with_all_support_retrieved": all_retrieved,
        "rollouts_with_any_support_exposed": any_exposed,
        "rollouts_with_all_support_exposed": all_exposed,
        "per_execution": per_execution,
        "interpretation": (
            "Exact-text audit. Exposed means retrieved evidence entered a following "
            "model input; it does not prove causal use in the answer."
        ),
    }


def summarize_job(
    checkpoint_root: Path,
    job: str,
    *,
    support_by_hotpot_id: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    job_dir = _job_dir(checkpoint_root, job)
    trajectories = job_dir / "trajectories"
    turns = _read_jsonl(trajectories / "stage3_final_turn.jsonl")
    traces = _read_jsonl(trajectories / "tool_calls.jsonl")
    experiences = _read_jsonl(job_dir / "buffer" / "explorer_output.jsonl")
    last_turns = _last_turn_by_execution(turns)
    groups = _signal_groups(last_turns) if job == SIGNAL_JOB else {}
    group_stds = {task_id: _pstdev(scores) for task_id, scores in groups.items()}
    nonzero = sum(std > 0 for std in group_stds.values())
    collapsed_turns = _collapse_stage3_rows(turns)
    format_turns = [row for row in collapsed_turns if "task_score" not in row]
    if not format_turns:
        format_turns = collapsed_turns
    summary = {
        "job": job,
        "present": job_dir.is_dir(),
        "stage3_rows": len(turns),
        "stage3_model_turns": len(collapsed_turns),
        "last_turns": len(last_turns),
        "last_turn_count": len(last_turns),
        "last_turns_missing_task_score": sum(
            row.get("task_score") is None for row in last_turns
        ),
        "found_answer_last": sum(bool(row.get("found_answer")) for row in last_turns),
        "has_answer_tag_last": sum(bool(row.get("has_answer_tag")) for row in last_turns),
        "repaired_rows": sum(bool(row.get("repaired")) for row in collapsed_turns),
        "found_answer_format_rows": sum(
            bool(row.get("found_answer")) for row in format_turns
        ),
        "format_row_count": len(format_turns),
        **_format_stats(turns),
        **_retrieve_stats(traces),
        **_tool_call_truncation_stats(traces, turns),
        "action_contract_join_failures": _join_failures(traces),
        "action_contract_join_failures_in_trace": _join_failures(traces),
        "action_contract": _audit_action_contract(
            experiences,
            traces,
            max_response_tokens=_max_response_tokens(job_dir),
        ),
        "receipts": _receipt_metrics(job_dir),
        "support_evidence": (
            _support_trace_stats(traces, turns, support_by_hotpot_id)
            if support_by_hotpot_id is not None
            else {"available": False, "reason": "HOTPOTQA_PATH was not provided"}
        ),
    }
    if job == SIGNAL_JOB:
        summary["per_task"] = [
            {
                "task_id": task_id,
                "k": len(scores),
                "last_step_f1": scores,
                "mean": _mean(scores),
                "group_std": group_stds[task_id],
            }
            for task_id, scores in sorted(groups.items())
        ]
        summary["tasks_with_group_std_gt_0"] = nonzero
        summary["task_count"] = len(groups)
        summary["fraction_group_std_gt_0"] = _ratio(nonzero, len(groups))
        summary["incomplete_k_groups"] = sum(len(scores) != 4 for scores in groups.values())
    scores = [
        float(row["task_score"])
        for row in last_turns
        if row.get("task_score") is not None
    ]
    summary["last_step_f1_mean"] = _mean(scores)
    return summary


def _print_value(label: str, value: Any) -> None:
    print(f"{label}: {value:.6f}" if isinstance(value, float) else f"{label}: {value}")


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"== {summary['job']} ==")
    if not summary.get("present"):
        print("MISSING_JOB_DIR")
        return
    for key in (
        "stage3_rows",
        "stage3_model_turns",
        "last_turns",
        "last_turns_missing_task_score",
        "found_answer_last",
        "has_answer_tag_last",
        "native_answer_tag_rate",
        "nudge_trigger_rate",
        "repair_trigger_rate",
        "repair_success_rate",
        "final_answer_tag_rate",
        "retrieve_attempted",
        "retrieve_used_by_following_response",
        "retrieve_disabled",
        "truncated_tool_call_trace_events",
        "suspected_truncated_tool_call_turns",
        "action_contract_join_failures_in_trace",
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
        for key in (
            "task_count",
            "tasks_with_group_std_gt_0",
            "fraction_group_std_gt_0",
            "incomplete_k_groups",
        ):
            _print_value(key, summary.get(key))
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
    print("-- action/token/logprob/policy audit --")
    contract = summary.get("action_contract") or {}
    for key in (
        "experience_file_present",
        "experience_count",
        "contract_experience_count",
        "rollout_count_in_experiences",
        "action_event_count",
        "unique_action_id_count",
        "policy_versions",
        "mixed_policy_task_count",
        "max_token_hit_rate",
        "stage3_max_token_hit_rate",
        "finished_tool_call_count",
        "unjoined_finished_tool_call_count",
        "unjoined_action_event_count",
        "action_contract_failure_count",
    ):
        _print_value(key, contract.get(key))
    for failure in contract.get("action_contract_failures") or []:
        print("CONTRACT_FAILURE", failure)
    print("-- supporting-fact exact-text audit --")
    support = summary.get("support_evidence") or {}
    if not support.get("available"):
        print("UNAVAILABLE", support.get("reason"))
    else:
        for key in (
            "rollout_count",
            "mapped_rollout_count",
            "unmapped_rollout_count",
            "support_fact_instances",
            "saved_support_fact_recall",
            "retrieved_support_fact_recall",
            "exposed_support_fact_recall",
            "rollouts_with_any_support_saved",
            "rollouts_with_all_support_saved",
            "rollouts_with_any_support_retrieved",
            "rollouts_with_all_support_retrieved",
            "rollouts_with_any_support_exposed",
            "rollouts_with_all_support_exposed",
        ):
            _print_value(key, support.get(key))
        print("support_interpretation", support.get("interpretation"))


def _strict_failures(summary: Mapping[str, Any], lock: Mapping[str, Any]) -> list[str]:
    if not summary.get("present"):
        return [f"missing job directory for {summary.get('job')}"]
    failures = []
    job = summary["job"]
    expected = (
        int(lock["train_size"]) * int(lock["signal_repeat_times"])
        if job == SIGNAL_JOB
        else 2
        if job == HELDOUT_JOB
        else int(lock["dev_size"])
    )
    if summary.get("last_turns") != expected:
        failures.append(f"{job}: expected {expected} rollouts, got {summary.get('last_turns')}")
    if summary.get("last_turns_missing_task_score"):
        failures.append(f"{job}: final F1 is missing for one or more rollouts")
    if job == SIGNAL_JOB and summary.get("incomplete_k_groups"):
        failures.append(f"{job}: incomplete K=4 groups")
    receipts = summary.get("receipts") or {}
    if not receipts or receipts.get("failed_count"):
        failures.append(f"{job}: missing receipt or nonzero failed_count")
    contract = summary.get("action_contract") or {}
    if not contract.get("experience_file_present"):
        failures.append(f"{job}: missing buffer/explorer_output.jsonl")
    if contract.get("action_contract_failure_count"):
        failures.append(f"{job}: action contract audit failed")
    if contract.get("execution_count_in_experiences") != summary.get("last_turns"):
        failures.append(f"{job}: Stage-3 executions do not all join persisted Experiences")
    if summary.get("action_contract_join_failures_in_trace"):
        failures.append(f"{job}: ActionContractError found in tool trace")
    support = summary.get("support_evidence") or {}
    if not support.get("available"):
        failures.append(f"{job}: supporting-fact audit is unavailable")
    elif support.get("unmapped_rollout_count"):
        failures.append(f"{job}: rollouts could not be mapped to frozen support labels")
    elif support.get("mapped_rollout_count") and not support.get(
        "support_fact_instances"
    ):
        failures.append(f"{job}: frozen supporting facts resolved to zero sentences")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print complete format-conditioned 4B diagnosis statistics."
    )
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--job", default="")
    parser.add_argument(
        "--hotpotqa-path",
        default=os.environ.get("HOTPOTQA_PATH", ""),
        help="DatasetDict used only for post-hoc supporting-fact matching.",
    )
    parser.add_argument("--json-output", default="")
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args(argv)
    checkpoint_root = Path(arguments.checkpoint_root).expanduser().resolve()
    lock = load_lock()
    jobs: Iterable[str] = (
        [resolve_job_alias(arguments.job)]
        if arguments.job
        else [
            lock["jobs"][key]
            for key in (
                "signal",
                "heldout",
                "mem_normal",
                "mem_no_retrieve",
                "mem_gold_support",
            )
        ]
    )
    jobs = list(jobs)
    hotpotqa_path = (
        Path(arguments.hotpotqa_path).expanduser().resolve()
        if arguments.hotpotqa_path
        else None
    )
    if hotpotqa_path is not None and not hotpotqa_path.is_dir():
        parser.error("--hotpotqa-path is not a DatasetDict directory")
    print("schema", lock["schema_version"])
    print("selection_status", lock.get("selection_status"))
    print(
        "interpretation: native format excludes nudge/repair; max-token hits are "
        "the truncation proxy; support matching is post-hoc exact text."
    )
    summaries = []
    for job in jobs:
        support = (
            _load_support_by_hotpot_id(hotpotqa_path, lock, job)
            if hotpotqa_path is not None
            else None
        )
        summaries.append(
            summarize_job(checkpoint_root, job, support_by_hotpot_id=support)
        )
    for summary in summaries:
        print_summary(summary)
    present = {item["job"]: item for item in summaries if item.get("present")}
    if all(job in present for job in (MEM_NORMAL_JOB, MEM_NO_RETRIEVE_JOB, MEM_GOLD_JOB)):
        print("== 32-dev memory-necessity ==")
        for job in (MEM_NORMAL_JOB, MEM_NO_RETRIEVE_JOB, MEM_GOLD_JOB):
            _print_value(
                f"{job}/last_step_f1_mean", present[job].get("last_step_f1_mean")
            )
    if HELDOUT_JOB in present:
        print("== 2-row held-out regression ==")
        _print_value("heldout/last_step_f1_mean", present[HELDOUT_JOB].get("last_step_f1_mean"))
    failures = [failure for item in summaries for failure in _strict_failures(item, lock)]
    payload = {
        "schema_version": "agemem.format_conditioned_diagnosis_report.v2",
        "lock_schema_version": lock["schema_version"],
        "selection_status": lock.get("selection_status"),
        "validation_failures": failures,
        "jobs": summaries,
    }
    if arguments.json_output:
        output_path = Path(arguments.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print("json_output", output_path)
    for failure in failures:
        label = "STRICT_FAILURE" if arguments.strict else "VALIDATION_WARNING"
        print(label, failure, file=sys.stderr)
    return 2 if arguments.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
