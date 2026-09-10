#!/usr/bin/env python3
"""Strict CPU replay of terminal, Flat-Oracle, and Oracle-DFA rewards.

The policy observations are immutable inputs produced by an earlier frozen
sampling job. Gold HotpotQA support is loaded only after sampling and is used
only by this reward audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AgeMem_code_agentscope.action_schema import ActionEvent  # noqa: E402
from trinity.common.e3_oracle_dfa import (  # noqa: E402
    ACTION_LOGIC_REWARD_CEILING,
    ACTION_MILESTONE_APS,
    DEFAULT_MAX_STEPS,
    FLAT_REWARD_VERSION,
    REWARD_VERSION,
    replay_hotpotqa_oracle_comparison,
)
from trinity.common.m8b_preflight import _canonical_json_sha256  # noqa: E402


REPORT_SCHEMA_VERSION = "agemem.e3_oracle_offline_comparison.v1"
EXPECTED_DIAGNOSTIC_SCHEMA = "agemem.bench_experience_audit.v1"
EXPECTED_ACTION_SCHEMA = "agemem.action_event.v2"
ARMS = ("terminal_only", "flat_oracle", "oracle_dfa")
STAGE1_MAX_SENTENCES_PER_TITLE = 10


def _observed_context_sentences(context: Mapping[str, Any]) -> list[str]:
    titles = context.get("title") or []
    sentence_groups = context.get("sentences") or []
    if not isinstance(titles, (list, tuple)) or not isinstance(
        sentence_groups, (list, tuple)
    ):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for sentences in sentence_groups[: len(titles)]:
        if not isinstance(sentences, (list, tuple)):
            continue
        for sentence in sentences[:STAGE1_MAX_SENTENCES_PER_TITLE]:
            text = str(sentence).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
    return result


def _supporting_sentences(row: Mapping[str, Any]) -> list[str]:
    supporting = row.get("supporting_facts")
    context = row.get("context")
    if not isinstance(supporting, Mapping) or not isinstance(context, Mapping):
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
    result: list[str] = []
    for title, raw_sentence_index in zip(support_titles, support_indices):
        title_index = title_to_index.get(str(title).strip())
        try:
            sentence_index = int(raw_sentence_index)
        except (TypeError, ValueError):
            continue
        if title_index is None or not 0 <= title_index < len(sentence_groups):
            continue
        sentences = sentence_groups[title_index]
        if isinstance(sentences, (list, tuple)) and 0 <= sentence_index < len(
            sentences
        ):
            result.append(str(sentences[sentence_index]))
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_turns(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for raw in rows:
        execution_id = str(raw.get("execution_id") or "")
        if not execution_id:
            raise RuntimeError("Stage-3 turn is missing execution_id")
        row = dict(raw)
        previous = latest.get(execution_id)
        coordinate = (int(row.get("round") or 0), bool(row.get("repaired")))
        previous_coordinate = (
            (int(previous.get("round") or 0), bool(previous.get("repaired")))
            if previous is not None
            else (-1, False)
        )
        if previous is None or coordinate >= previous_coordinate:
            latest[execution_id] = row
    return latest


def _load_locked_rows(
    *, hotpotqa_path: Path, lock: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    from datasets import load_from_disk

    selections = lock.get("fixed_train_rows")
    if not isinstance(selections, list) or not selections:
        raise RuntimeError("lock has no fixed_train_rows")
    dataset = load_from_disk(str(hotpotqa_path))
    result: dict[str, dict[str, Any]] = {}
    for selection in selections:
        if not isinstance(selection, Mapping):
            raise RuntimeError("fixed_train_rows contains a non-object")
        row = dict(dataset["train"][int(selection["source_index"])])
        hotpot_id = str(row.get("id") or "")
        if hotpot_id != str(selection.get("hotpot_id") or ""):
            raise RuntimeError(f"HotpotQA ID drift for {selection.get('hotpot_id')}")
        if _canonical_json_sha256(row) != selection.get("content_sha256"):
            raise RuntimeError(f"HotpotQA content drift for {hotpot_id}")
        result[hotpot_id] = row
    return result


def _experience_rollouts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[ActionEvent]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    actions: dict[str, list[ActionEvent]] = defaultdict(list)
    outcomes: dict[str, dict[str, Any]] = {}
    rollout_ids: dict[str, str] = {}
    seen_action_ids: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("diagnostic_schema_version") != EXPECTED_DIAGNOSTIC_SCHEMA:
            raise RuntimeError(f"experience[{index}] has wrong diagnostic schema")
        info = row.get("info")
        if not isinstance(info, Mapping):
            raise RuntimeError(f"experience[{index}] has no info object")
        execution_id = str(info.get("trace_execution_id") or "")
        if not execution_id:
            raise RuntimeError(f"experience[{index}] has no trace_execution_id")
        raw_events = info.get("agemem_action_events")
        if not isinstance(raw_events, list):
            raise RuntimeError(f"experience[{index}] has no ActionEvent list")
        eid = row.get("eid") if isinstance(row.get("eid"), Mapping) else {}
        expected_rollout = (
            f"{eid.get('batch', '')}/{eid.get('task', '')}/{eid.get('run', '')}"
        )
        previous_rollout = rollout_ids.setdefault(execution_id, expected_rollout)
        if previous_rollout != expected_rollout:
            raise RuntimeError("one execution maps to multiple rollout IDs")
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                raise RuntimeError("ActionEvent row is not an object")
            if raw_event.get("schema_version") != EXPECTED_ACTION_SCHEMA:
                raise RuntimeError("ActionEvent schema version drifted")
            event = ActionEvent.model_validate_json(
                json.dumps(raw_event, ensure_ascii=False, allow_nan=False)
            )
            if event.action_id in seen_action_ids:
                raise RuntimeError(f"duplicate ActionEvent {event.action_id}")
            seen_action_ids.add(event.action_id)
            actions[execution_id].append(event)

        outcome = outcomes.setdefault(execution_id, {})
        for key in ("task_score", "found_answer", "answer_exact_match"):
            value = info.get(key)
            if value is None:
                continue
            if key in outcome and outcome[key] != value:
                raise RuntimeError(f"inconsistent {key} within execution {execution_id}")
            outcome[key] = value

    for execution_id, sequence in actions.items():
        sequence.sort(key=lambda item: (item.assistant_turn_id, item.action_index_in_turn))
    for execution_id in rollout_ids:
        actions.setdefault(execution_id, [])
    return dict(actions), outcomes, rollout_ids


def _validate_trace_join(
    *,
    actions: Mapping[str, Sequence[ActionEvent]],
    trace_rows: Sequence[Mapping[str, Any]],
) -> None:
    finish_by_id: dict[str, Mapping[str, Any]] = {}
    for row in trace_rows:
        if str(row.get("phase") or "") != "finish":
            continue
        call_id = str(row.get("call_id") or "")
        if not call_id or call_id in finish_by_id:
            raise RuntimeError("finished trace call IDs must be non-empty and unique")
        finish_by_id[call_id] = row
    action_by_call: dict[str, tuple[str, ActionEvent]] = {}
    for execution_id, sequence in actions.items():
        for event in sequence:
            call_id = str(event.result.get("trace_call_id") or "")
            if not call_id or call_id in action_by_call:
                raise RuntimeError("ActionEvent trace_call_id values must be unique")
            action_by_call[call_id] = (execution_id, event)
    if set(finish_by_id) != set(action_by_call):
        missing_actions = sorted(set(finish_by_id) - set(action_by_call))
        missing_traces = sorted(set(action_by_call) - set(finish_by_id))
        raise RuntimeError(
            "ActionEvent/trace join is not exact: "
            f"trace_without_action={len(missing_actions)}, "
            f"action_without_trace={len(missing_traces)}"
        )
    for call_id, (execution_id, event) in action_by_call.items():
        trace = finish_by_id[call_id]
        if str(trace.get("execution_id") or "") != execution_id:
            raise RuntimeError(f"execution mismatch for trace call {call_id}")
        if str(trace.get("tool_name") or "") != event.action_type:
            raise RuntimeError(f"tool-name mismatch for trace call {call_id}")


def _group_statistics(
    rollouts: Sequence[Mapping[str, Any]], *, expected_k: int
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {arm: [] for arm in ARMS}
    )
    for row in rollouts:
        group = grouped[str(row["hotpot_id"])]
        for arm in ARMS:
            group[arm].append(float(row[arm]["training_total"]))
    incomplete = sorted(
        hotpot_id
        for hotpot_id, values in grouped.items()
        if any(len(values[arm]) != expected_k for arm in ARMS)
    )
    if incomplete:
        raise RuntimeError(
            f"expected K={expected_k}; incomplete groups: {', '.join(incomplete)}"
        )
    result: dict[str, Any] = {}
    for arm in ARMS:
        stds = [statistics.pstdev(values[arm]) for values in grouped.values()]
        result[arm] = {
            "task_count": len(stds),
            "groups_with_nonzero_std": sum(value > 0.0 for value in stds),
            "fraction_groups_with_nonzero_std": (
                sum(value > 0.0 for value in stds) / len(stds) if stds else 0.0
            ),
            "mean_training_total": statistics.fmean(
                float(row[arm]["training_total"]) for row in rollouts
            ),
        }
    return result


def build_report(
    *,
    experience_path: Path,
    trace_path: Path,
    stage3_turn_path: Path,
    hotpotqa_path: Path,
    lock_path: Path,
    seed: int,
    max_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise RuntimeError("protocol lock must be a JSON object")
    dataset_rows = _load_locked_rows(hotpotqa_path=hotpotqa_path, lock=lock)
    experience_rows = _read_jsonl(experience_path)
    trace_rows = _read_jsonl(trace_path)
    turns = _last_turns(_read_jsonl(stage3_turn_path))
    actions, outcomes, rollout_ids = _experience_rollouts(experience_rows)
    _validate_trace_join(actions=actions, trace_rows=trace_rows)

    expected_executions = set(turns)
    if set(rollout_ids) != expected_executions:
        raise RuntimeError(
            "Experience/Stage-3 execution sets differ: "
            f"experience_only={len(set(rollout_ids) - expected_executions)}, "
            f"turn_only={len(expected_executions - set(rollout_ids))}"
        )

    rollout_rows: list[dict[str, Any]] = []
    flat_credit_rows: list[dict[str, Any]] = []
    dfa_credit_rows: list[dict[str, Any]] = []
    for execution_id in sorted(expected_executions):
        turn = turns[execution_id]
        hotpot_id = str(turn.get("hotpot_id") or "")
        source = dataset_rows.get(hotpot_id)
        if source is None:
            raise RuntimeError(f"execution {execution_id} maps outside frozen train rows")
        outcome = outcomes.get(execution_id, {})
        if "task_score" not in outcome:
            outcome["task_score"] = turn.get("task_score")
        required = ("task_score", "found_answer", "answer_exact_match")
        missing = [key for key in required if outcome.get(key) is None]
        if missing:
            raise RuntimeError(
                f"execution {execution_id} is missing outcome fields: {missing}"
            )
        sequence = actions[execution_id]
        if sequence:
            task_id = sequence[0].task_id
            rollout_id = sequence[0].rollout_id
        else:
            rollout_id = rollout_ids[execution_id]
            task_id = rollout_id.rsplit("/", 1)[0]
        comparison = replay_hotpotqa_oracle_comparison(
            task_id=task_id,
            rollout_id=rollout_id,
            seed=seed,
            supporting_sentences=_supporting_sentences(source),
            observed_sentences=_observed_context_sentences(
                source.get("context") or {}
            ),
            action_events=sequence,
            exact_match=float(outcome["answer_exact_match"]),
            task_f1=float(outcome["task_score"]),
            found_answer=bool(outcome["found_answer"]),
            max_steps=max_steps,
        )
        terminal = comparison.terminal_total
        rollout_rows.append(
            {
                "execution_id": execution_id,
                "hotpot_id": hotpot_id,
                "task_id": task_id,
                "rollout_id": rollout_id,
                "action_count": comparison.action_count,
                "terminal_only": {
                    "logic_total": 0.0,
                    "training_total": terminal,
                },
                "flat_oracle": {
                    "logic_total": comparison.flat.logic_total,
                    "training_total": comparison.flat.training_total,
                    "milestone_total": comparison.flat.milestone_total,
                },
                "oracle_dfa": {
                    "logic_total": comparison.dfa.logic_total,
                    "training_total": comparison.dfa.training_total,
                    "milestone_total": comparison.dfa.milestone_total,
                    "final_state": comparison.dfa.final_state,
                    "final_status": comparison.dfa.final_status,
                    "accepted": comparison.dfa.accepted,
                },
            }
        )
        flat_credit_rows.extend(
            {"execution_id": execution_id, **credit.canonical_dict()}
            for credit in comparison.flat.credits
        )
        dfa_credit_rows.extend(
            {"execution_id": execution_id, **credit.canonical_dict()}
            for credit in comparison.dfa.credits
        )

    expected_k = int(lock.get("signal_repeat_times") or 0)
    expected_tasks = len(lock.get("fixed_train_rows") or [])
    if len(rollout_rows) != expected_tasks * expected_k:
        raise RuntimeError(
            f"expected {expected_tasks * expected_k} rollouts, got {len(rollout_rows)}"
        )
    action_count = sum(int(row["action_count"]) for row in rollout_rows)
    if len(flat_credit_rows) != action_count or len(dfa_credit_rows) != action_count:
        raise RuntimeError("per-action credit totals do not match real action count")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass",
        "interpretation": (
            "Post-hoc Oracle labels affect rewards only; policy observations and "
            "real ActionEvents are immutable. Answer correctness is rewarded once "
            "through terminal HotpotQA F1, not again as a logic milestone."
        ),
        "arms": list(ARMS),
        "action_milestone_aps": list(ACTION_MILESTONE_APS),
        "weights": {
            "logic_beta": 1.0,
            "milestone_weight": 0.25,
            "violation_weight": 0.0,
            "trend_weight": 0.0,
            "action_logic_reward_ceiling": ACTION_LOGIC_REWARD_CEILING,
        },
        "reward_versions": {
            "flat_oracle": FLAT_REWARD_VERSION,
            "oracle_dfa": REWARD_VERSION,
        },
        "source": {
            "experience_path": str(experience_path),
            "experience_sha256": _sha256(experience_path),
            "trace_path": str(trace_path),
            "trace_sha256": _sha256(trace_path),
            "stage3_turn_path": str(stage3_turn_path),
            "stage3_turn_sha256": _sha256(stage3_turn_path),
            "lock_path": str(lock_path),
            "lock_sha256": _sha256(lock_path),
        },
        "counts": {
            "task_count": expected_tasks,
            "rollout_count": len(rollout_rows),
            "action_event_count": action_count,
            "flat_credit_count": len(flat_credit_rows),
            "dfa_credit_count": len(dfa_credit_rows),
            "dfa_accepted_rollouts": sum(
                bool(row["oracle_dfa"]["accepted"]) for row in rollout_rows
            ),
            "flat_differs_from_terminal_rollouts": sum(
                row["flat_oracle"]["training_total"]
                != row["terminal_only"]["training_total"]
                for row in rollout_rows
            ),
            "dfa_differs_from_terminal_rollouts": sum(
                row["oracle_dfa"]["training_total"]
                != row["terminal_only"]["training_total"]
                for row in rollout_rows
            ),
            "flat_differs_from_dfa_rollouts": sum(
                row["flat_oracle"]["training_total"]
                != row["oracle_dfa"]["training_total"]
                for row in rollout_rows
            ),
        },
        "group_statistics": _group_statistics(rollout_rows, expected_k=expected_k),
        "rollouts": rollout_rows,
    }
    return report, flat_credit_rows, dfa_credit_rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    groups = report["group_statistics"]
    lines = [
        "# E3 Oracle reward offline comparison",
        "",
        f"Status: **{report['status']}**",
        "",
        str(report["interpretation"]),
        "",
        "| Arm | Mean total | Nonzero-std groups | Fraction |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        stats = groups[arm]
        lines.append(
            f"| {arm} | {stats['mean_training_total']:.6f} | "
            f"{stats['groups_with_nonzero_std']} | "
            f"{stats['fraction_groups_with_nonzero_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Tasks / rollouts / actions: {counts['task_count']} / "
            f"{counts['rollout_count']} / {counts['action_event_count']}",
            f"- Exact Flat/DFA credit joins: {counts['flat_credit_count']} / "
            f"{counts['dfa_credit_count']}",
            f"- DFA accepted rollouts: {counts['dfa_accepted_rollouts']}",
            f"- Flat differs from terminal: "
            f"{counts['flat_differs_from_terminal_rollouts']}",
            f"- DFA differs from terminal: "
            f"{counts['dfa_differs_from_terminal_rollouts']}",
            f"- Flat differs from DFA: {counts['flat_differs_from_dfa_rollouts']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experience-path", type=Path, required=True)
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--stage3-turn-path", type=Path, required=True)
    parser.add_argument("--hotpotqa-path", type=Path, required=True)
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "e1_4b_format_conditioned.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, flat_credits, dfa_credits = build_report(
        experience_path=args.experience_path.resolve(),
        trace_path=args.trace_path.resolve(),
        stage3_turn_path=args.stage3_turn_path.resolve(),
        hotpotqa_path=args.hotpotqa_path.resolve(),
        lock_path=args.lock_path.resolve(),
        seed=args.seed,
        max_steps=args.max_steps,
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output_dir / "report.md").write_text(
        _markdown(report), encoding="utf-8", newline="\n"
    )
    _write_jsonl(args.output_dir / "flat_oracle_credits.jsonl", flat_credits)
    _write_jsonl(args.output_dir / "oracle_dfa_credits.jsonl", dfa_credits)
    print(_markdown(report), end="")
    print(f"Report: {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
