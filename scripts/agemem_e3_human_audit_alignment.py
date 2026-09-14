#!/usr/bin/env python3
"""Join a private Chinese deduplicated audit to E3 semantic predictions.

Only aggregate confusion statistics are written. Candidate text, questions,
supporting facts, and per-action labels remain in the private input files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "agemem.e3_human_grounder_alignment.v1"
FINAL_LABELS = {"supports", "not_support", "unclear"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _confusion(rows: Sequence[tuple[str, bool]]) -> dict[str, Any]:
    confirmed = [(label, prediction) for label, prediction in rows if label != "unclear"]
    tp = sum(label == "supports" and prediction for label, prediction in confirmed)
    fp = sum(label == "not_support" and prediction for label, prediction in confirmed)
    fn = sum(label == "supports" and not prediction for label, prediction in confirmed)
    tn = sum(label == "not_support" and not prediction for label, prediction in confirmed)
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "confirmed_action_count": len(confirmed),
        "unclear_action_count": len(rows) - len(confirmed),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": _rate(tn, tn + fp),
    }


def _final_label_map(audit: Mapping[str, Any]) -> tuple[dict[str, str], Counter[str]]:
    groups = audit.get("去重审计记录")
    if not isinstance(groups, list):
        raise ValueError("audit is missing 去重审计记录 list")
    labels: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    for group_index, group in enumerate(groups, 1):
        if not isinstance(group, Mapping):
            raise ValueError(f"audit group {group_index} must be an object")
        review = group.get("人工复核")
        trace = group.get("去重追溯")
        if not isinstance(review, Mapping) or not isinstance(trace, Mapping):
            raise ValueError(f"audit group {group_index} lacks final review/trace")
        label = str(review.get("标签代码") or "")
        if label not in FINAL_LABELS:
            raise ValueError(f"audit group {group_index} has invalid final label {label!r}")
        source = str(review.get("标签来源") or "unknown")
        action_ids = trace.get("动作ID")
        if not isinstance(action_ids, list) or not action_ids:
            raise ValueError(f"audit group {group_index} has no action IDs")
        source_counts[source] += 1
        for raw_action_id in action_ids:
            action_id = str(raw_action_id or "")
            if not action_id or action_id in labels:
                raise ValueError(f"duplicate or empty audited action ID {action_id!r}")
            labels[action_id] = label
    metadata = audit.get("元数据")
    if isinstance(metadata, Mapping):
        expected = metadata.get("原始记录数")
        if isinstance(expected, int) and len(labels) != expected:
            raise ValueError(
                f"audit expands to {len(labels)} actions, expected {expected}"
            )
    return labels, source_counts


def build_alignment_report(
    *,
    audit_path: Path,
    semantic_audit_path: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping):
        raise ValueError("human audit root must be a JSON object")
    labels, source_counts = _final_label_map(audit)
    predictions: dict[str, tuple[bool, str]] = {}
    for row_number, row in enumerate(_read_jsonl(semantic_audit_path), 1):
        action_id = str(row.get("action_id") or "")
        if not action_id or action_id in predictions:
            raise ValueError(
                f"semantic audit row {row_number} has duplicate/empty action_id"
            )
        prediction = row.get("oracle_positive")
        if not isinstance(prediction, bool):
            raise ValueError(
                f"semantic audit row {row_number} lacks boolean oracle_positive"
            )
        predictions[action_id] = (prediction, str(row.get("action_type") or "unknown"))
    audited_ids = set(labels)
    predicted_ids = set(predictions)
    if audited_ids != predicted_ids:
        raise ValueError(
            "human/prediction action-ID sets differ: "
            f"human_only={len(audited_ids - predicted_ids)}, "
            f"prediction_only={len(predicted_ids - audited_ids)}"
        )

    joined = [
        (labels[action_id], predictions[action_id][0])
        for action_id in sorted(audited_ids)
    ]
    by_action_type: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for action_id in sorted(audited_ids):
        prediction, action_type = predictions[action_id]
        by_action_type[action_type].append((labels[action_id], prediction))
    label_counts = Counter(labels.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "interpretation": (
            "Final human labels take precedence; previously blank reviews use "
            "the recorded LLM fallback. Unclear labels are excluded from "
            "precision/recall/F1. The report contains aggregates only."
        ),
        "source": {
            "private_audit_filename": audit_path.name,
            "private_audit_sha256": _sha256(audit_path),
            "semantic_audit_filename": semantic_audit_path.name,
            "semantic_audit_sha256": _sha256(semantic_audit_path),
        },
        "counts": {
            "joined_action_count": len(joined),
            "final_label_counts": dict(sorted(label_counts.items())),
            "deduplicated_label_source_counts": dict(sorted(source_counts.items())),
        },
        "overall": _confusion(joined),
        "by_action_type": {
            action_type: _confusion(rows)
            for action_type, rows in sorted(by_action_type.items())
        },
    }


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def markdown(report: Mapping[str, Any]) -> str:
    overall = report["overall"]
    counts = report["counts"]
    lines = [
        "# E3 grounder alignment with final semantic audit",
        "",
        f"Status: **{report['status']}**",
        "",
        str(report["interpretation"]),
        "",
        f"- Exact action-ID joins: {counts['joined_action_count']}",
        f"- Final labels: {counts['final_label_counts']}",
        (
            "- TP / FP / FN / TN: "
            f"{overall['true_positive']} / {overall['false_positive']} / "
            f"{overall['false_negative']} / {overall['true_negative']}"
        ),
        f"- Precision: {_format_metric(overall['precision'])}",
        f"- Recall: {_format_metric(overall['recall'])}",
        f"- F1: {_format_metric(overall['f1'])}",
        "",
        "| Action type | Confirmed | Unclear | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for action_type, metrics in report["by_action_type"].items():
        lines.append(
            f"| {action_type} | {metrics['confirmed_action_count']} | "
            f"{metrics['unclear_action_count']} | "
            f"{_format_metric(metrics['precision'])} | "
            f"{_format_metric(metrics['recall'])} | "
            f"{_format_metric(metrics['f1'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--semantic-audit-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            raise SystemExit(f"refusing to overwrite alignment output: {output}")
    report = build_alignment_report(
        audit_path=args.audit_path.resolve(),
        semantic_audit_path=args.semantic_audit_path.resolve(),
    )
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.output_md.write_text(markdown(report), encoding="utf-8", newline="\n")
    if os.name != "nt":
        os.chmod(args.output_json, 0o600)
        os.chmod(args.output_md, 0o600)
    print(markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
