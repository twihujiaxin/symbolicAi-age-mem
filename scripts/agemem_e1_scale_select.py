#!/usr/bin/env python3
"""Select 18 extra HotpotQA train rows and freeze the 1.5B E1 scale YAML.

Keeps the original 6 M5 train IDs as a prefix. Does not train, does not enable
Stage-3 answer nudges, and does not modify the frozen dry-run YAML.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AgeMem_code_agentscope.hotpotqa_benchmark.adapter import (  # noqa: E402
    HotpotQADataAdapter,
    load_smoke_config,
)
from trinity.common.e1_scale import (  # noqa: E402
    LOCK_PATH,
    load_lock,
    selection_is_complete,
    write_generated_yaml,
)
from trinity.common.m8b_preflight import _canonical_json_sha256  # noqa: E402


def _row_record(dataset, split: str, source_index: int) -> dict:
    record = dataset[split][int(source_index)]
    return {
        "content_sha256": _canonical_json_sha256(record),
        "hotpot_id": str(record["id"]),
        "source_index": int(source_index),
    }


def select_scale_rows(hotpotqa_path: Path, lock: dict) -> dict:
    adapter = HotpotQADataAdapter(path=hotpotqa_path)
    smoke = load_smoke_config()
    extra_config = smoke.model_copy(
        update={
            "seed": int(lock["extra_selection_seed"]),
            "train_size": int(lock["extra_train_size"]),
        }
    )
    excluded = set(lock["source_train_prefix_ids"]) | set(lock["held_out_row_ids"])
    extra = adapter._select(
        source_split="train",
        benchmark_split="train",
        size=int(lock["extra_train_size"]),
        config=extra_config,
        excluded_ids=excluded,
    )
    dataset = adapter.dataset
    prefix_rows = []
    for expected in lock["prefix_train_rows"]:
        actual = _row_record(dataset, "train", expected["source_index"])
        if actual["hotpot_id"] != expected["hotpot_id"]:
            raise RuntimeError(
                f"prefix row {expected['source_index']} id drifted: "
                f"{actual['hotpot_id']} != {expected['hotpot_id']}"
            )
        if actual["content_sha256"] != expected["content_sha256"]:
            raise RuntimeError(
                f"prefix row {expected['hotpot_id']} content hash drifted"
            )
        prefix_rows.append(actual)
    extra_rows = [
        _row_record(dataset, "train", item.source_index) for item in extra
    ]
    extra_ids = [row["hotpot_id"] for row in extra_rows]
    if len(set(extra_ids)) != len(extra_ids):
        raise RuntimeError("extra train rows are not unique")
    if set(extra_ids) & excluded:
        raise RuntimeError("extra train rows collided with M5 prefix or held-out IDs")
    updated = dict(lock)
    updated["fixed_train_rows"] = prefix_rows + extra_rows
    updated["selection_status"] = "frozen"
    if not selection_is_complete(updated):
        raise RuntimeError("selected rows do not satisfy the E1 scale lock")
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze 24 E1 scale train rows from HOTPOTQA_PATH."
    )
    parser.add_argument(
        "--hotpotqa-path",
        default=os.environ.get("HOTPOTQA_PATH", ""),
    )
    parser.add_argument("--lock", default=str(LOCK_PATH))
    parser.add_argument(
        "--write-yaml",
        action="store_true",
        help="Write agemem_e1_scale.yaml and agemem_e1_scale_eval.yaml.",
    )
    arguments = parser.parse_args(argv)
    lock_path = Path(arguments.lock)
    if not lock_path.is_absolute():
        lock_path = REPOSITORY_ROOT / lock_path
    hotpotqa_path = Path(arguments.hotpotqa_path).expanduser()
    if not arguments.hotpotqa_path or not hotpotqa_path.is_dir():
        print("HOTPOTQA_PATH is missing or is not a DatasetDict directory.", file=sys.stderr)
        return 2
    lock = load_lock(lock_path)
    if lock.get("stage3_require_final_answer") or lock.get(
        "stage3_repair_untagged_answer"
    ):
        print("E1 scale must not enable Stage-3 answer nudges.", file=sys.stderr)
        return 2
    updated = select_scale_rows(hotpotqa_path, lock)
    lock_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if arguments.write_yaml:
        write_generated_yaml(updated)
    print("selection_status", updated["selection_status"])
    print("train_size", len(updated["fixed_train_rows"]))
    print("prefix", [row["hotpot_id"] for row in updated["fixed_train_rows"][:6]])
    print("extra", [row["hotpot_id"] for row in updated["fixed_train_rows"][6:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
