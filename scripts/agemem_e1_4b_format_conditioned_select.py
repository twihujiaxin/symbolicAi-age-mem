#!/usr/bin/env python3
"""Select 32-dev + 128-test validation rows for the format-conditioned 4B protocol.

Copies already-frozen 24 train rows from e1_scale. Writes 32-row memory-necessity
YAMLs after freeze. Windows cannot freeze without HOTPOTQA_PATH.
"""

from __future__ import annotations

import argparse
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
from trinity.common.e1_4b_format_conditioned import (  # noqa: E402
    LOCK_PATH,
    load_lock,
    load_scale_lock,
    selection_is_frozen,
    train_rows_match_scale,
    write_lock,
    write_mem_yamls,
)
from trinity.common.m8b_preflight import _canonical_json_sha256  # noqa: E402


def _row_record(dataset, split: str, source_index: int) -> dict:
    record = dataset[split][int(source_index)]
    return {
        "content_sha256": _canonical_json_sha256(record),
        "hotpot_id": str(record["id"]),
        "source_index": int(source_index),
    }


def _verify_known_rows(dataset, split: str, expected_rows: list[dict], label: str) -> list[dict]:
    verified = []
    for expected in expected_rows:
        actual = _row_record(dataset, split, expected["source_index"])
        if actual["hotpot_id"] != expected["hotpot_id"]:
            raise RuntimeError(
                f"{label} row {expected['source_index']} id drifted: "
                f"{actual['hotpot_id']} != {expected['hotpot_id']}"
            )
        if actual["content_sha256"] != expected["content_sha256"]:
            raise RuntimeError(f"{label} row {expected['hotpot_id']} content hash drifted")
        verified.append(actual)
    return verified


def select_format_conditioned_rows(hotpotqa_path: Path, lock: dict) -> dict:
    if not train_rows_match_scale(lock):
        raise RuntimeError("format-conditioned train rows must copy e1_scale.fixed_train_rows")
    adapter = HotpotQADataAdapter(path=hotpotqa_path)
    smoke = load_smoke_config()
    select_config = smoke.model_copy(
        update={
            "seed": int(lock["selection_seed"]),
            "dev_size": int(lock["dev_size"]),
            "test_size": int(lock["test_size"]),
        }
    )
    excluded = set(lock["excluded_validation_ids"])
    dataset = adapter.dataset
    train_rows = _verify_known_rows(
        dataset, "train", list(lock["fixed_train_rows"]), "train"
    )
    held_out_rows = _verify_known_rows(
        dataset, "validation", list(lock["held_out_rows"]), "held-out"
    )
    for expected in lock["excluded_validation_rows"]:
        actual = _row_record(dataset, "validation", expected["source_index"])
        if actual["hotpot_id"] != expected["hotpot_id"]:
            raise RuntimeError(
                f"excluded validation row {expected['source_index']} id drifted"
            )
    dev = adapter._select(
        source_split="validation",
        benchmark_split="dev",
        size=int(lock["dev_size"]),
        config=select_config,
        excluded_ids=excluded,
    )
    dev_rows = [_row_record(dataset, "validation", item.source_index) for item in dev]
    excluded_after_dev = excluded | {row["hotpot_id"] for row in dev_rows}
    test = adapter._select(
        source_split="validation",
        benchmark_split="test",
        size=int(lock["test_size"]),
        config=select_config,
        excluded_ids=excluded_after_dev,
    )
    test_rows = [_row_record(dataset, "validation", item.source_index) for item in test]
    updated = dict(lock)
    updated["fixed_train_rows"] = train_rows
    updated["held_out_rows"] = held_out_rows
    updated["held_out_row_ids"] = [row["hotpot_id"] for row in held_out_rows]
    updated["fixed_dev_rows"] = dev_rows
    updated["fixed_test_rows"] = test_rows
    updated["selection_status"] = "frozen"
    if not selection_is_frozen(updated):
        raise RuntimeError("selected rows do not satisfy the format-conditioned lock")
    if not train_rows_match_scale(updated, load_scale_lock()):
        raise RuntimeError("train rows drifted away from e1_scale.json during freeze")
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze 32-dev and 128-test validation rows for format-conditioned 4B."
    )
    parser.add_argument(
        "--hotpotqa-path",
        default=os.environ.get("HOTPOTQA_PATH", ""),
    )
    parser.add_argument("--lock", default=str(LOCK_PATH))
    parser.add_argument(
        "--write-yaml",
        action="store_true",
        help="Write the three 32-dev memory-necessity YAMLs after freeze.",
    )
    arguments = parser.parse_args(argv)
    lock_path = Path(arguments.lock)
    if not lock_path.is_absolute():
        lock_path = REPOSITORY_ROOT / lock_path
    hotpotqa_path = Path(arguments.hotpotqa_path).expanduser()
    if not arguments.hotpotqa_path or not hotpotqa_path.is_dir():
        print(
            "HOTPOTQA_PATH is missing or is not a DatasetDict directory. "
            "Windows cannot freeze 32+128 without local HotpotQA.",
            file=sys.stderr,
        )
        return 2
    lock = load_lock(lock_path)
    if not lock.get("stage3_require_final_answer") or not lock.get(
        "stage3_repair_untagged_answer"
    ):
        print("format-conditioned protocol must keep Stage-3 answer nudges.", file=sys.stderr)
        return 2
    if int(lock.get("selection_seed") or 0) in {20260802, 20260904}:
        print("selection_seed must not reuse smoke or scale seeds.", file=sys.stderr)
        return 2
    updated = select_format_conditioned_rows(hotpotqa_path, lock)
    if arguments.write_yaml:
        updated = write_mem_yamls(updated)
    write_lock(updated, lock_path)
    print("selection_status", updated["selection_status"])
    print("train_size", len(updated["fixed_train_rows"]))
    print("dev_size", len(updated["fixed_dev_rows"]))
    print("test_size", len(updated["fixed_test_rows"]))
    print("dev", [row["hotpot_id"] for row in updated["fixed_dev_rows"]])
    print("test", [row["hotpot_id"] for row in updated["fixed_test_rows"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
