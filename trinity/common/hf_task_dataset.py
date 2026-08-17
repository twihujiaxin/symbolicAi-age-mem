"""Helpers for loading deterministic Hugging Face task datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def is_saved_hf_dataset(path: str) -> bool:
    """Return whether *path* is a Hugging Face ``save_to_disk`` directory."""

    dataset_path = Path(path).expanduser()
    if not dataset_path.is_dir():
        return False
    return (dataset_path / "dataset_dict.json").is_file() or (
        dataset_path / "state.json"
    ).is_file()


def load_task_dataset(
    path: Optional[str], subset_name: Optional[str], split: str
) -> Dataset:
    """Load one task split without changing legacy ``load_dataset`` behavior.

    Hugging Face deliberately uses a different API for artifacts written by
    ``Dataset.save_to_disk`` / ``DatasetDict.save_to_disk``. Detect only those
    marker-bearing local directories and keep every other path (dataset
    scripts, Hub IDs, JSON/Parquet builders, and ordinary directories) on the
    existing ``load_dataset`` code path.
    """

    if not path:
        raise ValueError("task dataset path is required")

    if not is_saved_hf_dataset(path):
        return load_dataset(path, name=subset_name, split=split)

    if subset_name is not None:
        raise ValueError(
            "`subset_name` cannot be used with a local Hugging Face "
            "save_to_disk artifact"
        )

    saved_dataset = load_from_disk(str(Path(path).expanduser()))
    if isinstance(saved_dataset, DatasetDict):
        if split not in saved_dataset:
            available = ", ".join(sorted(saved_dataset.keys()))
            raise ValueError(
                f"split {split!r} is not present in saved DatasetDict; "
                f"available splits: {available or '<none>'}"
            )
        return saved_dataset[split]

    if isinstance(saved_dataset, Dataset):
        if split != "train":
            raise ValueError(
                "a local saved Dataset has no named splits; use split='train' "
                "or save a DatasetDict"
            )
        return saved_dataset

    raise TypeError(
        "load_from_disk returned an unsupported task dataset type: "
        f"{type(saved_dataset).__name__}"
    )


def select_task_rows(
    dataset: Dataset,
    row_indices: Optional[Sequence[int]],
    *,
    expected_row_ids: Optional[Sequence[str]] = None,
    row_id_key: Optional[str] = None,
    expected_dataset_fingerprint: Optional[str] = None,
) -> Dataset:
    """Select an ordered subset and verify its immutable source identity."""

    if expected_dataset_fingerprint is not None:
        if (
            not isinstance(expected_dataset_fingerprint, str)
            or not expected_dataset_fingerprint
        ):
            raise TypeError("`expected_dataset_fingerprint` must be a non-empty string")
        actual_fingerprint = getattr(dataset, "_fingerprint", None)
        if actual_fingerprint != expected_dataset_fingerprint:
            raise ValueError(
                "task dataset fingerprint mismatch: "
                f"{actual_fingerprint!r} != {expected_dataset_fingerprint!r}"
            )

    if row_indices is None:
        if expected_row_ids is not None or row_id_key is not None:
            raise ValueError(
                "row identity validation requires explicit `row_indices`"
            )
        return dataset
    indices = list(row_indices)
    if not indices:
        raise ValueError("`row_indices` must not be empty")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise TypeError("every `row_indices` value must be an integer")
    if len(indices) != len(set(indices)):
        raise ValueError("`row_indices` must not contain duplicates")

    invalid = [index for index in indices if index < 0 or index >= len(dataset)]
    if invalid:
        raise IndexError(
            f"row indices {invalid} are outside dataset bounds [0, {len(dataset)})"
        )
    selected = dataset.select(indices)

    if expected_row_ids is None:
        if row_id_key is not None:
            raise ValueError("`row_id_key` requires `expected_row_ids`")
        return selected
    expected_ids = list(expected_row_ids)
    if not expected_ids:
        raise ValueError("`expected_row_ids` must not be empty")
    if any(not isinstance(row_id, str) or not row_id for row_id in expected_ids):
        raise TypeError("every `expected_row_ids` value must be a non-empty string")
    if len(expected_ids) != len(indices):
        raise ValueError(
            "`expected_row_ids` length must equal `row_indices` length"
        )
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("`expected_row_ids` must not contain duplicates")
    if not isinstance(row_id_key, str) or not row_id_key:
        raise ValueError(
            "`row_id_key` must be a non-empty string when expected IDs are set"
        )
    if row_id_key not in selected.column_names:
        raise ValueError(f"task dataset has no row ID column {row_id_key!r}")
    actual_ids = list(selected[row_id_key])
    if actual_ids != expected_ids:
        raise ValueError(
            "selected task row IDs do not match the fixed manifest: "
            f"{actual_ids!r} != {expected_ids!r}"
        )
    return selected


__all__ = ["is_saved_hf_dataset", "load_task_dataset", "select_task_rows"]
