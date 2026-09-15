"""JSON-compatible YAML configuration loading with explicit shallow inheritance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import DynamicBuildConfig


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _merge(output[key], value)
        else:
            output[key] = value
    return output


def load_dynamic_config(path: Path) -> DynamicBuildConfig:
    path = path.resolve()

    def load_raw(current: Path, seen: set[Path]) -> dict[str, Any]:
        if current in seen:
            raise ValueError("dynamic config inheritance cycle")
        seen.add(current)
        value = json.loads(current.read_text(encoding="utf-8"))
        parent = value.pop("extends", None)
        if not parent:
            return value
        parent_path = (current.parent / parent).resolve()
        return _merge(load_raw(parent_path, seen), value)

    raw = load_raw(path, set())
    return DynamicBuildConfig.model_validate(raw)


__all__ = ["load_dynamic_config"]
