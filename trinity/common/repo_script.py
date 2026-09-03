"""Load AgeMem CLI modules without colliding with veRL's installed ``scripts`` package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_repo_script(module_name: str) -> ModuleType:
    """Import ``scripts/<module_name>.py`` by file path.

    veRL installs a top-level ``scripts`` package into site-packages. Tests must
    not use ``from scripts import ...``, which would bind that package instead of
    this repository's CLI files.
    """

    path = REPOSITORY_ROOT / "scripts" / f"{module_name}.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    unique_name = f"agemem_repo_cli_{module_name}"
    existing = sys.modules.get(unique_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load repository CLI: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module
