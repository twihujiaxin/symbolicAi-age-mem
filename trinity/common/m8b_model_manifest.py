"""Deterministic, offline provenance manifest for an M8b model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Sequence


MODEL_MANIFEST_SCHEMA_VERSION = "agemem.model_manifest.v1"
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IGNORED_DIRECTORY_NAMES = {".cache", ".git", "__pycache__"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model_manifest(
    model_path: Path,
    *,
    repository_id: str,
    revision: str,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Hash every material model file without importing or loading the model."""

    root = model_path.resolve()
    if not root.is_dir():
        raise ValueError("model_path must be an existing directory")
    if not isinstance(repository_id, str) or not repository_id.strip():
        raise ValueError("repository_id must be a non-empty string")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError("revision must be a lowercase 40-character commit ID")

    excluded = output_path.resolve() if output_path is not None else None
    entries: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(root)
        if any(part in _IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if excluded is not None and candidate.resolve() == excluded:
            continue
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"model artifact resolves outside model_path: {relative.as_posix()}"
            ) from exc
        size_bytes = resolved.stat().st_size
        if size_bytes <= 0:
            raise ValueError(
                f"model artifact is empty: {relative.as_posix()}"
            )
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": size_bytes,
                "sha256": _sha256_file(resolved),
            }
        )
    if not entries:
        raise ValueError("model directory contains no material files")
    return {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "repository_id": repository_id.strip(),
        "revision": revision,
        "files": entries,
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
    }


def write_model_manifest(
    manifest: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    output = output_path.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"model manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        manifest,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, output)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an offline SHA-256 provenance manifest for M8b."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)

    model_path = Path(arguments.model_path).expanduser().resolve()
    output = (
        Path(arguments.output).expanduser().resolve()
        if arguments.output
        else model_path / ".agemem_model_manifest.json"
    )
    manifest = build_model_manifest(
        model_path,
        repository_id=arguments.repository_id,
        revision=arguments.revision,
        output_path=output,
    )
    write_model_manifest(manifest, output, overwrite=arguments.force)
    print(
        "M8b model manifest written: "
        f"files={manifest['file_count']} bytes={manifest['total_size_bytes']} "
        f"path={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODEL_MANIFEST_SCHEMA_VERSION",
    "build_model_manifest",
    "main",
    "write_model_manifest",
]
