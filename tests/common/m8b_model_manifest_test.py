from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from trinity.common.m8b_model_manifest import (
    MODEL_MANIFEST_SCHEMA_VERSION,
    build_model_manifest,
    write_model_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def workspace_temp_directory():
    root = ROOT / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"m8b-model-manifest-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class ModelManifestTest(unittest.TestCase):
    def test_manifest_is_deterministic_and_excludes_itself(self):
        with workspace_temp_directory() as model:
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            output = model / ".agemem_model_manifest.json"
            revision = "a" * 40

            first = build_model_manifest(
                model,
                repository_id="Qwen/Qwen2.5-7B-Instruct",
                revision=revision,
                output_path=output,
            )
            write_model_manifest(first, output)
            second = build_model_manifest(
                model,
                repository_id="Qwen/Qwen2.5-7B-Instruct",
                revision=revision,
                output_path=output,
            )

            self.assertEqual(first, second)
            self.assertEqual(
                first["schema_version"], MODEL_MANIFEST_SCHEMA_VERSION
            )
            self.assertNotIn(
                ".agemem_model_manifest.json",
                {entry["path"] for entry in first["files"]},
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted, first)

    def test_invalid_revision_and_empty_file_are_rejected(self):
        with workspace_temp_directory() as model:
            (model / "empty.bin").touch()
            with self.assertRaisesRegex(ValueError, "revision"):
                build_model_manifest(
                    model,
                    repository_id="Qwen/Qwen2.5-7B-Instruct",
                    revision="main",
                )
            with self.assertRaisesRegex(ValueError, "empty"):
                build_model_manifest(
                    model,
                    repository_id="Qwen/Qwen2.5-7B-Instruct",
                    revision="b" * 40,
                )


if __name__ == "__main__":
    unittest.main()
