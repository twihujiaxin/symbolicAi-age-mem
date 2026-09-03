"""Contract tests for the Stage-3 final-answer probe.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_stage3_answer_probe.json"
SMOKE_LOCK_PATH = REPOSITORY_ROOT / "configs" / "m8b_autodl_preflight.json"
PROBE_YAML = (
    REPOSITORY_ROOT
    / "examples"
    / "agemem_hotpotqa"
    / "agemem_e1_stage3_answer_probe.yaml"
)
DRY_RUN = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_dry_run.yaml"
PROBE_SCRIPT = REPOSITORY_ROOT / "scripts" / "agemem_e1_stage3_answer_probe.sh"
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"
UTILS_PATH = (
    REPOSITORY_ROOT
    / "trinity"
    / "common"
    / "workflows"
    / "memory_context"
    / "utils.py"
)
PROMPT_PATH = (
    REPOSITORY_ROOT
    / "trinity"
    / "common"
    / "workflows"
    / "memory_context"
    / "workflow_prompt.py"
)


def _lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E1Stage3AnswerProbeTest(unittest.TestCase):
    def test_nudge_only_fires_on_the_last_stage3_round(self):
        utils = _load_module(UTILS_PATH, "e1_stage3_utils")
        nudge = utils.should_emit_stage3_final_answer_nudge
        self.assertTrue(
            nudge(enabled=True, round_index=1, max_rounds=2, found_answer=False)
        )
        self.assertFalse(
            nudge(enabled=True, round_index=0, max_rounds=2, found_answer=False)
        )
        self.assertFalse(
            nudge(enabled=False, round_index=1, max_rounds=2, found_answer=False)
        )
        self.assertFalse(
            nudge(enabled=True, round_index=1, max_rounds=2, found_answer=True)
        )

    def test_nudge_text_requires_answer_tags_and_forbids_tools(self):
        prompt = _load_module(PROMPT_PATH, "e1_stage3_prompt")
        text = prompt.STAGE3_FINAL_ANSWER_NUDGE
        self.assertIn("<answer>", text)
        self.assertIn("</answer>", text)
        self.assertIn("Do not call tools", text)
        self.assertNotIn("Retrieve_memory", text)
        self.assertNotIn("<tool_call>", text)
        lock = _lock()
        for row_id in lock["source_train_row_ids"]:
            self.assertNotIn(row_id, text)

    def test_probe_yaml_enables_nudge_without_reusing_smoke_job(self):
        lock = _lock()
        yaml_text = PROBE_YAML.read_text(encoding="utf-8")
        self.assertEqual(lock["schema_version"], "agemem.e1_stage3_answer_probe.lock.v1")
        self.assertEqual(lock["job_name"], "agemem-e1-stage3-answer-probe")
        self.assertTrue(lock["stage3_require_final_answer"])
        self.assertEqual(lock["stage3_max_rounds"], 2)
        self.assertIn('name: "agemem-e1-stage3-answer-probe"', yaml_text)
        self.assertNotIn(lock["smoke_job_name"], yaml_text)
        self.assertIn("stage3_require_final_answer: true", yaml_text)
        self.assertIn("stage3_max_rounds: 2", yaml_text)
        self.assertIn("mode: bench", yaml_text)
        self.assertIn("continue_from_checkpoint: false", yaml_text)
        for row_id in lock["source_train_row_ids"]:
            self.assertIn(row_id, yaml_text)

    def test_frozen_dry_run_does_not_enable_the_nudge(self):
        smoke_lock = json.loads(SMOKE_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(_source_digest(DRY_RUN), smoke_lock["source_files"]["config"]["sha256"])
        dry_run = DRY_RUN.read_text(encoding="utf-8")
        self.assertNotIn("stage3_require_final_answer: true", dry_run)
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        script = PROBE_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("e1_stage3_answer_probe_test", gate)
        self.assertNotIn("autodl_m8b_smoke.sh", script)
        self.assertNotIn("agemem_e1_dry_run.yaml", script)
        self.assertIn("agemem_e1_stage3_answer_probe.yaml", script)


if __name__ == "__main__":
    unittest.main()
