"""Tests for private final-audit/grounder aggregate alignment."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from scripts.agemem_e3_human_audit_alignment import build_alignment_report


class E3HumanAuditAlignmentTest(unittest.TestCase):
    def test_exact_join_and_unclear_exclusion(self) -> None:
        root = Path.cwd() / "tmp" / f"human-alignment-{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            audit_path = root / "private.json"
            semantic_path = root / "semantic.jsonl"
            audit = {
                "元数据": {"原始记录数": 4},
                "去重审计记录": [
                    {
                        "人工复核": {"标签代码": "supports", "标签来源": "人工填写"},
                        "去重追溯": {"动作ID": ["a1", "a2"]},
                    },
                    {
                        "人工复核": {"标签代码": "not_support", "标签来源": "大模型回填"},
                        "去重追溯": {"动作ID": ["a3"]},
                    },
                    {
                        "人工复核": {"标签代码": "unclear", "标签来源": "人工填写"},
                        "去重追溯": {"动作ID": ["a4"]},
                    },
                ],
            }
            audit_path.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
            predictions = [
                {"action_id": "a1", "action_type": "Add_memory", "oracle_positive": True},
                {"action_id": "a2", "action_type": "Retrieve_memory", "oracle_positive": False},
                {"action_id": "a3", "action_type": "Add_memory", "oracle_positive": True},
                {"action_id": "a4", "action_type": "Retrieve_memory", "oracle_positive": True},
            ]
            semantic_path.write_text(
                "".join(json.dumps(row) + "\n" for row in predictions),
                encoding="utf-8",
            )
            report = build_alignment_report(
                audit_path=audit_path,
                semantic_audit_path=semantic_path,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["counts"]["joined_action_count"], 4)
        self.assertEqual(report["overall"]["true_positive"], 1)
        self.assertEqual(report["overall"]["false_positive"], 1)
        self.assertEqual(report["overall"]["false_negative"], 1)
        self.assertEqual(report["overall"]["true_negative"], 0)
        self.assertEqual(report["overall"]["unclear_action_count"], 1)
        self.assertAlmostEqual(report["overall"]["precision"], 0.5)
        self.assertAlmostEqual(report["overall"]["recall"], 0.5)

    def test_mismatched_action_ids_fail_closed(self) -> None:
        root = Path.cwd() / "tmp" / f"human-alignment-{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            audit_path = root / "private.json"
            semantic_path = root / "semantic.jsonl"
            audit_path.write_text(
                json.dumps(
                    {
                        "元数据": {"原始记录数": 1},
                        "去重审计记录": [
                            {
                                "人工复核": {
                                    "标签代码": "supports",
                                    "标签来源": "人工填写",
                                },
                                "去重追溯": {"动作ID": ["human-only"]},
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            semantic_path.write_text(
                json.dumps(
                    {
                        "action_id": "prediction-only",
                        "action_type": "Add_memory",
                        "oracle_positive": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "action-ID sets differ"):
                build_alignment_report(
                    audit_path=audit_path,
                    semantic_audit_path=semantic_path,
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
