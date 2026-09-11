from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        name,
        REPOSITORY_ROOT / relative_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


memory_utils = load_module(
    "agemem_fact_memory_utils",
    "trinity/common/workflows/memory_context/utils.py",
)
workflow_prompt = load_module(
    "agemem_fact_memory_prompt",
    "trinity/common/workflows/memory_context/workflow_prompt.py",
)

TOOL_SCHEMA = memory_utils.TOOL_SCHEMA
build_tool_schema = memory_utils.build_tool_schema
validate_fact_memory_add = memory_utils.validate_fact_memory_add
STAGE1_FACT_MEMORY_INSTRUCTION = workflow_prompt.STAGE1_FACT_MEMORY_INSTRUCTION
build_tool_call_system_prompt = workflow_prompt.build_tool_call_system_prompt
FACT_DIAG_YAML = (
    REPOSITORY_ROOT
    / "examples"
    / "agemem_hotpotqa"
    / "agemem_e1_4b_fc_fact_memory_signal_diag.yaml"
)
LEGACY_DIAG_YAML = (
    REPOSITORY_ROOT
    / "examples"
    / "agemem_hotpotqa"
    / "agemem_e1_4b_fc_signal_diag.yaml"
)


class FactMemoryContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.titles = ["Northbridge Observatory"]
        self.sentences = [
            [
                "The Northbridge Observatory opened in 1987 in Lanton.",
                "It was founded by physicist Mira Vale.",
            ]
        ]
        self.valid_arguments = {
            "content": (
                "The Northbridge Observatory opened in 1987 in Lanton and "
                "was founded by physicist Mira Vale."
            ),
            "memory_type": "knowledge",
            "metadata": {
                "source_title": "Northbridge Observatory",
                "source_sentence_indices": [0, 1],
            },
        }

    def validate(self, arguments):
        return validate_fact_memory_add(
            arguments,
            source_titles=self.titles,
            source_sentence_groups=self.sentences,
        )

    def test_accepts_concrete_source_grounded_fact(self) -> None:
        self.assertIsNone(self.validate(self.valid_arguments))

    def test_rejects_topic_level_summary(self) -> None:
        arguments = dict(self.valid_arguments)
        arguments["content"] = "Summary of the observatory and its history"
        self.assertIn("topic-level", self.validate(arguments))

    def test_requires_knowledge_type_and_exact_source_title(self) -> None:
        arguments = dict(self.valid_arguments)
        arguments["memory_type"] = "general"
        self.assertIn("memory_type='knowledge'", self.validate(arguments))

        arguments = dict(self.valid_arguments)
        arguments["metadata"] = {"source_title": "Observatory"}
        self.assertIn("exactly match", self.validate(arguments))

    def test_rejects_non_string_content(self) -> None:
        arguments = dict(self.valid_arguments)
        arguments["content"] = {
            "fact": "The Northbridge Observatory opened in 1987."
        }
        error = self.validate(arguments)
        self.assertIn("non-empty string", error)

    def test_rejects_ungrounded_or_invalid_provenance(self) -> None:
        arguments = dict(self.valid_arguments)
        arguments["content"] = (
            "The Southern Archive launched a maritime museum in Port Azure."
        )
        self.assertIn("not sufficiently grounded", self.validate(arguments))

        arguments = dict(self.valid_arguments)
        arguments["metadata"] = {
            "source_title": "Northbridge Observatory",
            "source_sentence_indices": [2],
        }
        self.assertIn("out-of-range", self.validate(arguments))

        arguments = dict(self.valid_arguments)
        arguments["content"] = "Physicist Mira Vale founded the institution."
        arguments["metadata"] = {
            "source_title": "Northbridge Observatory",
            "source_sentence_indices": [0],
        }
        self.assertIn("not sufficiently grounded", self.validate(arguments))

    def test_rejects_duplicate_content(self) -> None:
        error = validate_fact_memory_add(
            self.valid_arguments,
            source_titles=self.titles,
            source_sentence_groups=self.sentences,
            existing_contents=[self.valid_arguments["content"].upper()],
        )
        self.assertIn("duplicate", error)

    def test_fact_schema_is_opt_in_and_does_not_mutate_base_schema(self) -> None:
        base_description = next(
            tool["description"]
            for tool in TOOL_SCHEMA
            if tool["name"] == "Add_memory"
        )
        ordinary = build_tool_schema(True)
        fact = build_tool_schema(True, fact_memory=True)
        ordinary_description = next(
            tool["description"]
            for tool in ordinary
            if tool["name"] == "Add_memory"
        )
        fact_description = next(
            tool["description"]
            for tool in fact
            if tool["name"] == "Add_memory"
        )
        fact_add = next(tool for tool in fact if tool["name"] == "Add_memory")

        self.assertEqual(ordinary_description, base_description)
        self.assertNotEqual(fact_description, base_description)
        self.assertIn("concrete", fact_description)
        self.assertEqual(
            fact_add["parameters"]["required"],
            ["content", "metadata", "memory_type"],
        )
        self.assertEqual(
            fact_add["parameters"]["properties"]["metadata"]["required"],
            ["source_title"],
        )
        self.assertEqual(
            next(
                tool["description"]
                for tool in TOOL_SCHEMA
                if tool["name"] == "Add_memory"
            ),
            base_description,
        )

    def test_prompt_is_explicit_and_uses_only_synthetic_example(self) -> None:
        self.assertIn("subject, relation, and object", STAGE1_FACT_MEMORY_INSTRUCTION)
        self.assertIn("source_title", STAGE1_FACT_MEMORY_INSTRUCTION)
        self.assertIn("fictional Northbridge Observatory", STAGE1_FACT_MEMORY_INSTRUCTION)
        self.assertIn("at most four Add_memory calls", STAGE1_FACT_MEMORY_INSTRUCTION)
        self.assertIn("under 80 words", STAGE1_FACT_MEMORY_INSTRUCTION)
        self.assertNotIn("M83", STAGE1_FACT_MEMORY_INSTRUCTION)

        ordinary = build_tool_call_system_prompt("[]")
        fact = build_tool_call_system_prompt("[]", fact_memory=True)
        self.assertIn("Strategy summary for reuse", ordinary)
        self.assertNotIn("Strategy summary for reuse", fact)
        self.assertNotIn("Solution approach for this type of problem", fact)
        self.assertIn('"source_title": "Northbridge Observatory"', fact)

    def test_fact_diagnostic_is_opt_in_and_legacy_config_stays_unchanged(self) -> None:
        fact_text = FACT_DIAG_YAML.read_text(encoding="utf-8")
        legacy_text = LEGACY_DIAG_YAML.read_text(encoding="utf-8")

        self.assertIn(
            'name: "agemem-e1-4b-fc-fact-memory-signal-diag"',
            fact_text,
        )
        self.assertEqual(fact_text.count("stage1_fact_memory_enabled: true"), 2)
        self.assertEqual(fact_text.count("stage1_fact_memory_validation: true"), 2)
        self.assertIn("reward_profile: terminal_only", fact_text)
        self.assertIn("repeat_times: 4", fact_text)
        self.assertNotIn("stage3_inject_gold_supporting", fact_text)
        self.assertNotIn("stage1_fact_memory_enabled", legacy_text)
        self.assertNotIn("stage1_fact_memory_validation", legacy_text)


if __name__ == "__main__":
    unittest.main()
