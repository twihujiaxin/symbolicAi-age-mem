"""Packaging gates for Trinity's M8 reuse of M2/M6 contracts."""

from __future__ import annotations

import unittest
from pathlib import Path

from setuptools import find_packages


ROOT = Path(__file__).resolve().parents[2]


class M8APackagingContractTest(unittest.TestCase):
    def test_wheel_includes_shared_agemem_contracts_and_pydantic(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('include = ["trinity*", "AgeMem_code_agentscope*"]', pyproject)
        self.assertIn('"pydantic>=2.0,<3"', pyproject)
        m8b_dependencies = pyproject.split("m8b = [", 1)[1].split("]", 1)[0]
        self.assertIn('"agentscope>=1.0.5,<2"', m8b_dependencies)
        self.assertIn('"mcp>=1.24,<2"', m8b_dependencies)
        self.assertIn(
            '"AgeMem_code_agentscope.toy_hotpotqa" = ["data/*.json"]',
            pyproject,
        )

        discovered = set(find_packages(where=str(ROOT)))
        self.assertIn("trinity.common", discovered)
        self.assertIn("AgeMem_code_agentscope", discovered)
        self.assertIn("AgeMem_code_agentscope.action_schema", discovered)
        packaged_fixture = (
            ROOT
            / "AgeMem_code_agentscope"
            / "toy_hotpotqa"
            / "data"
            / "stage2_context_challenges.json"
        )
        source_fixture = ROOT / "data" / "toy" / "stage2_context_challenges.json"
        self.assertEqual(packaged_fixture.read_bytes(), source_fixture.read_bytes())


if __name__ == "__main__":
    unittest.main()
