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

        discovered = set(find_packages(where=str(ROOT)))
        self.assertIn("trinity.common", discovered)
        self.assertIn("AgeMem_code_agentscope", discovered)
        self.assertIn("AgeMem_code_agentscope.action_schema", discovered)


if __name__ == "__main__":
    unittest.main()
