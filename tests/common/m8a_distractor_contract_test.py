import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY_ROOT / "trinity/common/workflows/memory_context/distractors.py"
SPEC = importlib.util.spec_from_file_location(
    "_agemem_m8a_distractors_for_test",
    MODULE_PATH,
)
distractors = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = distractors
SPEC.loader.exec_module(distractors)


class M8ADistractorContractTest(unittest.TestCase):
    def test_fixed_source_never_calls_provider(self):
        provider = mock.Mock(side_effect=AssertionError("provider must not run"))
        first = distractors.resolve_stage2_distractors(
            source="fixed",
            count=3,
            provider_generate=provider,
        )
        second = distractors.resolve_stage2_distractors(
            source="fixed",
            count=3,
            provider_generate=provider,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        provider.assert_not_called()

    def test_task_source_is_exact_and_provider_free(self):
        provider = mock.Mock(side_effect=AssertionError("provider must not run"))
        result = distractors.resolve_stage2_distractors(
            source="task",
            count=2,
            task_messages=["  frozen one  ", "frozen two", "unused"],
            provider_generate=provider,
        )
        self.assertEqual(result, ["frozen one", "frozen two"])
        provider.assert_not_called()

    def test_missing_or_short_sources_fail_closed(self):
        with self.assertRaises(distractors.DistractorContractError):
            distractors.resolve_stage2_distractors(source="task", count=1)
        with self.assertRaises(distractors.DistractorContractError):
            distractors.resolve_stage2_distractors(
                source="fixed",
                count=len(distractors.FIXED_DISTRACTOR_MESSAGES) + 1,
            )
        with self.assertRaises(distractors.DistractorContractError):
            distractors.resolve_stage2_distractors(source="unknown", count=1)

    def test_provider_is_only_called_for_explicit_provider_source(self):
        provider = mock.Mock(return_value=["one", "two"])
        result = distractors.resolve_stage2_distractors(
            source="provider",
            count=2,
            provider_generate=provider,
        )
        self.assertEqual(result, ["one", "two"])
        provider.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
