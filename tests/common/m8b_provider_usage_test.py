from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from trinity.common.auxiliary_provider import (
    AUXILIARY_PROVIDER_CALL_SCHEMA_VERSION,
    AUXILIARY_PROVIDER_SCHEMA_VERSION,
    AuxiliaryProviderCallError,
    AuxiliaryProviderResponseError,
    AuxiliaryProviderTelemetryError,
    AuxiliaryProviderTelemetryRecorder,
    AuxiliaryProviderUsageTracker,
    load_auxiliary_provider_config,
    resolve_auxiliary_provider_telemetry_path,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests/fixtures"


def load_memory_module():
    path = REPOSITORY_ROOT / "trinity/common/workflows/memory_context/memory_store.py"
    module_name = "_agemem_m8b_memory_store_for_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


memory_store = load_memory_module()
MemoryManager = memory_store.MemoryManager
chat_client = memory_store.chat_client


def valid_workflow_args():
    return {
        "auxiliary_provider": {
            "schema_version": AUXILIARY_PROVIDER_SCHEMA_VERSION,
            "provider": "dashscope",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "embedding_model": "text-embedding-v4",
            "embedding_dimensions": 256,
            "chat_model": "qwen-max",
            "usage_tracking": True,
        }
    }


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@contextmanager
def telemetry_file():
    path = FIXTURE_ROOT / f".m8b-provider-{uuid4().hex}.jsonl"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
        Path(f"{path}.lock").unlink(missing_ok=True)


class AuxiliaryProviderContractTest(unittest.TestCase):
    def test_terminal_only_requires_explicit_provider_lock(self):
        with self.assertRaisesRegex(ValueError, "explicit auxiliary_provider"):
            load_auxiliary_provider_config({}, required=True)

    def test_valid_lock_is_immutable_and_public(self):
        config = load_auxiliary_provider_config(valid_workflow_args(), required=True)
        self.assertEqual(config.provider, "dashscope")
        self.assertEqual(config.embedding_model, "text-embedding-v4")
        self.assertEqual(config.embedding_dimensions, 256)
        self.assertEqual(config.chat_model, "qwen-max")
        self.assertNotIn("api_key", config.public_dict())

    def test_credentials_and_unknown_fields_are_rejected(self):
        payload = valid_workflow_args()
        payload["auxiliary_provider"]["api_key"] = "must-not-be-here"
        with self.assertRaisesRegex(ValueError, "unknown field"):
            load_auxiliary_provider_config(payload, required=True)

        payload = valid_workflow_args()
        payload["auxiliary_provider"]["base_url"] = (
            "https://user:password@example.invalid/v1"
        )
        with self.assertRaisesRegex(ValueError, "must not contain credentials"):
            load_auxiliary_provider_config(payload, required=True)

    def test_tracking_cannot_be_disabled_for_terminal_only(self):
        payload = valid_workflow_args()
        payload["auxiliary_provider"]["usage_tracking"] = False
        with self.assertRaisesRegex(ValueError, "requires auxiliary usage tracking"):
            load_auxiliary_provider_config(payload, required=True)

    def test_terminal_only_contract_values_are_exact(self):
        alternatives = {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
            "embedding_model": "text-embedding-v3",
            "embedding_dimensions": 1024,
            "chat_model": "qwen-plus",
        }
        for field, value in alternatives.items():
            with self.subTest(field=field):
                payload = valid_workflow_args()
                payload["auxiliary_provider"][field] = value
                with self.assertRaisesRegex(ValueError, "frozen contract"):
                    load_auxiliary_provider_config(payload, required=True)

    def test_telemetry_path_is_independent_from_tool_trace(self):
        tool_path = FIXTURE_ROOT / "tools.jsonl"
        resolved = resolve_auxiliary_provider_telemetry_path(
            {"tool_trace_path": str(tool_path)}
        )
        self.assertEqual(
            Path(resolved),
            tool_path.parent / "auxiliary_provider_calls.jsonl",
        )
        self.assertNotEqual(Path(resolved), tool_path)


class AuxiliaryProviderUsageTrackerTest(unittest.TestCase):
    def setUp(self):
        self.config = load_auxiliary_provider_config(
            valid_workflow_args(), required=True
        )
        self.tracker = AuxiliaryProviderUsageTracker(self.config)

    def _persistent_tracker(self, path: Path, *, is_eval: bool = False):
        tracker = AuxiliaryProviderUsageTracker(
            self.config,
            telemetry_path=str(path),
        )
        tracker.start_rollout(
            task_id="task-1",
            rollout_id="batch/task-1/0",
            execution_id="execution-1",
            is_eval=is_eval,
        )
        return tracker

    def test_success_records_usage_without_payloads_or_fabricated_cost(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                    "ignored_provider_field": "not-copied",
                }
            )
        )
        returned = self.tracker.call(
            operation="chat_completion",
            model="qwen-max",
            invoke=lambda: response,
        )
        self.assertIs(returned, response)

        snapshot = self.tracker.snapshot()
        self.assertEqual(snapshot["total_calls"], 1)
        self.assertEqual(snapshot["failed_calls"], 0)
        self.assertEqual(snapshot["usage"]["total_tokens"], 15)
        self.assertIsNone(snapshot["cost"]["amount"])
        serialized = json.dumps(snapshot, sort_keys=True)
        self.assertNotIn("ignored_provider_field", serialized)
        self.assertNotIn("prompt", snapshot["calls"][0])
        self.assertNotIn("response", snapshot["calls"][0])
        self.assertNotIn("headers", snapshot["calls"][0])

    def test_failed_provider_call_is_sanitized_and_durably_persisted(self):
        secret = "sk-provider-error-must-not-leak"

        def fail():
            raise RuntimeError(f"Authorization: Bearer {secret}")

        with telemetry_file() as path:
            tracker = self._persistent_tracker(path, is_eval=True)
            with self.assertRaises(AuxiliaryProviderCallError) as raised:
                tracker.call(
                    operation="embedding",
                    model="text-embedding-v4",
                    invoke=fail,
                )

            self.assertNotIn(secret, str(raised.exception))
            records = read_jsonl(path)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(
                record["schema_version"],
                AUXILIARY_PROVIDER_CALL_SCHEMA_VERSION,
            )
            self.assertEqual(
                (
                    record["task_id"],
                    record["rollout_id"],
                    record["execution_id"],
                    record["call_index"],
                ),
                ("task-1", "batch/task-1/0", "execution-1", 0),
            )
            self.assertTrue(record["is_eval"])
            self.assertFalse(record["success"])
            self.assertEqual(record["outcome"], "provider_error")
            self.assertEqual(record["error_type"], "RuntimeError")
            serialized = json.dumps(record, sort_keys=True)
            self.assertNotIn(secret, serialized)
            for forbidden in ("prompt", "response", "headers", "api_key"):
                self.assertNotIn(forbidden, record)

    def test_malformed_and_empty_chat_response_are_failed_calls(self):
        secret = "sk-unit-test-secret-123456"
        malformed_responses = [
            SimpleNamespace(choices=[], usage=None),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
                usage=SimpleNamespace(total_tokens=1),
            ),
        ]
        with telemetry_file() as path:
            tracker = self._persistent_tracker(path)
            client = chat_client(api_key=secret, usage_tracker=tracker)
            client.client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_kwargs: malformed_responses.pop(0)
                    )
                )
            )

            for _ in range(2):
                with self.assertRaises(AuxiliaryProviderResponseError) as raised:
                    client.chat([{"role": "user", "content": "not persisted"}])
                self.assertNotIn(secret, str(raised.exception))

            records = read_jsonl(path)
            self.assertEqual([record["call_index"] for record in records], [0, 1])
            self.assertTrue(all(not record["success"] for record in records))
            self.assertTrue(
                all(record["outcome"] == "malformed_response" for record in records)
            )
            serialized = path.read_text(encoding="utf-8")
            self.assertNotIn(secret, serialized)
            self.assertNotIn("not persisted", serialized)

    def test_malformed_usage_does_not_hide_a_valid_provider_call(self):
        result = self.tracker.call(
            operation="chat_completion",
            model="qwen-max",
            invoke=lambda: {"usage": {"total_tokens": "not-an-integer"}},
            response_parser=lambda _response: "valid-body",
        )
        self.assertEqual(result, "valid-body")
        call = self.tracker.snapshot()["calls"][0]
        self.assertTrue(call["success"])
        self.assertEqual(call["usage_status"], "malformed")
        self.assertEqual(call["usage"], {})

    def test_persistence_failure_is_fail_closed_without_secret(self):
        secret = "sk-writer-secret-123456"

        class FailingRecorder:
            last_write_error_type = "OSError"

            def record(self, _event):
                raise AuxiliaryProviderTelemetryError(
                    f"unable to write api_key={secret}"
                )

        tracker = AuxiliaryProviderUsageTracker(
            self.config,
            telemetry_recorder=FailingRecorder(),
        )
        tracker.start_rollout(
            task_id="task-1",
            rollout_id="rollout-1",
            execution_id="execution-1",
            is_eval=False,
        )
        with self.assertRaises(AuxiliaryProviderTelemetryError) as raised:
            tracker.call(
                operation="embedding",
                model="text-embedding-v4",
                invoke=lambda: {"usage": {}},
            )
        self.assertNotIn(secret, str(raised.exception))

    def test_recorder_prepare_failure_has_a_generic_error_surface(self):
        blocker = FIXTURE_ROOT / f".m8b-provider-blocker-{uuid4().hex}"
        try:
            blocker.write_text("block", encoding="utf-8")
            with self.assertRaises(AuxiliaryProviderTelemetryError) as raised:
                AuxiliaryProviderTelemetryRecorder(
                    str(blocker / "provider.jsonl")
                )
            self.assertEqual(
                str(raised.exception),
                "unable to prepare auxiliary-provider telemetry",
            )
        finally:
            blocker.unlink(missing_ok=True)

    def test_call_indexes_are_thread_safe_and_reset_is_rollout_scoped(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    self.tracker.call,
                    operation="embedding",
                    model="text-embedding-v4",
                    invoke=lambda: {"usage": {"total_tokens": 1}},
                )
                for _ in range(32)
            ]
            for future in futures:
                future.result()
        snapshot = self.tracker.snapshot()
        self.assertEqual(snapshot["total_calls"], 32)
        self.assertEqual(
            [call["call_index"] for call in snapshot["calls"]],
            list(range(32)),
        )
        self.tracker.reset()
        self.assertEqual(self.tracker.snapshot()["total_calls"], 0)

    def test_in_flight_call_cannot_pollute_a_new_rollout_snapshot(self):
        entered = threading.Event()
        release = threading.Event()

        def delayed_response():
            entered.set()
            release.wait(timeout=5)
            return {"usage": {"total_tokens": 1}}

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.tracker.call,
                operation="embedding",
                model="text-embedding-v4",
                invoke=delayed_response,
            )
            self.assertTrue(entered.wait(timeout=5))
            self.tracker.reset()
            release.set()
            future.result()
        self.assertEqual(self.tracker.snapshot()["total_calls"], 0)

    def test_openai_clients_disable_sdk_retries(self):
        manager = MemoryManager(
            embedding_model="text-embedding-v4",
            embedding_dim=256,
            api_key="unit-test-key",
            usage_tracker=self.tracker,
        )
        client = chat_client(api_key="unit-test-key", usage_tracker=self.tracker)
        self.assertEqual(manager.client.max_retries, 0)
        self.assertEqual(client.client.max_retries, 0)

    def test_missing_or_empty_api_key_has_no_secret_example(self):
        with patch.dict(os.environ, {}, clear=True):
            for supplied_key in (None, "", "   "):
                with self.subTest(supplied_key=supplied_key):
                    with self.assertRaises(ValueError) as raised:
                        MemoryManager(
                            embedding_model="text-embedding-v4",
                            embedding_dim=256,
                            api_key=supplied_key,
                        )
                    self.assertEqual(
                        str(raised.exception),
                        "DASHSCOPE_API_KEY environment variable is not set",
                    )


if __name__ == "__main__":
    unittest.main()
