import copy
import json
import unittest
import importlib.util
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from tests.stream_dynamic_v2 import first_action_probe_test as fixtures
from AgeMem_code_agentscope.streaming_memory.dynamic.sequential_probe import build_sequential_plan, execute_sequential_plan


class SequentialProbeTest(unittest.TestCase):
    def test_cpu_cli_prepare_and_identity_with_resource_doubles(self):
        h, account, budget, sampling = self.fixture()
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("sequential_cli_test", root / "scripts/agemem_dynamic_v2_sequential_probe.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory(dir=root / "runs") as scratch:
            folder = Path(scratch)
            model = folder / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "tokenizer.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"not-real-weights")
            source = fixtures.build_plan(h, account, budget, sampling, comparison="no_example_vs_structure_v4")
            source.update({"model": {"policy_path": str(model), "tokenizer_path": str(model),
                                     "tokenizer_revision": "resource-double", "enable_thinking": False},
                           "physical_gpu_id": 1, "backend": {}, "git_commit": cli.common.git_commit(),
                           "code_identity": cli.common.code_identity(),
                           "file_identity": cli.common.file_identity(model, model),
                           "weight_file_stats": cli.common.weight_file_stats(model)})
            source["plan_sha256"] = cli.digest(source)
            path = folder / "source.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            import io
            from contextlib import redirect_stdout
            with patch.object(cli.common, "accounting_for", return_value=account), redirect_stdout(io.StringIO()):
                cli.prepare(types.SimpleNamespace(source_plan=path, output_dir=folder / "prepared"))
            plan = json.loads((folder / "prepared/plan.public.json").read_text(encoding="utf-8"))
            cli.verify_identity(plan)
            self.assertEqual(plan["model_call_upper_bound"], 18)
            self.assertEqual(plan["source_plan_sha256"], source["plan_sha256"])
            self.assertEqual(plan["history"]["chunks"], source["history"]["chunks"][:3])
            self.assertFalse(plan["full_weight_digest_verified"])
            # Identity failures never create a second plan.
            source["model"]["enable_thinking"] = True
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                cli.prepare(types.SimpleNamespace(source_plan=path, output_dir=folder / "bad"))
            self.assertFalse((folder / "bad").exists())

    def fixture(self):
        return fixtures.FirstActionProbeTest().multichunk_fixture()

    def test_prefix_budget_isolation_feedback_and_persistent_memories(self):
        h, account, budget, sampling = self.fixture()
        plan = build_sequential_plan(h, account, budget, sampling)
        self.assertEqual(plan["prefix_indices"], [0, 1, 2])
        self.assertEqual(plan["model_call_upper_bound"], 18)
        self.assertEqual(plan["response_token_upper_bound"], 9216)
        self.assertNotIn("FACT_3", str(plan["history"]))
        calls = []
        def generate(ids, params):
            turn = len(calls) % 9
            chunk, decision = divmod(turn, 3)
            calls.append(params)
            args = {"memory_id": f"m{chunk}", "content": f"SECOND_FACT_{chunk}", "source_refs": [f"source-{chunk}-second"]}
            action = ({"name": "ADD", "arguments": args} if decision == 0 else
                      {"name": "RETRIEVE", "arguments": {"memory_id": f"m{chunk}"}} if decision == 1 else
                      {"name": "NEXT", "arguments": {}})
            return {"prompt_token_ids": ids, "response_text": json.dumps([action]), "response_token_ids": [1] * 80}
        rows, snapshots, report = execute_sequential_plan(plan, account, generate)
        self.assertEqual(report["model_call_count"], 18)
        self.assertEqual(report["admitted_writes"], 6)
        self.assertEqual(len(snapshots), 2)
        for snapshot in snapshots:
            self.assertEqual(len(snapshot["snapshot"]["active_memories"]), 3)
            self.assertEqual(len(snapshot["checkpoints"]), 3)
        self.assertIn('"memory_id":"m0"', str(rows[1]["prompt_messages"]))
        self.assertNotIn('"memory_id":"m0"', str(rows[9]["prompt_messages"]))
        self.assertTrue(all(r["memory_tokens"] <= budget["persistent_memory_tokens"] for r in rows))
        self.assertFalse(report["learning_effectiveness_checked"])
        self.assertFalse(report["experience_action_contract_checked"])

    def test_invalid_response_receives_feedback_without_repair(self):
        h, account, budget, sampling = self.fixture()
        plan = build_sequential_plan(h, account, budget, sampling)
        count = 0
        def generate(ids, params):
            nonlocal count
            response = '{"ACTION":"ADD"}' if count % 2 == 0 else '[{"name":"NEXT","arguments":{}}]'
            count += 1
            return {"prompt_token_ids": ids, "response_text": response, "response_token_ids": [1] * 20}
        rows, snapshots, report = execute_sequential_plan(plan, account, generate)
        self.assertEqual(report["model_call_count"], 12)
        self.assertEqual(report["admitted_writes"], 0)
        self.assertTrue(all(not r["action_result"]["admitted"] for r in rows[::2]))
        self.assertIn("JSON array", str(rows[1]["prompt_messages"]))
        self.assertTrue(all(not s["snapshot"]["active_memories"] for s in snapshots))

    def test_tampering_and_actual_token_overflow_fail(self):
        h, account, budget, sampling = self.fixture()
        plan = build_sequential_plan(h, account, budget, sampling)
        changed = copy.deepcopy(plan)
        changed["rollout_seeds"] = [17, 27]
        with self.assertRaises(ValueError):
            execute_sequential_plan(changed, account, lambda *_: self.fail("no model call"))
        with self.assertRaises(ValueError):
            build_sequential_plan(h, account, {**budget, "max_decisions_per_chunk": 4}, sampling)
        with self.assertRaisesRegex(ValueError, "token budget"):
            execute_sequential_plan(plan, account, lambda ids, _: {
                "prompt_token_ids": ids, "response_text": "[]", "response_token_ids": [1] * 513})
