"""Execute Explorer's real finish-eval body with CPU-only boundary doubles."""

import ast
import asyncio
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from trinity.common.diagnostic_bench import should_persist_diagnostic_bench


ROOT = Path(__file__).resolve().parents[2]


class BenchPersistenceTest(unittest.TestCase):
    def test_shared_gate_preserves_old_and_rejects_ordinary_eval(self):
        for workflow in (
            "AgeMem_hotpot_workflow_training",
            "AgeMem_dynamic_multiquery_v2_training",
        ):
            self.assertTrue(should_persist_diagnostic_bench("bench", workflow))
            self.assertFalse(should_persist_diagnostic_bench("both", workflow))
        self.assertFalse(should_persist_diagnostic_bench("bench", "math_workflow"))

    def run_finish(self, mode, workflow, failed=False):
        tree = ast.parse((ROOT / "trinity/explorer/explorer.py").read_text())
        explorer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "Explorer")
        method = next(node for node in explorer.body
                      if isinstance(node, ast.AsyncFunctionDef)
                      and node.name == "_finish_eval_step")
        calls, receipts = [], []
        namespace = {
            "Optional": Optional,
            "time": time,
            "should_persist_diagnostic_bench": should_persist_diagnostic_bench,
            "gather_metrics": lambda *args: {},
            "write_benchmark_receipt": lambda *args, **kwargs: receipts.append(kwargs),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT), "exec"),
             namespace)
        experience = object()

        async def get_results(batch_id):
            return [SimpleNamespace(ok=not failed, metric={})], [experience]

        async def persist(exps):
            calls.append(exps)
            return {"pipeline/diagnostic_experience_count": len(exps)}

        config = SimpleNamespace(
            mode=mode,
            checkpoint_job_dir="unused-test-directory",
            buffer=SimpleNamespace(explorer_input=SimpleNamespace(
                default_eval_workflow_type=workflow)),
        )
        actor = SimpleNamespace(
            config=config,
            pending_eval_tasks=deque([(0, "fixture")]),
            explore_step_num=0,
            model_version=0,
            scheduler=SimpleNamespace(get_results=get_results),
            experience_pipeline=SimpleNamespace(
                persist_diagnostic_input=SimpleNamespace(remote=persist)),
            monitor=SimpleNamespace(log=lambda *args, **kwargs: None),
        )
        asyncio.run(namespace["_finish_eval_step"](actor, prefix="bench"))
        return calls, receipts, experience

    def test_dynamic_bench_invokes_persistence_before_receipt(self):
        calls, receipts, experience = self.run_finish(
            "bench", "AgeMem_dynamic_multiquery_v2_training")
        self.assertEqual(calls, [[experience]])
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["task_summaries"][0]["failed_count"], 0)

    def test_ordinary_eval_does_not_persist(self):
        calls, receipts, _ = self.run_finish(
            "both", "AgeMem_dynamic_multiquery_v2_training")
        self.assertEqual(calls, [])
        self.assertEqual(len(receipts), 1)

    def test_failed_task_cannot_produce_success_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "evaluation tasks failed"):
            self.run_finish("bench", "AgeMem_dynamic_multiquery_v2_training", True)
