import copy
import importlib.util
import json
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

from AgeMem_code_agentscope.streaming_memory.dynamic.environment import DYNAMIC_INGEST_SYSTEM
from AgeMem_code_agentscope.streaming_memory.dynamic.first_action_probe import (
    SINGLE_ACTION_SYSTEM, STRUCTURE_ONLY_SYSTEM, TASK_EXPLICIT_SYSTEM, build_plan, execute_probe,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import DynamicHistoryPublic, DynamicPublicChunk
from AgeMem_code_agentscope.streaming_memory.token_budget import DebugLexicalTokenizer, TokenAccounting


class FirstActionProbeTest(unittest.TestCase):
    def test_v4_structure_only_changes_system_and_preserves_v3_baseline(self):
        h, accounting, budget, sampling = self.multichunk_fixture()
        old = build_plan(h, accounting, budget, sampling, comparison="single_vs_no_example_v3")
        plan = build_plan(h, accounting, budget, sampling, comparison="no_example_vs_structure_v4")
        self.assertEqual(plan["probe_version"], "agemem.dynamic.first_action_probe.v4")
        self.assertEqual(plan["chunk_indices"], old["chunk_indices"])
        self.assertEqual(plan["sampling"], old["sampling"])
        self.assertEqual(plan["response_token_upper_bound"], 6144)
        self.assertTrue(STRUCTURE_ONLY_SYSTEM.startswith(SINGLE_ACTION_SYSTEM))
        self.assertNotIn("FIRST_FACT", STRUCTURE_ONLY_SYSTEM)
        self.assertNotIn("SECOND_FACT", STRUCTURE_ONLY_SYSTEM)
        for previous, control, treatment in zip(old["cases"][1::2], plan["cases"][::2], plan["cases"][1::2]):
            self.assertEqual(previous, control)
            self.assertEqual(control["seed"], treatment["seed"])
            self.assertEqual(control["messages"][1:], treatment["messages"][1:])
            self.assertNotIn("ADD FORMAT EXAMPLE", str(treatment["messages"]))
            self.assertIn('"memory_id":"<fresh_id>"', treatment["messages"][0]["content"])

    def test_v4_missing_id_is_rejected_not_repaired_and_valid_content_is_stored(self):
        h, accounting, budget, sampling = self.multichunk_fixture()
        plan = build_plan(h, accounting, budget, sampling, comparison="no_example_vs_structure_v4")
        def generate(prompts, params):
            outputs = []
            for case, ids in zip(plan["cases"], prompts):
                index = case["chunk_index"]
                args = {"content": f"SECOND_FACT_{index}", "source_refs": [f"source-{index}-second"]}
                if case["profile"] == "structure_only_no_example_v4":
                    args["memory_id"] = "m1"
                outputs.append({"prompt_token_ids": ids, "response_text": json.dumps([{"name": "ADD", "arguments": args}]),
                                "response_token_ids": [1] * 100, "finish_reason": "stop"})
            return outputs
        rows, report = execute_probe(plan, accounting, generate)
        self.assertTrue(all(not row["action_result"]["admitted"] for row in rows[::2]))
        self.assertEqual(report["arms"]["single_action_no_example_v3"]["admitted_writes"], 0)
        self.assertEqual(report["arms"]["structure_only_no_example_v4"]["admitted_writes"], 6)
        self.assertFalse(any(row["concrete_add_example_present"] for row in rows))
        self.assertFalse(report["learning_effectiveness_checked"])

    def multichunk_fixture(self):
        h, accounting, budget, sampling = self.fixture()
        chunks = tuple(DynamicPublicChunk(chunk_id=f"chunk-{i}", observed_at=i,
                       text=f"FIRST_FACT_{i}\nSECOND_FACT_{i}",
                       source_refs=(f"source-{i}-first", f"source-{i}-second"), content_token_count=10)
                       for i in range(5))
        return h.model_copy(update={"chunks": chunks}), accounting, budget, sampling

    def test_v3_chunk_selection_example_ablation_and_isolation(self):
        h, accounting, budget, sampling = self.multichunk_fixture()
        plan = build_plan(h, accounting, budget, sampling, comparison="single_vs_no_example_v3")
        self.assertEqual(plan["chunk_indices"], [0, 2, 4])
        self.assertEqual(plan["model_call_count"], 12)
        self.assertEqual(plan["response_token_upper_bound"], 6144)
        self.assertEqual(plan["probe_version"], "agemem.dynamic.first_action_probe.v3")
        for with_example, without_example in zip(plan["cases"][::2], plan["cases"][1::2]):
            self.assertEqual(with_example["seed"], without_example["seed"])
            self.assertEqual(with_example["messages"][0], without_example["messages"][0])
            self.assertEqual(with_example["messages"][-1], without_example["messages"][-1])
            self.assertEqual(with_example["messages"][1]["content"].split("\nADD FORMAT EXAMPLE")[0],
                             without_example["messages"][1]["content"])
            self.assertNotIn("ADD FORMAT EXAMPLE", str(without_example["messages"]))
            index = with_example["chunk_index"]
            self.assertIn(f"SECOND_FACT_{index}", str(without_example["messages"]))
            for other in set(range(5)) - {index}:
                self.assertNotIn(f"FACT_{other}", str(without_example["messages"]))

    def test_v3_nonfirst_fact_metrics_and_zero_write_are_honest(self):
        h, accounting, budget, sampling = self.multichunk_fixture()
        plan = build_plan(h, accounting, budget, sampling, comparison="single_vs_no_example_v3")
        def generate(prompts, params):
            outputs = []
            for case, ids in zip(plan["cases"], prompts):
                index = case["chunk_index"]
                text = '[{"name":"NEXT","arguments":{}}]' if case["profile"] == "single_action_probe_v2" else json.dumps([
                    {"name": "ADD", "arguments": {"memory_id": "m1", "content": f"SECOND_FACT_{index}",
                                                   "source_refs": [f"source-{index}-second"]}}])
                outputs.append({"prompt_token_ids": ids, "response_text": text, "response_token_ids": [1] * 80})
            return outputs
        rows, report = execute_probe(plan, accounting, generate)
        self.assertEqual(len(rows), 12)
        self.assertEqual(report["arms"]["single_action_probe_v2"]["admitted_writes"], 0)
        arm = report["arms"]["single_action_no_example_v3"]
        self.assertEqual(arm["sample_count"], 6)
        self.assertEqual(arm["admitted_writes_by_chunk"], {"0": 2, "2": 2, "4": 2})
        self.assertEqual(arm["first_sentence_count"], 0)
        self.assertEqual(arm["nonfirst_fact_count"], 6)
        self.assertFalse(report["sequential_ingest_checked"])
        self.assertEqual(report["independent_history_count"], 1)

    def test_v3_selection_tamper_and_too_few_chunks_rejected(self):
        h, accounting, budget, sampling = self.multichunk_fixture()
        plan = build_plan(h, accounting, budget, sampling, comparison="single_vs_no_example_v3")
        plan["chunk_indices"] = [0, 1, 4]
        with self.assertRaises(ValueError):
            execute_probe(plan, accounting, lambda *_: self.fail("no model call"))
        with self.assertRaisesRegex(ValueError, "at least three"):
            build_plan(h.model_copy(update={"chunks": h.chunks[:2]}), accounting, budget, sampling,
                       comparison="single_vs_no_example_v3")

    def test_v2_comparison_changes_only_single_action_system_suffix(self):
        h, accounting, budget, sampling = self.fixture()
        old = build_plan(h, accounting, budget, sampling)
        plan = build_plan(h, accounting, budget, sampling, comparison="task_vs_single_v2")
        self.assertNotIn("comparison", old)
        self.assertEqual(old["probe_version"], "agemem.dynamic.first_action_probe.v1")
        self.assertEqual(plan["probe_version"], "agemem.dynamic.first_action_probe.v2")
        self.assertTrue(SINGLE_ACTION_SYSTEM.startswith(TASK_EXPLICIT_SYSTEM))
        self.assertEqual(plan["sampling"], old["sampling"])
        self.assertEqual(plan["budget"], old["budget"])
        self.assertEqual(plan["response_token_upper_bound"], 4096)
        for baseline, treatment in zip(plan["cases"][::2], plan["cases"][1::2]):
            self.assertEqual(baseline["profile"], "task_explicit_probe_v1")
            self.assertEqual(treatment["profile"], "single_action_probe_v2")
            self.assertEqual(baseline["seed"], treatment["seed"])
            self.assertEqual(baseline["messages"][1:], treatment["messages"][1:])
        self.assertEqual(plan["cases"][0]["messages"], old["cases"][1]["messages"])

    def test_v2_multi_action_is_not_repaired_and_single_fact_can_be_admitted(self):
        h, accounting, budget, sampling = self.fixture()
        plan = build_plan(h, accounting, budget, sampling, comparison="task_vs_single_v2")
        action = {"name": "ADD", "arguments": {"memory_id": "m1",
                  "content": h.chunks[0].text, "source_refs": ["public-src"]}}
        def generate(prompts, params):
            return [{"prompt_token_ids": ids,
                     "response_text": json.dumps([action, action] if index % 2 == 0 else [action]),
                     "response_token_ids": [1] * 150, "finish_reason": "stop"}
                    for index, ids in enumerate(prompts)]
        rows, report = execute_probe(plan, accounting, generate)
        self.assertTrue(all(row["action_result"] is None for row in rows[::2]))
        self.assertEqual(report["arms"]["task_explicit_probe_v1"]["admitted_writes"], 0)
        self.assertEqual(report["arms"]["single_action_probe_v2"]["admitted_writes"], 4)
        self.assertEqual(report["arms"]["single_action_probe_v2"]["exact_visible_source_body_count"], 4)
        self.assertEqual(report["comparison"], "task_vs_single_v2")
        self.assertFalse(report["learning_effectiveness_checked"])

    def test_v2_comparison_and_version_tampering_fail_before_model(self):
        h, accounting, budget, sampling = self.fixture()
        with self.assertRaisesRegex(ValueError, "unknown probe comparison"):
            build_plan(h, accounting, budget, sampling, comparison="unregistered")
        plan = build_plan(h, accounting, budget, sampling, comparison="task_vs_single_v2")
        for key, value in (("comparison", "legacy_vs_task_v1"),
                           ("probe_version", "agemem.dynamic.first_action_probe.v1")):
            changed = copy.deepcopy(plan)
            changed[key] = value
            with self.assertRaises(ValueError):
                execute_probe(changed, accounting, lambda *_: self.fail("must not sample"))

    def fixture(self):
        text = "自第 1 日起，示例项目的负责人为成员甲。"
        history = DynamicHistoryPublic(
            history_id="public-h", history_family_id="public-family", context_budget_tokens=2048,
            memory_budget_tokens=1024, chunks=(DynamicPublicChunk(
                chunk_id="public-chunk", observed_at=0, text=text, source_refs=("public-src",),
                content_token_count=20),))
        accounting = TokenAccounting.from_tokenizer(DebugLexicalTokenizer())
        budget = {"context_total_tokens": 2048, "persistent_memory_tokens": 1024,
                  "ingest_max_new_tokens": 512, "max_decisions_per_chunk": 3,
                  "answer_tail_tokens": 64, "retrieved_payload_tokens": 400}
        return history, accounting, budget, {"temperature": 0.6, "top_p": 1.0, "top_k": -1}

    def test_plan_is_eight_matched_first_actions_only_and_legacy_prompt_preserved(self):
        h, accounting, budget, sampling = self.fixture()
        plan = build_plan(h, accounting, budget, sampling)
        self.assertEqual(plan["model_call_count"], 8)
        self.assertEqual(plan["reader_call_count"], 0)
        self.assertEqual(plan["optimizer_update_count"], 0)
        self.assertEqual(plan["response_token_upper_bound"], 4096)
        self.assertTrue(TASK_EXPLICIT_SYSTEM.endswith(DYNAMIC_INGEST_SYSTEM))
        for old, new in zip(plan["cases"][::2], plan["cases"][1::2]):
            self.assertEqual(old["seed"], new["seed"])
            self.assertEqual(old["messages"][1:], new["messages"][1:])
            self.assertIn("Active memory handles: []", str(old["messages"]))
            self.assertNotIn('"question"', json.dumps(old["messages"]))

    def test_later_chunk_is_not_in_first_action_observation(self):
        h, accounting, budget, sampling = self.fixture()
        h = h.model_copy(update={"chunks": (*h.chunks, DynamicPublicChunk(
            chunk_id="later", observed_at=1, text="UNSEEN_LATER_STREAM_TEXT", source_refs=("later-src",),
            content_token_count=4))})
        plan = build_plan(h, accounting, budget, sampling)
        for case in plan["cases"]:
            self.assertNotIn("UNSEEN_LATER_STREAM_TEXT", json.dumps(case["messages"]))
            self.assertNotIn("later-src", json.dumps(case["messages"]))

    def test_valid_and_invalid_actions_are_independent_and_body_is_not_pointer_credit(self):
        h, accounting, budget, sampling = self.fixture()
        plan = build_plan(h, accounting, budget, sampling)
        def generate(prompts, params):
            self.assertEqual(len(prompts), 8)
            self.assertEqual(params[0]["seed"], params[1]["seed"])
            outputs = []
            for index, ids in enumerate(prompts):
                if index % 2 == 0:
                    text = '[{"name":"NEXT","arguments":{}}]'
                else:
                    text = json.dumps([{"name": "ADD", "arguments": {
                        "memory_id": "m1", "source_refs": ["public-src"],
                        "content": h.chunks[0].text if index != 3 else "unrelated topic summary"}}])
                outputs.append({"prompt_token_ids": ids, "response_text": text,
                                "response_token_ids": [1] * 90, "finish_reason": "stop"})
            return outputs
        rows, report = execute_probe(plan, accounting, generate)
        self.assertEqual(report["arms"]["legacy_v3"]["actions"], {"NEXT": 4})
        self.assertEqual(report["arms"]["task_explicit_probe_v1"]["admitted_writes"], 4)
        self.assertEqual(report["arms"]["task_explicit_probe_v1"]["exact_visible_source_body_count"], 3)
        self.assertTrue(rows[3]["source_refs_visible"])
        self.assertFalse(rows[3]["exact_visible_source_body"])
        self.assertFalse(report["learning_effectiveness_checked"])
        self.assertFalse(report["oracle_semantics_checked"])
        self.assertTrue(all(row["memory_tokens"] <= 1024 for row in rows))

    def test_invalid_response_never_executes_and_all_next_is_reported_not_fake_success(self):
        h, accounting, budget, sampling = self.fixture()
        plan = build_plan(h, accounting, budget, sampling)
        def generate(prompts, params):
            return [{"prompt_token_ids": ids, "response_text": '{"ACTION":"ADD","arguments":{}}',
                     "response_token_ids": [1] * 20} for ids in prompts]
        rows, report = execute_probe(plan, accounting, generate)
        self.assertTrue(all(row["action_result"] is None for row in rows))
        self.assertTrue(all(row["memory_tokens"] == 0 for row in rows))
        self.assertEqual(report["arms"]["legacy_v3"]["parse_codes"], {"not_array": 4})

    def test_changed_plan_and_model_token_overflow_are_rejected(self):
        h, accounting, budget, sampling = self.fixture()
        plan = build_plan(h, accounting, budget, sampling)
        changed = copy.deepcopy(plan)
        changed["cases"][0]["messages"][0]["content"] = "tampered"
        with self.assertRaises(ValueError):
            execute_probe(changed, accounting, lambda *_: self.fail("must not call model"))
        def overflow(prompts, params):
            return [{"prompt_token_ids": ids, "response_text": "[]", "response_token_ids": [1] * 513}
                    for ids in prompts]
        with self.assertRaisesRegex(ValueError, "token cap"):
            execute_probe(plan, accounting, overflow)

    def test_small_c_and_invalid_sampling_fail_before_model(self):
        h, accounting, budget, sampling = self.fixture()
        with self.assertRaises(ValueError):
            build_plan(h, accounting, budget, {**sampling, "temperature": float("nan")})
        budget = {**budget, "context_total_tokens": 512}
        h = h.model_copy(update={"context_budget_tokens": 512})
        with self.assertRaises(ValueError):
            build_plan(h, accounting, budget, sampling)

    def test_source_bench_sampling_and_thinking_are_resolved_not_training_args(self):
        path = Path(__file__).resolve().parents[2] / "scripts/agemem_dynamic_v2_first_action_probe.py"
        spec = importlib.util.spec_from_file_location("first_action_probe_cli_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        taskset = Path("public-taskset")
        launcher = {"mode": "bench", "model": {"enable_thinking": True},
            "explorer": {"rollout_model": {"enable_thinking": False}}, "buffer": {"explorer_input": {
            "taskset": {"rollout_args": {"temperature": 0.99}},
            "eval_tasksets": [{"path": str(taskset), "rollout_args": {"temperature": 0.6}}]}}}
        sampling, thinking, _ = module.resolve_settings(launcher, taskset)
        self.assertEqual(sampling["temperature"], 0.6)
        self.assertFalse(thinking)
        launcher["explorer"]["rollout_model"]["enable_thinking"] = True
        self.assertTrue(module.resolve_settings(launcher, taskset)[1])
        for invalid in (None, "False", 0, 1):
            launcher["explorer"]["rollout_model"]["enable_thinking"] = invalid
            with self.assertRaisesRegex(ValueError,
                    "source launcher must explicitly lock explorer.rollout_model.enable_thinking"):
                module.resolve_settings(launcher, taskset)
        del launcher["explorer"]["rollout_model"]["enable_thinking"]
        with self.assertRaisesRegex(ValueError, "explorer.rollout_model.enable_thinking"):
            module.resolve_settings(launcher, taskset)
        class Tokenizer(DebugLexicalTokenizer):
            def apply_chat_template(self, *a, enable_thinking, **kw):
                self.last_thinking = enable_thinking
                return super().apply_chat_template(*a, **kw)
        tokenizer = Tokenizer()
        view = module.ThinkingTokenizerView(tokenizer, False)
        view.apply_chat_template([], tokenize=True, add_generation_prompt=True)
        self.assertFalse(tokenizer.last_thinking)

    def test_prepare_and_run_cli_boundaries_freeze_plan_and_emit_eight_rows(self):
        # Complete CLI function flow with explicit resource doubles, not a
        # production tokenizer/model/GPU validation.
        h, accounting, budget, _ = self.fixture()
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "first_action_probe_cli_boundary_test", root / "scripts/agemem_dynamic_v2_first_action_probe.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=root / "runs") as scratch:
            folder = Path(scratch)
            model_path = folder / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text('{}')
            (model_path / "tokenizer.json").write_text('{}')
            (model_path / "model.safetensors").write_bytes(b"resource-double-not-model-weights")
            taskset = folder / "taskset"
            taskset.mkdir()
            config_path = folder / "config.json"
            config_path.write_text('{}')
            launcher_path = folder / "launcher.yaml"
            launcher_path.write_text(json.dumps({"mode": "bench", "model": {
                "model_path": str(model_path)},
                "explorer": {"rollout_model": {"enable_thinking": False}}, "buffer": {"explorer_input": {
                    "eval_tasksets": [{"path": str(taskset), "expected_dataset_fingerprint": "frozen",
                        "expected_row_ids": [h.history_id], "rollout_args": {"temperature": 0.6}}]}}}))
            class Dataset(list):
                _fingerprint = "frozen"
            datasets = types.ModuleType("datasets")
            datasets.load_from_disk = lambda _: {"train": Dataset([h.model_dump(mode="json")])}
            cfg = types.SimpleNamespace(
                model={"policy_path": str(model_path), "tokenizer_path": str(model_path),
                       "policy_revision": "declared", "tokenizer_revision": "declared"},
                runtime={"gpu_ids": [1, 2]}, experiment={"seed": 7}, budget=budget)
            prepared = folder / "prepared"
            args = types.SimpleNamespace(config=config_path, launcher=launcher_path, taskset=taskset,
                                         gpu_id=1, output_dir=prepared)
            import io
            from contextlib import redirect_stdout
            with patch.object(module, "load_dynamic_config", return_value=cfg), \
                    patch.object(module, "accounting_for", return_value=accounting), \
                    patch.dict("sys.modules", {"datasets": datasets}), redirect_stdout(io.StringIO()):
                module.prepare(args)
            plan_path = prepared / "plan.public.json"
            self.assertEqual(json.loads(plan_path.read_text(encoding="utf-8"))["sampling"]["temperature"], 0.6)
            self.assertIs(json.loads(plan_path.read_text(encoding="utf-8"))["model"]["enable_thinking"], False)
            torch = types.ModuleType("torch")
            torch.cuda = types.SimpleNamespace(device_count=lambda: 1, get_device_name=lambda _: "DOUBLE")
            transformers = types.ModuleType("transformers")
            transformers.PreTrainedTokenizerBase = type("Base", (), {"all_special_tokens": []})
            vllm = types.ModuleType("vllm")
            class LLM:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                def generate(self, prompts, params, use_tqdm):
                    assert len(prompts) == len(params) == 8
                    assert all(param.n == 1 and param.max_tokens == 512 for param in params)
                    return [types.SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[
                        types.SimpleNamespace(text='[{"name":"NEXT","arguments":{}}]',
                                              token_ids=[1] * 20, finish_reason="stop")]) for p in prompts]
            vllm.LLM = LLM
            vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
            output = folder / "results"
            with patch.object(module, "accounting_for", return_value=accounting), \
                    patch.dict("sys.modules", {"torch": torch, "transformers": transformers, "vllm": vllm}), \
                    patch.dict("os.environ", {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"}), \
                    redirect_stdout(io.StringIO()):
                module.run(types.SimpleNamespace(plan=plan_path, output_dir=output))
            rows = json.loads((output / "actions.public.json").read_text(encoding="utf-8"))
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 8)
            self.assertEqual(report["arms"]["task_explicit_probe_v1"]["actions"], {"NEXT": 4})
            self.assertEqual(report["arms"]["task_explicit_probe_v1"]["admitted_writes"], 0)
            self.assertFalse(report["full_weight_digest_verified"])
