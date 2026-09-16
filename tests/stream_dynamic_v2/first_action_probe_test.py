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
    TASK_EXPLICIT_SYSTEM, build_plan, execute_probe,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import DynamicHistoryPublic, DynamicPublicChunk
from AgeMem_code_agentscope.streaming_memory.token_budget import DebugLexicalTokenizer, TokenAccounting


class FirstActionProbeTest(unittest.TestCase):
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
        launcher = {"mode": "bench", "model": {"enable_thinking": False}, "buffer": {"explorer_input": {
            "taskset": {"rollout_args": {"temperature": 0.99}},
            "eval_tasksets": [{"path": str(taskset), "rollout_args": {"temperature": 0.6}}]}}}
        sampling, thinking, _ = module.resolve_settings(launcher, taskset)
        self.assertEqual(sampling["temperature"], 0.6)
        self.assertFalse(thinking)
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
                "enable_thinking": False, "model_path": str(model_path)}, "buffer": {"explorer_input": {
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
