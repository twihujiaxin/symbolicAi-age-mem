"""CPU preparation and explicitly authorized one-GPU bounded sequential ingest."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from AgeMem_code_agentscope.streaming_memory.dynamic.first_action_probe import digest
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import DynamicHistoryPublic
from AgeMem_code_agentscope.streaming_memory.dynamic.sequential_probe import (
    build_sequential_plan, execute_sequential_plan, validate_sequential_plan,
)

spec = importlib.util.spec_from_file_location("first_probe_common", ROOT / "scripts/agemem_dynamic_v2_first_action_probe.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def code_identity():
    result = common.code_identity()
    for name in ("scripts/agemem_dynamic_v2_sequential_probe.py",
                 "AgeMem_code_agentscope/streaming_memory/dynamic/sequential_probe.py"):
        result[name] = common.sha(ROOT / name)
    return result


def verify_identity(plan):
    unsigned = dict(plan)
    seal = unsigned.pop("plan_sha256")
    if digest(unsigned) != seal:
        raise ValueError("plan digest changed")
    model = plan["model"]
    if plan["code_identity"] != code_identity() or plan["git_commit"] != common.git_commit():
        raise ValueError("sequential code or HEAD changed after prepare")
    if plan["file_identity"] != common.file_identity(model["policy_path"], model["tokenizer_path"]):
        raise ValueError("model config/tokenizer identity changed")
    if plan["weight_file_stats"] != common.weight_file_stats(model["policy_path"]):
        raise ValueError("weight stats changed (not full cryptographic hash)")


def prepare(args):
    source = json.loads(args.source_plan.read_text(encoding="utf-8"))
    unsigned = dict(source)
    if unsigned.pop("plan_sha256") != digest(unsigned):
        raise ValueError("source probe plan digest mismatch")
    if source["probe_version"] != "agemem.dynamic.first_action_probe.v4":
        raise ValueError("use the frozen v4 model/input identity")
    if source["code_identity"] != common.code_identity():
        raise ValueError("original probe code changed; do not silently reinterpret its input")
    model = source["model"]
    if source["file_identity"] != common.file_identity(model["policy_path"], model["tokenizer_path"]):
        raise ValueError("source tokenizer/config identity changed")
    if source["weight_file_stats"] != common.weight_file_stats(model["policy_path"]):
        raise ValueError("source model weight stats changed")
    plan = build_sequential_plan(DynamicHistoryPublic.model_validate(source["history"]),
                                 common.accounting_for(model), source["budget"], source["sampling"])
    plan.update({"model": model, "physical_gpu_id": source["physical_gpu_id"], "backend": source["backend"],
                 "file_identity": source["file_identity"], "weight_file_stats": source["weight_file_stats"],
                 "source_plan_sha256": source["plan_sha256"], "source_file_sha256": common.sha(args.source_plan),
                 "source_git_commit": source["git_commit"], "git_commit": common.git_commit(),
                 "code_identity": code_identity(), "full_weight_digest_verified": False})
    plan["plan_sha256"] = digest(plan)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plan.public.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "sequential_prepared_cpu", "plan": str(args.output_dir / "plan.public.json"),
                      "plan_sha256": plan["plan_sha256"], "profile": plan["profile"],
                      "prefix_indices": plan["prefix_indices"], "model_call_upper_bound": 18,
                      "response_token_upper_bound": plan["response_token_upper_bound"],
                      "first_prompt_plus_output": len(plan["first_prompt_token_ids"]) + plan["sampling"]["max_tokens"],
                      "future_state_dependent_prompts_checked_at_each_call": True,
                      "actual_model_calls": 0, "reader_calls": 0, "optimizer_updates": 0}, indent=2))


def run(args):
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    verify_identity(plan)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(plan["physical_gpu_id"]) or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("exact prepared physical GPU and PCI_BUS_ID required")
    accounting = common.accounting_for(plan["model"])
    # Rebuild the fixed prefix contract before loading vLLM. Dynamic C/B is checked on every call.
    validate_sequential_plan(plan, accounting)
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible CUDA device required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
    from vllm import LLM, SamplingParams
    started = time.perf_counter()
    engine = LLM(model=plan["model"]["policy_path"], tokenizer=plan["model"]["tokenizer_path"],
                 tensor_parallel_size=1, max_model_len=plan["budget"]["context_total_tokens"], dtype="bfloat16",
                 gpu_memory_utilization=0.55, enforce_eager=True, seed=7, trust_remote_code=False)
    startup = time.perf_counter() - started
    generation_seconds = 0.0
    def generate(ids, params):
        nonlocal generation_seconds
        begin = time.perf_counter()
        outputs = engine.generate([{"prompt_token_ids": ids}], SamplingParams(**params, skip_special_tokens=True), use_tqdm=False)
        generation_seconds += time.perf_counter() - begin
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ValueError("one completion per serial request required")
        output, completion = outputs[0], outputs[0].outputs[0]
        return {"prompt_token_ids": list(output.prompt_token_ids), "response_text": completion.text,
                "response_token_ids": list(completion.token_ids), "finish_reason": completion.finish_reason}
    with (args.output_dir / "actions.public.jsonl").open("x", encoding="utf-8") as stream:
        def record(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({k: row[k] for k in ("rollout_index", "chunk_index", "turn", "parse_code", "action_name", "action_result")}), flush=True)
        rows, snapshots, report = execute_sequential_plan(plan, accounting, generate, record)
    report.update({"git_commit": plan["git_commit"], "plan_sha256": plan["plan_sha256"], "model": plan["model"],
                   "engine_startup_seconds": startup, "sampling_seconds": generation_seconds,
                   "total_runtime_seconds": time.perf_counter() - started,
                   "physical_gpu_id": plan["physical_gpu_id"], "full_weight_digest_verified": False})
    (args.output_dir / "snapshots.public.json").write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("prepare")
    cpu.add_argument("--source-plan", type=Path, required=True)
    cpu.add_argument("--output-dir", type=Path, required=True)
    gpu = commands.add_parser("run")
    gpu.add_argument("--plan", type=Path, required=True)
    gpu.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
