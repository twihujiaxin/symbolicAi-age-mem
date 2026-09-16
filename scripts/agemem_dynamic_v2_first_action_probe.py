"""Prepare on CPU, then sample locked first actions with one local vLLM GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from AgeMem_code_agentscope.streaming_memory.dynamic.config import load_dynamic_config
from AgeMem_code_agentscope.streaming_memory.dynamic.first_action_probe import (
    COMPARISONS, DEFAULT_COMPARISON, build_plan, digest, execute_probe,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import DynamicHistoryPublic
from AgeMem_code_agentscope.streaming_memory.token_budget import TokenAccounting, load_tokenizer


class ThinkingTokenizerView:
    """Use exactly the same thinking flag when counting and passing model IDs."""
    def __init__(self, tokenizer, enable_thinking):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, enable_thinking=self.enable_thinking, **kwargs)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_identity(model_path, tokenizer_path):
    # Full weights are not rehashed; this boundary is recorded in report.
    files = [Path(model_path) / "config.json"]
    files += [path for path in Path(model_path).glob("*.index.json") if path.is_file()]
    files += [Path(tokenizer_path) / name for name in (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "chat_template.jinja", "vocab.json", "merges.txt")
        if (Path(tokenizer_path) / name).is_file()]
    return {str(path.resolve()): sha(path) for path in files}


def weight_file_stats(model_path):
    paths = sorted([*Path(model_path).glob("*.safetensors"), *Path(model_path).glob("pytorch_model*.bin")])
    if not paths or any(path.stat().st_size == 0 for path in paths):
        raise ValueError("local nonempty model weights required")
    return {path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in paths}


def code_identity():
    paths = [Path(__file__)] + [ROOT / "AgeMem_code_agentscope/streaming_memory" / name for name in (
        "dynamic/first_action_probe.py", "dynamic/environment.py", "dynamic/action_parser.py",
        "dynamic/schema.py", "token_budget.py")]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def resolve_settings(launcher, taskset):
    if launcher.get("mode") != "bench":
        raise ValueError("probe source launcher must be bench")
    datasets = launcher["buffer"]["explorer_input"]["eval_tasksets"]
    selected = [item for item in datasets if Path(item["path"]).resolve() == taskset.resolve()]
    if len(selected) != 1:
        raise ValueError("expected one matching frozen bench taskset")
    rollout = selected[0].get("rollout_args") or {}
    # Mirror GenerationConfig defaults; do not inherit training taskset args.
    sampling = {"temperature": rollout.get("temperature", 1.0),
                "top_p": rollout.get("top_p", 1.0), "top_k": rollout.get("top_k", -1)}
    model = launcher["explorer"]["rollout_model"]
    if not isinstance(model.get("enable_thinking"), bool):
        raise ValueError("source launcher must explicitly lock explorer.rollout_model.enable_thinking")
    return sampling, model["enable_thinking"], selected[0]


def accounting_for(model):
    tokenizer = load_tokenizer(model["tokenizer_path"], model["tokenizer_revision"])
    return TokenAccounting.from_tokenizer(
        ThinkingTokenizerView(tokenizer, model["enable_thinking"]),
        name=model["tokenizer_path"], revision=model["tokenizer_revision"])


def prepare(args):
    config = load_dynamic_config(args.config)
    if config.model.get("tokenizer_path") in (None, "", "debug-lexical"):
        raise ValueError("GPU probe preparation requires frozen production tokenizer")
    ids = config.runtime.get("gpu_ids")
    if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
            or len(ids) != len(set(ids)) or args.gpu_id not in ids):
        raise ValueError("physical probe GPU must belong to nonempty frozen runtime.gpu_ids")
    launcher = json.loads(args.launcher.read_text(encoding="utf-8"))
    sampling, thinking, taskset_settings = resolve_settings(launcher, args.taskset)
    if taskset_settings.get("split", "train") != "train":
        raise ValueError("probe must not read dev/test")
    if taskset_settings.get("row_indices") not in (None, [0]):
        raise ValueError("probe only supports the one-history smoke row")
    model = {"policy_path": str(Path(config.model["policy_path"]).resolve()),
             "policy_declared_revision": config.model.get("policy_revision"),
             "tokenizer_path": str(Path(config.model["tokenizer_path"]).resolve()),
             "tokenizer_revision": config.model["tokenizer_revision"], "enable_thinking": thinking}
    if not model["policy_declared_revision"] or "unresolved" in str(model["policy_declared_revision"]).lower():
        raise ValueError("frozen policy declared revision required")
    if Path(launcher["model"]["model_path"]).resolve() != Path(model["policy_path"]):
        raise ValueError("source launcher policy differs from frozen dynamic config")
    from datasets import load_from_disk
    train = load_from_disk(str(args.taskset))["train"]
    fingerprint = taskset_settings.get("expected_dataset_fingerprint")
    if not fingerprint or train._fingerprint != fingerprint:
        raise ValueError("source bench fingerprint missing or differs from frozen taskset")
    if len(train) != 1:
        raise ValueError("probe expects the existing one-history frozen train taskset")
    history = DynamicHistoryPublic.model_validate(train[0])
    expected_ids = taskset_settings.get("expected_row_ids")
    if expected_ids is not None and expected_ids != [history.history_id]:
        raise ValueError("source bench history ID differs from frozen taskset")
    plan = build_plan(history, accounting_for(model), config.budget, sampling,
                      seed=int(config.experiment["seed"]),
                      comparison=getattr(args, "comparison", DEFAULT_COMPARISON))
    plan.update({"model": model, "physical_gpu_id": args.gpu_id,
                 "backend": {"name": "direct_vllm_llm", "tensor_parallel_size": 1,
                             "gpu_memory_utilization": 0.55, "dtype": "bfloat16",
                             "enforce_eager": True},
                 "git_commit": git_commit(), "code_identity": code_identity(),
                 "file_identity": file_identity(model["policy_path"], model["tokenizer_path"]),
                 "weight_file_stats": weight_file_stats(model["policy_path"]),
                 "source_config_sha256": sha(args.config), "source_launcher_sha256": sha(args.launcher),
                 "source_taskset_fingerprint": train._fingerprint,
                 "full_weight_digest_verified": False})
    plan["plan_sha256"] = digest(plan)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plan.public.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared_cpu", "plan": str(args.output_dir / "plan.public.json"),
                      "sampling": plan["sampling"], "enable_thinking": thinking,
                      "model_call_count": plan["model_call_count"], "response_token_upper_bound": plan["response_token_upper_bound"],
                      "max_prompt_plus_output": max(len(case["prompt_token_ids"]) for case in plan["cases"])
                          + plan["sampling"]["max_tokens"],
                      "reader_calls": 0, "optimizer_updates": 0}, indent=2))


def run(args):
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    unsigned = dict(plan)
    expected = unsigned.pop("plan_sha256")
    if digest(unsigned) != expected:
        raise ValueError("plan digest mismatch")
    if code_identity() != plan["code_identity"] or git_commit() != plan["git_commit"]:
        raise ValueError("probe code changed after CPU prepare")
    model = plan["model"]
    if file_identity(model["policy_path"], model["tokenizer_path"]) != plan["file_identity"]:
        raise ValueError("model config/tokenizer files changed after prepare")
    if weight_file_stats(model["policy_path"]) != plan["weight_file_stats"]:
        raise ValueError("weight files changed after prepare (stats check, not full digest)")
    if (os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != str(plan["physical_gpu_id"])):
        raise ValueError("set PCI_BUS_ID and exactly the one prepared physical GPU")
    accounting = accounting_for(model)
    # Revalidate prompt IDs and C before loading the GPU engine.
    rebuilt = build_plan(DynamicHistoryPublic.model_validate(plan["history"]), accounting,
                         plan["budget"], plan["sampling"], plan["cases"][0]["seed"],
                         comparison=plan.get("comparison", DEFAULT_COMPARISON))
    if rebuilt["cases"] != plan["cases"] or rebuilt["prompt_sha256"] != plan["prompt_sha256"]:
        raise ValueError("prompt or tokenizer mismatch")
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError("probe requires exactly one visible CUDA GPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Reuse the repository's existing tokenizer compatibility shim without Ray.
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
    from vllm import LLM, SamplingParams
    started = time.perf_counter()
    engine = LLM(model=model["policy_path"], tokenizer=model["tokenizer_path"],
                 tensor_parallel_size=1, max_model_len=int(plan["budget"]["context_total_tokens"]),
                 gpu_memory_utilization=0.55, dtype="bfloat16", enforce_eager=True,
                 seed=plan["cases"][0]["seed"], trust_remote_code=False)
    startup_seconds = time.perf_counter() - started
    sampling_seconds = 0.0

    def generate(prompts, params):
        nonlocal sampling_seconds
        sample_started = time.perf_counter()
        outputs = engine.generate(
            [{"prompt_token_ids": ids} for ids in prompts],
            [SamplingParams(**item, skip_special_tokens=True) for item in params], use_tqdm=False)
        sampling_seconds = time.perf_counter() - sample_started
        rows = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise ValueError("one completion per request required")
            completion = output.outputs[0]
            rows.append({"prompt_token_ids": list(output.prompt_token_ids),
                         "response_text": completion.text, "response_token_ids": list(completion.token_ids),
                         "finish_reason": completion.finish_reason})
        return rows

    rows, report = execute_probe(plan, accounting, generate)
    report.update({"git_commit": plan["git_commit"], "plan_sha256": expected, "model": model,
                   "backend": plan["backend"], "physical_gpu_id": plan["physical_gpu_id"],
                   "gpu_name": torch.cuda.get_device_name(0), "full_weight_digest_verified": False,
                   "engine_startup_seconds": startup_seconds, "sampling_seconds": sampling_seconds,
                   "total_runtime_seconds": time.perf_counter() - started,
                   "vllm_version": getattr(sys.modules["vllm"], "__version__", "unavailable")})
    (args.output_dir / "actions.public.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("prepare")
    cpu.add_argument("--config", type=Path, required=True)
    cpu.add_argument("--launcher", type=Path, required=True)
    cpu.add_argument("--taskset", type=Path, required=True)
    cpu.add_argument("--gpu-id", type=int, default=1)
    cpu.add_argument("--comparison", choices=tuple(COMPARISONS), default=DEFAULT_COMPARISON)
    cpu.add_argument("--output-dir", type=Path, required=True)
    gpu = commands.add_parser("run")
    gpu.add_argument("--plan", type=Path, required=True)
    gpu.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
