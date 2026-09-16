"""CPU-only audit of source-labelled ingest prompts; never generates model tokens."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AgeMem_code_agentscope.streaming_memory.dynamic.config import load_dynamic_config
from AgeMem_code_agentscope.streaming_memory.dynamic.environment import DynamicMemoryEnvironment
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import DynamicHistoryPublic
from AgeMem_code_agentscope.streaming_memory.token_budget import TokenAccounting, load_tokenizer


def check_history(history, accounting, budget):
    env = DynamicMemoryEnvironment(
        history, accounting=accounting, read_rollout_id="cpu-prompt-preflight",
        policy_version="no-model", ingest_max_new_tokens=int(budget["ingest_max_new_tokens"]),
        max_decisions_per_chunk=int(budget["max_decisions_per_chunk"]),
        answer_tail_tokens=int(budget["answer_tail_tokens"]),
        retrieval_payload_tokens=int(budget["retrieved_payload_tokens"]),
    )
    maximum = 0
    for _ in history.chunks:
        messages = env.admit_next_chunk()
        maximum = max(maximum, accounting.count_chat(messages) + env.ingest_max_new_tokens)
        env.commit_chunk()
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    args = parser.parse_args()
    config = load_dynamic_config(args.config)
    tokenizer = load_tokenizer(config.model["tokenizer_path"], config.model["tokenizer_revision"])
    accounting = TokenAccounting.from_tokenizer(tokenizer, revision=config.model["tokenizer_revision"])
    from datasets import load_from_disk
    dataset = load_from_disk(str(args.taskset))["train"]
    maxima = [
        check_history(DynamicHistoryPublic.model_validate(row), accounting, config.budget)
        for row in dataset
    ]
    if not maxima:
        raise ValueError("empty smoke taskset")
    print(json.dumps({
        "status": "pass", "histories": len(maxima), "max_prompt_plus_output": max(maxima),
        "scope": "public ingest with empty memory and no action receipts; runtime rechecks every call",
        "model_generation": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
