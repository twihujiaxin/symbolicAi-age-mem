"""Read-only CPU response classification; never repairs or replays actions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from AgeMem_code_agentscope.streaming_memory.dynamic.action_parser import parse_public_action


def classify_rows(rows):
    codes = Counter()
    overridden = 0
    for row in rows:
        text = row.get("response_text") or ""
        codes[parse_public_action(text).code] += 1
        payload = text.strip()
        if payload.startswith("<tool_call>") and payload.endswith("</tool_call>"):
            payload = payload[len("<tool_call>"):-len("</tool_call>")].strip()
        try:
            value = json.loads(payload)
            call = value[0] if isinstance(value, list) and len(value) == 1 else None
            args = call.get("arguments") if isinstance(call, dict) else None
            if isinstance(args, dict) and "type" in args:
                overridden += 1
        except (ValueError, TypeError):
            pass
    return {"experience_count": len(rows), "strict_parse_codes": dict(codes),
            "responses_with_legacy_type_override": overridden,
            "action_execution": False, "source_semantics_checked": False}


def verify_runtime(rows):
    """Audit persisted summaries/joins, not missing raw model tensors or semantics."""
    infos = [row.get("info") or {} for row in rows]
    receipts = [info["dynamic_group_receipt"] for info in infos if "dynamic_group_receipt" in info]
    assert len(receipts) == 1, f"expected one receipt, got {len(receipts)}"
    receipt = receipts[0]
    assert receipt["complete_bundle_count"] == 1
    assert receipt["read_rollout_count"] == 2 and receipt["query_branch_count"] == 4
    assert receipt["reader_actor_loss_tokens"] == 0 and receipt["model_used"] and receipt["reader_frozen"]
    assert receipt["action_interface_version"] == "agemem.dynamic.action_interface.v3"
    events = []
    writes = 0
    for row, info in zip(rows, infos):
        assert info.get("phase") == "ingest"
        assert info.get("dynamic_action_interface") == receipt["action_interface_version"]
        parsed = parse_public_action(row.get("response_text") or "")
        assert info.get("dynamic_action_parse_code") == parsed.code
        joined = info.get("agemem_action_events", [])
        if parsed.code != "ok":
            assert not joined, "invalid response has an executable ActionEvent"
            continue
        assert info["dynamic_action_type"] == parsed.name
        assert len(joined) == 1, "valid public action missing its unique ActionEvent"
        event = joined[0]
        assert event["action_type"] == parsed.name
        assert event["action_id"] == info["dynamic_action_id"]
        assert event["policy_version"] == receipt["policy_version"]
        events.extend(joined)
        writes += int(parsed.name in {"ADD", "UPDATE"} and bool(info.get("dynamic_action_admitted")))
    ids = [event["action_id"] for event in events]
    assert ids and len(ids) == len(set(ids)), "missing/duplicate action_id"
    assert receipt["admitted_memory_write_count"] == writes
    assert writes, "no admitted memory writes"
    assert receipt["invalid_response_count"] == sum(
        parse_public_action(row.get("response_text") or "").code != "ok" for row in rows
    )
    assert len({info["dynamic_snapshot_id"] for info in infos}) == 2
    assert len({branch["query_branch_id"] for info in infos
                for branch in info["dynamic_query_branches"]}) == 4
    blob = json.dumps(infos)
    assert '"answer_text"' not in blob and '"retrieved_payloads"' not in blob
    return {"action_interface_smoke": "pass", "action_event_count": len(events),
            "admitted_memory_write_count": writes, "receipt": receipt,
            "learning_effectiveness_checked": False, "source_semantics_checked": False,
            "raw_token_logprob_tensors_checked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experience-file", type=Path, required=True)
    parser.add_argument("--verify-runtime", action="store_true", help="Require persisted V3 K2/m2 action smoke gate")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.experience_file.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    print(json.dumps(classify_rows(rows), ensure_ascii=False, indent=2))
    if args.verify_runtime:
        print(json.dumps(verify_runtime(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
