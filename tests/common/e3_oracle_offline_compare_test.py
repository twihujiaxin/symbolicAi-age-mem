"""End-to-end fixture test for the CPU-only real-action reward report."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path


try:
    from datasets import Dataset, DatasetDict, load_from_disk

    from AgeMem_code_agentscope.action_schema import ActionEvent
    from scripts.agemem_e3_oracle_offline_compare import build_report
    from trinity.common.action_event_contract import stable_action_id
    from trinity.common.m8b_preflight import _canonical_json_sha256
except ModuleNotFoundError:
    Dataset = None
    DatasetDict = None
    load_from_disk = None
    ActionEvent = None
    build_report = None
    stable_action_id = None
    _canonical_json_sha256 = None


@unittest.skipUnless(build_report is not None, "offline report dependencies unavailable")
class E3OracleOfflineCompareTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _action(*, rollout_id: str, call_id: str, content: str) -> dict:
        task_id = rollout_id.rsplit("/", 1)[0]
        action_id = stable_action_id(
            rollout_id=rollout_id,
            stage_id=1,
            timestep=0,
            assistant_turn_id=0,
            action_index_in_turn=0,
        )
        return ActionEvent(
            action_id=action_id,
            task_id=task_id,
            rollout_id=rollout_id,
            stage_id=1,
            timestep=0,
            assistant_turn_id=0,
            action_index_in_turn=0,
            source="llm",
            action_type="Add_memory",
            action_text="<tool_call>Add_memory</tool_call>",
            arguments={"content": content},
            result={
                "trace_call_id": call_id,
                "status": "ok",
                "output": {"memory_id": call_id, "outcome": "added"},
                "error": None,
            },
            response_token_ids=(1, 2),
            token_start=0,
            token_end=1,
            old_logprobs=(-0.1, -0.2),
            policy_version="model_version:0",
        ).model_dump(mode="json")

    def test_build_report_joins_real_actions_and_compares_all_three_arms(self):
        support = "Alice was born in Paris."
        source_row = {
            "id": "hotpot-1",
            "question": "Where was Alice born?",
            "answer": "Paris",
            "context": {
                "title": ["Alice"],
                "sentences": [[support, "Another sentence."]],
            },
            "supporting_facts": {"title": ["Alice"], "sent_id": [0]},
        }
        test_tmp_root = Path.cwd() / "tmp"
        test_tmp_root.mkdir(parents=True, exist_ok=True)
        root = test_tmp_root / f"e3-offline-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            dataset_path = root / "dataset"
            DatasetDict({"train": Dataset.from_list([source_row])}).save_to_disk(
                str(dataset_path)
            )
            stored_row = dict(load_from_disk(str(dataset_path))["train"][0])
            lock_path = root / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "signal_repeat_times": 2,
                        "fixed_train_rows": [
                            {
                                "hotpot_id": "hotpot-1",
                                "source_index": 0,
                                "content_sha256": _canonical_json_sha256(stored_row),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            experience_rows = []
            trace_rows = []
            turn_rows = []
            for run, (content, score) in enumerate(((support, 1.0), ("noise", 0.0))):
                execution_id = f"execution-{run}"
                rollout_id = f"batch/task/{run}"
                call_id = f"call-{run}"
                experience_rows.append(
                    {
                        "diagnostic_schema_version": "agemem.bench_experience_audit.v1",
                        "eid": {
                            "batch": "batch",
                            "task": "task",
                            "run": run,
                            "step": 0,
                        },
                        "info": {
                            "trace_execution_id": execution_id,
                            "task_score": score,
                            "found_answer": True,
                            "answer_exact_match": score,
                            "agemem_action_events": [
                                self._action(
                                    rollout_id=rollout_id,
                                    call_id=call_id,
                                    content=content,
                                )
                            ],
                        },
                    }
                )
                trace_rows.append(
                    {
                        "phase": "finish",
                        "call_id": call_id,
                        "execution_id": execution_id,
                        "tool_name": "Add_memory",
                    }
                )
                turn_rows.append(
                    {
                        "execution_id": execution_id,
                        "hotpot_id": "hotpot-1",
                        "task_id": "task",
                        "round": 0,
                        "repaired": False,
                        "task_score": score,
                    }
                )

            experience_path = root / "experiences.jsonl"
            trace_path = root / "traces.jsonl"
            turn_path = root / "turns.jsonl"
            self._write_jsonl(experience_path, experience_rows)
            self._write_jsonl(trace_path, trace_rows)
            self._write_jsonl(turn_path, turn_rows)
            (
                report,
                flat_credits,
                dfa_credits,
                semantic_audit_rows,
                positive_control_rows,
            ) = build_report(
                experience_path=experience_path,
                trace_path=trace_path,
                stage3_turn_path=turn_path,
                hotpotqa_path=dataset_path,
                lock_path=lock_path,
                seed=7,
                max_steps=48,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["counts"]["rollout_count"], 2)
        self.assertEqual(report["counts"]["action_event_count"], 2)
        self.assertEqual(len(flat_credits), 2)
        self.assertEqual(len(dfa_credits), 2)
        self.assertEqual(len(semantic_audit_rows), 2)
        self.assertEqual(len(positive_control_rows), 4)
        self.assertEqual(report["positive_controls"]["status"], "pass")
        self.assertEqual(report["positive_controls"]["failure_count"], 0)
        self.assertEqual(
            report["real_action_semantic_audit"]["oracle_positive_action_count"],
            1,
        )
        self.assertEqual(
            report["group_statistics"]["terminal_only"][
                "groups_with_nonzero_std"
            ],
            1,
        )
        self.assertGreater(
            report["rollouts"][0]["flat_oracle"]["logic_total"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
