import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from AgeMem_code_agentscope.memory_extraction.false_reject_audit import (
    FalseRejectAuditError,
    M6FalseRejectAuditReport,
    _read_aps,
    _read_credits,
    _read_steps,
    _replay_credit_dfa,
    _validate_action_stream,
    _validate_ap_grounding,
    build_m6_false_reject_audit,
    write_m6_false_reject_audit,
)
from AgeMem_code_agentscope.memory_extraction.benchmark import M6BenchmarkConfig
from AgeMem_code_agentscope.memory_extraction.models import canonical_digest
from AgeMem_code_agentscope.memory_oracle import RewardConfig
from AgeMem_code_agentscope.action_schema import load_migration_manifest


ROOT = Path(__file__).resolve().parents[2]
HAS_INPUTS = all(
    path.exists()
    for path in (
        ROOT / "artifacts/m6_extraction_benchmark/extraction_benchmark.json",
        ROOT / "runs/m6_schema_v2/migration_manifest.json",
        ROOT / "runs/m6_extraction_benchmark/human_backed_mock/action_credits",
        ROOT / "runs/m6_extraction_benchmark/controlled_error/action_credits",
    )
)


@unittest.skipUnless(HAS_INPUTS, "canonical M6 runtime artifacts are required")
class M6FalseRejectAuditTest(unittest.TestCase):
    def test_five_false_rejects_are_fully_explained(self):
        first = build_m6_false_reject_audit(repository=ROOT)
        second = build_m6_false_reject_audit(repository=ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertTrue(first.m7_entry_gate_passed)
        self.assertEqual(first.real_llm_call_count, 0)
        self.assertEqual(len(first.cases), 5)
        self.assertEqual(
            {case.injection_fact_id for case in first.cases},
            set(first.relevant_drop_fact_ids),
        )
        self.assertEqual(len(first.irrelevant_corrupt_fact_ids), 2)
        self.assertEqual(
            (
                first.unexplained_count,
                first.state_tracker_error_count,
                first.ap_grounding_error_count,
                first.action_alignment_error_count,
                first.dfa_implementation_error_count,
            ),
            (0, 0, 0, 0, 0),
        )
        expected = {
            "hotpot-5a74b19355429916b01641dd": (
                "Sunye",
                "birth_date",
                "1989-08-12",
            ),
            "hotpot-5a83df2655429933447460a1": (
                "Grant O'Riley",
                "played_for",
                "Fitzroy Football Club",
            ),
            "hotpot-5a85aaee5542991dd0999e84": (
                "Junko Noda",
                "known_for",
                "Love Hina",
            ),
            "hotpot-5a8ac7d055429950cd6afb8f": (
                "Veitchia",
                "plant_family",
                "Arecaceae",
            ),
            "hotpot-5abecbed5542997719eab5c5": (
                "Coming of Age",
                "cancelled_with",
                "Ideal",
            ),
        }
        for case in first.cases:
            self.assertTrue(case.rollout_id.endswith("-gold"))
            self.assertEqual(case.missing_triples, (expected[case.task_id],))
            self.assertEqual(case.first_divergent_timestep, 0)
            self.assertTrue(case.first_divergent_action_id.endswith(":call:0"))
            self.assertEqual(
                case.propagation[0].oracle_only_aps,
                ("stored_supporting_fact",),
            )
            self.assertTrue(case.missing_state_fact_ids)
            self.assertEqual(case.missing_state_fact_ids, case.expected_state_fact_ids)
            self.assertTrue(case.grounding_evidence_ap_ids)
            self.assertEqual(
                case.dfa_checked_action_count, 2 * len(case.oracle_ap_trace)
            )
            self.assertTrue(case.state_tracker_correct)
            self.assertTrue(case.ap_grounding_correct)
            self.assertTrue(case.action_id_alignment_correct)
            self.assertTrue(case.dfa_implementation_correct)
            self.assertFalse(case.dfa_definition_too_strict)
            self.assertEqual(case.oracle_final_status, "accepted")
            self.assertEqual(case.extracted_final_status, "rejected")
            self.assertAlmostEqual(case.oracle_total_reward, 2.0)
            self.assertAlmostEqual(case.extracted_total_reward, 1.25)
            self.assertAlmostEqual(case.total_reward_error, -0.75)
            self.assertEqual(case.classification, "expected_extractor_omission")
            self.assertTrue(case.causal_chain_complete)

    def test_report_output_is_byte_stable_and_schema_is_strict(self):
        temporary = ROOT / "runs" / f"m6_fr_audit_test_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, temporary, True)
        first = write_m6_false_reject_audit(
            repository=ROOT,
            output_root=temporary / "first",
            docs_path=temporary / "first.md",
        )
        second = write_m6_false_reject_audit(
            repository=ROOT,
            output_root=temporary / "second",
            docs_path=temporary / "second.md",
        )
        self.assertEqual(first, second)
        self.assertEqual(
            (temporary / "first/false_reject_audit.json").read_bytes(),
            (temporary / "second/false_reject_audit.json").read_bytes(),
        )
        self.assertEqual(
            (temporary / "first/false_reject_audit.md").read_bytes(),
            (temporary / "second/false_reject_audit.md").read_bytes(),
        )
        markdown = (temporary / "first/false_reject_audit.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Audit schema: `agemem.m6_false_reject_audit.v2`", markdown)
        self.assertIn("DFA/reward action checks: `74`", markdown)
        payload = first.model_dump(mode="python")
        payload["unexpected"] = True
        with self.assertRaises(ValidationError):
            M6FalseRejectAuditReport.model_validate(payload)

    def test_cross_digest_tamper_fails_closed(self):
        config = M6BenchmarkConfig.from_json(
            ROOT / "configs/m6_extraction_benchmark.json"
        )
        tampered = config.model_copy(update={"seed": config.seed + 1})
        with patch(
            "AgeMem_code_agentscope.memory_extraction.false_reject_audit."
            "M6BenchmarkConfig.from_json",
            return_value=tampered,
        ):
            with self.assertRaisesRegex(
                FalseRejectAuditError, "config_digest does not match"
            ):
                build_m6_false_reject_audit(repository=ROOT)

        report = build_m6_false_reject_audit(repository=ROOT)
        payload = report.model_dump(mode="python")
        payload["m6_report_config_digest"] = "0" * 64
        payload["digest"] = canonical_digest(
            {key: value for key, value in payload.items() if key != "digest"}
        )
        with self.assertRaisesRegex(ValidationError, "M7 entry gate"):
            M6FalseRejectAuditReport.model_validate(payload)

    def test_action_ap_and_dfa_semantic_tampering_fails_closed(self):
        manifest = load_migration_manifest(
            ROOT / "runs/m6_schema_v2/migration_manifest.json"
        )
        item = manifest.files[0]
        steps = _read_steps(ROOT / "runs/m6_schema_v2" / item.target_trajectory_path)
        credit_path = (
            ROOT
            / "runs/m6_extraction_benchmark/human_backed_mock"
            / item.target_credit_path
        )
        credits = _read_credits(credit_path)
        first_action = (
            steps[0]
            .actions[0]
            .model_copy(
                update={"assistant_turn_id": steps[-1].actions[0].assistant_turn_id + 1}
            )
        )
        tampered_steps = (
            steps[0].model_copy(update={"actions": (first_action,)}),
            *steps[1:],
        )
        with self.assertRaisesRegex(FalseRejectAuditError, "coordinates"):
            _validate_action_stream(tampered_steps, credits)

        relative = Path(item.target_credit_path)
        ap_path = (
            ROOT
            / "runs/m6_extraction_benchmark/human_backed_mock/ap_records"
            / Path(*relative.parts[1:])
        )
        aps = _read_aps(ap_path)
        tampered_evidence = dict(credits[0].atomic_proposition_evidence)
        tampered_evidence[next(iter(tampered_evidence))] = ("0" * 64,)
        tampered_credits = (
            credits[0].model_copy(
                update={"atomic_proposition_evidence": tampered_evidence}
            ),
            *credits[1:],
        )
        with self.assertRaisesRegex(FalseRejectAuditError, "exactly equal"):
            _validate_ap_grounding(aps, tampered_credits)

        reward = credits[0].reward_breakdown.model_copy(
            update={"milestone": credits[0].reward_breakdown.milestone + 0.125}
        )
        tampered_dfa_credits = (
            credits[0].model_copy(update={"reward_breakdown": reward}),
            *credits[1:],
        )
        profile = RewardConfig.from_json(ROOT / "configs/m4_reward.json").profile(
            "terminal_dfa"
        )
        with self.assertRaisesRegex(FalseRejectAuditError, "reward recomputation"):
            _replay_credit_dfa(
                steps,
                tampered_dfa_credits,
                seed=20260802,
                reward_profile=profile,
            )


if __name__ == "__main__":
    unittest.main()
