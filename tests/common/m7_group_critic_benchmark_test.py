import shutil
import unittest
import uuid
from pathlib import Path

from pydantic import ValidationError

from AgeMem_code_agentscope.group_critic.metrics import (
    AcceptanceObservation,
    M7CriticUsageMetrics,
    score_acceptance,
    score_reward_error,
    score_strata,
    stability_digest,
    usage_from_texts,
)
from AgeMem_code_agentscope.group_critic.benchmark import (
    M7BenchmarkError,
    M7GroupCriticConfig,
    M7ProfileReport,
    _markdown,
    _resolve_repo_path,
    _profile_rows,
    _renumber_adversarial_rows,
    _validate_profile_credit_digests,
    _validate_source_lineage,
    build_baseline_report,
    build_group_inputs,
    write_m7_offline_report,
)
from AgeMem_code_agentscope.action_schema import load_migration_manifest
from AgeMem_code_agentscope.hotpotqa_benchmark import HotpotQADataAdapter, load_manifest
from AgeMem_code_agentscope.hotpotqa_benchmark.metrics import OracleBenchmarkReport
from AgeMem_code_agentscope.memory_extraction.benchmark import (
    M6ExtractionBenchmarkReport,
)
from AgeMem_code_agentscope.memory_extraction.false_reject_audit import (
    M6FalseRejectAuditReport,
)


ROOT = Path(__file__).resolve().parents[2]
FULLWIKI = ROOT.parent / "data" / "hotpot_qa" / "fullwiki"
HAS_CANONICAL_INPUTS = all(
    path.exists()
    for path in (
        FULLWIKI,
        ROOT / "artifacts/m5_hotpotqa_smoke/oracle_benchmark.json",
        ROOT / "artifacts/m6_extraction_benchmark/extraction_benchmark.json",
        ROOT / "artifacts/m6_extraction_benchmark/false_reject_audit.json",
        ROOT / "runs/m6_schema_v2/migration_manifest.json",
    )
)


class M7GroupCriticMetricsTest(unittest.TestCase):
    def setUp(self):
        self.rows = (
            AcceptanceObservation(
                task_id="task-a",
                rollout_id="task-a-gold",
                expected_accepted=True,
                predicted_accepted=True,
                question_type="bridge",
                action_count=7,
            ),
            AcceptanceObservation(
                task_id="task-a",
                rollout_id="task-a-wrong",
                expected_accepted=False,
                predicted_accepted=False,
                question_type="bridge",
                action_count=7,
            ),
            AcceptanceObservation(
                task_id="task-b",
                rollout_id="task-b-gold",
                expected_accepted=True,
                predicted_accepted=False,
                question_type="comparison",
                action_count=9,
            ),
            AcceptanceObservation(
                task_id="task-b",
                rollout_id="task-b-wrong",
                expected_accepted=False,
                predicted_accepted=True,
                question_type="comparison",
                action_count=8,
            ),
        )

    def test_acceptance_and_exact_strata(self):
        metrics = score_acceptance(self.rows)
        self.assertEqual(metrics.false_accept_numerator, 1)
        self.assertEqual(metrics.false_accept_denominator, 2)
        self.assertEqual(metrics.false_accept_rate, 0.5)
        self.assertEqual(metrics.false_reject_numerator, 1)
        self.assertEqual(metrics.false_reject_denominator, 2)
        self.assertEqual(metrics.false_reject_rate, 0.5)
        strata = score_strata(self.rows)
        self.assertEqual(
            {(item.dimension, item.value) for item in strata},
            {
                ("question_type", "bridge"),
                ("question_type", "comparison"),
                ("action_count", "7"),
                ("action_count", "8"),
                ("action_count", "9"),
            },
        )

    def test_action_reward_join_and_duplicate_rollouts_fail_closed(self):
        reward = score_reward_error({"a0": 0.25, "a1": 1.25}, {"a0": 0.0, "a1": 1.0})
        self.assertEqual(reward.count, 2)
        self.assertEqual(reward.absolute_error_total, 0.5)
        self.assertEqual(reward.bias, -0.25)
        with self.assertRaisesRegex(ValueError, "same non-empty action_id"):
            score_reward_error({"a0": 0.0}, {"a1": 0.0})
        with self.assertRaisesRegex(ValueError, "unique rollout_id"):
            score_acceptance((self.rows[0], self.rows[0]))

    def test_usage_is_explicitly_heuristic_and_provider_free(self):
        usage = usage_from_texts(
            cold_calls=1,
            cache_hits=5,
            cache_misses=1,
            inputs=("abc", "中"),
            outputs=("{}",),
        )
        self.assertEqual(usage.input_characters, 4)
        self.assertEqual(usage.input_utf8_bytes, 6)
        self.assertEqual(usage.heuristic_input_tokens, 2)
        self.assertIsNone(usage.provider_input_tokens)
        self.assertIsNone(usage.provider_output_tokens)
        self.assertIsNone(usage.provider_cost)
        self.assertEqual(usage.real_llm_call_count, 0)
        with self.assertRaises(ValidationError):
            M7CriticUsageMetrics(
                cold_calls=0,
                cache_hits=0,
                cache_misses=0,
                input_characters=0,
                output_characters=0,
                input_utf8_bytes=4,
                output_utf8_bytes=0,
                heuristic_input_tokens=0,
                heuristic_output_tokens=0,
            )

    def test_digest_is_byte_stable(self):
        value = [item.model_dump(mode="json") for item in self.rows]
        self.assertEqual(stability_digest(value), stability_digest(value))
        self.assertEqual(len(stability_digest(value)), 64)

    def test_profile_metric_denominators_fail_closed(self):
        baseline = score_acceptance(
            AcceptanceObservation(
                task_id=f"task-{index}",
                rollout_id=f"rollout-{index}",
                expected_accepted=index < 10,
                predicted_accepted=index < 10,
                question_type="bridge",
                action_count=7,
            )
            for index in range(30)
        )
        rewards = score_reward_error(
            {f"a{index}": 0.0 for index in range(224)},
            {f"a{index}": 0.0 for index in range(224)},
        )
        profile = M7ProfileReport(
            name="oracle",
            hand_dfa_acceptance=baseline,
            hand_dfa_reward_error=rewards,
            critic_dfa_acceptance=baseline,
            critic_hand_reward_error=rewards,
            critic_hand_acceptance_agreement_count=30,
            critic_hand_acceptance_agreement_rate=1.0,
            strata=(),
            action_credit_digest="a" * 64,
            failure_count=0,
        )
        payload = profile.model_dump(mode="python")
        acceptance = dict(payload["hand_dfa_acceptance"])
        acceptance.update(count=29, true_negative=19, false_accept_denominator=19)
        payload["hand_dfa_acceptance"] = acceptance
        with self.assertRaisesRegex(ValidationError, "every profile rollout"):
            M7ProfileReport.model_validate(payload)

        payload = profile.model_dump(mode="python")
        reward = dict(payload["hand_dfa_reward_error"])
        reward["count"] = 223
        payload["hand_dfa_reward_error"] = reward
        with self.assertRaisesRegex(ValidationError, "every profile action"):
            M7ProfileReport.model_validate(payload)

    def test_configured_repository_path_escape_fails_closed(self):
        with self.assertRaisesRegex(M7BenchmarkError, "escapes repository"):
            _resolve_repo_path(ROOT, "../outside.json")


@unittest.skipUnless(HAS_CANONICAL_INPUTS, "canonical M5/M6 inputs are required")
class M7GroupCriticBaselineIntegrationTest(unittest.TestCase):
    def test_adversarial_rows_are_replay_valid_tool_level_actions(self):
        config = M7GroupCriticConfig.from_json(ROOT / "configs/m7_group_critic.json")
        manifest = load_migration_manifest(
            ROOT / config.migration_root / "migration_manifest.json"
        )
        item = next(row for row in manifest.files if row.policy == "gold")
        steps, credits = _profile_rows(ROOT, config, item, "oracle")
        pairs = tuple(zip(steps, credits))
        add_index = next(
            index
            for index, (_, credit) in enumerate(pairs)
            if "stored_supporting_fact" in credit.atomic_propositions
        )
        retrieve_index = next(
            index
            for index, (_, credit) in enumerate(pairs)
            if "supporting_coverage_complete" in credit.atomic_propositions
            and "retrieved_supporting_fact" in credit.atomic_propositions
        )

        for index, suffix, repeat_count in (
            (add_index, "duplicate-add", 1),
            (retrieve_index, "loop-retrieve", 2),
        ):
            with self.subTest(suffix=suffix):
                candidate_steps, candidate_credits, injected_ids = (
                    _renumber_adversarial_rows(
                        pairs,
                        inject_after=index,
                        suffix=suffix,
                        repeat_count=repeat_count,
                    )
                )
                self.assertEqual(len(injected_ids), repeat_count)
                self.assertTrue(
                    all(
                        left.memory_after == right.memory_before
                        for left, right in zip(candidate_steps, candidate_steps[1:])
                    )
                )
                by_id = {
                    step.actions[0].action_id: (step, credit)
                    for step, credit in zip(candidate_steps, candidate_credits)
                }
                for action_id in injected_ids:
                    step, credit = by_id[action_id]
                    self.assertEqual(step.actions[0].result["tool_call_id"], action_id)
                    self.assertEqual(step.memory_before, step.memory_after)
                    if suffix == "duplicate-add":
                        self.assertEqual(credit.atomic_propositions, ())
                        self.assertFalse(step.actions[0].result["metadata"]["success"])
                    else:
                        self.assertIn(
                            "retrieved_supporting_fact", credit.atomic_propositions
                        )

    def test_k3_groups_hide_answers_and_preserve_action_provenance(self):
        groups = build_group_inputs(repository=ROOT)
        self.assertEqual(len(groups), 30)
        for profile in ("oracle", "human_backed_mock", "controlled_error"):
            selected = [item for item in groups if item.ap_profile == profile]
            self.assertEqual(len(selected), 10)
            self.assertTrue(all(len(item.rollouts) == 3 for item in selected))
        m5 = __import__("json").loads(
            (ROOT / "artifacts/m5_hotpotqa_smoke/oracle_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = load_manifest(ROOT / "data/splits/hotpotqa_smoke_manifest.json")
        adapter = HotpotQADataAdapter(FULLWIKI)
        expected = {
            f"hotpot-{selection.hotpot_id}": adapter.row(
                selection.source_split, selection.source_index
            )
            for selection in manifest.selections
        }
        # Task descriptions are adapter questions, never answers/support labels.
        for group in groups:
            self.assertEqual(group.task_description, expected[group.task_id].question)
            self.assertNotEqual(group.task_description, expected[group.task_id].answer)
            self.assertNotIn("supporting_fact_ids", group.task_description)
            self.assertNotIn("oracle_labels", group.task_description)
            action_ids = [
                action.evidence.action_id
                for rollout in group.rollouts
                for action in rollout.actions
            ]
            self.assertEqual(len(action_ids), len(set(action_ids)))
        self.assertEqual(
            {item.task_id for item in groups}, {x["task_id"] for x in m5["records"]}
        )

    def test_canonical_hand_baseline_and_stability_gate(self):
        first = build_baseline_report(repository=ROOT)
        second = build_baseline_report(repository=ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first.to_json(), second.to_json())
        profiles = {item.name: item for item in first.profiles}
        self.assertEqual(
            (
                profiles["oracle"].hand_dfa_acceptance.false_accept_numerator,
                profiles["oracle"].hand_dfa_acceptance.false_reject_numerator,
            ),
            (0, 0),
        )
        self.assertEqual(
            (
                profiles[
                    "human_backed_mock"
                ].hand_dfa_acceptance.false_accept_numerator,
                profiles[
                    "human_backed_mock"
                ].hand_dfa_acceptance.false_reject_numerator,
            ),
            (0, 0),
        )
        self.assertEqual(
            (
                profiles["controlled_error"].hand_dfa_acceptance.false_accept_numerator,
                profiles["controlled_error"].hand_dfa_acceptance.false_reject_numerator,
            ),
            (0, 5),
        )
        self.assertTrue(first.stability.stable)
        self.assertEqual(first.stability.repeat_checks, 150)
        self.assertEqual(first.stability.permutation_checks, 180)
        self.assertEqual(first.hand_replay_count, 90)
        self.assertEqual(first.hand_replay_exact_match_count, 90)
        self.assertEqual(first.reward_farming.scenario_count, 20)
        self.assertTrue(first.reward_farming.passed)
        self.assertEqual(first.reward_farming.automaton_source, "hand_dfa")
        self.assertEqual(first.reward_farming.dfa_spec_id, first.hand_dfa_spec_id)
        duplicate_records = tuple(
            item
            for item in first.reward_farming_records
            if ":duplicate-add:" in item.injected_action_ids[0]
        )
        loop_records = tuple(
            item
            for item in first.reward_farming_records
            if ":loop-retrieve:" in item.injected_action_ids[0]
        )
        self.assertEqual((len(duplicate_records), len(loop_records)), (10, 10))
        self.assertTrue(
            all(len(item.injected_action_ids) == 1 for item in duplicate_records)
        )
        self.assertTrue(
            all(len(item.injected_action_ids) == 2 for item in loop_records)
        )
        self.assertTrue(
            all(
                item.dfa_spec_id == first.hand_dfa_spec_id
                for item in first.reward_farming_records
            )
        )
        self.assertEqual(first.validator_valid_count, 25)
        self.assertEqual(first.validator_invalid_count, 25)
        self.assertEqual(first.mock_critic_unavailable_count, 5)
        self.assertEqual(first.explicit_fallback_count, 30)
        self.assertEqual(first.usage.cache_misses, 30)
        self.assertEqual(first.usage.cache_hits, 30)
        self.assertEqual(first.usage.cold_calls, 360)
        self.assertEqual(first.silent_adoption_count, 0)
        self.assertEqual(first.evidence_coverage, 1.0)
        self.assertEqual(first.real_llm_call_count, 0)
        self.assertIsNone(first.provider_cost)
        self.assertEqual(first.interference.stage_1_interference_count, 6)
        self.assertEqual(first.interference.stage_2_interference_count, 3)
        markdown = _markdown(first)
        self.assertIn("Critic+fallback FA", markdown)
        self.assertIn("Critic + explicit fallback pipeline", markdown)
        self.assertIn("Mock calls/cache hits/cache misses: `360/30/30`", markdown)
        self.assertIn("not provider billing", markdown)

    def test_credit_digest_tamper_fails_closed_without_touching_sources(self):
        cfg = M7GroupCriticConfig.from_json(ROOT / "configs/m7_group_critic.json")
        m6 = M6ExtractionBenchmarkReport.model_validate_json(
            (ROOT / cfg.m6_report_path).read_text(encoding="utf-8")
        )
        manifest = load_migration_manifest(
            ROOT / cfg.migration_root / "migration_manifest.json"
        )
        temporary = ROOT / "runs" / f"m7_digest_test_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, temporary, True)
        migration = temporary / "migration"
        runtime = temporary / "runtime"
        for item in manifest.files:
            oracle_source = ROOT / cfg.migration_root / item.target_credit_path
            oracle_target = migration / item.target_credit_path
            oracle_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(oracle_source, oracle_target)
            for profile in ("human_backed_mock", "controlled_error"):
                source = (
                    ROOT
                    / cfg.extraction_runtime_root
                    / profile
                    / item.target_credit_path
                )
                target = runtime / profile / item.target_credit_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        copied_cfg = cfg.model_copy(
            update={
                "migration_root": migration.relative_to(ROOT).as_posix(),
                "extraction_runtime_root": runtime.relative_to(ROOT).as_posix(),
            }
        )
        self.assertEqual(
            _validate_profile_credit_digests(ROOT, copied_cfg, m6, manifest)[
                "human_backed_mock"
            ],
            next(
                item.action_credit_digest
                for item in m6.profiles
                if item.name == "human_backed_mock"
            ),
        )
        tampered = runtime / "human_backed_mock" / manifest.files[0].target_credit_path
        lines = tampered.read_text(encoding="utf-8").splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        tampered.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(
            M7BenchmarkError, "human_backed_mock action-credit digest mismatch"
        ):
            _validate_profile_credit_digests(ROOT, copied_cfg, m6, manifest)

    def test_false_reject_audit_must_bind_to_loaded_source_chain(self):
        cfg = M7GroupCriticConfig.from_json(ROOT / "configs/m7_group_critic.json")
        m5 = OracleBenchmarkReport.model_validate_json(
            (ROOT / cfg.m5_report_path).read_text(encoding="utf-8")
        )
        m6 = M6ExtractionBenchmarkReport.model_validate_json(
            (ROOT / cfg.m6_report_path).read_text(encoding="utf-8")
        )
        manifest = load_migration_manifest(
            ROOT / cfg.migration_root / "migration_manifest.json"
        )
        audit = M6FalseRejectAuditReport.model_validate_json(
            (ROOT / cfg.m6_false_reject_audit_path).read_text(encoding="utf-8")
        )
        _validate_source_lineage(ROOT, cfg, m5, m6, manifest, audit)

        internally_consistent_but_unbound = audit.model_copy(
            update={
                "config_digest": "0" * 64,
                "m6_report_config_digest": "0" * 64,
            }
        )
        with self.assertRaisesRegex(
            M7BenchmarkError,
            "False Reject audit lineage mismatch",
        ):
            _validate_source_lineage(
                ROOT,
                cfg,
                m5,
                m6,
                manifest,
                internally_consistent_but_unbound,
            )

    def test_report_outputs_are_byte_stable_and_complete(self):
        temporary = ROOT / "runs" / f"m7_report_test_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, temporary, True)
        first = write_m7_offline_report(
            repository=ROOT,
            output_root=temporary / "first",
            docs_path=temporary / "first.md",
        )
        second = write_m7_offline_report(
            repository=ROOT,
            output_root=temporary / "second",
            docs_path=temporary / "second.md",
        )
        self.assertEqual(first, second)
        for name in (
            "offline_validation.json",
            "offline_validation.md",
            "validation_failures.jsonl",
            "reward_farming.jsonl",
        ):
            self.assertEqual(
                (temporary / "first" / name).read_bytes(),
                (temporary / "second" / name).read_bytes(),
            )
        self.assertEqual(
            len(
                (temporary / "first/validation_failures.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            5,
        )
        self.assertEqual(
            len(
                (temporary / "first/reward_farming.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            20,
        )


if __name__ == "__main__":
    unittest.main()
