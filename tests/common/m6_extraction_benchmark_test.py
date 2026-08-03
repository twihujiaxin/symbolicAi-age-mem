import hashlib
import shutil
import unittest
import uuid
from pathlib import Path

from AgeMem_code_agentscope.action_schema import (
    ActionCreditRecord,
    TrajectoryStepV2,
    load_migration_manifest,
)
from AgeMem_code_agentscope.hotpotqa_benchmark import HotpotQADataAdapter
from AgeMem_code_agentscope.memory_extraction.benchmark import (
    M6BenchmarkConfig,
    M6BenchmarkError,
    M6ExtractionBenchmark,
    default_benchmark_config_path,
    run_default_m6_benchmark,
)
from AgeMem_code_agentscope.memory_extraction.models import APRecord


ROOT = Path(__file__).resolve().parents[2]
FULLWIKI = ROOT.parent / "data" / "hotpot_qa" / "fullwiki"
M5_REPORT = ROOT / "artifacts" / "m5_hotpotqa_smoke" / "oracle_benchmark.json"
M5_RUNTIME = ROOT / "runs" / "m5_hotpotqa_smoke"
M6_MIGRATION = ROOT / "runs" / "m6_schema_v2"
HAS_CANONICAL_INPUTS = all(
    path.exists()
    for path in (
        FULLWIKI,
        M5_REPORT,
        M5_RUNTIME,
        M6_MIGRATION / "migration_manifest.json",
    )
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _workspace():
    path = ROOT / "runs" / f"m6_benchmark_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


@unittest.skipUnless(
    HAS_CANONICAL_INPUTS,
    "canonical M5 runtime, M6 migration, and local fullwiki data are required",
)
class M6ExtractionBenchmarkIntegrationTest(unittest.TestCase):
    def test_canonical_benchmark_is_joined_grounded_and_byte_stable(self):
        manifest = load_migration_manifest(M6_MIGRATION / "migration_manifest.json")
        protected_paths = [M5_REPORT, M6_MIGRATION / "migration_manifest.json"]
        for item in manifest.files:
            protected_paths.extend(
                (
                    M5_RUNTIME / item.source_trajectory_path,
                    M5_RUNTIME / item.source_reward_path,
                    M6_MIGRATION / item.target_trajectory_path,
                    M6_MIGRATION / item.target_credit_path,
                )
            )
        before_hashes = {path: _sha256(path) for path in protected_paths}

        temporary_root = _workspace()
        self.addCleanup(shutil.rmtree, temporary_root, True)
        try:
            run_roots = (temporary_root / "first", temporary_root / "second")
            artifacts = []
            for run_root in run_roots:
                artifacts.append(
                    run_default_m6_benchmark(
                        data_path=FULLWIKI,
                        output_root=run_root / "artifacts",
                        docs_path=run_root / "docs.md",
                        runtime_output=run_root / "runtime",
                    )
                )

            self.assertEqual(artifacts[0].report, artifacts[1].report)
            self.assertEqual(_tree_bytes(run_roots[0]), _tree_bytes(run_roots[1]))
            report = artifacts[0].report
            self.assertEqual(
                report.digest,
                "e803f7752dc9e7357284887cf7716273bbd5396f62db1fc438d7cad95a2f9f92",
            )
            self.assertEqual(report.real_llm_call_count, 0)
            self.assertEqual(report.llm_evaluation, "not_run")
            self.assertEqual(report.canonical_rollout_count, 30)
            self.assertEqual(report.canonical_action_count, 224)
            self.assertEqual(
                (
                    report.annotation_validation.task_count,
                    report.annotation_validation.record_count,
                    report.annotation_validation.triple_count,
                    report.annotation_validation.relevant_fact_count,
                    report.annotation_validation.irrelevant_fact_count,
                ),
                (10, 34, 37, 24, 10),
            )

            profiles = {item.name: item for item in report.profiles}
            human = profiles["human_backed_mock"]
            controlled = profiles["controlled_error"]
            self.assertEqual(human.triple_metrics.micro.f1, 1.0)
            self.assertAlmostEqual(human.ap_metrics.micro.f1, 0.9760765550239235)
            self.assertEqual(human.ap_metrics.micro.false_positive, 0)
            self.assertEqual(human.ap_metrics.micro.false_negative, 10)
            self.assertEqual(human.acceptance.false_accept_rate, 0.0)
            self.assertEqual(human.acceptance.false_reject_rate, 0.0)
            self.assertEqual(human.reward_propagation.action_total.mae, 0.0)
            self.assertEqual(
                human.reward_propagation.trajectory_absolute_error_total, 0.0
            )
            self.assertEqual((human.accepted_rollouts, human.failure_count), (10, 10))

            self.assertAlmostEqual(
                controlled.triple_metrics.micro.f1, 0.8695652173913043
            )
            self.assertAlmostEqual(controlled.ap_metrics.micro.f1, 0.8369565217391304)
            self.assertEqual(controlled.ap_metrics.micro.false_positive, 0)
            self.assertEqual(controlled.ap_metrics.micro.false_negative, 60)
            self.assertEqual(controlled.acceptance.false_accept_rate, 0.0)
            self.assertEqual(controlled.acceptance.false_reject_rate, 0.5)
            self.assertAlmostEqual(
                controlled.reward_propagation.action_total.mae,
                0.056919642857142856,
            )
            self.assertEqual(
                controlled.reward_propagation.trajectory_absolute_error_total,
                7.25,
            )
            self.assertEqual(
                (controlled.accepted_rollouts, controlled.failure_count), (5, 20)
            )

            for profile in report.profiles:
                self.assertEqual(profile.provenance_integrity_rate, 1.0)
                credit_count = 0
                ap_count = 0
                for item in manifest.files:
                    relative = Path(item.target_credit_path)
                    credit_path = run_roots[0] / "runtime" / profile.name / relative
                    ap_path = (
                        run_roots[0]
                        / "runtime"
                        / profile.name
                        / "ap_records"
                        / Path(*relative.parts[1:])
                    )
                    trajectory_path = M6_MIGRATION / item.target_trajectory_path
                    steps = tuple(
                        TrajectoryStepV2.model_validate_json(line)
                        for line in trajectory_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    )
                    credits = tuple(
                        ActionCreditRecord.model_validate_json(line)
                        for line in credit_path.read_text(encoding="utf-8").splitlines()
                    )
                    aps = tuple(
                        APRecord.model_validate_json(line)
                        for line in ap_path.read_text(encoding="utf-8").splitlines()
                    )
                    ap_by_id = {ap.ap_id: ap for ap in aps}
                    self.assertEqual(len(ap_by_id), len(aps))
                    self.assertEqual(
                        tuple(step.actions[0].action_id for step in steps),
                        tuple(credit.action_id for credit in credits),
                    )
                    for credit in credits:
                        self.assertEqual(
                            set(credit.atomic_proposition_evidence),
                            set(credit.atomic_propositions),
                        )
                        for (
                            proposition,
                            ap_ids,
                        ) in credit.atomic_proposition_evidence.items():
                            self.assertEqual(len(ap_ids), 1)
                            ap = ap_by_id[ap_ids[0]]
                            self.assertEqual(ap.action_id, credit.action_id)
                            self.assertEqual(ap.proposition, proposition)
                            self.assertIn(credit.action_id, ap.evidence_action_ids)
                    credit_count += len(credits)
                    ap_count += len(aps)
                self.assertEqual(credit_count, 224)
                self.assertEqual(ap_count, profile.provenance_total_count)

        finally:
            self.assertEqual(
                before_hashes,
                {path: _sha256(path) for path in protected_paths},
            )

    def test_manifest_hash_tamper_fails_closed(self):
        temporary_root = _workspace()
        self.addCleanup(shutil.rmtree, temporary_root, True)
        copied_migration = temporary_root / "migration"
        shutil.copytree(M6_MIGRATION, copied_migration)
        manifest = load_migration_manifest(copied_migration / "migration_manifest.json")
        target = copied_migration / manifest.files[0].target_trajectory_path
        target.write_bytes(target.read_bytes() + b"\n")
        config = M6BenchmarkConfig.from_json(default_benchmark_config_path())
        config = config.model_copy(
            update={"m6_migration_root": copied_migration.relative_to(ROOT).as_posix()}
        )
        benchmark = M6ExtractionBenchmark(
            config=config,
            adapter=HotpotQADataAdapter(FULLWIKI),
            repository=ROOT,
        )
        with self.assertRaisesRegex(M6BenchmarkError, "manifest hash mismatch"):
            benchmark.run(output_root=temporary_root / "out")


if __name__ == "__main__":
    unittest.main()
