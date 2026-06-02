import unittest
import json
import tempfile
from pathlib import Path

from bayesian_agent.benchmarks.evolution import (
    build_benchmark_posterior_context,
    build_benchmark_skill_context,
    classify_failure,
    save_skill_evolution_snapshot,
)
from bayesian_agent.benchmarks.sop_lifelong import build_lifelong_prompt
from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.core.registry import BayesianSkillRegistry


class BenchmarkEvolutionTests(unittest.TestCase):
    def test_single_failure_mode_is_audit_only_not_active_patch(self):
        registry = BayesianSkillRegistry.in_memory()
        registry.record(
            TrajectoryEvidence(
                task_id="sop_02",
                skill_id="benchmark/sop_bench",
                context="sop_bench",
                outcome="failure",
                failure_mode="left_expected_output_blank",
                total_tokens=100,
            )
        )

        context = build_benchmark_skill_context("sop_bench", registry)

        self.assertIn("Benchmark SOP Guardrails", context)
        self.assertIn("rows[row_index - 1]", context)
        self.assertIn("raw category string", context)
        self.assertNotIn("Bayesian Failure-Mode Patches", context)
        self.assertNotIn("failure_mode=left_expected_output_blank", context)
        self.assertNotIn("Bayesian Skill Context", context)
        self.assertNotIn("Bayesian Posterior Audit", context)
        self.assertNotIn("posterior_success=", context)
        self.assertNotIn("alpha=", context)

    def test_failure_mode_patch_rules_are_rendered_from_evidence(self):
        registry = BayesianSkillRegistry.in_memory()
        for idx in range(2):
            registry.record(
                TrajectoryEvidence(
                    task_id=f"sop_{idx}",
                    skill_id="benchmark/sop_bench",
                    context="sop_bench",
                    outcome="failure",
                    failure_mode="left_expected_output_blank",
                    total_tokens=100,
                )
            )

        context = build_benchmark_skill_context("sop_bench", registry)

        self.assertIn("Bayesian Failure-Mode Patches", context)
        self.assertIn("failure_mode=left_expected_output_blank observed=2", context)
        self.assertIn("confirm the target row's `expected_output` is non-empty", context)

    def test_benchmark_skill_context_is_strictly_benchmark_scoped(self):
        registry = BayesianSkillRegistry.in_memory()
        registry.record(
            TrajectoryEvidence(
                task_id="sop_01",
                skill_id="benchmark/sop_bench",
                context="sop_bench",
                outcome="success",
                total_tokens=100,
            )
        )
        registry.record(
            TrajectoryEvidence(
                task_id="lifelong_0",
                skill_id="benchmark/lifelong_agentbench",
                context="lifelong_agentbench",
                outcome="success",
                total_tokens=100,
            )
        )

        context = build_benchmark_skill_context("sop_bench", registry)

        self.assertIn("Benchmark SOP Guardrails: sop_bench", context)
        self.assertNotIn("benchmark/lifelong_agentbench", context)
        self.assertNotIn("Lifelong", context)

        posterior_context = build_benchmark_posterior_context("sop_bench", registry)

        self.assertIn("benchmark/sop_bench", posterior_context)
        self.assertNotIn("benchmark/lifelong_agentbench", posterior_context)

    def test_classify_lifelong_transcript_failure(self):
        failure = classify_failure("lifelong_agentbench", {"success": False, "got_sql": "Turn 1 🛠️ tool output"})

        self.assertEqual(failure, "wrote_transcript_instead_of_sql_after_workspace_confusion")

    def test_classify_sop_wrong_decision_as_unverified_target_row(self):
        failure = classify_failure("sop_bench", {"success": False, "got": "fulfill_immediately", "expected": "reject"})

        self.assertEqual(failure, "computed_decision_for_wrong_or_unverified_target_row")

    def test_lifelong_prompt_forbids_unrequested_id_columns_on_insert(self):
        entry = {"instruction": "Insert a new payment record.", "table_info": {"name": "payments"}}

        prompt = build_lifelong_prompt(entry, "/tmp/task")

        self.assertIn("Do not include id or primary-key columns", prompt)

    def test_skill_evolution_snapshots_persist_before_and_after_context(self):
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            registry = BayesianSkillRegistry(out_root / "beliefs.json")
            before_context = build_benchmark_skill_context("sop_bench", registry)

            before = save_skill_evolution_snapshot(
                out_root=out_root,
                benchmark="sop_bench",
                task_id="sop_01",
                stage="before",
                registry=registry,
                context=before_context,
            )

            self.assertIn("skill_context_before.md", before["context_path"])
            self.assertIn("posterior_context_before.md", before["posterior_context_path"])
            self.assertTrue((out_root / "skill_evolution" / "sop_bench" / "sop_01" / "skill_context_before.md").exists())
            self.assertTrue((out_root / "skill_evolution" / "sop_bench" / "sop_01" / "posterior_context_before.md").exists())
            self.assertIn("Benchmark SOP Guardrails", (out_root / "skill_evolution" / "sop_bench" / "sop_01" / "skill_context_before.md").read_text())

            result = {"task_id": "sop_01", "success": True, "total_tokens": 123, "output_contract": "csv_expected_output"}
            registry.record(
                TrajectoryEvidence(
                    task_id="sop_01",
                    skill_id="benchmark/sop_bench",
                    context="sop_bench",
                    outcome="success",
                    total_tokens=123,
                    metadata={"output_contract": "csv_expected_output"},
                )
            )
            after_context = build_benchmark_skill_context("sop_bench", registry)
            after = save_skill_evolution_snapshot(
                out_root=out_root,
                benchmark="sop_bench",
                task_id="sop_01",
                stage="after",
                registry=registry,
                context=after_context,
                result=result,
            )

            self.assertIn("skill_context_after.md", after["context_path"])
            self.assertIn("posterior_context_after.md", after["posterior_context_path"])
            skill_context_after = (out_root / "skill_evolution" / "sop_bench" / "sop_01" / "skill_context_after.md").read_text()
            posterior_context_after = (out_root / "skill_evolution" / "sop_bench" / "sop_01" / "posterior_context_after.md").read_text()
            self.assertIn("Benchmark SOP Guardrails", skill_context_after)
            self.assertNotIn("Bayesian Skill Context", skill_context_after)
            self.assertNotIn("Bayesian Posterior Audit", skill_context_after)
            self.assertNotIn("posterior_success=", skill_context_after)
            self.assertIn("Bayesian Posterior Audit", posterior_context_after)
            self.assertIn("posterior_success=0.667", posterior_context_after)
            after_belief = json.loads((out_root / "skill_evolution" / "sop_bench" / "sop_01" / "belief_after.json").read_text())
            self.assertTrue(after_belief["known"])
            self.assertEqual(after_belief["belief"]["observations"], 1)

            index = json.loads((out_root / "skill_evolution" / "index.json").read_text())
            self.assertEqual([item["stage"] for item in index["snapshots"]], ["before", "after"])
            self.assertTrue(all("posterior_context_path" in item for item in index["snapshots"]))


if __name__ == "__main__":
    unittest.main()
