import unittest

from bayesian_agent.benchmarks.evolution import build_benchmark_skill_context, classify_failure
from bayesian_agent.benchmarks.sop_lifelong import build_lifelong_prompt
from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.core.registry import BayesianSkillRegistry


class BenchmarkEvolutionTests(unittest.TestCase):
    def test_sop_failure_context_is_owned_by_bayesian_agent(self):
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

        self.assertIn("Bayesian Skill Context", context)
        self.assertIn("rows[row_index - 1]", context)
        self.assertIn("raw category string", context)

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


if __name__ == "__main__":
    unittest.main()
