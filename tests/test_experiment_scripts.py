import unittest
import os
import tempfile
from pathlib import Path

from bayesian_agent.benchmarks.sop_lifelong import compact_baseline_run, incremental_task_filter
from experiments.run_sop_lifelong import build_run_plan, load_env_file


class SopLifelongExperimentTests(unittest.TestCase):
    def test_all_mode_plans_baseline_full_and_incremental(self):
        plan = build_run_plan("all", Path("/tmp/sop_lifelong"), [])

        self.assertEqual([run.name for run in plan], ["baseline", "bayesian_full", "bayesian_incremental"])
        self.assertEqual([run.mode for run in plan], ["baseline", "bayesian-full", "bayesian-incremental"])
        self.assertEqual(plan[2].baseline_paths, ["/tmp/sop_lifelong/baseline/results.json"])

    def test_incremental_filter_runs_zero_tasks_for_bench_without_failures(self):
        baseline_results = {"sop_bench": [{"task_id": "sop_01", "success": True}]}
        failed = {}

        self.assertEqual(incremental_task_filter(baseline_results, failed, "sop_bench"), set())
        self.assertIsNone(incremental_task_filter({}, failed, "sop_bench"))

    def test_baseline_compaction_drops_heavy_transcripts(self):
        compacted = compact_baseline_run(
            {"task_id": "sop_01", "success": True, "transcript": "large", "usage_events": [1], "exit_reason": "verbose"}
        )

        self.assertEqual(compacted, {"task_id": "sop_01", "success": True})

    def test_load_env_file_uses_standard_library(self):
        old_value = os.environ.pop("BAYESIAN_AGENT_TEST_ENV", None)
        try:
            with tempfile.TemporaryDirectory() as td:
                env_path = Path(td) / ".env"
                env_path.write_text("BAYESIAN_AGENT_TEST_ENV='ok'\n", encoding="utf-8")

                load_env_file(env_path)

                self.assertEqual(os.environ["BAYESIAN_AGENT_TEST_ENV"], "ok")
        finally:
            if old_value is None:
                os.environ.pop("BAYESIAN_AGENT_TEST_ENV", None)
            else:
                os.environ["BAYESIAN_AGENT_TEST_ENV"] = old_value


if __name__ == "__main__":
    unittest.main()
