import unittest
import os
import tempfile
from argparse import Namespace
from pathlib import Path

from bayesian_agent.adapters.bayesian_agent import NativeBayesianAgentAdapter
from bayesian_agent.adapters.claude_code import ClaudeCodeAdapter
from bayesian_agent.adapters.generic_agent import GenericAgentAdapter
from bayesian_agent.adapters.mini_swe_agent import MiniSWEAgentAdapter
from bayesian_agent.benchmarks.sop_lifelong import compact_baseline_run, incremental_task_filter, prepare_belief_store, replay_skill_evolution_artifacts
from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.harness import AgentHarness
from experiments import run_benchmarks
from experiments.run_benchmarks import build_run_plan, load_env_file


class SopLifelongExperimentTests(unittest.TestCase):
    def test_unified_benchmark_runner_module_exists(self):
        plan = build_run_plan("all", Path("/tmp/realfin"), [])

        self.assertEqual([run.name for run in plan], ["baseline", "bayesian_full", "bayesian_incremental"])
        self.assertEqual(plan[2].baseline_paths, ["/tmp/realfin/baseline/results.json"])

    def test_all_mode_plans_baseline_full_and_incremental(self):
        plan = build_run_plan("all", Path("/tmp/sop"), [])

        self.assertEqual([run.name for run in plan], ["baseline", "bayesian_full", "bayesian_incremental"])
        self.assertEqual([run.mode for run in plan], ["baseline", "bayesian-full", "bayesian-incremental"])
        self.assertEqual(plan[2].baseline_paths, ["/tmp/sop/baseline/results.json"])

    def test_core_benchmark_selection_uses_separate_output_roots(self):
        default_specs = run_benchmarks.build_benchmark_runs("core", "deepseek-v4-flash", "")
        explicit_specs = run_benchmarks.build_benchmark_runs("core", "deepseek-v4-flash", "/tmp/ba-core")

        self.assertEqual([spec.bench for spec in default_specs], ["sop", "lifelong"])
        self.assertEqual(
            [spec.out_root for spec in default_specs],
            [Path("results/sop_deepseek_v4_flash"), Path("results/lifelong_deepseek_v4_flash")],
        )
        self.assertEqual(
            [spec.out_root for spec in explicit_specs],
            [Path("/tmp/ba-core/sop"), Path("/tmp/ba-core/lifelong")],
        )

    def test_build_adapter_selects_claude_code_or_genericagent(self):
        base = {
            "model": "deepseek-v4-flash",
            "genericagent_root": "../GenericAgent",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com",
            "anthropic_base_url": "https://api.deepseek.com/anthropic",
            "protocol": "openai",
            "disable_ssl_verify": False,
            "host_header": "",
            "claude_cli": "claude",
            "claude_permission_mode": "bypassPermissions",
            "claude_timeout": 900,
            "claude_max_budget_usd": 0.0,
            "mini_swe_agent_root": "../mini-swe-agent",
            "mini_swe_config": "default",
            "mini_swe_env_timeout": 60,
            "mini_swe_wall_time_limit": 0,
            "native_timeout": 180,
            "native_max_tokens": 4096,
            "native_temperature": 0.0,
        }

        native = run_benchmarks.build_adapter(Namespace(**{**base, "harness": "bayesian-agent"}))
        self.assertIsInstance(run_benchmarks.build_adapter(Namespace(**{**base, "harness": "genericagent"})), GenericAgentAdapter)
        claude = run_benchmarks.build_adapter(Namespace(**{**base, "harness": "claude-code"}))
        mini = run_benchmarks.build_adapter(Namespace(**{**base, "harness": "mini-swe-agent"}))

        self.assertIsInstance(native, NativeBayesianAgentAdapter)
        self.assertIn("first-party", native.integration_note())
        self.assertIsInstance(claude, ClaudeCodeAdapter)
        self.assertEqual(claude.model, "deepseek-v4-flash")
        self.assertIsInstance(mini, MiniSWEAgentAdapter)
        self.assertEqual(mini.model, "deepseek-v4-flash")

    def test_build_harness_wraps_selected_adapter(self):
        class Adapter:
            pass

        harness = run_benchmarks.build_harness(Adapter())

        self.assertIsInstance(harness, AgentHarness)
        self.assertIsInstance(harness.adapter, Adapter)

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

    def test_bayesian_full_starts_from_empty_belief_store(self):
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            registry = prepare_belief_store(out_root, "bayesian-full")
            registry.record(TrajectoryEvidence(task_id="old", skill_id="benchmark/sop_bench", context="sop_bench", outcome="success"))

            reset = prepare_belief_store(out_root, "bayesian-full")

            self.assertEqual(reset.beliefs(), [])

    def test_replay_skill_evolution_artifacts_from_results_payload(self):
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            artifacts = replay_skill_evolution_artifacts(
                out_root,
                {
                    "results": {
                        "sop_bench": [
                            {
                                "task_id": "sop_01",
                                "success": True,
                                "total_tokens": 10,
                                "output_contract": "csv_expected_output",
                            }
                        ]
                    }
                },
            )

            self.assertTrue((artifacts / "index.json").exists())
            self.assertTrue((artifacts / "sop_bench" / "sop_01" / "skill_context_before.md").exists())
            self.assertTrue((artifacts / "sop_bench" / "sop_01" / "skill_context_after.md").exists())
            self.assertTrue((artifacts / "sop_bench" / "sop_01" / "posterior_context_before.md").exists())
            self.assertTrue((artifacts / "sop_bench" / "sop_01" / "posterior_context_after.md").exists())
            skill_context_after = (artifacts / "sop_bench" / "sop_01" / "skill_context_after.md").read_text()
            posterior_context_after = (artifacts / "sop_bench" / "sop_01" / "posterior_context_after.md").read_text()
            self.assertIn("Benchmark SOP Guardrails", skill_context_after)
            self.assertNotIn("posterior_success=", skill_context_after)
            self.assertIn("posterior_success=0.667", posterior_context_after)

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
