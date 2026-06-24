import json
import tempfile
import unittest
from pathlib import Path

from bayesian_agent.harness import AgentHarness, HarnessTask


class HarnessCoreTests(unittest.TestCase):
    def test_harness_wraps_adapter_run_and_writes_artifacts(self):
        class Adapter:
            def run_task(self, *, prompt, workspace, max_turns):
                self.prompt = prompt
                self.workspace = Path(workspace)
                self.max_turns = max_turns
                return {"transcript": "ok", "input_tokens": 2, "output_tokens": 3}

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "task"
            adapter = Adapter()
            harness = AgentHarness(adapter)

            run = harness.run_task(HarnessTask(task_id="t1", prompt="solve", workspace=workspace, max_turns=4))

            self.assertEqual(adapter.prompt, "solve")
            self.assertEqual(adapter.workspace, workspace.resolve())
            self.assertEqual(adapter.max_turns, 4)
            self.assertEqual(run["task_id"], "t1")
            self.assertEqual(run["total_tokens"], 5)
            self.assertTrue((workspace / "harness_task.json").exists())
            self.assertTrue((workspace / "harness_run.json").exists())
            saved = json.loads((workspace / "harness_run.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["task_id"], "t1")

    def test_harness_injects_skill_and_three_layer_memory_context(self):
        class Adapter:
            def run_task(self, *, prompt, workspace, max_turns):
                return {"transcript": prompt}

        with tempfile.TemporaryDirectory() as td:
            harness = AgentHarness(Adapter(), memory_enabled=True)
            harness.memory.hippocampus.remember("recent clue")
            harness.memory.state.set("phase", "draft")
            harness.memory.cortex.remember("stable rule")

            run = harness.run_task(
                HarnessTask(
                    task_id="t1",
                    prompt="solve",
                    workspace=Path(td),
                    skill_context="skill patch",
                    memory_context=True,
                )
            )

            self.assertIn("skill patch", run["transcript"])
            self.assertIn("### Hippocampus", run["transcript"])
            self.assertIn("recent clue", run["transcript"])
            self.assertIn("phase: draft", run["transcript"])
            self.assertIn("stable rule", run["transcript"])
            self.assertTrue(run["harness_memory_context"])

    def test_harness_memory_is_disabled_by_default_even_when_task_requests_context(self):
        class Adapter:
            def run_task(self, *, prompt, workspace, max_turns):
                return {"transcript": prompt}

        with tempfile.TemporaryDirectory() as td:
            harness = AgentHarness(Adapter())
            harness.memory.hippocampus.remember("recent clue")
            harness.memory.state.set("phase", "draft")
            harness.memory.cortex.remember("stable rule")

            run = harness.run_task(
                HarnessTask(
                    task_id="t1",
                    prompt="solve",
                    workspace=Path(td),
                    skill_context="skill patch",
                    memory_context=True,
                )
            )

            self.assertIn("skill patch", run["transcript"])
            self.assertNotIn("### Hippocampus", run["transcript"])
            self.assertNotIn("recent clue", run["transcript"])
            self.assertFalse(run["harness_memory_context"])

    def test_harness_records_verified_outcome_to_cortex(self):
        class Adapter:
            def run_task(self, *, prompt, workspace, max_turns):
                return {"transcript": "ok", "total_tokens": 11}

        with tempfile.TemporaryDirectory() as td:
            harness = AgentHarness(Adapter(), registry_path=Path(td) / "beliefs.json", memory_enabled=True)
            run = harness.run_task(HarnessTask(task_id="sop_01", prompt="solve", workspace=Path(td) / "task"))

            belief = harness.record_outcome(
                run,
                skill_id="benchmark/sop_bench",
                context="sop_bench",
                success=False,
                failure_mode="left_expected_output_blank",
            )

            self.assertEqual(belief.observations, 1)
            self.assertEqual(belief.failure_modes["left_expected_output_blank"], 1)
            self.assertEqual(harness.memory.state.get("last_outcome"), "failure")

    def test_disabled_memory_still_records_verified_outcome_to_registry(self):
        class Adapter:
            def run_task(self, *, prompt, workspace, max_turns):
                return {"transcript": "ok", "total_tokens": 11}

        with tempfile.TemporaryDirectory() as td:
            harness = AgentHarness(Adapter(), registry_path=Path(td) / "beliefs.json")
            run = harness.run_task(HarnessTask(task_id="sop_01", prompt="solve", workspace=Path(td) / "task"))

            belief = harness.record_outcome(
                run,
                skill_id="benchmark/sop_bench",
                context="sop_bench",
                success=False,
                failure_mode="left_expected_output_blank",
            )

            self.assertEqual(belief.observations, 1)
            self.assertEqual(belief.failure_modes["left_expected_output_blank"], 1)
            self.assertIsNone(harness.memory.state.get("last_outcome"))


if __name__ == "__main__":
    unittest.main()
