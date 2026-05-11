import tempfile
import unittest
from pathlib import Path

from bayesian_agent.adapters.base import AgentAdapter
from bayesian_agent.adapters.generic_agent import GenericAgentAdapter


class AdapterTests(unittest.TestCase):
    def test_agent_adapter_protocol_is_runtime_checkable(self):
        class Dummy:
            def run(self, task, skill_context):
                return {"task_id": task["task_id"], "success": True, "skill_context": skill_context}

        self.assertIsInstance(Dummy(), AgentAdapter)

    def test_generic_agent_adapter_does_not_import_generic_agent_eagerly(self):
        adapter = GenericAgentAdapter(root="/tmp/not-installed")

        self.assertEqual(adapter.root, "/tmp/not-installed")
        self.assertIn("GenericAgent", adapter.integration_note())

    def test_generic_agent_adapter_builds_model_config_without_benchmark_runner(self):
        adapter = GenericAgentAdapter(root="/tmp/genericagent", model="deepseek-v4-flash")

        config = adapter.model_configs("test-key")

        self.assertIn("native_oai_config_bayesian_agent", config)
        self.assertEqual(config["native_oai_config_bayesian_agent"]["model"], "deepseek-v4-flash")
        self.assertEqual(config["native_oai_config_bayesian_agent"]["apikey"], "test-key")

    def test_generic_agent_adapter_has_task_level_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = GenericAgentAdapter(root="/tmp/genericagent")

            task = adapter.build_task(prompt="hello", workspace=Path(td), max_turns=3)

            self.assertEqual(task["prompt"], "hello")
            self.assertEqual(task["workspace"], str(Path(td).resolve()))
            self.assertEqual(task["max_turns"], 3)


if __name__ == "__main__":
    unittest.main()
