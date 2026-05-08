import unittest

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


if __name__ == "__main__":
    unittest.main()
