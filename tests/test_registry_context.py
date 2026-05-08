import tempfile
import unittest
from pathlib import Path

from bayesian_agent import BayesianSkillRegistry, TrajectoryEvidence
from bayesian_agent.core.context import SkillContextBuilder
from bayesian_agent.core.policy import RewritePolicy


class RegistryContextTests(unittest.TestCase):
    def test_registry_persists_beliefs(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "beliefs.json"
            registry = BayesianSkillRegistry(path)
            registry.record(TrajectoryEvidence(task_id="t1", skill_id="skill/a", context="ctx", outcome="success", total_tokens=7))

            reloaded = BayesianSkillRegistry(path)

            self.assertEqual(reloaded.get("skill/a").observations, 1)
            self.assertEqual(reloaded.get("skill/a").mean_tokens, 7.0)

    def test_context_builder_orders_by_posterior_and_cost(self):
        registry = BayesianSkillRegistry.in_memory()
        registry.record(TrajectoryEvidence(task_id="a1", skill_id="skill/a", context="ctx", outcome="success", total_tokens=100))
        registry.record(TrajectoryEvidence(task_id="a2", skill_id="skill/a", context="ctx", outcome="success", total_tokens=120))
        registry.record(TrajectoryEvidence(task_id="b1", skill_id="skill/b", context="ctx", outcome="failure", total_tokens=10, failure_mode="bad"))

        context = SkillContextBuilder(registry).render(task_context="ctx", limit=2)

        self.assertIn("Bayesian Skill Context", context)
        self.assertLess(context.find("skill/a"), context.find("skill/b"))
        self.assertIn("posterior_success", context)

    def test_rewrite_policy_selects_actions(self):
        registry = BayesianSkillRegistry.in_memory()
        for i in range(4):
            registry.record(TrajectoryEvidence(task_id=f"f{i}", skill_id="skill/failing", context="ctx", outcome="failure", failure_mode="same_bug"))
        failing = registry.get("skill/failing")

        decision = RewritePolicy().decide(failing)

        self.assertEqual(decision.action, "retire")


if __name__ == "__main__":
    unittest.main()
