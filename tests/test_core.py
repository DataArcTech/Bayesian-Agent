import unittest

from bayesian_agent import SkillBelief, TrajectoryEvidence
from bayesian_agent.core.belief import RewriteDecision


class CoreModelTests(unittest.TestCase):
    def test_trajectory_evidence_round_trips(self):
        evidence = TrajectoryEvidence(
            task_id="sop_12",
            skill_id="benchmark/sop_bench",
            context="sop_bench",
            outcome="failure",
            input_tokens=100,
            output_tokens=20,
            failure_mode="xml_wrapped_answer",
            metadata={"expected": "manual_review"},
        )

        restored = TrajectoryEvidence.from_dict(evidence.to_dict())

        self.assertEqual(restored.task_id, "sop_12")
        self.assertEqual(restored.total_tokens, 120)
        self.assertEqual(restored.metadata["expected"], "manual_review")

    def test_skill_belief_updates_beta_posterior_and_cost(self):
        belief = SkillBelief(skill_id="benchmark/sop_bench")
        belief.update(TrajectoryEvidence(task_id="sop_01", skill_id=belief.skill_id, context="sop", outcome="success", total_tokens=10))
        belief.update(TrajectoryEvidence(task_id="sop_02", skill_id=belief.skill_id, context="sop", outcome="failure", total_tokens=30, failure_mode="blank_cell"))

        self.assertEqual(belief.alpha, 2.0)
        self.assertEqual(belief.beta, 2.0)
        self.assertEqual(belief.observations, 2)
        self.assertEqual(belief.failure_modes["blank_cell"], 1)
        self.assertEqual(belief.mean_tokens, 20.0)
        self.assertAlmostEqual(belief.success_probability, 0.5)

    def test_rewrite_decision_is_serializable(self):
        decision = RewriteDecision(action="patch", reason="failures cluster", confidence=0.75)

        self.assertEqual(decision.to_dict()["action"], "patch")


if __name__ == "__main__":
    unittest.main()
