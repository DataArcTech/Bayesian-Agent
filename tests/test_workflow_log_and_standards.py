import json
import tempfile
import unittest
from pathlib import Path

from bayesian_agent.adapters.workflow_log import evidence_from_jsonl, workflow_record_to_evidence
from bayesian_agent.core.standards import DEFAULT_AGENTIC_STANDARDS, WorkflowStandard, evaluate_standards
from bayesian_agent.cli import main


class WorkflowLogAndStandardsTests(unittest.TestCase):
    def test_workflow_record_to_evidence_accepts_agent_log_aliases(self):
        event = workflow_record_to_evidence(
            {
                "run_id": "grade-1",
                "workflow_id": "openclaw/grading/rubric_feedback",
                "task_family": "grading",
                "status": "completed",
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "steps": 4,
                "duration_seconds": 3.5,
                "summary": "drafted rubric feedback",
            }
        )

        self.assertEqual(event.task_id, "grade-1")
        self.assertEqual(event.skill_id, "openclaw/grading/rubric_feedback")
        self.assertEqual(event.context, "grading")
        self.assertTrue(event.success)
        self.assertEqual(event.total_tokens, 125)
        self.assertEqual(event.turns, 4)

    def test_evolve_workflow_log_cli_updates_registry(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "runs.jsonl"
            registry = Path(td) / "beliefs.json"
            context = Path(td) / "context.md"
            records = [
                {"id": "a", "workflow": "coding", "sop_id": "openclaw/coding/test_first", "success": True, "total_tokens": 20},
                {"id": "b", "workflow": "coding", "sop_id": "openclaw/coding/test_first", "success": False, "failure_mode": "missing_test", "total_tokens": 30},
            ]
            src.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

            code = main(["evolve-workflow-log", "--jsonl", str(src), "--registry", str(registry), "--context-out", str(context)])

            self.assertEqual(code, 0)
            raw = json.loads(registry.read_text())
            belief = raw["skills"]["openclaw/coding/test_first"]
            self.assertEqual(belief["observations"], 2)
            self.assertIn("posterior_std", context.read_text())

    def test_workflow_standard_evaluation(self):
        standard = WorkflowStandard("must_verify", "verification required", required_signals=["verified"])

        failed = standard.evaluate({"signals": []})
        passed = standard.evaluate({"signals": ["verified"]})

        self.assertFalse(failed["passed"])
        self.assertTrue(passed["passed"])
        self.assertTrue(evaluate_standards({"signals": ["verified", "failure_mode_recorded"]}, DEFAULT_AGENTIC_STANDARDS))


if __name__ == "__main__":
    unittest.main()
