import json
import tempfile
import unittest
from pathlib import Path

from bayesian_agent.core.repair import failed_task_ids, merge_repairs, summarize_incremental_lift
from bayesian_agent.cli import main


class RepairCliTests(unittest.TestCase):
    def test_failed_task_ids_and_merge_repairs(self):
        baseline = {
            "sop_bench": [
                {"task_id": "sop_01", "success": True, "input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
                {"task_id": "sop_02", "success": False, "input_tokens": 20, "output_tokens": 2, "total_tokens": 22},
            ]
        }
        repairs = {
            "sop_bench": [
                {"task_id": "sop_02", "success": True, "input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
            ]
        }

        self.assertEqual(failed_task_ids(baseline), {"sop_bench": {"sop_02"}})
        self.assertTrue(all(run["success"] for run in merge_repairs(baseline, repairs)["sop_bench"]))
        self.assertEqual(summarize_incremental_lift(baseline, repairs)["sop_bench"]["accuracy"], 1.0)

    def test_cli_summarize_writes_json(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "results.json"
            out = Path(td) / "summary.json"
            src.write_text(json.dumps({"results": {"bench": [{"task_id": "a", "success": True, "total_tokens": 10}]}}), encoding="utf-8")

            code = main(["summarize", "--results", str(src), "--out", str(out)])

            self.assertEqual(code, 0)
            self.assertTrue(out.exists())
            self.assertEqual(json.loads(out.read_text())["bench"]["successes"], 1)


if __name__ == "__main__":
    unittest.main()
