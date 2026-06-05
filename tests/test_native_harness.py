import tempfile
import unittest
from pathlib import Path

from bayesian_agent.harness.native import NativeBayesianAgentAdapter
from bayesian_agent.harness.tools import WorkspaceToolbox


class NativeHarnessTests(unittest.TestCase):
    def test_native_harness_runs_llm_tool_loop_without_external_harness(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def chat(self, messages, tools):
                self.calls.append((messages, tools))
                if len(self.calls) == 1:
                    return {
                        "content": "I will write the file.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "file_write",
                                "arguments": {"path": "answer.txt", "content": "ok"},
                            }
                        ],
                        "usage": {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12},
                        "finish_reason": "tool_calls",
                    }
                return {
                    "content": "Finished.",
                    "tool_calls": [],
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                    "finish_reason": "stop",
                }

        with tempfile.TemporaryDirectory() as td:
            adapter = NativeBayesianAgentAdapter(client=FakeClient(), model="unit-test-model")

            run = adapter.run_task(prompt="write ok", workspace=Path(td), max_turns=4)

            self.assertEqual((Path(td) / "answer.txt").read_text(encoding="utf-8"), "ok")
            self.assertEqual(run["exit_reason"], "stop")
            self.assertEqual(run["total_tokens"], 17)
            self.assertEqual(run["api_calls"], 2)
            self.assertIn("file_write", run["transcript"])
            self.assertIn("first-party", adapter.integration_note())

    def test_workspace_toolbox_executes_code_inside_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            toolbox = WorkspaceToolbox(Path(td))

            result = toolbox.dispatch("code_run", {"code": "open('made.txt','w').write('ok')", "language": "python"})

            self.assertEqual(result["status"], "success")
            self.assertEqual((Path(td) / "made.txt").read_text(encoding="utf-8"), "ok")

    def test_workspace_toolbox_rejects_file_access_outside_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            toolbox = WorkspaceToolbox(Path(td))

            result = toolbox.dispatch("file_read", {"path": "../outside.txt"})

            self.assertEqual(result["status"], "error")
            self.assertIn("outside workspace", result["error"])

    def test_workspace_toolbox_reads_workspace_symlinked_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "workspace"
            cache = root / "cache"
            workspace.mkdir()
            cache.mkdir()
            (cache / "data.txt").write_text("cached", encoding="utf-8")
            (workspace / "api_cache").symlink_to(cache, target_is_directory=True)
            toolbox = WorkspaceToolbox(workspace)

            result = toolbox.dispatch("file_read", {"path": "api_cache/data.txt"})

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["content"], "cached")


if __name__ == "__main__":
    unittest.main()
