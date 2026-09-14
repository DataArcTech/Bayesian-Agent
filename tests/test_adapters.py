import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

from bayesian_agent.adapters.aider import AiderAdapter
from bayesian_agent.adapters.base import AgentAdapter
from bayesian_agent.adapters.claude_code import ClaudeCodeAdapter
from bayesian_agent.adapters.generic_agent import GenericAgentAdapter
from bayesian_agent.adapters.mini_swe_agent import MiniSWEAgentAdapter


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

    def test_mini_swe_agent_adapter_does_not_import_harness_eagerly(self):
        adapter = MiniSWEAgentAdapter(root="/tmp/not-installed")

        self.assertEqual(adapter.root, "/tmp/not-installed")
        self.assertIn("mini-swe-agent", adapter.integration_note())

    def test_mini_swe_agent_adapter_builds_litellm_openai_config(self):
        adapter = MiniSWEAgentAdapter(root="/tmp/mini-swe-agent", model="deepseek-v4-flash")

        config = adapter.model_config("test-key")

        self.assertEqual(config["model_name"], "openai/deepseek-v4-flash")
        self.assertEqual(config["model_kwargs"]["api_key"], "test-key")
        self.assertEqual(config["model_kwargs"]["api_base"], "https://api.deepseek.com")
        self.assertTrue(config["model_kwargs"]["drop_params"])
        self.assertEqual(config["cost_tracking"], "ignore_errors")

    def test_mini_swe_agent_adapter_has_task_level_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = MiniSWEAgentAdapter(root="/tmp/mini-swe-agent")

            task = adapter.build_task(prompt="hello", workspace=Path(td), max_turns=3)

            self.assertEqual(task["prompt"], "hello")
            self.assertEqual(task["workspace"], str(Path(td).resolve()))
            self.assertEqual(task["max_turns"], 3)

    def test_mini_swe_agent_adapter_resolves_source_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            package = root / "src" / "minisweagent"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            adapter = MiniSWEAgentAdapter(root=str(root))

            self.assertEqual(adapter.resolve_root(), root.resolve())

    def test_claude_code_adapter_parses_model_usage_tokens(self):
        adapter = ClaudeCodeAdapter(model="deepseek-v4-flash")
        raw = {
            "type": "result",
            "result": "done",
            "total_cost_usd": 0.12,
            "modelUsage": {
                "deepseek-v4-flash": {
                    "inputTokens": 100,
                    "cacheReadInputTokens": 20,
                    "cacheCreationInputTokens": 5,
                    "outputTokens": 7,
                    "costUSD": 0.12,
                }
            },
        }

        parsed = adapter.parse_result(raw)

        self.assertEqual(parsed["transcript"], "done")
        self.assertEqual(parsed["input_tokens"], 125)
        self.assertEqual(parsed["output_tokens"], 7)
        self.assertEqual(parsed["total_tokens"], 132)
        self.assertEqual(parsed["total_cost_usd"], 0.12)

    def test_claude_code_adapter_builds_noninteractive_command(self):
        adapter = ClaudeCodeAdapter(model="deepseek-v4-pro[1m]", cli_path="/usr/local/bin/claude")

        command = adapter.build_command()

        self.assertIn("/usr/local/bin/claude", command)
        self.assertIn("--print", command)
        self.assertIn("--output-format", command)
        self.assertIn("json", command)
        self.assertIn("--model", command)
        self.assertIn("deepseek-v4-pro[1m]", command)

    def test_claude_code_adapter_can_load_existing_workspace_log(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            (workspace / "model_response_log.txt").write_text(
                '{"result":"cached","modelUsage":{"deepseek-v4-flash":{"inputTokens":3,"outputTokens":2}}}',
                encoding="utf-8",
            )
            adapter = ClaudeCodeAdapter(model="deepseek-v4-flash")

            run = adapter.load_run_from_workspace(workspace)

            self.assertEqual(run["transcript"], "cached")
            self.assertEqual(run["total_tokens"], 5)

    @unittest.skipIf(os.name == "nt", "Claude Code timeout handling relies on POSIX os.killpg")
    def test_claude_code_adapter_returns_error_on_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "sleep_cli.sh"
            script.write_text("#!/usr/bin/env bash\nsleep 5\n", encoding="utf-8")
            script.chmod(script.stat().st_mode | stat.S_IXUSR)
            adapter = ClaudeCodeAdapter(model="deepseek-v4-flash", cli_path=str(script), timeout_seconds=1)

            run = adapter.run_task(prompt="hello", workspace=Path(td) / "workspace")

            self.assertNotEqual(run["exit_code"], 0)
            self.assertTrue(run["is_error"])
            self.assertIn("timed out", run["error"])

    @unittest.skipIf(os.name == "nt", "Claude Code timeout handling relies on POSIX os.killpg")
    def test_claude_code_adapter_kills_child_processes_on_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "spawn_child.py"
            script.write_text(
                "#!/usr/bin/env python3\n"
                "import os\n"
                "import subprocess\n"
                "import time\n"
                "child = subprocess.Popen(['sleep', '30'])\n"
                "with open('child.pid', 'w', encoding='utf-8') as handle:\n"
                "    handle.write(str(child.pid))\n"
                "    handle.flush()\n"
                "    os.fsync(handle.fileno())\n"
                "time.sleep(30)\n",
                encoding="utf-8",
            )
            script.chmod(script.stat().st_mode | stat.S_IXUSR)
            workspace = Path(td) / "workspace"
            adapter = ClaudeCodeAdapter(model="deepseek-v4-flash", cli_path=str(script), timeout_seconds=2)

            run = adapter.run_task(prompt="hello", workspace=workspace)

            self.assertNotEqual(run["exit_code"], 0)
            for _ in range(20):
                if (workspace / "child.pid").exists():
                    break
                time.sleep(0.1)
            child_pid = int((workspace / "child.pid").read_text(encoding="utf-8").strip())
            for _ in range(20):
                if not _pid_exists(child_pid):
                    break
                time.sleep(0.1)
            self.assertFalse(_pid_exists(child_pid))

    def test_aider_adapter_does_not_import_aider_eagerly(self):
        adapter = AiderAdapter(cli_path="/tmp/not-installed-aider")

        self.assertEqual(adapter.cli_path, "/tmp/not-installed-aider")
        self.assertIn("Aider", adapter.integration_note())

    def test_aider_adapter_builds_noninteractive_command(self):
        adapter = AiderAdapter(model="deepseek-v4-pro[1m]", cli_path="/usr/local/bin/aider")

        command = adapter.build_command()

        self.assertIn("/usr/local/bin/aider", command)
        self.assertIn("--model", command)
        self.assertIn("deepseek-v4-pro[1m]", command)
        self.assertIn("--yes-always", command)
        self.assertIn("--no-auto-commits", command)
        self.assertIn("--no-stream", command)
        self.assertIn("--message-file", command)
        self.assertNotIn("--message", command)

        # Also test auto_commits=True and yes_always=False
        adapter_custom = AiderAdapter(auto_commits=True, yes_always=False)
        command_custom = adapter_custom.build_command()
        self.assertIn("--auto-commits", command_custom)
        self.assertNotIn("--no-auto-commits", command_custom)
        self.assertNotIn("--yes-always", command_custom)

    def test_aider_adapter_parses_token_and_cost_usage(self):
        adapter = AiderAdapter(model="deepseek-v4-flash")
        sample_stdout = (
            "Aider v0.70.0\n"
            "Model: deepseek-v4-flash\n"
            "Applied 1 edit successfully.\n"
            "Tokens: 1.5k sent, 250 received. Cost: $0.02 message, $0.05 session.\n"
        )

        parsed = adapter.parse_result(sample_stdout)

        self.assertIn("Applied 1 edit successfully", parsed["transcript"])
        self.assertEqual(parsed["input_tokens"], 1500)
        self.assertEqual(parsed["output_tokens"], 250)
        self.assertEqual(parsed["total_tokens"], 1750)
        self.assertEqual(parsed["total_cost_usd"], 0.05)
        self.assertFalse(parsed["is_error"])
        self.assertEqual(parsed["errors"], [])

        # Parse realistic comma-formatted usage
        sample_stdout_comma = (
            "Model: deepseek-v4-flash\n"
            "Tokens: 1,200 sent, 350 received. Cost: $0.03\n"
        )
        parsed_comma = adapter.parse_result(sample_stdout_comma)
        self.assertEqual(parsed_comma["input_tokens"], 1200)
        self.assertEqual(parsed_comma["output_tokens"], 350)
        self.assertEqual(parsed_comma["total_tokens"], 1550)
        self.assertEqual(parsed_comma["total_cost_usd"], 0.03)

        # Graceful handling when usage info is absent
        empty_parsed = adapter.parse_result("No tokens mentioned here.")
        self.assertEqual(empty_parsed["input_tokens"], 0)
        self.assertEqual(empty_parsed["output_tokens"], 0)
        self.assertEqual(empty_parsed["total_tokens"], 0)
        self.assertEqual(empty_parsed["total_cost_usd"], 0.0)

    def test_aider_adapter_has_task_level_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = AiderAdapter(model="deepseek-v4-flash")

            task = adapter.build_task(prompt="solve task", workspace=Path(td), max_turns=5)

            self.assertEqual(task["prompt"], "solve task")
            self.assertEqual(task["workspace"], str(Path(td).resolve()))
            self.assertEqual(task["max_turns"], 5)

    def test_aider_adapter_returns_error_on_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            if os.name == "nt":
                script = Path(td) / "sleep_cli.bat"
                script.write_text(f'@"{sys.executable}" -c "import time; time.sleep(5)"\n', encoding="utf-8")
            else:
                script = Path(td) / "sleep_cli.sh"
                script.write_text("#!/usr/bin/env bash\nsleep 5\n", encoding="utf-8")
                script.chmod(script.stat().st_mode | stat.S_IXUSR)
            adapter = AiderAdapter(model="deepseek-v4-flash", cli_path=str(script), timeout_seconds=1)

            run = adapter.run_task(prompt="hello", workspace=Path(td) / "workspace")

            self.assertNotEqual(run["exit_code"], 0)
            self.assertTrue(run["is_error"])
            self.assertIn("timed out", run["error"])

    def test_aider_adapter_kills_child_processes_on_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            py_code = (
                "import os\n"
                "import subprocess\n"
                "import sys\n"
                "import time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                "with open('child.pid', 'w', encoding='utf-8') as handle:\n"
                "    handle.write(str(child.pid))\n"
                "    handle.flush()\n"
                "    os.fsync(handle.fileno())\n"
                "time.sleep(30)\n"
            )
            if os.name == "nt":
                py_script = Path(td) / "spawn_child.py"
                py_script.write_text(py_code, encoding="utf-8")
                script = Path(td) / "spawn_child.bat"
                script.write_text(f'@"{sys.executable}" "{py_script}"\n', encoding="utf-8")
            else:
                script = Path(td) / "spawn_child.py"
                script.write_text(f"#!/usr/bin/env python3\n{py_code}", encoding="utf-8")
                script.chmod(script.stat().st_mode | stat.S_IXUSR)

            workspace = Path(td) / "workspace"
            adapter = AiderAdapter(model="deepseek-v4-flash", cli_path=str(script), timeout_seconds=2)

            run = adapter.run_task(prompt="hello", workspace=workspace)

            self.assertNotEqual(run["exit_code"], 0)
            for _ in range(20):
                if (workspace / "child.pid").exists():
                    break
                time.sleep(0.1)
            child_pid = int((workspace / "child.pid").read_text(encoding="utf-8").strip())
            for _ in range(20):
                if not _pid_exists(child_pid):
                    break
                time.sleep(0.1)
            self.assertFalse(_pid_exists(child_pid))

    def test_aider_adapter_can_load_existing_workspace_log(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            (workspace / "model_response_log.txt").write_text(
                "Completed edits.\nTokens: 120 sent, 45 received. Cost: $0.01 message, $0.01 session.\n",
                encoding="utf-8",
            )
            adapter = AiderAdapter(model="deepseek-v4-flash")

            run = adapter.load_run_from_workspace(workspace)

            self.assertIsNotNone(run)
            self.assertIn("Completed edits", run["transcript"])
            self.assertEqual(run["input_tokens"], 120)
            self.assertEqual(run["output_tokens"], 45)
            self.assertEqual(run["total_tokens"], 165)
            self.assertEqual(run["total_cost_usd"], 0.01)
            self.assertTrue(run["recovered_from_workspace"])


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    return True


if __name__ == "__main__":
    unittest.main()

