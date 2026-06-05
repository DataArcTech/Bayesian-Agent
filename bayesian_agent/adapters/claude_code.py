"""Claude Code CLI adapter."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union


@dataclass
class ClaudeCodeAdapter:
    """Run one task in a workspace using the local `claude` CLI."""

    model: str = "deepseek-v4-flash"
    cli_path: str = "claude"
    permission_mode: str = "bypassPermissions"
    timeout_seconds: int = 900
    max_budget_usd: Optional[float] = None
    system_prompt: str = ""

    def integration_note(self) -> str:
        return (
            "Claude Code integration is optional. Claude Code executes task prompts; "
            "Bayesian-Agent owns benchmark orchestration and result grading."
        )

    def run(self, task: Mapping[str, Any], skill_context: str = "") -> Mapping[str, Any]:
        prompt = str(task["prompt"])
        if skill_context:
            prompt = f"{skill_context}\n{prompt}"
        return self.run_task(
            prompt=prompt,
            workspace=task["workspace"],
            max_turns=int(task.get("max_turns", 8) or 8),
        )

    def build_task(self, *, prompt: str, workspace: Union[str, Path], max_turns: int = 8) -> Mapping[str, Any]:
        return {"prompt": prompt, "workspace": str(Path(workspace).resolve()), "max_turns": int(max_turns)}

    def build_command(self) -> list[str]:
        command = [
            self.cli_path,
            "--print",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--permission-mode",
            self.permission_mode,
            "--no-session-persistence",
        ]
        if self.system_prompt:
            command.extend(["--append-system-prompt", self.system_prompt])
        if self.max_budget_usd is not None:
            command.extend(["--max-budget-usd", str(self.max_budget_usd)])
        return command

    def run_task(self, *, prompt: str, workspace: Union[str, Path], max_turns: int = 8) -> Mapping[str, Any]:
        workspace_path = Path(workspace).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        command = self.build_command()
        started = time.time()
        raw_stdout = ""
        raw_stderr = ""
        try:
            process = subprocess.Popen(
                command,
                cwd=str(workspace_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                text=True,
            )
            raw_stdout, raw_stderr = process.communicate(input=prompt, timeout=self.timeout_seconds)
            elapsed = time.time() - started
            raw_stdout = raw_stdout or ""
            raw_stderr = raw_stderr or ""
            exit_code = process.returncode
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            try:
                timeout_stdout, timeout_stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                timeout_stdout, timeout_stderr = process.communicate()
            elapsed = time.time() - started
            raw_stdout = _decode_timeout_output(exc.stdout) or _decode_timeout_output(timeout_stdout)
            raw_stderr = _decode_timeout_output(exc.stderr) or _decode_timeout_output(timeout_stderr)
            exit_code = 124
            raw = {
                "type": "result",
                "is_error": True,
                "result": raw_stdout,
                "errors": [f"Claude Code timed out after {self.timeout_seconds} seconds."],
            }
            parsed = self.parse_result(raw)
            parsed["elapsed_seconds"] = elapsed
            parsed["exit_code"] = exit_code
            parsed["error"] = "; ".join(str(item) for item in parsed.get("errors") or [])[:2000]
            self._write_run_artifacts(workspace_path, command, raw_stdout, raw_stderr, parsed)
            return parsed
        (workspace_path / "claude_command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
        (workspace_path / "model_response_log.txt").write_text(raw_stdout, encoding="utf-8")
        if raw_stderr:
            (workspace_path / "claude_stderr.txt").write_text(raw_stderr, encoding="utf-8")
        try:
            raw = json.loads(raw_stdout)
        except json.JSONDecodeError:
            raw = {
                "type": "result",
                "is_error": True,
                "result": raw_stdout,
                "errors": [raw_stderr or "Claude Code returned non-JSON output."],
            }
        parsed = self.parse_result(raw)
        parsed["elapsed_seconds"] = elapsed
        parsed["exit_code"] = exit_code
        if exit_code != 0:
            errors = list(parsed.get("errors") or [])
            if raw_stderr:
                errors.append(raw_stderr[-2000:])
            parsed["errors"] = errors
            parsed["error"] = "; ".join(str(item) for item in errors)[:2000]
        (workspace_path / "transcript.txt").write_text(str(parsed.get("transcript") or ""), encoding="utf-8")
        return parsed

    def load_run_from_workspace(self, workspace: Union[str, Path]) -> Optional[Mapping[str, Any]]:
        log_path = Path(workspace).resolve() / "model_response_log.txt"
        if not log_path.exists():
            return None
        try:
            raw = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        parsed = dict(self.parse_result(raw))
        parsed["elapsed_seconds"] = 0.0
        parsed["exit_code"] = 0
        parsed["recovered_from_workspace"] = True
        return parsed

    def parse_result(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        model_usage = dict(raw.get("modelUsage") or {})
        input_tokens = 0
        output_tokens = 0
        cost = float(raw.get("total_cost_usd") or 0.0)
        for usage in model_usage.values():
            usage = dict(usage or {})
            input_tokens += int(usage.get("inputTokens") or 0)
            input_tokens += int(usage.get("cacheReadInputTokens") or 0)
            input_tokens += int(usage.get("cacheCreationInputTokens") or 0)
            output_tokens += int(usage.get("outputTokens") or 0)
            cost += float(usage.get("costUSD") or 0.0)
        if raw.get("total_cost_usd") is not None:
            cost = float(raw.get("total_cost_usd") or 0.0)
        transcript = str(raw.get("result") or raw.get("content") or raw.get("message") or "")
        errors = list(raw.get("errors") or [])
        return {
            "transcript": transcript,
            "exit_reason": str(raw.get("stop_reason") or raw.get("subtype") or ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "total_cost_usd": cost,
            "usage_events": [{"source": "claude_code", "model_usage": model_usage}],
            "model_usage": model_usage,
            "session_id": str(raw.get("session_id") or ""),
            "claude_uuid": str(raw.get("uuid") or ""),
            "is_error": bool(raw.get("is_error")),
            "errors": errors,
        }

    def _write_run_artifacts(
        self,
        workspace_path: Path,
        command: list[str],
        raw_stdout: str,
        raw_stderr: str,
        parsed: Mapping[str, Any],
    ) -> None:
        (workspace_path / "claude_command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
        (workspace_path / "model_response_log.txt").write_text(raw_stdout, encoding="utf-8")
        if raw_stderr:
            (workspace_path / "claude_stderr.txt").write_text(raw_stderr, encoding="utf-8")
        (workspace_path / "transcript.txt").write_text(str(parsed.get("transcript") or ""), encoding="utf-8")


def _decode_timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
