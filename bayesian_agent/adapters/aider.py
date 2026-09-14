"""Aider CLI adapter."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union


@dataclass
class AiderAdapter:
    """Run one task in a workspace using the local `aider` CLI."""

    model: str = "deepseek-v4-flash"
    cli_path: str = "aider"
    timeout_seconds: int = 900
    auto_commits: bool = False
    yes_always: bool = True

    def integration_note(self) -> str:
        return (
            "Aider integration is optional. Aider executes task prompts; "
            "Bayesian-Agent owns benchmark orchestration and result grading. "
            "Aider code is not copied or vendored."
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

    def build_command(self, message_file: Optional[Union[str, Path]] = "aider_prompt.txt") -> list[str]:
        # We pass the prompt via `--message-file` rather than argv `--message` because:
        # 1. Non-interactive scripted runs need to avoid OS argv length limits (ARG_MAX) with large prompts/skill contexts.
        # 2. `--message-file` explicitly instructs Aider to process the reply and exit, avoiding terminal TTY / prompt_toolkit hangs.
        # 3. It leaves an exact, inspectable prompt audit trail in the task workspace.
        command = [
            self.cli_path,
            "--model",
            self.model,
        ]
        if self.yes_always:
            command.append("--yes-always")
        if self.auto_commits:
            command.append("--auto-commits")
        else:
            command.append("--no-auto-commits")
        command.extend([
            "--no-stream",
            "--no-check-update",
            "--no-analytics",
            "--no-pretty",
        ])
        if message_file is not None:
            command.extend(["--message-file", str(message_file)])
        return command

    def run_task(self, *, prompt: str, workspace: Union[str, Path], max_turns: int = 8) -> Mapping[str, Any]:
        workspace_path = Path(workspace).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        prompt_file = workspace_path / "aider_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        command = self.build_command(message_file=prompt_file)
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
            raw_stdout, raw_stderr = process.communicate(timeout=self.timeout_seconds)
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
                "transcript": raw_stdout,
                "is_error": True,
                "errors": [f"Aider timed out after {self.timeout_seconds} seconds."],
            }
            parsed = self.parse_result(raw)
            parsed["elapsed_seconds"] = elapsed
            parsed["exit_code"] = exit_code
            parsed["error"] = "; ".join(str(item) for item in parsed.get("errors") or [])[:2000]
            self._write_run_artifacts(workspace_path, command, raw_stdout, raw_stderr, parsed)
            return parsed

        raw = {
            "transcript": raw_stdout,
            "stdout": raw_stdout,
            "stderr": raw_stderr,
            "is_error": exit_code != 0,
            "errors": [raw_stderr] if (exit_code != 0 and raw_stderr) else [],
        }
        parsed = self.parse_result(raw)
        parsed["elapsed_seconds"] = elapsed
        parsed["exit_code"] = exit_code
        if exit_code != 0:
            errors = list(parsed.get("errors") or [])
            if raw_stderr and raw_stderr[-2000:] not in errors:
                errors.append(raw_stderr[-2000:])
            parsed["errors"] = errors
            parsed["error"] = "; ".join(str(item) for item in errors)[:2000]
            parsed["is_error"] = True
        self._write_run_artifacts(workspace_path, command, raw_stdout, raw_stderr, parsed)
        return parsed

    def load_run_from_workspace(self, workspace: Union[str, Path]) -> Optional[Mapping[str, Any]]:
        log_path = Path(workspace).resolve() / "model_response_log.txt"
        if not log_path.exists():
            return None
        raw_text = log_path.read_text(encoding="utf-8")
        parsed = dict(self.parse_result(raw_text))
        parsed["elapsed_seconds"] = 0.0
        parsed["exit_code"] = 0
        parsed["recovered_from_workspace"] = True
        return parsed

    def parse_result(self, raw: Union[Mapping[str, Any], str]) -> Mapping[str, Any]:
        if isinstance(raw, str):
            text = raw
            transcript = raw
            is_error = False
            errors: list[str] = []
            exit_reason = ""
            session_id = ""
        else:
            transcript = str(raw.get("transcript") or raw.get("stdout") or raw.get("result") or "")
            text = f"{transcript}\n{raw.get('stdout') or ''}"
            is_error = bool(raw.get("is_error", False))
            errors = list(raw.get("errors") or [])
            exit_reason = str(raw.get("exit_reason") or raw.get("stop_reason") or "")
            session_id = str(raw.get("session_id") or "")

        input_tokens = 0
        output_tokens = 0
        cost = 0.0

        # Parse token usage:
        # e.g.: "Tokens: 1,500 sent, 250 received. Cost: $0.02 message, $0.05 session."
        # or:   "Tokens: 1.5k sent, 250 received."
        # or:   "Tokens: 2.1k sent, 1.2k cache write, 500 cache hit, 300 received."
        match_sent = re.search(r"Tokens:.*?(\d+(?:[.,]\d+)?)\s*([kKmM])?\s*sent", text, re.IGNORECASE | re.DOTALL)
        if not match_sent:
            match_sent = re.search(r"(\d+(?:[.,]\d+)?)\s*([kKmM])?\s*sent", text, re.IGNORECASE)
        if match_sent:
            input_tokens = _parse_token_count(match_sent.group(1), match_sent.group(2) or "")

        match_recv = re.search(r"(\d+(?:[.,]\d+)?)\s*([kKmM])?\s*received", text, re.IGNORECASE)
        if match_recv:
            output_tokens = _parse_token_count(match_recv.group(1), match_recv.group(2) or "")

        # Parse cost:
        # Prefer session cost if reported, otherwise message cost or general cost
        match_cost_session = re.search(r"Cost:\s*.*?\$([0-9]+(?:\.[0-9]+)?)\s*session", text, re.IGNORECASE | re.DOTALL)
        if match_cost_session:
            try:
                cost = float(match_cost_session.group(1))
            except ValueError:
                cost = 0.0
        else:
            match_cost = re.search(r"Cost:\s*\$([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            if match_cost:
                try:
                    cost = float(match_cost.group(1))
                except ValueError:
                    cost = 0.0

        total_tokens = input_tokens + output_tokens
        return {
            "transcript": transcript,
            "exit_reason": exit_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": cost,
            "usage_events": [
                {
                    "source": "aider",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "total_cost_usd": cost,
                }
            ],
            "session_id": session_id,
            "is_error": is_error,
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
        (workspace_path / "aider_command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
        (workspace_path / "model_response_log.txt").write_text(raw_stdout, encoding="utf-8")
        if raw_stderr:
            (workspace_path / "aider_stderr.txt").write_text(raw_stderr, encoding="utf-8")
        (workspace_path / "transcript.txt").write_text(str(parsed.get("transcript") or ""), encoding="utf-8")


def _parse_token_count(val: str, unit: str = "") -> int:
    val = val.replace(",", "").strip()
    multiplier = 1.0
    unit = unit.lower().strip()
    if unit == "k":
        multiplier = 1000.0
    elif unit == "m":
        multiplier = 1000000.0
    try:
        return int(float(val) * multiplier)
    except (ValueError, TypeError):
        return 0


def _decode_timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
            process.terminate()
    except (ProcessLookupError, OSError):
        return


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
            process.kill()
    except (ProcessLookupError, OSError):
        return
