"""Workspace tools for the first-party Bayesian-Agent harness."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read a UTF-8 text file inside the current task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path or absolute path inside workspace."},
                    "start": {"type": "integer", "description": "1-based start line.", "default": 1},
                    "count": {"type": "integer", "description": "Maximum lines to return.", "default": 200},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write UTF-8 text to a file inside the current task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path or absolute path inside workspace."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_run",
            "description": "Run Python or shell code in the current task workspace and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code or shell command to run."},
                    "language": {"type": "string", "enum": ["python", "bash", "sh"], "default": "python"},
                    "timeout": {"type": "integer", "description": "Timeout seconds.", "default": 60},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the task after required files have been written and verified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Short completion note."},
                },
                "required": ["message"],
            },
        },
    },
]


class WorkspaceToolbox:
    """Small deterministic tool dispatcher scoped to one workspace."""

    def __init__(self, workspace: Path, *, default_timeout: int = 60) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.default_timeout = int(default_timeout or 60)

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        arguments = dict(arguments or {})
        try:
            if name == "file_read":
                return self.file_read(
                    arguments.get("path", ""),
                    start=int(arguments.get("start") or 1),
                    count=int(arguments.get("count") or 200),
                )
            if name == "file_write":
                return self.file_write(arguments.get("path", ""), str(arguments.get("content") or ""))
            if name == "code_run":
                return self.code_run(
                    str(arguments.get("code") or ""),
                    language=str(arguments.get("language") or "python"),
                    timeout=int(arguments.get("timeout") or self.default_timeout),
                )
            if name == "finish":
                return {"status": "success", "finished": True, "message": str(arguments.get("message") or "")}
            return {"status": "error", "error": f"Unknown tool: {name}"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def file_read(self, path: Any, *, start: int = 1, count: int = 200) -> Dict[str, Any]:
        resolved = self._resolve_read_path(path)
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start or 1))
        count = max(1, min(int(count or 200), 1000))
        chunk = lines[start - 1 : start - 1 + count]
        return {
            "status": "success",
            "path": str(resolved),
            "start": start,
            "line_count": len(lines),
            "content": "\n".join(chunk),
        }

    def file_write(self, path: Any, content: str) -> Dict[str, Any]:
        resolved = self._resolve_write_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return {"status": "success", "path": str(resolved), "bytes": len(content.encode("utf-8"))}

    def code_run(self, code: str, *, language: str = "python", timeout: int = 60) -> Dict[str, Any]:
        if not code.strip():
            return {"status": "error", "error": "code is empty"}
        timeout = max(1, min(int(timeout or self.default_timeout), 300))
        language = language.lower()
        tmp_path = None
        if language in {"python", "py"}:
            handle = tempfile.NamedTemporaryFile(
                suffix=".ba_harness.py",
                delete=False,
                mode="w",
                encoding="utf-8",
                dir=str(self.workspace),
            )
            handle.write(code)
            handle.close()
            tmp_path = Path(handle.name)
            command = [sys.executable, "-X", "utf8", "-u", str(tmp_path)]
        elif language in {"bash", "sh", "shell"}:
            command = ["bash", "-lc", code]
        else:
            return {"status": "error", "error": f"Unsupported language: {language}"}

        started = subprocess.Popen(
            command,
            cwd=str(self.workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = started.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_group(started)
            try:
                stdout, stderr = started.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(started)
                stdout, stderr = started.communicate()
            return {
                "status": "error",
                "exit_code": 124,
                "stdout": _limit_text(stdout),
                "stderr": _limit_text(stderr),
                "error": f"code_run timed out after {timeout} seconds",
            }
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        status = "success" if started.returncode == 0 else "error"
        return {
            "status": status,
            "exit_code": int(started.returncode or 0),
            "stdout": _limit_text(stdout),
            "stderr": _limit_text(stderr),
        }

    def _workspace_lexical_path(self, path: Any) -> Path:
        raw = str(path or "").strip()
        if not raw:
            raise ValueError("path is required")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        lexical = Path(os.path.abspath(str(candidate)))
        if not _is_relative_to(lexical, self.workspace):
            raise ValueError(f"path is outside workspace: {raw}")
        return lexical

    def _resolve_read_path(self, path: Any) -> Path:
        # RealFin exposes read-only cache data as a symlink inside each workspace.
        # The lexical path must stay under the workspace, but the target may be an
        # external cache directory.
        return self._workspace_lexical_path(path)

    def _resolve_write_path(self, path: Any) -> Path:
        candidate = self._workspace_lexical_path(path)
        resolved = candidate.resolve()
        if not _is_relative_to(resolved, self.workspace):
            raise ValueError(f"path is outside workspace: {path}")
        return resolved


def compact_tool_result(result: Mapping[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _limit_text(value: Any, limit: int = 12000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n...[truncated]...\n{text[-half:]}"


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
