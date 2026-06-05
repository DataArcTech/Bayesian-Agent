"""Shared harness types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Union


@dataclass
class HarnessTask:
    """One normalized task envelope executed by a backend adapter."""

    task_id: str
    prompt: str
    workspace: Union[str, Path]
    max_turns: int = 8
    skill_context: str = ""
    memory_context: bool = False
    task_context: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def workspace_path(self) -> Path:
        return Path(self.workspace).resolve()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "prompt": str(self.prompt),
            "workspace": str(self.workspace_path()),
            "max_turns": int(self.max_turns or 0),
            "skill_context": str(self.skill_context or ""),
            "memory_context": bool(self.memory_context),
            "task_context": str(self.task_context or ""),
            "metadata": _json_safe(dict(self.metadata or {})),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
