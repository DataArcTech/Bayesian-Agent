"""Trajectory evidence for Bayesian Skill/SOP evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, Mapping, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TrajectoryEvidence:
    """Action-verified evidence emitted by an agent run."""

    task_id: str
    skill_id: str
    context: str
    outcome: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    turns: int = 0
    elapsed_seconds: float = 0.0
    failure_mode: str = ""
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.total_tokens:
            self.total_tokens = int(self.input_tokens or 0) + int(self.output_tokens or 0)
        self.input_tokens = int(self.input_tokens or 0)
        self.output_tokens = int(self.output_tokens or 0)
        self.total_tokens = int(self.total_tokens or 0)
        self.turns = int(self.turns or 0)
        self.elapsed_seconds = float(self.elapsed_seconds or 0.0)

    @property
    def success(self) -> bool:
        return self.outcome.strip().lower() == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "skill_id": self.skill_id,
            "context": self.context,
            "outcome": self.outcome,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "turns": self.turns,
            "elapsed_seconds": self.elapsed_seconds,
            "failure_mode": self.failure_mode,
            "summary": self.summary,
            "metadata": _json_safe(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrajectoryEvidence":
        return cls(
            task_id=str(raw.get("task_id") or ""),
            skill_id=str(raw.get("skill_id") or raw.get("benchmark") or "unknown"),
            context=str(raw.get("context") or raw.get("benchmark") or ""),
            outcome=str(raw.get("outcome") or ("success" if raw.get("success") else "failure")),
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
            turns=int(raw.get("turns") or 0),
            elapsed_seconds=float(raw.get("elapsed_seconds") or 0.0),
            failure_mode=str(raw.get("failure_mode") or raw.get("error") or ""),
            summary=str(raw.get("summary") or ""),
            metadata=_json_safe(dict(raw.get("metadata") or {})),
            created_at=str(raw.get("created_at") or utc_now()),
        )

    @classmethod
    def from_run(
        cls,
        run: Mapping[str, Any],
        *,
        skill_id: str,
        context: str,
        failure_mode: Optional[str] = None,
    ) -> "TrajectoryEvidence":
        return cls(
            task_id=str(run.get("task_id") or ""),
            skill_id=skill_id,
            context=context,
            outcome="success" if run.get("success") else "failure",
            input_tokens=int(run.get("input_tokens") or 0),
            output_tokens=int(run.get("output_tokens") or 0),
            total_tokens=int(run.get("total_tokens") or 0),
            turns=int(run.get("turns") or 0),
            elapsed_seconds=float(run.get("elapsed_seconds") or 0.0),
            failure_mode=str(failure_mode if failure_mode is not None else run.get("failure_mode") or run.get("error") or ""),
            summary=str(run.get("summary") or run.get("task_id") or ""),
            metadata=_json_safe({k: v for k, v in run.items() if k not in {"transcript", "usage_events"}}),
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
