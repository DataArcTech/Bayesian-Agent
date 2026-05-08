"""Agent adapter protocol."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class AgentAdapter(Protocol):
    """Minimal boundary an external agent must satisfy."""

    def run(self, task: Mapping[str, Any], skill_context: str) -> Mapping[str, Any]:
        """Run one task with Bayesian Skill context and return a trajectory-like mapping."""
        ...
