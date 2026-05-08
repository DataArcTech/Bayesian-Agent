"""Optional GenericAgent integration boundary.

This module intentionally does not vendor or eagerly import GenericAgent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class GenericAgentAdapter:
    """Thin placeholder adapter for local GenericAgent checkouts."""

    root: str

    def integration_note(self) -> str:
        return (
            "GenericAgent integration is optional. Set root to a local GenericAgent checkout; "
            "Bayesian-Agent does not copy or vendor GenericAgent code."
        )

    def run(self, task: Mapping[str, Any], skill_context: str) -> Mapping[str, Any]:
        raise NotImplementedError(
            "GenericAgentAdapter is a boundary placeholder for v0.3. "
            "Wire this method to a local GenericAgent runner in your environment."
        )
