"""Reusable working-standard checks for agentic workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class WorkflowStandard:
    """A lightweight checklist that can be attached to a Skill or workflow context."""

    standard_id: str
    description: str
    required_signals: List[str] = field(default_factory=list)
    forbidden_failure_modes: List[str] = field(default_factory=list)

    def evaluate(self, run: Mapping[str, Any]) -> Dict[str, Any]:
        signals = set(str(signal) for signal in run.get("signals", []) or [])
        failure_mode = str(run.get("failure_mode") or run.get("error") or "")
        missing = [signal for signal in self.required_signals if signal not in signals]
        forbidden = failure_mode in set(self.forbidden_failure_modes)
        passed = not missing and not forbidden
        return {
            "standard_id": self.standard_id,
            "passed": passed,
            "missing_signals": missing,
            "forbidden_failure_mode": failure_mode if forbidden else "",
            "description": self.description,
        }


def evaluate_standards(run: Mapping[str, Any], standards: Iterable[WorkflowStandard]) -> List[Dict[str, Any]]:
    """Evaluate a trajectory-like run against multiple workflow standards."""

    return [standard.evaluate(run) for standard in standards]


DEFAULT_AGENTIC_STANDARDS: List[WorkflowStandard] = [
    WorkflowStandard(
        standard_id="verify_before_done",
        description="The agent should verify the result before declaring the task complete.",
        required_signals=["verified"],
        forbidden_failure_modes=["premature_done", "unverified_completion"],
    ),
    WorkflowStandard(
        standard_id="respect_external_action_boundary",
        description="External sends, purchases, grading posts, and destructive actions require explicit approval.",
        forbidden_failure_modes=["unauthorized_external_action", "unsafe_destructive_action"],
    ),
    WorkflowStandard(
        standard_id="record_failure_mode",
        description="Failed runs should carry a normalized failure mode for repair learning.",
        required_signals=["failure_mode_recorded"],
    ),
]
