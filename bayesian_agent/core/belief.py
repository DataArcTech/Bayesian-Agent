"""Belief state for Bayesian Skill/SOP hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from bayesian_agent.core.evidence import TrajectoryEvidence, utc_now


MAX_EVIDENCE = 100


@dataclass
class RewriteDecision:
    action: str
    reason: str
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "confidence": float(self.confidence)}


@dataclass
class SkillBelief:
    """Posterior belief for one Skill/SOP hypothesis."""

    skill_id: str
    alpha: float = 1.0
    beta: float = 1.0
    contexts: Dict[str, int] = field(default_factory=dict)
    failure_modes: Dict[str, int] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    observations: int = 0
    mean_tokens: float = 0.0
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_elapsed_seconds: float = 0.0
    last_updated: str = field(default_factory=utc_now)

    @property
    def success_probability(self) -> float:
        denom = self.alpha + self.beta
        return self.alpha / denom if denom else 0.0

    def update(self, event: TrajectoryEvidence) -> "SkillBelief":
        outcome = event.outcome.strip().lower()
        if outcome == "success":
            self.alpha += 1.0
        elif outcome in {"failure", "failed", "error"}:
            self.beta += 1.0

        context = event.context or "unknown"
        self.contexts[context] = self.contexts.get(context, 0) + 1
        if event.failure_mode:
            self.failure_modes[event.failure_mode] = self.failure_modes.get(event.failure_mode, 0) + 1

        self.observations += 1
        n = float(self.observations)
        self.mean_tokens += (event.total_tokens - self.mean_tokens) / n
        self.mean_input_tokens += (event.input_tokens - self.mean_input_tokens) / n
        self.mean_output_tokens += (event.output_tokens - self.mean_output_tokens) / n
        self.mean_elapsed_seconds += (event.elapsed_seconds - self.mean_elapsed_seconds) / n
        self.evidence.append(event.to_dict())
        self.evidence = self.evidence[-MAX_EVIDENCE:]
        self.last_updated = utc_now()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "posterior_success": self.success_probability,
            "contexts": self.contexts,
            "failure_modes": self.failure_modes,
            "evidence": self.evidence[-MAX_EVIDENCE:],
            "observations": self.observations,
            "mean_tokens": self.mean_tokens,
            "mean_input_tokens": self.mean_input_tokens,
            "mean_output_tokens": self.mean_output_tokens,
            "mean_elapsed_seconds": self.mean_elapsed_seconds,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, skill_id: str, raw: Mapping[str, Any]) -> "SkillBelief":
        return cls(
            skill_id=str(raw.get("skill_id") or skill_id),
            alpha=float(raw.get("alpha", 1.0)),
            beta=float(raw.get("beta", 1.0)),
            contexts=dict(raw.get("contexts") or {}),
            failure_modes=dict(raw.get("failure_modes") or {}),
            evidence=list(raw.get("evidence") or []),
            observations=int(raw.get("observations") or 0),
            mean_tokens=float(raw.get("mean_tokens") or 0.0),
            mean_input_tokens=float(raw.get("mean_input_tokens") or 0.0),
            mean_output_tokens=float(raw.get("mean_output_tokens") or 0.0),
            mean_elapsed_seconds=float(raw.get("mean_elapsed_seconds") or 0.0),
            last_updated=str(raw.get("last_updated") or utc_now()),
        )
