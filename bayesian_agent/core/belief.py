"""Belief state for Bayesian Skill/SOP hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from bayesian_agent.core.algorithms import DEFAULT_ALGORITHM, SUPPORTED_ALGORITHMS, is_categorical_bayes, is_frequentist, normalize_algorithm
from bayesian_agent.core.algorithms.beta_bernoulli import BetaBernoulliState
from bayesian_agent.core.algorithms.categorical_bayes import CategoricalBayesState
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
    algorithm: str = DEFAULT_ALGORITHM
    alpha: float = 1.0
    beta: float = 1.0
    categorical_bayes: CategoricalBayesState = field(default_factory=CategoricalBayesState)
    contexts: Dict[str, int] = field(default_factory=dict)
    failure_modes: Dict[str, int] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    observations: int = 0
    mean_tokens: float = 0.0
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_elapsed_seconds: float = 0.0
    last_updated: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.algorithm = normalize_algorithm(self.algorithm)
        if is_frequentist(self.algorithm) and self.observations == 0 and self.alpha == 1.0 and self.beta == 1.0:
            self.alpha = 0.0
            self.beta = 0.0

    @property
    def success_probability(self) -> float:
        if is_categorical_bayes(self.algorithm):
            return self.categorical_bayes.predict_success()
        if is_frequentist(self.algorithm):
            total = self.alpha + self.beta
            return self.alpha / total if total else 0.0
        return self.beta_state.success_probability

    @property
    def beta_state(self) -> BetaBernoulliState:
        return BetaBernoulliState(alpha=self.alpha, beta=self.beta)

    def predict_success_probability(self, context: str = "", features: Optional[Mapping[str, Any]] = None) -> float:
        if not is_categorical_bayes(self.algorithm):
            return self.success_probability
        merged = dict(features or {})
        if context:
            merged.setdefault("context", context)
        return self.categorical_bayes.predict_success(merged)

    def predict_failure_probability(self, context: str = "", features: Optional[Mapping[str, Any]] = None) -> float:
        if not is_categorical_bayes(self.algorithm):
            return 1.0 - self.success_probability
        merged = dict(features or {})
        if context:
            merged.setdefault("context", context)
        return self.categorical_bayes.predict_proba(merged).get("failure", 0.0)

    def update(self, event: TrajectoryEvidence) -> "SkillBelief":
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported belief algorithm: {self.algorithm}")
        self.algorithm = normalize_algorithm(self.algorithm)

        outcome = event.outcome.strip().lower()
        if outcome == "success":
            self.alpha += 1.0
        elif outcome in {"failure", "failed", "error"}:
            self.beta += 1.0
        if is_categorical_bayes(self.algorithm):
            self.categorical_bayes.update(event)

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
            "algorithm": normalize_algorithm(self.algorithm),
            "alpha": self.alpha,
            "beta": self.beta,
            "posterior_success": self.success_probability,
            "beta_bernoulli": self.beta_state.to_dict(),
            "frequentist": {
                "successes": float(self.alpha),
                "failures": float(self.beta),
                "success_rate": self.success_probability,
            } if is_frequentist(self.algorithm) else {},
            "categorical_bayes": self.categorical_bayes.to_dict() if is_categorical_bayes(self.algorithm) else {},
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
    def from_dict(cls, skill_id: str, raw: Mapping[str, Any], algorithm: Optional[str] = None) -> "SkillBelief":
        raw = dict(raw or {})
        resolved_algorithm = normalize_algorithm(str(raw.get("algorithm") or (algorithm if not raw else "beta_bernoulli") or DEFAULT_ALGORITHM))
        if resolved_algorithm not in SUPPORTED_ALGORITHMS:
            resolved_algorithm = DEFAULT_ALGORITHM
        default_count = 0.0 if is_frequentist(resolved_algorithm) else 1.0
        categorical_raw = raw.get("categorical_bayes") or raw.get("naive_bayes") or {}
        return cls(
            skill_id=str(raw.get("skill_id") or skill_id),
            algorithm=resolved_algorithm,
            alpha=float(raw.get("alpha", default_count)),
            beta=float(raw.get("beta", default_count)),
            categorical_bayes=CategoricalBayesState.from_dict(categorical_raw),
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
