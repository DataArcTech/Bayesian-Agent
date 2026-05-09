"""Skill ranking strategies for posterior-weighted context selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from bayesian_agent.core.belief import SkillBelief


@dataclass(frozen=True)
class RankingStrategy:
    """A named strategy for ranking Skill beliefs."""

    name: str
    description: str
    scorer: Callable[[SkillBelief, str], float]

    def score(self, belief: SkillBelief, context: str = "") -> float:
        return float(self.scorer(belief, context))


def _context_bonus(belief: SkillBelief, context: str) -> float:
    if not context:
        return 0.0
    if context in belief.contexts:
        return 1.0
    # Lightweight partial match for hierarchical contexts such as "openclaw/grading".
    return 0.25 if any(context in known or known in context for known in belief.contexts) else 0.0


def _safe_mean_tokens(belief: SkillBelief) -> float:
    return max(float(belief.mean_tokens or 0.0), 1.0)


def exploit_score(belief: SkillBelief, context: str = "") -> float:
    """Prefer proven, context-matching, low-uncertainty Skills."""

    return belief.success_probability + (0.15 * _context_bonus(belief, context)) - (0.25 * belief.posterior_std)


def explore_score(belief: SkillBelief, context: str = "") -> float:
    """Prefer uncertain Skills with some contextual relevance."""

    return belief.posterior_std + (0.10 * _context_bonus(belief, context)) + min(belief.observations, 3) * 0.01


def cost_aware_score(belief: SkillBelief, context: str = "") -> float:
    """Prefer success per token, while retaining a small context bonus."""

    return (belief.success_probability / _safe_mean_tokens(belief)) * 1000.0 + (0.10 * _context_bonus(belief, context))


def context_aware_score(belief: SkillBelief, context: str = "") -> float:
    """Prefer Skills proven in the same or nearby task context."""

    return belief.success_probability + (0.35 * _context_bonus(belief, context)) - (0.10 * belief.posterior_std)


STRATEGIES: Dict[str, RankingStrategy] = {
    "exploit": RankingStrategy("exploit", "Prefer proven, low-uncertainty Skills.", exploit_score),
    "explore": RankingStrategy("explore", "Prefer Skills that need more evidence.", explore_score),
    "cost_aware": RankingStrategy("cost_aware", "Prefer high-success, low-token Skills.", cost_aware_score),
    "context_aware": RankingStrategy("context_aware", "Prefer Skills proven in similar contexts.", context_aware_score),
}


def get_strategy(name: str = "exploit") -> RankingStrategy:
    """Return a ranking strategy by name."""

    normalized = (name or "exploit").strip().lower().replace("-", "_")
    if normalized not in STRATEGIES:
        available = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown ranking strategy '{name}'. Available: {available}")
    return STRATEGIES[normalized]
