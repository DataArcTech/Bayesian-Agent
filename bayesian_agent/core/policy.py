"""Rewrite policy for Bayesian Skill/SOP evolution."""

from __future__ import annotations

from bayesian_agent.core.belief import RewriteDecision, SkillBelief


class RewritePolicy:
    """Map posterior belief state to Skill rewrite actions."""

    def decide(self, belief: SkillBelief) -> RewriteDecision:
        p = belief.success_probability
        if belief.observations == 0:
            return RewriteDecision("explore", "no verified evidence yet", confidence=0.1)
        if belief.beta >= 4 and p < 0.45:
            return RewriteDecision("retire", "posterior failures dominate", confidence=min(0.95, belief.beta / (belief.alpha + belief.beta)))
        if belief.failure_modes and max(belief.failure_modes.values()) >= 2:
            return RewriteDecision("patch", "failures cluster around a recurring mode", confidence=0.75)
        if len(belief.contexts) >= 3 and belief.observations >= 4:
            return RewriteDecision("split", "evidence spans multiple contexts", confidence=0.65)
        if belief.observations >= 3 and p >= 0.72:
            return RewriteDecision("compress", "success evidence is stable", confidence=p)
        return RewriteDecision("explore", "posterior remains uncertain", confidence=0.35)
