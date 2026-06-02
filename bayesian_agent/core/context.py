"""Posterior audit rendering for Skill beliefs."""

from __future__ import annotations

from bayesian_agent.core.policy import RewritePolicy
from bayesian_agent.core.registry import BayesianSkillRegistry


class SkillContextBuilder:
    """Render concise posterior audit summaries from Skill beliefs."""

    def __init__(self, registry: BayesianSkillRegistry, policy: RewritePolicy = None):
        self.registry = registry
        self.policy = policy or RewritePolicy()

    def render(self, task_context: str = "", limit: int = 5, strict_context: bool = False) -> str:
        beliefs = self.registry.top(limit=limit, context=task_context, strict_context=strict_context)
        if not beliefs:
            return ""
        lines = [
            "### Bayesian Posterior Audit",
            "Posterior summaries are for ranking, rewrite decisions, and debugging; model-facing prompts should use executable Skill/SOP text.",
        ]
        for belief in beliefs:
            decision = self.policy.decide(belief)
            failures = ", ".join(f"{k}={v}" for k, v in sorted(belief.failure_modes.items())[:3]) or "none"
            context_success = belief.predict_success_probability(context=task_context) if task_context else belief.success_probability
            lines.append(
                "- "
                f"{belief.skill_id}: algorithm={belief.algorithm}, "
                f"posterior_success={belief.success_probability:.3f}, "
                f"context_success={context_success:.3f}, "
                f"alpha={belief.alpha:.1f}, beta={belief.beta:.1f}, "
                f"observations={belief.observations}, mean_tokens={belief.mean_tokens:.1f}, "
                f"rewrite={decision.action}, failures={failures}"
            )
        lines.append("Current task files and runtime feedback remain authoritative.")
        return "\n".join(lines)
