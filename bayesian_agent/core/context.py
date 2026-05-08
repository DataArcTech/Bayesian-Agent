"""Posterior-weighted Skill context rendering."""

from __future__ import annotations

from bayesian_agent.core.policy import RewritePolicy
from bayesian_agent.core.registry import BayesianSkillRegistry


class SkillContextBuilder:
    """Render concise Skill/SOP context from posterior beliefs."""

    def __init__(self, registry: BayesianSkillRegistry, policy: RewritePolicy = None):
        self.registry = registry
        self.policy = policy or RewritePolicy()

    def render(self, task_context: str = "", limit: int = 5) -> str:
        beliefs = self.registry.top(limit=limit, context=task_context)
        if not beliefs:
            return ""
        lines = [
            "### Bayesian Skill Context",
            "Use these posterior-weighted Skills/SOPs as hypotheses, not as unquestioned instructions.",
        ]
        for belief in beliefs:
            decision = self.policy.decide(belief)
            failures = ", ".join(f"{k}={v}" for k, v in sorted(belief.failure_modes.items())[:3]) or "none"
            lines.append(
                "- "
                f"{belief.skill_id}: posterior_success={belief.success_probability:.3f}, "
                f"alpha={belief.alpha:.1f}, beta={belief.beta:.1f}, "
                f"observations={belief.observations}, mean_tokens={belief.mean_tokens:.1f}, "
                f"rewrite={decision.action}, failures={failures}"
            )
        lines.append("Current task files and runtime feedback remain authoritative.")
        return "\n".join(lines)
