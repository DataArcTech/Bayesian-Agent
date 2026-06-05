"""Optional agent adapters."""

from bayesian_agent.adapters.base import AgentAdapter
from bayesian_agent.adapters.bayesian_agent import NativeBayesianAgentAdapter
from bayesian_agent.adapters.claude_code import ClaudeCodeAdapter
from bayesian_agent.adapters.generic_agent import GenericAgentAdapter
from bayesian_agent.adapters.mini_swe_agent import MiniSWEAgentAdapter

__all__ = [
    "AgentAdapter",
    "ClaudeCodeAdapter",
    "GenericAgentAdapter",
    "MiniSWEAgentAdapter",
    "NativeBayesianAgentAdapter",
]
