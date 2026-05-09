"""Optional agent adapters."""

from bayesian_agent.adapters.base import AgentAdapter
from bayesian_agent.adapters.generic_agent import GenericAgentAdapter
from bayesian_agent.adapters.workflow_log import evidence_from_jsonl, workflow_record_to_evidence

__all__ = ["AgentAdapter", "GenericAgentAdapter", "evidence_from_jsonl", "workflow_record_to_evidence"]
