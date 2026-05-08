# Adapters

Bayesian-Agent is designed to integrate with external agent harnesses without copying their code.

## Adapter Contract

An external harness should satisfy the `AgentAdapter` protocol:

```python
from typing import Any, Mapping, Protocol

class AgentAdapter(Protocol):
    def run(self, task: Mapping[str, Any], skill_context: str) -> Mapping[str, Any]:
        ...
```

The adapter receives:

- a task object from the external benchmark or application
- posterior-weighted Skill context from Bayesian-Agent

It returns:

- a trajectory-like mapping that can be converted into `TrajectoryEvidence`

## GenericAgent Boundary

The v0.4 GenericAgent adapter is intentionally thin:

```python
from bayesian_agent.adapters.generic_agent import GenericAgentAdapter

adapter = GenericAgentAdapter(root="/path/to/GenericAgent")
print(adapter.integration_note())
```

It does not eagerly import GenericAgent and does not vendor GenericAgent source code.

## Why This Boundary Matters

Bayesian-Agent should be usable with more than one agent framework. The durable contract is the trajectory schema, not a copied harness implementation.

External systems should emit:

- task identity
- success or failure outcome
- failure mode
- token usage
- runtime metadata

Bayesian-Agent can then update beliefs and render the next Skill context.

## MinimalAgent Status

MinimalAgent adapter support is intentionally not included in v0.4.

The recommended path is:

1. stabilize the GenericAgent boundary
2. keep the core trace schema portable
3. add more adapters only after the adapter contract has enough real usage
