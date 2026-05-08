# Adapters

Bayesian-Agent is designed to integrate with external agent harnesses without copying their code. This is one of the main reasons the project is not just another agent framework: the Bayesian layer can improve whichever harness emits verified trajectories.

## Adaptation Advantage

Bayesian-Agent separates Skill evolution from task execution:

```text
Harness executes -> Bayesian-Agent learns -> Adapter injects posterior Skill context -> Harness reruns
```

That separation enables three deployment styles:

- run a full benchmark from scratch with Bayesian Skill evolution enabled
- repair only the failed tasks from an existing agent run
- reuse the same Skill belief registry across compatible harnesses

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

## Planned Bayesian-Agent Harness

Current experiments use GenericAgent as the backend harness. A dedicated Bayesian-Agent harness is planned so users can run the full loop without depending on GenericAgent, while still keeping GenericAgent and other frameworks as optional backends.

## MinimalAgent Status

MinimalAgent adapter support is intentionally not included in v0.4.

The recommended path is:

1. stabilize the GenericAgent boundary
2. keep the core trace schema portable
3. upload the dedicated Bayesian-Agent harness
4. add more adapters only after the adapter contract has enough real usage
