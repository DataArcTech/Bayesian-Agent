# Architecture

Bayesian-Agent is intentionally small. The framework core is independent from any specific agent harness, so the same Bayesian Skill/SOP evolution loop can support full runs, incremental repair, and cross-harness adaptation.

<div align="center">
  <img src="assets/bayesian_agent_framework_v2.svg" width="900" alt="Bayesian-Agent framework"/>
  <br/>
  <em>Bayesian Skill Evolution framework.</em>
</div>

## Data Flow

```text
Any Compatible Harness Run
      |
      v
TrajectoryEvidence
      |
      v
BayesianSkillRegistry
      |
      v
SkillBelief + RewritePolicy
      |
      v
SkillContextBuilder
      |
      v
Adapter
      |
      v
Any Compatible Harness Next Run
```

## Package Layout

```text
bayesian_agent/
  core/
    evidence.py    # TrajectoryEvidence model
    belief.py      # SkillBelief and RewriteDecision
    registry.py    # JSON-backed BayesianSkillRegistry
    policy.py      # default rewrite policy
    context.py     # posterior-weighted context renderer
    repair.py      # result normalization and repair summaries
  adapters/
    base.py        # AgentAdapter protocol
    generic_agent.py
  cli.py
```

## Core Boundaries

`bayesian_agent.core` is framework-agnostic. It knows nothing about GenericAgent, benchmark runners, browser tools, or model APIs.

`bayesian_agent.adapters` defines how external harnesses can connect. The GenericAgent adapter in v0.4 is intentionally a boundary placeholder, not a vendored copy of GenericAgent.

This separation is what prevents Bayesian-Agent from being swallowed by the agent framework category. It is a reusable Bayesian evolution layer that can sit beside multiple harnesses rather than competing with all of them as another monolithic runtime.

`schemas/` defines portable JSON shapes for trajectories and Skill beliefs.

`artifacts/` contains result files from the initial GenericAgent validation. GenericAgent is the current experimental harness; a dedicated Bayesian-Agent harness is planned.

## Persistence Model

`BayesianSkillRegistry` persists beliefs as JSON:

```python
registry = BayesianSkillRegistry("temp/beliefs.json")
registry.record(event)
registry.save()
```

The registry can also run in memory:

```python
registry = BayesianSkillRegistry.in_memory()
```

## Context Rendering

`SkillContextBuilder` selects top posterior beliefs and renders concise context:

```python
context = SkillContextBuilder(registry).render(task_context="sop_bench", limit=5)
```

The rendered context tells the downstream agent to treat Skills as hypotheses rather than unquestioned instructions.
