# Architecture

Bayesian-Agent is intentionally small. The framework core is independent from any specific external agent harness, so the same Bayesian Skill/SOP evolution loop can support first-party native runs, GenericAgent-backed runs, incremental repair, and cross-harness adaptation.

<div align="center">
  <img src="assets/bayesian_agent_framework_v2.svg" width="900" alt="Bayesian-Agent framework"/>
  <br/>
  <em>Bayesian Skill Evolution framework.</em>
</div>

## Data Flow

```text
Native BA Harness or Compatible External Harness
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
Native Harness / Adapter
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
    context.py     # posterior audit and Skill context rendering
    repair.py      # result normalization and repair summaries
  harness/
    native.py      # first-party LLM/tool turn loop
    llm.py         # OpenAI-compatible chat client
    tools.py       # workspace-scoped tools
    core.py        # task envelope, artifacts, optional three-layer memory bridge
  memory/
    layers.py      # hippocampus, intermediate state, cortex
  adapters/
    base.py        # AgentAdapter protocol
    bayesian_agent.py
    generic_agent.py
    mini_swe_agent.py
    claude_code.py
  cli.py
```

## Core Boundaries

`bayesian_agent.core` is framework-agnostic. It knows nothing about GenericAgent, benchmark runners, browser tools, or model APIs.

`bayesian_agent.harness` contains the first-party native harness. It runs the OpenAI-compatible LLM loop, dispatches workspace tools, captures trajectories, and can optionally bridge into the three-layer memory system. Native memory is disabled by default; verified outcomes still update the Bayesian Skill registry.

`bayesian_agent.adapters` defines how external harnesses can connect. GenericAgent, mini-swe-agent, and Claude Code are optional compatibility backends, not vendored runtimes.

This separation is what prevents Bayesian-Agent from being swallowed by the agent framework category. It is a reusable Bayesian evolution layer that can sit beside multiple harnesses rather than competing with all of them as another monolithic runtime.

`schemas/` defines portable JSON shapes for trajectories and Skill beliefs.

`artifacts/` contains result files from the initial GenericAgent validation. `results/native_harness_deepseek_v4_flash_full/` and `results/native_harness_deepseek_v4_pro_full/` contain local full-sample results from the first-party native harness.

## Persistence Model

`BayesianSkillRegistry` persists beliefs as JSON:

```python
registry = BayesianSkillRegistry("temp/beliefs.json", algorithm="categorical_bayes")
registry.record(event)
registry.save()
```

The registry can also run in memory:

```python
registry = BayesianSkillRegistry.in_memory()
```

Use `algorithm="beta_bernoulli"` for the optional legacy global success-rate posterior.

## Context Rendering

`SkillContextBuilder` selects top posterior beliefs and renders concise context:

```python
context = SkillContextBuilder(registry).render(task_context="sop_bench", limit=5)
```

The rendered context tells the downstream agent to treat Skills as hypotheses rather than unquestioned instructions.
