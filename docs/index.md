# Bayesian-Agent

<div align="center">
  <img src="assets/banner.png" width="920" alt="Bayesian-Agent banner"/>
</div>

Bayesian-Agent is a standalone Bayesian self-evolving agent framework for turning verified agent trajectories into reusable, evidence-weighted Skills and SOPs.

The project focuses on the inference side of agent improvement. Instead of changing base model parameters, it changes the agent's inference environment by maintaining posterior beliefs over Skills, failure modes, token cost, and context-specific reliability.

<div align="center">
  <img src="assets/bayesian_agent_overview.png" width="900" alt="Bayesian-Agent overview"/>
  <br/>
  <em>Verified trajectories become posterior-weighted Skills and SOPs.</em>
</div>

## Why It Exists

LLM agent engineering has moved through three layers:

1. **Prompt Engineering**: make task instructions clearer.
2. **Context Engineering**: decide what evidence the model can see.
3. **Harness Engineering**: put the model inside a runnable, observable, recoverable system.

Skills and SOPs are the durable memory of that harness. Bayesian-Agent makes their evolution evidence-driven:

```text
Trajectory -> Verifier -> Evidence -> Posterior Skill Belief -> Better Context -> Next Run
```

## What v0.4 Includes

- Bayesian Skill registry with Beta posterior updates.
- Evidence schema for agent trajectories.
- Posterior-weighted Skill context rendering.
- Failure-mode-aware repair planning.
- CLI utilities for trace ingestion, summarization, and incremental repair.
- GenericAgent integration boundary without copying or vendoring GenericAgent.
- SOP-Bench and Lifelong AgentBench result artifacts.

## Install

```bash
git clone https://github.com/DataArcTech/Bayesian-Agent.git
cd Bayesian-Agent
python -m pip install -e .
```

For documentation development:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

## Next Steps

- Start with the [Quick Start](quick-start.md).
- Read the [Core Concepts](core-concepts.md) if you want the Bayesian framing.
- Use the [CLI](cli.md) to update a registry from traces.
- Inspect [Experiments](experiments/index.md) for the GenericAgent + `deepseek-v4-flash` validation.
