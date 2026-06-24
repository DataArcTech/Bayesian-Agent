# Bayesian-Agent: A Bayesian Self-Evolving Agent Framework with Cross-Harness Adaptation

<div align="center">
  <img src="assets/banner.png" width="920" alt="Bayesian-Agent banner"/>
</div>

<p align="center">
  📚 <a href="https://dataarctech.github.io/Bayesian-Agent/">Docs</a> |
  🐙 <a href="https://github.com/DataArcTech/Bayesian-Agent">GitHub</a> |
  📄 <a href="http://arxiv.org/abs/2606.08348">arXiv:2606.08348</a>
</p>

Bayesian-Agent is a Bayesian self-evolving layer for turning verified agent trajectories into reusable, evidence-weighted Skills and SOPs across agent frameworks and execution harnesses.

The project focuses on the inference side of agent improvement. Instead of changing base model parameters, it changes the agent's inference environment by maintaining posterior beliefs over Skills, failure modes, token cost, and context-specific reliability.

In v0.5, the default Bayesian core is a per-Skill **Bayesian Evidence Model** over verified success/failure evidence and context/runtime features. The current implementation uses a categorical likelihood backend exposed as `categorical_bayes`; the old `naive_bayes` name remains a compatibility alias. The earlier Beta-Bernoulli update remains available as an optional ablation backend. Fuller Bayesian model selection and uncertainty-aware Skill selection are planned in the roadmap.

Agent runs are expensive: tokens are expensive, latency is high, benchmark cases are limited, and real production failures are even rarer. When samples are scarce, each sample is costly, and we cannot wait for large-sample statistics to stabilize, Bayesian modeling lets Bayesian-Agent combine prior belief, uncertainty, and new verified evidence into more stable Skill rewrite decisions.

Bayesian-Agent is designed to avoid being just another monolithic agent framework:

- **Full-run evolution from scratch**: run tasks without prior traces and evolve Skills online.
- **Incremental repair**: attach to an existing agent, learn from failed trajectories, and rerun only failed tasks.
- **Native-first, cross-harness adaptation**: run with the lightweight BA native harness, or integrate with GenericAgent and other agent frameworks through adapters.

## News

- **2026-06-09:** The Bayesian-Agent paper is now available on arXiv: [arXiv:2606.08348](http://arxiv.org/abs/2606.08348).

<div align="center">
  <img src="assets/bayesian_agent_overview.png" width="900" alt="Bayesian-Agent overview"/>
  <br/>
  <em>Verified trajectories from compatible harnesses become evidence-ranked Skills and executable SOP patches.</em>
</div>

## Why It Exists

LLM agent engineering has moved through three layers:

1. **Prompt Engineering**: make task instructions clearer.
2. **Context Engineering**: decide what evidence the model can see.
3. **Harness Engineering**: put the model inside a runnable, observable, recoverable system.

Skills and SOPs are the durable memory of a harness. Bayesian-Agent makes their evolution evidence-driven and portable:

```text
Trajectory -> Verifier -> Evidence -> Posterior Skill Belief -> Better Context -> Next Run
```

## What v0.5 Includes

- Bayesian Skill registry with Bayesian Evidence Model updates and optional Beta-Bernoulli updates.
- Evidence schema for agent trajectories.
- Posterior-weighted Skill context rendering.
- Failure-mode-aware repair planning.
- First-party native harness with a minimal LLM loop, workspace tools, optional three-layer memory, and trajectory capture. Native memory is disabled by default; enable it with `--native-memory`.
- CLI utilities for trace ingestion, summarization, and incremental repair.
- GenericAgent, mini-swe-agent, and Claude Code integration boundaries without copying or vendoring those runtimes.
- Three operating patterns: full self-evolution, incremental repair, and cross-harness adaptation.
- SOP-Bench, Lifelong AgentBench, and RealFin result artifacts.

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
- Read [Why Bayesian for Skill Evolution](articles/why-bayesian-for-skill-evolution.md) for the small-sample, cost-sensitive motivation.
- Read [Native Harness](native-harness.md) for the first-party harness design.
- Use the [CLI](cli.md) to update a registry from traces.
- Read [Adapters](adapters.md) to understand how Bayesian-Agent moves across harnesses.
- Inspect [Experiments](experiments/index.md) for native-harness full-sample results and GA-backed validation.
