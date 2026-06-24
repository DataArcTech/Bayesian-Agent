# Quick Start

This guide shows the shortest path from installation to a Bayesian Skill registry and its audit/context outputs.

Bayesian-Agent supports three paths:

- run the first-party native harness and evolve Skills during a full run
- repair only failed tasks from an existing agent run
- adapt the same Bayesian Skill registry to another harness through an adapter

## Install

```bash
git clone https://github.com/DataArcTech/Bayesian-Agent.git
cd Bayesian-Agent
python -m pip install -e .
```

The core package has no runtime dependencies beyond the Python standard library.

## Run The Native Harness

The benchmark runner defaults to the Bayesian-Agent harness. It owns the LLM loop, workspace tools, optional three-layer memory, and trajectory capture:

```bash
export DEEPSEEK_API_KEY="sk-..."
python experiments/run_benchmarks.py \
  --harness bayesian-agent \
  --model deepseek-v4-flash \
  --bench core \
  --mode all \
  --limit 1
```

Use `--bench realfin` for RealFin-Bench. External compatibility backends remain available with `--harness genericagent`, `--harness mini-swe-agent`, or `--harness claude-code`.

Native memory prompt injection and hippocampus/state updates are off by default. Add `--native-memory` to enable them. Bayesian Skill registry updates remain enabled in Bayesian modes either way.

## Update a Skill Registry

Use `evolve` to ingest one or more result files and update a persistent Bayesian Skill registry:

```bash
bayesian-agent evolve \
  --results artifacts/ga_deepseek_baseline/sop_results.json \
  --registry temp/bayesian_skill_beliefs.json \
  --context-out temp/skill_context.md
```

This command:

- reads benchmark or agent run traces
- converts each run into `TrajectoryEvidence`
- updates the corresponding Skill posterior
- optionally renders a posterior audit context for inspection or custom adapter use

## Plan Incremental Repair

Use `repair-plan` to extract failed task ids from a baseline run:

```bash
bayesian-agent repair-plan \
  --baseline artifacts/ga_deepseek_baseline/sop_results.json \
  --out temp/failed_tasks.json
```

This is useful when Bayesian-Agent is attached to another agent as a repair layer.

## Summarize Results

```bash
bayesian-agent summarize \
  --results artifacts/bayesian_incremental/results.json \
  --out temp/summary.json
```

For baseline plus repair traces:

```bash
bayesian-agent incremental-summary \
  --baseline artifacts/ga_deepseek_baseline/sop_results.json \
  --repairs artifacts/bayesian_incremental/results.json \
  --out temp/incremental_summary.json
```

## Python Example

```python
from bayesian_agent import BayesianSkillRegistry, SkillContextBuilder, TrajectoryEvidence

registry = BayesianSkillRegistry("temp/beliefs.json")
registry.record(
    TrajectoryEvidence(
        task_id="sop_12",
        skill_id="benchmark/sop_bench",
        context="sop_bench",
        outcome="failure",
        failure_mode="xml_wrapped_answer",
        input_tokens=70123,
        output_tokens=4242,
    )
)

print(SkillContextBuilder(registry).render(task_context="sop_bench"))
```

## Expected Output Shape

The generic `SkillContextBuilder` rendering is intentionally short. Treat it as an audit/debug view of the posterior state unless your adapter explicitly wants this format:

```text
### Bayesian Posterior Audit
Posterior summaries are for ranking, rewrite decisions, and debugging; model-facing prompts should use executable Skill/SOP text.
- benchmark/sop_bench: algorithm=categorical_bayes, posterior_success=0.333, context_success=0.333, alpha=1.0, beta=2.0, observations=1, mean_tokens=74365.0, rewrite=explore, failures=xml_wrapped_answer=1
Current task files and runtime feedback remain authoritative.
```
