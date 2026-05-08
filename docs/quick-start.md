# Quick Start

This guide shows the shortest path from installation to a posterior-weighted Skill context.

## Install

```bash
git clone https://github.com/DataArcTech/Bayesian-Agent.git
cd Bayesian-Agent
python -m pip install -e .
```

Bayesian-Agent v0.4 has no runtime dependencies beyond the Python standard library.

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
- optionally renders reusable Skill context for a future run

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

Rendered context is intentionally short:

```text
### Bayesian Skill Context
Use these posterior-weighted Skills/SOPs as hypotheses, not as unquestioned instructions.
- benchmark/sop_bench: posterior_success=0.333, alpha=1.0, beta=2.0, observations=1, mean_tokens=74365.0, rewrite=explore, failures=xml_wrapped_answer=1
Current task files and runtime feedback remain authoritative.
```
