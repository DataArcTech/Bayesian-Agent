# Artifacts

This directory contains result artifacts from the GenericAgent prototype validation.

- `ga_deepseek_baseline/`: original GA + `deepseek-v4-flash` SOP-Bench and Lifelong AgentBench runs.
- `bayesian_full/`: Bayesian self-evolving run from scratch.
- `bayesian_incremental/`: Bayesian incremental repair run using GA failures as evidence.

These artifacts are included so users can inspect the result format and reproduce summary calculations before the GenericAgent adapter is wired to their local environment.
