# Roadmap

Bayesian-Agent v0.4 is an early standalone release. The core package is usable for evidence ingestion, belief updates, context rendering, repair planning, and result summarization.

The roadmap is organized around the project's main advantage: Bayesian-Agent should support full self-evolution from scratch, incremental repair for existing agents, and cross-harness adaptation instead of becoming another isolated agent framework.

## Completed

- Refactored the GenericAgent prototype into a standalone package core.
- Defined a common trace schema for agent runs.
- Implemented the Bayesian Skill registry.
- Implemented full self-evolving primitives.
- Implemented incremental repair utilities.
- Added a GenericAgent optional adapter boundary without vendoring GenericAgent.
- Released experiment result artifacts.
- Added bilingual README files.
- Added MkDocs documentation and GitHub Pages deployment.

## Next

- Add executable benchmark runners for external checkouts.
- Add richer rewrite policies and adapter examples.
- Add adapters for more agent harnesses after the GenericAgent boundary stabilizes.
- Add more examples for project-specific failure-mode taxonomies.
- Add documentation for operating Bayesian-Agent in a continuous evaluation pipeline.
- Upload our own Bayesian-Agent harness. Current experiments use GenericAgent as the backend harness, but the project will provide a first-party harness for users who want the complete loop out of the box.

## Bayesian Algorithm Roadmap

The current v0.x implementation uses per-Skill Beta-Bernoulli posterior updates:

```text
p_k | D_k ~ Beta(alpha_0 + s_k, beta_0 + f_k)
```

Future releases will move toward more complete Bayesian inference:

- **Posterior model selection over Skill hypotheses**: compare competing Skills directly with `P(h_k | D) ∝ P(D | h_k)P(h_k)`.
- **Hierarchical and contextual Bayesian reliability**: share evidence across related contexts while still learning context-specific Skill performance.
- **Thompson sampling and Bayesian bandits**: balance exploration and exploitation when choosing which Skill/SOP variant to inject.
- **Bayesian decision theory**: select actions by expected utility, jointly considering accuracy, token cost, latency, and repair risk.
- **Dirichlet-Multinomial failure-mode modeling**: estimate distributions over recurring failure modes instead of simple counts.
- **Bayesian optimization for prompt/SOP variants**: search rewrite variants with sample-efficient uncertainty-aware optimization.
- **Bayesian model averaging**: combine multiple compatible Skill variants when posterior uncertainty remains high.
- **Sequential Bayesian drift detection**: detect when a once-reliable Skill degrades because the task distribution or harness behavior changed.

## Non-Goals for v0.4

- Bayesian-Agent does not train or fine-tune base models.
- Bayesian-Agent does not replace GenericAgent.
- Bayesian-Agent does not copy or vendor GenericAgent.
- Bayesian-Agent is not limited to GenericAgent.
- MinimalAgent adapter support is not included yet.
