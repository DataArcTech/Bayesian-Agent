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

## Bayesian Algorithm Direction

The current v0.x implementation uses per-Skill Beta-Bernoulli posterior updates:

```text
p_k | D_k ~ Beta(alpha_0 + s_k, beta_0 + f_k)
```

Future releases will move toward broader Bayesian reasoning:

- **Richer Skill hypothesis inference**: compare, combine, and specialize competing Skill/SOP hypotheses with posterior evidence.
- **Context-aware Bayesian structure**: model relationships among tasks, contexts, tools, failure modes, and Skills, including Bayesian Networks.
- **Uncertainty-aware Skill selection**: use posterior uncertainty to balance exploration, exploitation, and repair cost.
- **Bayesian decision policies**: choose rewrite, rerun, retire, or transfer actions by expected utility over accuracy, token cost, latency, and risk.
- **Online adaptation**: detect distribution shift and update Skill beliefs continuously as harnesses, models, and tasks change.

## Non-Goals for v0.4

- Bayesian-Agent does not train or fine-tune base models.
- Bayesian-Agent does not replace GenericAgent.
- Bayesian-Agent does not copy or vendor GenericAgent.
- Bayesian-Agent is not limited to GenericAgent.
- MinimalAgent adapter support is not included yet.
