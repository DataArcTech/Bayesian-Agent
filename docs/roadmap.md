# Roadmap

Bayesian-Agent v0.5 is a native-first early release. The core package is usable for evidence ingestion, belief updates, context rendering, repair planning, result summarization, and first-party benchmark execution.

The roadmap is organized around the project's main advantage: Bayesian-Agent should support full self-evolution from scratch, incremental repair for existing agents, and cross-harness adaptation instead of becoming another isolated agent framework.

## Completed

- Refactored the GenericAgent prototype into a standalone package core.
- Defined a common trace schema for agent runs.
- Implemented the Bayesian Skill registry.
- Implemented full self-evolving primitives.
- Implemented incremental repair utilities.
- Added a GenericAgent optional adapter boundary without vendoring GenericAgent.
- Added the first-party Bayesian-Agent native harness.
- Added optional mini-swe-agent and Claude Code backend boundaries.
- Added opt-in automatic failure discovery and trajectory-to-Skill distillation while retaining catalog-first behavior by default.
- Released experiment result artifacts.
- Added bilingual README files.
- Added MkDocs documentation and GitHub Pages deployment.

## Next

- Harden executable benchmark runners for native and external backends.
- Add richer rewrite policies and adapter examples.
- Add adapters for more agent harnesses after the current compatibility boundaries stabilize.
- Add more examples for project-specific failure-mode taxonomies and automatic discovery overrides.
- Add documentation for operating Bayesian-Agent in a continuous evaluation pipeline.

## Bayesian Algorithm Direction

The current v0.5 implementation defaults to a per-Skill Bayesian Evidence Model with a categorical likelihood backend and keeps Beta-Bernoulli as an optional compatibility backend.

Future releases will move toward broader Bayesian reasoning:

- **Skill hypothesis inference**: compare, combine, and specialize competing Skill/SOP hypotheses with posterior evidence.
- **Bayesian Networks and graphical structure**: model dependencies among tasks, contexts, tools, failure modes, and Skills.
- **Uncertainty-aware online decisions**: choose selection, rewrite, rerun, retire, and transfer actions under uncertainty as harnesses, models, and tasks change.

## Non-Goals for v0.5

- Bayesian-Agent does not train or fine-tune base models.
- Bayesian-Agent does not replace GenericAgent.
- Bayesian-Agent does not copy or vendor GenericAgent.
- Bayesian-Agent is not limited to GenericAgent.
- MinimalAgent adapter support is not included yet.
