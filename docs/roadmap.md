# Roadmap

Bayesian-Agent v0.4 is an early standalone release. The core package is usable for evidence ingestion, belief updates, context rendering, repair planning, and result summarization.

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

## Non-Goals for v0.4

- Bayesian-Agent does not train or fine-tune base models.
- Bayesian-Agent does not replace GenericAgent.
- Bayesian-Agent does not copy or vendor GenericAgent.
- MinimalAgent adapter support is not included yet.
