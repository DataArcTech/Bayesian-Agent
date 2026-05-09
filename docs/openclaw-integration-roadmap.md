# OpenClaw Integration Roadmap

This roadmap sketches how Bayesian-Agent can become useful for persistent assistant systems such as OpenClaw while staying framework-agnostic.

## Why OpenClaw Is a Good Testbed

OpenClaw-style assistants repeatedly execute durable workflows: coding repair, research review, browser tasks, grading support, inbox triage, and project maintenance. These workflows naturally produce trajectories, verifier signals, and reusable procedures.

Bayesian-Agent can help by turning those repeated procedures into evidence-weighted Skills instead of relying on unfiltered memory or anecdotal prompt edits.

## Target Use Cases

1. **Workflow reliability**
   - Track which SOPs actually succeed for recurring assistant tasks.
   - Prefer high-posterior, low-cost procedures for similar future contexts.

2. **Failure-mode learning**
   - Count recurring errors such as missing tests, stale browser refs, premature grading, or unsafe external actions.
   - Route clustered failures to `patch`, `split`, or `retire` decisions.

3. **Incremental repair**
   - When a batch task partially fails, rerun only the failed units with posterior-weighted context.
   - Useful for grading batches, code-review batches, literature-screening batches, and benchmark tasks.

4. **Cross-harness memory discipline**
   - Keep the Bayesian registry separate from the execution harness.
   - Let OpenClaw, CLI agents, and future harnesses share the same evidence format.

## Minimal Adapter Shape

An OpenClaw adapter should export verified run records into `TrajectoryEvidence` fields:

- `task_id`: stable id for the workflow unit, such as `grading/assignment/student_id` or `repo/pr/check_id`.
- `skill_id`: SOP or skill used, such as `openclaw/grading/rubric_feedback`.
- `context`: coarse task family, such as `grading`, `coding`, `browser`, `research`.
- `outcome`: verifier result: `success`, `failure`, or `error`.
- `failure_mode`: short normalized failure label.
- token/runtime metadata where available.

## First Contribution Thread

The first practical improvement is uncertainty visibility. Posterior mean alone can over-rank sparse evidence. Rendering posterior standard deviation helps downstream harnesses distinguish:

- high-confidence successful skills;
- promising but under-tested skills;
- unstable skills that need exploration or splitting.

## Suggested Next PRs

1. Add posterior uncertainty fields and render them in context. *(small, self-contained)*
2. Add configurable ranking strategies: exploit, explore, cost-aware, and context-match.
3. Add an `openclaw` example exporter that converts a small JSONL task log into `TrajectoryEvidence`.
4. Add richer repair-plan output with failure-mode clusters, not only failed task ids.
5. Add reproducible benchmark runner scripts for published artifacts.
