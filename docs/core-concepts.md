# Core Concepts

Bayesian-Agent treats Skill evolution as Bayesian inference over operational hypotheses.

## Inference Environment

A base model samples from:

```text
P(X | theta)
```

An agent system samples from:

```text
P(X | theta, C)
```

`theta` is the base model parameter state. `C` is the inference environment: prompts, tools, memory, retrieved context, Skills, SOPs, benchmark traces, verifier feedback, and runtime constraints.

Bayesian-Agent improves `C`. It does not train or fine-tune the base model.

## Skill as Hypothesis

A Skill or SOP is a hypothesis about how to make an agent succeed under a task context:

```text
P(success | theta, C, skill)
```

The same Skill may work in one context and fail in another. That is why Bayesian-Agent records both outcomes and context distribution.

## Trajectory Evidence

Each agent run should emit verified evidence:

- task id
- skill id
- task context
- success or failure outcome
- input, output, and total tokens
- turns and elapsed time
- failure mode
- summary and metadata

Evidence should come from a benchmark grader, test suite, deterministic checker, or other action-grounded verifier.

## Posterior Belief

Each Skill uses a Beta posterior:

```text
success: alpha += 1
failure: beta += 1
posterior_success = alpha / (alpha + beta)
```

The registry also tracks mean token cost, failure modes, and context counts.

## Rewrite Policy

The default policy maps posterior state to small, inspectable actions:

| Signal | Action |
|---|---|
| no evidence | `explore` |
| stable success | `compress` |
| repeated failure mode | `patch` |
| mixed contexts | `split` |
| dominant failures | `retire` |

These actions are recommendations. External harnesses decide how to rewrite, rerun, or retire Skills.
