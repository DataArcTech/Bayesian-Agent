# Core Concepts

Bayesian-Agent treats Skill evolution as Bayesian inference over operational hypotheses. Its core contribution is not another closed agent loop, but a portable evolution layer that can run from scratch, repair existing agents incrementally, and adapt across harnesses.

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

Bayesian-Agent improves `C`. It does not train or fine-tune the base model, and it does not require replacing the user's existing agent framework.

## Skill as Hypothesis

A Skill or SOP is a hypothesis about how to make an agent succeed under a task context:

```text
P(success | theta, C, skill)
```

The same Skill may work in one context and fail in another. That is why Bayesian-Agent records both outcomes and context distribution.

## Current Bayesian Assumption

Bayesian-Agent v0.x models each Skill/SOP independently with a Beta-Bernoulli posterior:

```text
p_k = P(success | h_k, context)
p_k ~ Beta(alpha_0, beta_0)
y_i ~ Bernoulli(p_k)
```

After `s_k` verified successes and `f_k` verified failures:

```text
p_k | D_k ~ Beta(alpha_0 + s_k, beta_0 + f_k)
posterior_success = E[p_k | D_k]
                  = (alpha_0 + s_k) / (alpha_0 + beta_0 + s_k + f_k)
```

This is a lightweight conjugate Bayesian update. It should not be confused with a full Bayesian model-selection layer over multiple competing Skill hypotheses:

```text
P(h_k | D) ∝ P(D | h_k) P(h_k)
```

Full model selection is on the roadmap.

## Three Operating Patterns

Bayesian-Agent is meant to be used in three complementary ways:

| Pattern | What it does | Why it matters |
|---|---|---|
| Full self-evolution | Runs tasks from scratch and updates Skill beliefs online. | Tests whether Skills can emerge without prior traces. |
| Incremental repair | Reads baseline failures and reruns only failed tasks. | Improves existing agents with small additional inference cost. |
| Cross-harness adaptation | Uses a common trajectory schema and adapters. | Lets Bayesian Skill evolution move across agent frameworks. |

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
