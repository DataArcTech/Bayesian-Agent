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

Bayesian-Agent v0.5 models each Skill/SOP independently. The default backend is a feature-conditioned **Bayesian Evidence Model** over verified success/failure labels. Its current implementation is a categorical likelihood model:

```text
D_k = {(x_i, y_i)}
P(y | h_k) = (N_y + alpha) / (N + alpha * |Y|)
P(x_j = v | y, h_k) = (N_{j,v,y} + alpha) / (N_{j,y} + alpha * |V_j|)
P(success | h_k, x) ∝ P(success | h_k) * Π_j P(x_j | success, h_k)
```

`x_i` includes five fixed likelihood terms plus optional short metadata terms:

| Term | Purpose |
|---|---|
| `context` | Models benchmark, task family, or harness-specific context. |
| `failure_mode` | Records repeated, actionable error patterns. |
| `token_bucket` | Models token efficiency and search-heavy trajectories. |
| `turn_bucket` | Models interaction complexity and recovery loops. |
| `latency_bucket` | Models slow tool, data, or API paths. |
| `metadata.*` | Adds harness-specific short scalar diagnostics. |

So the current model has `5 fixed terms + 0..N metadata terms`, not a fixed total feature count. Metadata is included only for short scalar values (`str`, `int`, `float`, `bool`, with `len(str(value)) <= 80`). Runtime values are bucketed to keep the categorical likelihood stable under small online datasets.

The implementation uses `alpha = 1` Laplace smoothing. The public algorithm name is `categorical_bayes`; `naive_bayes` is accepted as a legacy alias for the same factorized categorical likelihood.

The earlier Beta-Bernoulli backend remains available as an optional global success-rate model:

```text
p_k | D_k ~ Beta(alpha_0 + s_k, beta_0 + f_k)
posterior_success = E[p_k | D_k]
```

Both are lightweight Bayesian updates. They should not be confused with a full Bayesian model-selection layer over multiple competing Skill hypotheses:

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

Each Skill stores the selected belief algorithm and its posterior state:

```text
algorithm = categorical_bayes  # default Bayesian Evidence Model
algorithm = beta_bernoulli     # optional global success-rate baseline
```

The registry also tracks mean token cost, failure modes, and context counts.

## Rewrite Policy

The default policy maps posterior state to five small, inspectable actions:

| Action | Current trigger | Rationale |
|---|---|---|
| `explore` | no observations, or no stronger rule fires | Keep collecting evidence while the posterior is sparse or uncertain. |
| `retire` | `beta >= 4` and `success_probability < 0.45` | Avoid retiring after one or two unlucky failures, but remove clearly harmful Skills. |
| `patch` | one failure mode appears at least twice | Convert recurring failures into concrete guardrails while avoiding one-off overfitting. |
| `split` | at least 3 contexts and at least 4 observations | Separate broad Skills when one SOP spans incompatible contexts. |
| `compress` | at least 3 observations and `success_probability >= 0.72` | Distill stable Skills to reduce future token cost. |

The checks run in implementation order: no evidence, retire, patch, split, compress, then fallback explore. These actions are recommendations. External harnesses decide how to rewrite, rerun, or retire Skills.

The bundled SOP-Bench and Lifelong runners implement one concrete `patch` behavior: recurring known failure modes are converted into short failure-mode-specific guardrails in the next prompt. A single failure is recorded in `belief_*.json` and `posterior_context_*.md` as candidate evidence, but it is not promoted into model-facing patch text until the same failure mode has at least two verified occurrences. This keeps the current v0.5 implementation honest: it patches the inference context for the same Skill belief, rather than silently creating a separate child Skill hypothesis.
