# Bayesian Self-Evolving Agent Method

Bayesian-Agent treats each Skill or SOP as a hypothesis about agent success under a task context. The method is harness-agnostic: it can bootstrap Skills in a full run, repair existing agents incrementally, or transfer the same posterior Skill registry across compatible harnesses.

```text
P(success | theta, C, h)
```

- `theta`: frozen base model parameters
- `C`: inference condition, including prompt, memory, tools, retrieved context, and harness feedback
- `h`: Skill/SOP hypothesis

The framework does not train the base model and does not require replacing the agent runtime. It changes the inference environment by using posterior beliefs to rank Skills, decide rewrites, and generate model-facing Skill/SOP text through adapters.

## Bayesian Formulation in v0.5

The current default implementation uses a **Bayesian Evidence Model** for each Skill/SOP. The v0.5 backend is a feature-conditioned categorical likelihood model: verified trajectories update the probability that a Skill will succeed under an observed context and runtime signature. It is not yet a full Bayesian model-selection system over competing Skill hypotheses.

For Skill hypothesis `h_k`, let `D_k = {(x_i, y_i)}` be verified evidence. `y_i` is either `success` or `failure`, and `x_i` contains discrete evidence features such as context, failure mode, token bucket, turn bucket, latency bucket, and selected metadata:

```text
P(y | h_k) = (N_y + alpha) / (N + alpha * |Y|)
P(x_j = v | y, h_k) = (N_{j,v,y} + alpha) / (N_{j,y} + alpha * |V_j|)
P(y | h_k, x) ∝ P(y | h_k) * Π_j P(x_j | y, h_k)
```

Bayesian-Agent v0.5 uses `alpha = 1` Laplace smoothing. The resulting `P(success | h_k, x)` is used for Skill ranking, posterior audit rendering, and rewrite policy decisions. The public algorithm name is `categorical_bayes`; `naive_bayes` remains accepted as a legacy alias.

### Current Likelihood Terms

The current implementation extracts five fixed categorical likelihood terms from each verified trajectory, plus optional `metadata.*` terms:

| Term | Source | Reason |
|---|---|---|
| `context` | `TrajectoryEvidence.context` | Captures benchmark, task family, or harness context. |
| `failure_mode` | `TrajectoryEvidence.failure_mode`, or `__none__` when empty | Captures repeated, actionable failure patterns. |
| `token_bucket` | bucketed total tokens | Captures whether success or failure was cheap, expensive, or search-heavy. |
| `turn_bucket` | bucketed interaction turns | Captures planning instability and recovery loops. |
| `latency_bucket` | bucketed elapsed seconds | Captures slow tool/data/API paths. |
| `metadata.*` | short scalar metadata values | Allows harness-specific diagnostics without hardcoding one harness schema. |

The total number of likelihood terms is therefore:

```text
5 fixed terms + 0..N metadata terms
```

`metadata.*` terms are included only for values of type `str`, `int`, `float`, or `bool`, and only when `len(str(value)) <= 80`. Lists, dicts, long strings, transcripts, and usage events should stay out of the likelihood model. This keeps the evidence model readable and reduces high-cardinality overfitting.

The built-in buckets are:

| Runtime signal | Buckets |
|---|---|
| `token_bucket` | `0`, `1_1k`, `1k_10k`, `10k_100k`, `100k_1m`, `1m_plus` |
| `turn_bucket` | `0`, `1_2`, `3_5`, `6_10`, `11_20`, `20_plus` |
| `latency_bucket` | `0s`, `0s_10s`, `10s_60s`, `1m_5m`, `5m_30m`, `30m_plus` |

The bucketed design is intentional: exact token counts or latencies are too sparse in early online evidence, while buckets preserve the operational signal needed for Skill rewrite decisions.

The original Beta-Bernoulli backend is still available for compatibility and ablation:

```text
p_k | D_k ~ Beta(alpha_0 + s_k, beta_0 + f_k)
E[p_k | D_k] = (alpha_0 + s_k) / (alpha_0 + beta_0 + s_k + f_k)
```

The general Bayesian model-selection form we plan to add later is:

```text
P(h_k | D) ∝ P(D | h_k) P(h_k)
```

That would let the framework compare multiple Skill hypotheses directly, rather than only tracking each Skill's success probability independently.

## Evidence

Each agent run emits `TrajectoryEvidence`:

- task id
- skill id
- context
- success or failure outcome
- token counts
- latency and turns
- failure mode
- task metadata

Evidence should be action-verified. For example, a benchmark grader, unit test, or deterministic checker should decide whether a run succeeded.

## Belief Update

Each Skill stores a selected belief backend:

```text
algorithm = categorical_bayes  # default Bayesian Evidence Model
algorithm = beta_bernoulli     # optional compatibility backend
```

For `categorical_bayes`, each verified event increments the class count and each extracted feature value count for the observed label. For Beta-Bernoulli, each verified success increments `alpha` and each verified failure increments `beta`.

The registry also tracks cost, context distribution, and failure modes. These statistics guide which executable Skill/SOP text is selected or patched for future model-facing context.

## Rewrite Policy

The default policy maps posterior state to five actions:

| Action | Current trigger | Rationale |
|---|---|---|
| `explore` | `observations == 0`, or no stronger rule fires | Keep collecting evidence while the posterior is sparse or uncertain. |
| `retire` | `beta >= 4` and `success_probability < 0.45` | Require multiple failures before retiring a Skill, then remove clearly harmful or stale guidance. |
| `patch` | `max(failure_mode_count) >= 2` | Promote only repeated failure patterns into model-facing patch text. |
| `split` | at least 3 observed contexts and at least 4 observations | Separate broad Skills when evidence shows they span multiple operational contexts. |
| `compress` | at least 3 observations and `success_probability >= 0.72` | Distill stable Skills after enough positive evidence, reducing future token cost. |

The checks run in implementation order: no evidence, retire, patch, split, compress, then fallback explore. These thresholds are conservative v0.5 heuristics. They are meant to make the default policy inspectable and robust against one-off failures, not to claim an optimal decision rule.

For the built-in SOP-Bench and Lifelong AgentBench runners, `patch` is not only a label in the posterior audit context. Observed benchmark failure modes are mapped to concrete patch rules, but they are injected into the next prompt under `Bayesian Failure-Mode Patches` only after the same failure mode has at least two verified occurrences. A single failure remains candidate evidence in `belief_*.json` and `posterior_context_*.md`, which reduces overfitting to one-off mistakes. The prompt does not include raw posterior numbers such as `posterior_success`, `alpha`, or `beta`. For example, repeated `left_expected_output_blank` failures add a CSV writeback verification rule, and repeated `invented_unrequested_column` failures add SQL column-use constraints. v0.5 records post-patch evidence back to the same benchmark Skill; later releases may split recurring patches into separate child Skill hypotheses.

## Full Mode

Full self-evolving mode runs all tasks and updates Skill beliefs online. This mode tests whether Bayesian Skill Evolution can improve an agent from scratch.

## Incremental Repair Mode

Incremental repair mode starts from a baseline agent's traces:

```text
baseline traces -> failure ids -> Bayesian context -> rerun failures -> merged final result
```

This mode is the recommended production path because it adds Bayesian-Agent as a plug-in repair layer instead of replacing the base agent.
