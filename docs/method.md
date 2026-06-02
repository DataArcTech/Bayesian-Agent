# Bayesian Self-Evolving Agent Method

Bayesian-Agent treats each Skill or SOP as a hypothesis about agent success under a task context. The method is harness-agnostic: it can bootstrap Skills in a full run, repair existing agents incrementally, or transfer the same posterior Skill registry across compatible harnesses.

```text
P(success | theta, C, h)
```

- `theta`: frozen base model parameters
- `C`: inference condition, including prompt, memory, tools, retrieved context, and harness feedback
- `h`: Skill/SOP hypothesis

The framework does not train the base model and does not require replacing the agent runtime. It changes the inference environment by using posterior beliefs to rank Skills, decide rewrites, and generate model-facing Skill/SOP text through adapters.

## Bayesian Formulation in v0.x

The current default implementation uses a **Bayesian Evidence Model** for each Skill/SOP. The v0.x backend is a feature-conditioned categorical likelihood model: verified trajectories update the probability that a Skill will succeed under an observed context and runtime signature. It is not yet a full Bayesian model-selection system over competing Skill hypotheses.

For Skill hypothesis `h_k`, let `D_k = {(x_i, y_i)}` be verified evidence. `y_i` is either `success` or `failure`, and `x_i` contains discrete evidence features such as context, failure mode, token bucket, turn bucket, latency bucket, and selected metadata:

```text
P(y | h_k) = (N_y + alpha) / (N + alpha * |Y|)
P(x_j = v | y, h_k) = (N_{j,v,y} + alpha) / (N_{j,y} + alpha * |V_j|)
P(y | h_k, x) ∝ P(y | h_k) * Π_j P(x_j | y, h_k)
```

Bayesian-Agent v0.x uses `alpha = 1` Laplace smoothing. The resulting `P(success | h_k, x)` is used for Skill ranking, posterior audit rendering, and rewrite policy decisions. The public algorithm name is `categorical_bayes`; `naive_bayes` remains accepted as a legacy alias.

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

The default policy maps posterior state to actions:

- `compress`: repeated success suggests the Skill is stable
- `patch`: failures cluster around a recurring failure mode
- `split`: evidence spans different contexts
- `retire`: failures dominate the posterior
- `explore`: evidence is still sparse or uncertain

The policy is intentionally small in v0.4. It is designed to be replaced by project-specific policies.

For the built-in SOP-Bench and Lifelong AgentBench runners, `patch` is not only a label in the posterior audit context. Observed benchmark failure modes are mapped to concrete patch rules and injected into the next prompt under `Bayesian Failure-Mode Patches`. The prompt does not include raw posterior numbers such as `posterior_success`, `alpha`, or `beta`; those stay in `belief_*.json` and `posterior_context_*.md` artifacts. For example, `left_expected_output_blank` adds a CSV writeback verification rule, and `invented_unrequested_column` adds SQL column-use constraints. v0.x records post-patch evidence back to the same benchmark Skill; later releases may split recurring patches into separate child Skill hypotheses.

## Full Mode

Full self-evolving mode runs all tasks and updates Skill beliefs online. This mode tests whether Bayesian Skill Evolution can improve an agent from scratch.

## Incremental Repair Mode

Incremental repair mode starts from a baseline agent's traces:

```text
baseline traces -> failure ids -> Bayesian context -> rerun failures -> merged final result
```

This mode is the recommended production path because it adds Bayesian-Agent as a plug-in repair layer instead of replacing the base agent.
