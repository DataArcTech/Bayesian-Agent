# Bayesian Self-Evolving Agent Method

Bayesian-Agent treats each Skill or SOP as a hypothesis about agent success under a task context. The method is harness-agnostic: it can bootstrap Skills in a full run, repair existing agents incrementally, or transfer the same posterior Skill registry across compatible harnesses.

```text
P(success | theta, C, h)
```

- `theta`: frozen base model parameters
- `C`: inference condition, including prompt, memory, tools, retrieved context, and harness feedback
- `h`: Skill/SOP hypothesis

The framework does not train the base model and does not require replacing the agent runtime. It changes the inference environment by maintaining posterior-weighted Skill context that can be injected through adapters.

## Bayesian Formulation in v0.x

The current implementation uses a Beta-Bernoulli Bayesian update for each Skill/SOP. It is Bayesian in the posterior-belief sense, but it is not yet a full Bayesian model-selection system over competing Skill hypotheses.

For Skill hypothesis `h_k`, let:

```text
p_k = P(y = 1 | h_k, c)
y_i ~ Bernoulli(p_k)
p_k ~ Beta(alpha_0, beta_0)
```

where `y_i = 1` means a verified success and `c` is the task context. Given evidence `D_k` with `s_k` successes and `f_k` failures:

```text
p_k | D_k ~ Beta(alpha_0 + s_k, beta_0 + f_k)
E[p_k | D_k] = (alpha_0 + s_k) / (alpha_0 + beta_0 + s_k + f_k)
```

Bayesian-Agent v0.x uses `alpha_0 = beta_0 = 1`. The posterior mean is used for Skill ranking, context rendering, and rewrite policy decisions.

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

Each Skill uses a Beta posterior:

```text
success: alpha_i += 1
failure: beta_i += 1
E[p_i] = alpha_i / (alpha_i + beta_i)
```

The registry also tracks cost, context distribution, and failure modes. These statistics guide what gets injected into future context.

## Rewrite Policy

The default policy maps posterior state to actions:

- `compress`: repeated success suggests the Skill is stable
- `patch`: failures cluster around a recurring failure mode
- `split`: evidence spans different contexts
- `retire`: failures dominate the posterior
- `explore`: evidence is still sparse or uncertain

The policy is intentionally small in v0.4. It is designed to be replaced by project-specific policies.

## Full Mode

Full self-evolving mode runs all tasks and updates Skill beliefs online. This mode tests whether Bayesian Skill Evolution can improve an agent from scratch.

## Incremental Repair Mode

Incremental repair mode starts from a baseline agent's traces:

```text
baseline traces -> failure ids -> Bayesian context -> rerun failures -> merged final result
```

This mode is the recommended production path because it adds Bayesian-Agent as a plug-in repair layer instead of replacing the base agent.
