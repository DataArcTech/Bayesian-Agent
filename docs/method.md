# Bayesian Self-Evolving Agent Method

Bayesian-Agent treats each Skill or SOP as a hypothesis about agent success under a task context.

```text
P(success | theta, C, h)
```

- `theta`: frozen base model parameters
- `C`: inference condition, including prompt, memory, tools, retrieved context, and harness feedback
- `h`: Skill/SOP hypothesis

The framework does not train the base model. It changes the inference environment by maintaining posterior-weighted Skill context.

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
