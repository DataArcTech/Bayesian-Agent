# Experiments

The first Bayesian-Agent prototype was validated inside GenericAgent with `deepseek-v4-flash`. These experiments demonstrate two advantages at once: Bayesian-Agent can run a full self-evolving loop from scratch, and it can also act as an incremental repair layer for an existing agent.

## Baseline

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA | deepseek-v4-flash | 80% | 1.34M | 57k | 1.39M | 11.47 |
| Lifelong AgentBench | GA | deepseek-v4-flash | 90% | 649k | 42k | 690k | 26.07 |

## Full Self-Evolving Mode

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+Bayesian | deepseek-v4-flash | 100% | 1.07M | 52k | 1.12M | 17.86 |
| Lifelong AgentBench | GA+Bayesian | deepseek-v4-flash | 95% | 666k | 44k | 710k | 26.77 |

## Incremental Repair Mode

Bayesian-Agent read the GA baseline traces and reran only failed tasks.

| Benchmark | Agent | Model | Final Accuracy | Incremental Input | Incremental Output | Incremental Total | Incremental Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 216k | 10k | 226k | 17.73 |
| Lifelong AgentBench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 71k | 7k | 78k | 25.57 |

## Interpretation

The full mode result shows that Bayesian Skill Evolution can improve SOP-Bench accuracy while reducing token usage.

The incremental mode result is the more practical path: Bayesian-Agent can attach to a baseline agent, inspect failed tasks, and spend only a small amount of additional inference to reach 100% final accuracy on the reported runs.

The cross-harness implication is the larger product direction. GenericAgent is the current experimental harness, but the Bayesian-Agent core only requires verified trajectory evidence and an adapter, so the same method can be applied to other agent frameworks.

## Artifacts

Result artifacts are stored in:

```text
artifacts/
  ga_deepseek_baseline/
  bayesian_full/
  bayesian_incremental/
```

These files are included so users can inspect result formats and reproduce summary calculations before wiring a live benchmark runner.
