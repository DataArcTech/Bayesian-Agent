# Experiments

The first prototype was validated inside GenericAgent with `deepseek-v4-flash`.

These experiments are meant to show two deployment paths: Bayesian-Agent can run a full self-evolving loop from scratch, and it can also attach to an existing agent as an incremental repair layer. GenericAgent is the current experimental harness, not a hard dependency of the Bayesian-Agent method.

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

Artifacts are stored under `artifacts/`.

The cross-harness path depends on the same evidence format: any agent framework that emits verified trajectories can feed the Bayesian Skill registry and receive posterior-weighted Skill context through an adapter.
