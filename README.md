# Bayesian-Agent

Bayesian Self-Evolving Agent Framework for turning agent failures into reusable, evidence-weighted Skills and SOPs.

> Code is coming soon. The current prototype was validated inside GenericAgent and is being refactored into a standalone open-source framework.

## Why Bayesian-Agent

Large language model engineering is moving through three layers:

1. **Prompt Engineering**: make the task instruction clearer.
2. **Context Engineering**: decide what information the model can see at inference time.
3. **Harness Engineering**: place the model inside a runnable, observable, recoverable system.

Prompting helps a model answer better in one turn. Context helps it make better decisions with the right evidence. Harness Engineering matters when we want an agent to work for many steps in a real environment with tools, files, tests, logs, memory, and failure recovery.

In this setting, **Skills** and **SOPs** become first-class engineering assets. A good Skill is not just a longer prompt. It is compressed operational knowledge:

- what to inspect first
- which tools to use
- how to verify progress
- which failure modes to avoid
- when to stop, retry, or rewrite the procedure

The open question is: how should Skills evolve?

Bayesian-Agent treats each Skill or SOP as a hypothesis about agent success:

```text
P(success | model, context, skill)
```

Instead of blindly appending every anecdote to memory, Bayesian-Agent records action-verified evidence and updates posterior beliefs about which Skills work, in which contexts, and at what token cost.

## Core Idea

Most LLM engineering interventions fall into two MECE routes:

1. Change the model parameters, such as pretraining, fine-tuning, and reinforcement learning.
2. Change the inference conditions, such as prompts, context, RAG, tools, memory, and harnesses.

Bayesian-Agent focuses on the second route.

If a base model samples from:

```text
P(X | theta)
```

then an agent system samples from:

```text
P(X | theta, C)
```

where `C` is the inference environment. Skills, SOPs, tools, memory, and execution feedback are all part of `C`.

Bayesian-Agent makes the Skill layer self-evolving by maintaining a posterior belief for each Skill:

```text
Skill h ~ hypothesis
Evidence e ~ verified trajectory outcome
Posterior ~ P(h works | e_1, e_2, ..., e_n)
```

## How It Works

For each Skill or benchmark SOP, Bayesian-Agent maintains:

- a Beta posterior over success probability
- verified success and failure evidence
- failure mode counts
- token and latency statistics
- context distribution
- rewrite policy

At execution time, the agent uses posterior-weighted Skill context. After execution, the framework records evidence and updates the Skill registry.

Typical rewrite policies include:

- **reinforce or compress** when repeated verified successes raise confidence
- **patch** when failures cluster around a specific failure mode
- **specialize or split** when a Skill works in one context but fails in another
- **retire or rewrite** when posterior failures dominate
- **keep exploratory** when evidence is insufficient

This turns Skill evolution from prompt folklore into an evidence-tracking process.

## Two Modes

### 1. Full Self-Evolving Mode

Bayesian-Agent starts from scratch, runs the benchmark tasks, collects evidence, and evolves Skills during the run.

This mode tests whether Bayesian Skill Evolution can improve an agent without relying on prior benchmark traces.

### 2. Incremental Repair Mode

Bayesian-Agent can also be attached to an existing agent.

The base agent runs first. Bayesian-Agent reads the base agent's success and failure traces, updates posterior Skill beliefs, then reruns only the failed tasks.

This mode is designed for practical use:

```text
Base Agent -> Failure Traces -> Bayesian Skill Evolution -> Rerun Failures -> Higher Accuracy
```

It does not require retraining the model or replacing the original agent.

## Experimental Results

We validated the prototype on GenericAgent with `deepseek-v4-flash`.

### Baseline: GenericAgent + deepseek-v4-flash

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA | deepseek-v4-flash | 80% | 1.34M | 57k | 1.39M | 11.47 |
| Lifelong AgentBench | GA | deepseek-v4-flash | 90% | 649k | 42k | 690k | 26.07 |

### Full Self-Evolving Run

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+Bayesian | deepseek-v4-flash | 100% | 1.07M | 52k | 1.12M | 17.86 |
| Lifelong AgentBench | GA+Bayesian | deepseek-v4-flash | 95% | 666k | 44k | 710k | 26.77 |

In full mode, Bayesian-Agent improved SOP-Bench from 80% to 100% while reducing token usage from 1.39M to 1.12M. Lifelong AgentBench improved from 90% to 95% with similar token cost.

### Incremental Repair Run

In incremental mode, Bayesian-Agent only reran the failed GA tasks:

- SOP-Bench: 4 failed tasks, all repaired
- Lifelong AgentBench: 2 failed tasks, all repaired

| Benchmark | Agent | Model | Final Accuracy | Incremental Input | Incremental Output | Incremental Total | Incremental Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 216k | 10k | 226k | 17.73 |
| Lifelong AgentBench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 71k | 7k | 78k | 25.57 |

This shows Bayesian-Agent can work as a plug-in repair layer: it can take an existing agent that has not reached 100% accuracy and improve it using only a small amount of incremental inference.

## Design Goals

Bayesian-Agent is being refactored around these principles:

- **Framework-agnostic**: integrate with existing agent systems instead of replacing them.
- **Evidence-first**: evolve Skills only from verified execution traces.
- **Token-aware**: track the cost of every Skill hypothesis.
- **Failure-mode aware**: distinguish different types of failures instead of treating all failures as identical.
- **Incremental by default**: improve existing agents by repairing their failures.
- **Portable Skill registry**: make evolved Skills reusable across agents, models, and tasks.

## Roadmap

- [ ] Refactor the GenericAgent prototype into a standalone package.
- [ ] Define a common trace schema for agent runs.
- [ ] Implement the Bayesian Skill registry.
- [ ] Implement full self-evolving mode.
- [ ] Implement incremental repair mode.
- [ ] Add benchmark runners for SOP-Bench and Lifelong AgentBench.
- [ ] Add documentation for integrating with external agents.
- [ ] Release reproducible experiment scripts and result artifacts.

## Status

The first prototype has been validated inside GenericAgent. The public repository currently contains project documentation only. Source code will be released after the framework is refactored out of the GenericAgent-specific implementation.

## Repository

GitHub: <https://github.com/DataArcTech/Bayesian-Agent>

## License

This project is released under the MIT License.
