# Bayesian-Agent: A Bayesian Self-Evolving Agent Framework with Cross-Harness Adaptation

<div align="center">
  <img src="assets/banner.png" width="920" alt="Bayesian-Agent banner"/>
</div>

<p align="center">
  🌐 <a href="README.md">English</a> | 🇨🇳 <a href="README_ZH.md">中文</a> |
  📚 <a href="https://dataarctech.github.io/Bayesian-Agent/">Docs</a> |
  🐙 <a href="https://github.com/DataArcTech/Bayesian-Agent">GitHub</a> |
  📄 arXiv Coming Soon
</p>

Bayesian-Agent is a Bayesian self-evolving layer for turning agent failures into reusable, evidence-weighted Skills and SOPs across agent frameworks and execution harnesses.

It is designed to stand out from monolithic agent frameworks in three ways:

- **Run from scratch**: start with no prior traces and evolve Skills during full benchmark or production runs.
- **Repair incrementally**: attach to an existing agent, read its failed trajectories, and rerun only the tasks that need repair.
- **Adapt across harnesses**: integrate with GenericAgent today, and with other agent frameworks through a portable trajectory schema and adapter boundary.

> v0.4 is the first standalone release. It includes the core Bayesian Skill Evolution package, schemas, CLI utilities, experiment artifacts, and a clean GenericAgent integration boundary. GenericAgent itself is not copied, vendored, or forked.

## 📅 News

- **2026-05-09:** Released Bayesian-Agent v0.4 as a standalone cross-harness Bayesian Skill Evolution package with schemas, CLI utilities, and experiment artifacts.
- **2026-05-09:** Added the optional GenericAgent adapter boundary without copying or vendoring GenericAgent.
- **2026-05-09:** Published bilingual project documentation and the Bayesian-Agent framework diagram.

## 🌟 Overview

Agent engineering is moving through three layers:

1. **Prompt Engineering**: write better task instructions.
2. **Context Engineering**: decide what evidence the model can see at inference time.
3. **Harness Engineering**: put the model inside an observable, executable, recoverable system.

Prompting can improve one answer. Context can improve one decision. Harness Engineering is what lets an agent work across tools, files, tests, memory, logs, and failure recovery.

In that setting, **Skills** and **SOPs** become first-class engineering assets. A good Skill is not just a longer prompt. It is compressed operational knowledge:

- what to inspect first
- which tools to use
- how to verify progress
- which failure modes to avoid
- when to stop, retry, or rewrite the procedure

Bayesian-Agent asks a simple question: if Skills are hypotheses about how to solve tasks, why should they evolve by anecdote instead of evidence? The answer is a framework-agnostic evolution layer that can bootstrap Skills from scratch, repair existing agents incrementally, and move across harnesses as long as they emit verified trajectories.

<div align="center">
  <img src="assets/bayesian_agent_overview.png" width="900" alt="Bayesian-Agent overview"/>
  <br/>
  <em>Bayesian-Agent turns verified trajectories from any compatible harness into posterior-weighted Skills and SOPs.</em>
</div>

## 🧠 Core Idea

Most LLM engineering interventions fall into two MECE routes:

1. Change the model parameter distribution, such as pretraining, fine-tuning, and reinforcement learning.
2. Change the inference condition, such as prompts, context, RAG, tools, memory, and harnesses.

Bayesian-Agent focuses on the second route.

If a base model samples from:

```text
P(X | theta)
```

then an agent system samples from:

```text
P(X | theta, C)
```

where `C` is the inference environment. Skills, SOPs, tools, memory, retrieved evidence, execution traces, and verifier feedback are all part of `C`.

Bayesian-Agent treats each Skill or SOP as a hypothesis about success:

```text
P(success | theta, C, skill)
```

After each verified trajectory, the framework updates a posterior belief over that Skill. The next run receives posterior-weighted Skill context instead of unfiltered memory.

## 📋 Core Features

- **Evidence-weighted Skill evolution**: update Skill beliefs from verified success and failure trajectories.
- **Bayesian Skill registry**: maintain Beta posteriors, failure modes, token cost, latency, turns, and context distribution.
- **Failure-mode-aware repair**: identify recurring errors and generate focused repair plans.
- **Token-aware context building**: inject only the Skills with useful posterior evidence.
- **Full self-evolution from scratch**: run all tasks, collect evidence online, and evolve Skills without prior traces.
- **Incremental repair for existing agents**: consume failed trajectories from a baseline agent and rerun only the failed tasks.
- **Cross-harness adaptation**: integrate with GenericAgent today and other agent frameworks through adapters instead of vendoring their code.
- **Standard-library-first core**: v0.4 has no runtime dependency beyond Python.

## 🧬 Self-Evolution Mechanism

<div align="center">
  <img src="assets/bayesian_agent_framework_v2.svg" width="900" alt="Bayesian-Agent framework"/>
  <br/>
  <em>Bayesian Skill Evolution framework.</em>
</div>

```text
[Agent Trajectory]
      |
      v
[Verifier / Benchmark Grader]
      |
      v
[TrajectoryEvidence: success, failure mode, tokens, turns, latency]
      |
      v
[Bayesian Skill Registry: posterior + cost + contexts]
      |
      v
[Rewrite Policy: compress, patch, split, retire, explore]
      |
      v
[Posterior-Weighted Skill Context]
      |
      v
[Next Agent Run]
```

For each Skill or benchmark SOP, Bayesian-Agent maintains:

- a Beta posterior over success probability
- verified success and failure evidence
- failure mode counts
- input, output, and total token statistics
- latency and turn statistics
- context distribution
- rewrite policy recommendations

The default rewrite policy is intentionally small:

| Posterior signal | Policy action |
|---|---|
| repeated verified success | compress or reinforce |
| clustered failures | patch |
| mixed outcomes across contexts | split or specialize |
| dominant failures | retire or rewrite |
| sparse evidence | explore |

## 🚀 Install

```bash
git clone https://github.com/DataArcTech/Bayesian-Agent.git
cd Bayesian-Agent
python -m pip install -e .
```

The package currently requires Python 3.9+ and has no runtime dependencies beyond the Python standard library.

## ⚡ Quick Start

Update a Bayesian Skill registry from existing agent results:

```bash
bayesian-agent evolve \
  --results artifacts/ga_deepseek_baseline/sop_results.json \
  --registry temp/bayesian_skill_beliefs.json \
  --context-out temp/skill_context.md
```

Find failed tasks for incremental repair:

```bash
bayesian-agent repair-plan \
  --baseline artifacts/ga_deepseek_baseline/sop_results.json \
  --out temp/failed_tasks.json
```

Summarize a run:

```bash
bayesian-agent summarize \
  --results artifacts/bayesian_incremental/results.json \
  --out temp/summary.json
```

## 🐍 Python API

```python
from bayesian_agent import BayesianSkillRegistry, SkillContextBuilder, TrajectoryEvidence

registry = BayesianSkillRegistry("temp/beliefs.json")
registry.record(
    TrajectoryEvidence(
        task_id="sop_12",
        skill_id="benchmark/sop_bench",
        context="sop_bench",
        outcome="failure",
        failure_mode="xml_wrapped_answer",
        input_tokens=70123,
        output_tokens=4242,
        total_tokens=74365,
    )
)

skill_context = SkillContextBuilder(registry).render(task_context="sop_bench")
print(skill_context)
```

## 🔁 Three Operating Patterns

### 🌱 Full Self-Evolving Mode

Bayesian-Agent starts from scratch, runs benchmark tasks, collects verified evidence, and evolves Skills during the run.

This mode tests whether Bayesian Skill Evolution can improve an agent without relying on prior traces.

### 🛠️ Incremental Repair Mode

Bayesian-Agent can also attach to an existing agent. The base agent runs first. Bayesian-Agent reads its success and failure traces, updates posterior Skill beliefs, then reruns only the failed tasks.

```text
Base Agent -> Failure Traces -> Bayesian Skill Evolution -> Rerun Failures -> Higher Accuracy
```

This is the recommended production path because it improves an existing agent without retraining the model or replacing the original harness.

### 🔌 Cross-Harness Adaptation Mode

Bayesian-Agent is not tied to a single agent runtime. Any agent framework can become a backend if it emits the common trajectory schema and accepts posterior-weighted Skill context through an adapter.

```text
Any Agent Harness -> Trajectory Schema -> Bayesian Skill Registry -> Adapter -> Next Harness Run
```

This makes Bayesian-Agent a portable Skill/SOP evolution layer rather than another closed agent framework.

## 📊 Experimental Results

The v0.4 prototype was validated with GenericAgent and `deepseek-v4-flash` on SOP-Bench and Lifelong AgentBench.

### 🧱 Baseline: GenericAgent + deepseek-v4-flash

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA | deepseek-v4-flash | 80% | 1.34M | 57k | 1.39M | 11.47 |
| Lifelong AgentBench | GA | deepseek-v4-flash | 90% | 649k | 42k | 690k | 26.07 |

### 🌱 Full Self-Evolving Run

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+Bayesian | deepseek-v4-flash | 100% | 1.07M | 52k | 1.12M | 17.86 |
| Lifelong AgentBench | GA+Bayesian | deepseek-v4-flash | 95% | 666k | 44k | 710k | 26.77 |

In full mode, Bayesian-Agent improved SOP-Bench from 80% to 100% while reducing token usage from 1.39M to 1.12M. Lifelong AgentBench improved from 90% to 95% with similar token cost.

### 🛠️ Incremental Repair Run

In incremental mode, Bayesian-Agent only reran failed GenericAgent tasks:

- SOP-Bench: 4 failed tasks, all repaired
- Lifelong AgentBench: 2 failed tasks, all repaired

| Benchmark | Agent | Model | Final Accuracy | Incremental Input | Incremental Output | Incremental Total | Incremental Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 216k | 10k | 226k | 17.73 |
| Lifelong AgentBench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 71k | 7k | 78k | 25.57 |

The result shows that Bayesian-Agent can work as a plug-in repair layer: it can take an existing agent below 100% accuracy and improve it with a small amount of incremental inference. This is the practical advantage over one-off benchmark agents: Bayesian-Agent can sit beside a harness, learn from its failures, and improve it without replacing it.

Experiment artifacts are stored under [`artifacts/`](artifacts/), and the method note is in [`docs/method.md`](docs/method.md).

## 🔌 GenericAgent and Cross-Harness Adaptation

The first prototype was validated inside GenericAgent, but Bayesian-Agent is not a GenericAgent fork and not just a GenericAgent add-on.

The open-source structure is:

- `bayesian_agent/core/`: framework-agnostic Bayesian Skill Evolution logic
- `bayesian_agent/adapters/base.py`: minimal adapter contract for external agents
- `bayesian_agent/adapters/generic_agent.py`: optional GenericAgent boundary
- `schemas/`: portable trajectory and Skill belief schemas
- `artifacts/`: reproducible benchmark result files

GenericAgent remains the current experimental backend. Users can integrate Bayesian-Agent with their own agent harness by emitting the common trajectory schema and implementing the adapter boundary.

The long-term direction is to make Bayesian-Agent the Bayesian Skill/SOP evolution layer for many agent runtimes: GenericAgent, our own upcoming Agent harness, and other external frameworks.

MinimalAgent adapter support is intentionally not included in v0.4.

## 🗂️ Repository Layout

```text
bayesian_agent/
  core/                 # Evidence, beliefs, registry, policy, context, repair
  adapters/             # Adapter contract and optional GenericAgent boundary
schemas/                # JSON schemas for trajectories and Skill beliefs
artifacts/              # Baseline, full-mode, and incremental-mode result artifacts
docs/                   # Method and experiment notes
examples/               # Integration notes
tests/                  # Standard-library unittest suite
```

## 🧭 Roadmap

- [x] Refactor the GenericAgent prototype into a standalone package core.
- [x] Define a common trace schema for agent runs.
- [x] Implement the Bayesian Skill registry.
- [x] Implement full self-evolving primitives.
- [x] Implement incremental repair utilities.
- [x] Add a GenericAgent optional adapter boundary without vendoring GenericAgent.
- [x] Release experiment result artifacts.
- [x] Add English and Chinese project READMEs.
- [ ] Add executable benchmark runners for external checkouts.
- [ ] Add richer rewrite policies and adapter examples.
- [ ] Add adapters for more agent harnesses after the GenericAgent boundary stabilizes.
- [ ] Release our own Agent harness for Bayesian-Agent; current experiments use GenericAgent as the backend harness.


## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DataArcTech/Bayesian-Agent&type=Date)](https://www.star-history.com/#DataArcTech/Bayesian-Agent&Date)

## 📝 Citation

If you use Bayesian-Agent in your research or projects, please cite it as:

```bibtex
@software{bayesian_agent_2026,
  title = {Bayesian-Agent: A Bayesian Self-Evolving Agent Framework with Cross-Harness Adaptation},
  author = {{Xiaojun Wu}},
  year = {2026},
  url = {https://github.com/DataArcTech/Bayesian-Agent}
}
```

## 📄 License

MIT License. See [`LICENSE`](LICENSE).
