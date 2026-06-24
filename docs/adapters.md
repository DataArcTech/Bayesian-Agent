# Adapters

Bayesian-Agent now has a first-party native harness, and it still integrates with external agent harnesses without copying their code. This is one of the main reasons the project is not just another monolithic framework: the Bayesian layer can improve whichever harness emits verified trajectories.

## Adaptation Advantage

Bayesian-Agent separates Skill evolution from task execution:

```text
Native or external harness executes -> Bayesian-Agent learns -> Skill/SOP text updates -> Harness reruns
```

That separation enables three deployment styles:

- run a full benchmark from scratch with Bayesian Skill evolution enabled
- repair only the failed tasks from an existing agent run
- reuse the same Skill belief registry across compatible harnesses

## Native Harness First

The default execution backend is now the Bayesian-Agent native harness:

```bash
python experiments/run_benchmarks.py \
  --harness bayesian-agent \
  --model deepseek-v4-flash \
  --bench core \
  --mode all \
  --limit 1
```

The native harness owns only the minimal execution substrate:

- LLM: a small OpenAI-compatible chat client.
- Tools: workspace-scoped `file_read`, `file_write`, `code_run`, and `finish`.
- Memory: optional three layers, `hippocampus`, intermediate `state`, and persistent `cortex`; disabled by default unless `--native-memory` is passed.
- Loop: turn execution, tool dispatch, transcript capture, usage accounting, and trajectory persistence.

The harness layer is intentionally simple and efficient. Most capability improvement is meant to come from Bayesian Skill/SOP evolution, where verified trajectories update reusable procedures instead of hiding behavior inside a large runtime.

## Adapter Contract

An external harness should satisfy the `AgentAdapter` protocol:

```python
from typing import Any, Mapping, Protocol

class AgentAdapter(Protocol):
    def run(self, task: Mapping[str, Any], skill_context: str) -> Mapping[str, Any]:
        ...
```

The adapter receives:

- a task object from the external benchmark or application
- model-facing Skill/SOP text selected or patched by Bayesian-Agent

It returns:

- a trajectory-like mapping that can be converted into `TrajectoryEvidence`

## GenericAgent Adapter

The GenericAgent adapter is intentionally thin: it runs one prompt in one workspace. Benchmark loops and Bayesian Skill evolution stay in Bayesian-Agent.

```python
from bayesian_agent.adapters.generic_agent import GenericAgentAdapter

adapter = GenericAgentAdapter(root="/path/to/GenericAgent", model="deepseek-v4-flash")
result = adapter.run(
    {
        "prompt": "Solve the task in this workspace.",
        "workspace": "temp/task_01",
        "max_turns": 8,
    },
    skill_context="### Bayesian Failure-Mode Patches\n...",
)
```

It does not eagerly import GenericAgent and does not vendor GenericAgent source code. The experiment script `experiments/run_benchmarks.py` uses this adapter for task execution while Bayesian-Agent owns SOP-Bench, Lifelong AgentBench, and RealFin-Bench orchestration, evidence collection, posterior updates, and incremental repair.

## Why This Boundary Matters

Bayesian-Agent should be usable with more than one agent framework. The durable contract is the trajectory schema, not a copied harness implementation.

External systems should emit:

- task identity
- success or failure outcome
- failure mode
- token usage
- runtime metadata

Bayesian-Agent can then update beliefs, keep posterior audit artifacts, and render the next model-facing Skill/SOP text.

## Optional Compatibility Backends

External harnesses remain useful for comparison and transfer. Current optional backend names are:

```bash
--harness genericagent
--harness mini-swe-agent
--harness claude-code
```

Each backend should emit enough trajectory evidence for Bayesian-Agent to update Skill beliefs: task identity, outcome, failure mode, token usage, tool/runtime metadata, and artifacts.

## MinimalAgent Status

MinimalAgent adapter support is intentionally not included in v0.5.

The recommended path is:

1. keep the native harness small and inspectable
2. keep the core trace schema portable
3. use GA, mini-swe-agent, and Claude Code as compatibility backends
4. add more adapters only after the adapter contract has enough real usage
