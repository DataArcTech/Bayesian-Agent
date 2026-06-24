# Native Harness

Bayesian-Agent now includes a first-party native harness. It is intentionally small: the harness owns the executable loop, while most long-term capability improvement is pushed into Bayesian Skill evolution.

The goal is not to rebuild a large monolithic agent runtime. The goal is to provide a lean, observable substrate where verified trajectories can become better Skills and SOPs.

## Minimal Design

The native harness has three core parts:

| Layer | File | Responsibility |
|---|---|---|
| LLM | `bayesian_agent/harness/llm.py` | A tiny OpenAI-compatible chat-completions client for DeepSeek/OpenAI-style APIs. |
| Tools | `bayesian_agent/harness/tools.py` | Workspace-scoped `file_read`, `file_write`, `code_run`, and `finish`. |
| Loop | `bayesian_agent/harness/native.py` | Turn loop, tool dispatch, usage accounting, transcript capture, and trajectory persistence. |

When enabled, native harness memory is deliberately kept to three layers:

| Memory | Role |
|---|---|
| Hippocampus | Fast episodic notes for the current run. |
| Intermediate State | Current task/run state such as last task, workspace, and outcome. |
| Cortex | Durable notes and the persistent Bayesian Skill belief registry. |

The harness stays simple:

- no vendored GenericAgent code
- no browser/runtime framework dependency
- no hidden global memory stack
- no model-specific reasoning framework
- no large prompt-engineering surface inside the harness

The harness is expected to execute, observe, and record. Bayesian Skill evolution is expected to learn reusable procedures from verified outcomes.

Native memory prompt injection and hippocampus/state updates are disabled by default. This keeps baseline and Bayesian runs from silently carrying extra prompt context. Enable the three-layer memory context explicitly with `--native-memory`. The Bayesian Skill registry is not part of this switch: verified outcomes still update Skill beliefs when native memory is off.

## Why Keep The Harness Small?

A large harness can improve short-term task performance, but it also makes the learning layer harder to inspect. Bayesian-Agent keeps the native harness small for three reasons:

1. **Attribution**: when a run improves, the change should be traceable to a Skill/SOP update, not an opaque runtime behavior.
2. **Portability**: the same Skill evolution logic should transfer to GA, mini-swe-agent, Claude Code, or another harness.
3. **Efficiency**: the native harness should spend tokens on task evidence and verification, not on maintaining a complicated harness persona.

This means the harness provides the minimum executable environment:

```text
Prompt + Skill context
      |
      v
LLM turn
      |
      v
Workspace tool call
      |
      v
Observation / transcript
      |
      v
Verifier result
      |
      v
Bayesian Skill update
```

## CLI Usage

`bayesian-agent` is now the default harness:

```bash
python experiments/run_benchmarks.py \
  --harness bayesian-agent \
  --model deepseek-v4-flash \
  --bench core \
  --mode all \
  --limit 1
```

Add `--native-memory` only when you want the native harness to inject three-layer memory context and maintain hippocampus/state during the run:

```bash
python experiments/run_benchmarks.py \
  --harness bayesian-agent \
  --native-memory \
  --model deepseek-v4-flash \
  --bench core \
  --mode all
```

External harnesses remain available:

```bash
--harness genericagent
--harness mini-swe-agent
--harness claude-code
```

Use `--dry-run` to confirm the runner is first-party:

```json
{
  "harness": "bayesian-agent",
  "ba_harness_core": true,
  "native_first_party": true,
  "native_memory": false,
  "harness_root": ".../Bayesian-Agent"
}
```

For RealFin-Bench, the native harness can read cache symlinks inside the workspace, while writes remain restricted to the task workspace.

## Trajectory Artifacts

Every native run writes:

```text
<task-workspace>/
  harness_task.json
  harness_run.json
  native_harness_trajectory.json
  model_response_log.txt
  transcript.txt
```

`native_harness_trajectory.json` records the BA-owned turns, tool calls, observations, usage, and final run metadata. This is the evidence surface consumed by benchmark grading and Bayesian Skill evolution.

## Full-Sample Results

The table below uses full benchmark samples rather than one-task debug checks. SOP-Bench and Lifelong AgentBench contain 20 tasks each; RealFin-Bench contains 40 tasks.

| Benchmark | Model | Mode | Score | Total Tokens | Evidence |
|---|---|---|---:|---:|---|
| SOP-Bench | deepseek-v4-flash | baseline | 19/20 (95.0%) | 1.05M | `results/native_harness_deepseek_v4_flash_full/sop` |
| SOP-Bench | deepseek-v4-flash | bayesian_full | 20/20 (100.0%) | 870k | `results/native_harness_deepseek_v4_flash_full/sop` |
| SOP-Bench | deepseek-v4-flash | bayesian_incremental | 20/20 final, 1/1 repaired | 45k incremental | `results/native_harness_deepseek_v4_flash_full/sop` |
| Lifelong AgentBench | deepseek-v4-flash | baseline | 19/20 (95.0%) | 538k | `results/native_harness_deepseek_v4_flash_full/lifelong` |
| Lifelong AgentBench | deepseek-v4-flash | bayesian_full | 20/20 (100.0%) | 514k | `results/native_harness_deepseek_v4_flash_full/lifelong` |
| Lifelong AgentBench | deepseek-v4-flash | bayesian_incremental | 20/20 final, 1/1 repaired | 65k incremental | `results/native_harness_deepseek_v4_flash_full/lifelong` |
| SOP-Bench | deepseek-v4-pro | baseline | 20/20 (100.0%) | 744k | `results/native_harness_deepseek_v4_pro_full/sop` |
| SOP-Bench | deepseek-v4-pro | bayesian_full | 20/20 (100.0%) | 739k | `results/native_harness_deepseek_v4_pro_full/sop` |
| Lifelong AgentBench | deepseek-v4-pro | baseline | 20/20 (100.0%) | 422k | `results/native_harness_deepseek_v4_pro_full/lifelong` |
| Lifelong AgentBench | deepseek-v4-pro | bayesian_full | 20/20 (100.0%) | 437k | `results/native_harness_deepseek_v4_pro_full/lifelong` |
| RealFin-Bench | deepseek-v4-flash | baseline | 25/40 (62.5%) | 10.29M | `results/native_harness_deepseek_v4_flash_full/realfin` |
| RealFin-Bench | deepseek-v4-flash | bayesian_full | 28/40 (70.0%) | 10.89M | `results/native_harness_deepseek_v4_flash_full/realfin` |
| RealFin-Bench | deepseek-v4-flash | bayesian_incremental | 29/40 final, 4/15 repaired | 3.76M incremental | `results/native_harness_deepseek_v4_flash_full/realfin` |
| RealFin-Bench | deepseek-v4-pro | baseline | 26/40 (65.0%) | 9.54M | `results/native_harness_deepseek_v4_pro_full/realfin_retry` |
| RealFin-Bench | deepseek-v4-pro | bayesian_full | 28/40 (70.0%) | 9.91M | `results/native_harness_deepseek_v4_pro_full/realfin_retry` |
| RealFin-Bench | deepseek-v4-pro | bayesian_incremental | 31/40 final, 5/14 repaired | 4.59M incremental | `results/native_harness_deepseek_v4_pro_full/realfin_retry` |

Interpretation:

- The native harness can now run every benchmark itself and emit its own verified trajectories.
- Native memory is optional and disabled by default; Bayesian-mode runs still use the Bayesian Skill registry for evidence updates.
- On SOP/Lifelong, the minimal native harness reaches 95-100% full-sample accuracy and uses less token budget than the historical GA-backed full runs.
- On RealFin, BA native improves `deepseek-v4-pro` final accuracy from the historical GA-backed 68% to 77.5%, but spends more tokens because the first-party harness keeps domain logic minimal and lets the model inspect cached market data directly.
- The intended design tradeoff is deliberate: the harness remains simple and observable, while long-term capability improvement is pushed into Bayesian Skill/SOP evolution.

## Published GA Validation

The original GA-backed validation remains useful because it covers larger benchmark slices:

| Benchmark | Agent | Accuracy | Total Tokens | Efficiency |
|---|---|---:|---:|---:|
| SOP-Bench | GA baseline | 80% | 1.39M | 11.47 |
| SOP-Bench | GA+Bayesian full | 100% | 1.12M | 17.86 |
| SOP-Bench | GA+Bayesian incremental | 100% final | 268k incremental | 14.93 |
| Lifelong AgentBench | GA baseline | 90% | 690k | 26.07 |
| Lifelong AgentBench | GA+Bayesian full | 95% | 710k | 26.77 |
| Lifelong AgentBench | GA+Bayesian incremental | 100% final | 139k incremental | 14.41 |
| RealFin-Bench | GA baseline | 60% | 3.72M | 6.46 |
| RealFin-Bench | GA+Bayesian full | 65% | 3.70M | 7.02 |
| RealFin-Bench | GA+Bayesian incremental | 68% final | 1.72M incremental | 1.74 |

The native harness result is therefore best read as a harness-minimality result: BA can now run tasks itself, emit trajectories itself, and still rely on the same Bayesian Skill evolution layer for durable improvement.
