# Experiments

The first prototype was validated inside GenericAgent with `deepseek-v4-flash`.

These experiments are meant to show two deployment paths: Bayesian-Agent can run a full self-evolving loop from scratch, and it can also attach to an existing agent as an incremental repair layer. GenericAgent is the current experimental harness, not a hard dependency of the Bayesian-Agent method.

## Running SOP-Bench and Lifelong AgentBench

The repository includes one model-agnostic script for SOP-Bench and Lifelong AgentBench. It uses GenericAgent only as the execution harness; Bayesian-Agent owns benchmark orchestration and Skill evolution.

```bash
cd Bayesian-Agent
export GENERICAGENT_ROOT="/path/to/GenericAgent"
export DEEPSEEK_API_KEY="sk-..."
export MODEL="deepseek-v4-flash"
"$GENERICAGENT_ROOT/.venv/bin/python" \
  experiments/run_sop_lifelong.py \
  --genericagent-root "$GENERICAGENT_ROOT" \
  --model "$MODEL" \
  --mode all \
  --bench core \
  --out-root "temp/sop_lifelong_${MODEL//-/_}"
```

Default `--mode all` runs:

- `baseline`: GenericAgent on SOP-Bench and Lifelong AgentBench.
- `bayesian_full`: Bayesian full self-evolution from scratch.
- `bayesian_incremental`: Bayesian repair using the fresh baseline and rerunning only failed tasks.

Bayesian modes now persist per-task Skill evolution artifacts under:

```text
<run-root>/skill_evolution/
  index.json
  sop_bench/
    sop_01/
      skill_context_before.md
      skill_context_after.md
      posterior_context_before.md
      posterior_context_after.md
      belief_before.json
      belief_after.json
      snapshot_before.json
      snapshot_after.json
```

`skill_context_before.md` is the exact model-facing Skill/SOP text injected into that task. For the built-in benchmarks, it contains stable benchmark guardrails and any active `Bayesian Failure-Mode Patches`. A patch becomes active only after the same failure mode has at least two verified occurrences, so single failures stay audit-only. `skill_context_after.md` is the next model-facing Skill/SOP text after verifier feedback is recorded.

`posterior_context_before.md` and `posterior_context_after.md` are audit artifacts for the Bayesian belief state. They may include posterior summaries such as `posterior_success`, `alpha`, `beta`, observations, and rewrite decisions, but those numeric summaries are not injected into the benchmark prompt.

For older result directories that only contain `results.json`, rebuild the Skill evolution trail without rerunning the model:

```bash
bayesian-agent replay-skill-artifacts \
  --results results/sop_lifelong_deepseek_v4_flash/bayesian_full/results.json
```

Use `--limit 1` for a smoke test before full runs. To switch to `deepseek-v4-pro`, set `MODEL=deepseek-v4-pro`; the script itself is the same.

To repair an existing GA baseline instead of using a fresh baseline from the same run, pass the baseline result files:

```bash
"$GENERICAGENT_ROOT/.venv/bin/python" \
  experiments/run_sop_lifelong.py \
  --genericagent-root "$GENERICAGENT_ROOT" \
  --model "$MODEL" \
  --mode bayesian-incremental \
  --bench core \
  --baseline-results artifacts/ga_deepseek_baseline/sop_results.json \
  --baseline-results artifacts/ga_deepseek_baseline/lifelong_results.json \
  --out-root "temp/sop_lifelong_${MODEL//-/_}_incremental_from_ga"
```

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
| SOP-Bench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 254k | 14k | 268k | 14.93 |
| Lifelong AgentBench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 129k | 10k | 139k | 14.41 |

Published example artifacts are stored under `artifacts/`. New live runs write their own result and Skill evolution artifacts under the selected `--out-root`.

The cross-harness path depends on the same evidence format: any agent framework that emits verified trajectories can feed the Bayesian Skill registry and receive model-facing Skill/SOP text through an adapter.
