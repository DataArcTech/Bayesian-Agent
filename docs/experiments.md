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

Artifacts are stored under `artifacts/`.

The cross-harness path depends on the same evidence format: any agent framework that emits verified trajectories can feed the Bayesian Skill registry and receive posterior-weighted Skill context through an adapter.
