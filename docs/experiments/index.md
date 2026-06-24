# Experiments

The first prototype was validated inside GenericAgent with `deepseek-v4-flash`. Bayesian-Agent now also includes a first-party native harness, so experiments can run either inside BA itself or through an external compatibility backend.

These experiments are meant to show three deployment paths: Bayesian-Agent can run a minimal native harness, run a full self-evolving loop from scratch, and attach to an existing agent as an incremental repair layer.

## Running Benchmarks

The repository includes one model-agnostic script for SOP-Bench, Lifelong AgentBench, and RealFin-Bench. By default it uses the first-party BA native harness. Select the benchmark with `--bench core`, `--bench sop`, `--bench lifelong`, or `--bench realfin`.

```bash
cd Bayesian-Agent
export DEEPSEEK_API_KEY="sk-..."
export MODEL="deepseek-v4-flash"
python \
  experiments/run_benchmarks.py \
  --harness bayesian-agent \
  --model "$MODEL" \
  --mode all \
  --bench core
```

External backends remain available:

```bash
--harness genericagent
--harness mini-swe-agent
--harness claude-code
```

`--bench core` selects SOP-Bench and Lifelong AgentBench together, but it does not write a combined result root. It fans out to `results/sop_${MODEL//-/_}` and `results/lifelong_${MODEL//-/_}`. If you pass `--out-root temp/core_${MODEL//-/_}`, that path is treated as a parent and the benchmark roots become `temp/core_${MODEL//-/_}/sop` and `temp/core_${MODEL//-/_}/lifelong`.

Default `--mode all` runs:

- `baseline`: selected harness without Bayesian Skill context.
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
  --results results/sop_deepseek_v4_flash/bayesian_full/results.json
```

Use `--limit 1` for a smoke test before full runs. To switch to `deepseek-v4-pro`, set `MODEL=deepseek-v4-pro`; the script itself is the same. For RealFin-Bench, use the same entrypoint with `--bench realfin`.

To repair an existing GA baseline instead of using a fresh baseline from the same run, pass the baseline result files:

```bash
"$GENERICAGENT_ROOT/.venv/bin/python" \
  experiments/run_benchmarks.py \
  --harness genericagent \
  --genericagent-root "$GENERICAGENT_ROOT" \
  --model "$MODEL" \
  --mode bayesian-incremental \
  --bench core \
  --baseline-results artifacts/ga_deepseek_baseline/sop_results.json \
  --baseline-results artifacts/ga_deepseek_baseline/lifelong_results.json
```

## Native Harness Full-Sample Results

These are local full-sample checks with the first-party BA native harness. SOP-Bench and Lifelong AgentBench contain 20 tasks each; RealFin-Bench contains 40 tasks.

For the focused RealFin comparison between Bayesian Skill evolution and the frequentist control, including task-level case analysis, see [Bayesian vs Frequentist on RealFin-Bench](bayesian-vs-frequentist-realfin.md).

For the focused model-size ablation on RealFin-Bench with the mini-swe-agent backend, see [Model Scaling on RealFin-Bench](model-scaling-realfin.md).

| Benchmark | Model | Mode | Score | Total Tokens | Evidence |
|---|---|---|---:|---:|---|
| SOP-Bench | deepseek-v4-flash | baseline | 19/20 (95.0%) | 1.05M | `results/native_harness_deepseek_v4_flash_full/sop` |
| SOP-Bench | deepseek-v4-flash | bayesian_full | 20/20 (100.0%) | 870k | `results/native_harness_deepseek_v4_flash_full/sop` |
| SOP-Bench | deepseek-v4-flash | bayesian_incremental | 20/20 final, 1/1 repaired | 45k incremental | `results/native_harness_deepseek_v4_flash_full/sop` |
| Lifelong AgentBench | deepseek-v4-flash | baseline | 19/20 (95.0%) | 538k | `results/native_harness_deepseek_v4_flash_full/lifelong` |
| Lifelong AgentBench | deepseek-v4-flash | bayesian_full | 20/20 (100.0%) | 514k | `results/native_harness_deepseek_v4_flash_full/lifelong` |
| Lifelong AgentBench | deepseek-v4-flash | bayesian_incremental | 20/20 final, 1/1 repaired | 65k incremental | `results/native_harness_deepseek_v4_flash_full/lifelong` |
| RealFin-Bench | deepseek-v4-flash | baseline | 25/40 (62.5%) | 10.29M | `results/native_harness_deepseek_v4_flash_full/realfin` |
| RealFin-Bench | deepseek-v4-flash | bayesian_full | 28/40 (70.0%) | 10.89M | `results/native_harness_deepseek_v4_flash_full/realfin` |
| RealFin-Bench | deepseek-v4-flash | bayesian_incremental | 29/40 final, 4/15 repaired | 3.76M incremental | `results/native_harness_deepseek_v4_flash_full/realfin` |
| SOP-Bench | deepseek-v4-pro | baseline | 20/20 (100.0%) | 744k | `results/native_harness_deepseek_v4_pro_full/sop` |
| SOP-Bench | deepseek-v4-pro | bayesian_full | 20/20 (100.0%) | 739k | `results/native_harness_deepseek_v4_pro_full/sop` |
| Lifelong AgentBench | deepseek-v4-pro | baseline | 20/20 (100.0%) | 422k | `results/native_harness_deepseek_v4_pro_full/lifelong` |
| Lifelong AgentBench | deepseek-v4-pro | bayesian_full | 20/20 (100.0%) | 437k | `results/native_harness_deepseek_v4_pro_full/lifelong` |
| RealFin-Bench | deepseek-v4-pro | baseline | 26/40 (65.0%) | 9.54M | `results/native_harness_deepseek_v4_pro_full/realfin_retry` |
| RealFin-Bench | deepseek-v4-pro | bayesian_full | 28/40 (70.0%) | 9.91M | `results/native_harness_deepseek_v4_pro_full/realfin_retry` |
| RealFin-Bench | deepseek-v4-pro | bayesian_incremental | 31/40 final, 5/14 repaired | 4.59M incremental | `results/native_harness_deepseek_v4_pro_full/realfin_retry` |

The native harness is intentionally simple: LLM, workspace tools, trajectory capture, and optional three-layer memory. Native memory is disabled by default; Bayesian Skill registry updates still run in Bayesian modes. Its job is to execute and observe. More capability improvement is pushed into Bayesian Skill/SOP evolution.

## Published GA Validation

The earlier published validation used GenericAgent as the execution backend on larger benchmark slices.

### Baseline

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA | deepseek-v4-flash | 80% | 1.34M | 57k | 1.39M | 11.47 |
| Lifelong AgentBench | GA | deepseek-v4-flash | 90% | 649k | 42k | 690k | 26.07 |

### Full Self-Evolving Mode

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+Bayesian | deepseek-v4-flash | 100% | 1.07M | 52k | 1.12M | 17.86 |
| Lifelong AgentBench | GA+Bayesian | deepseek-v4-flash | 95% | 666k | 44k | 710k | 26.77 |

### Incremental Repair Mode

Bayesian-Agent read the GA baseline traces and reran only failed tasks.

| Benchmark | Agent | Model | Final Accuracy | Incremental Input | Incremental Output | Incremental Total | Incremental Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 254k | 14k | 268k | 14.93 |
| Lifelong AgentBench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 129k | 10k | 139k | 14.41 |

## Historical GA-Backed RealFin Run

The earlier RealFin validation used GenericAgent as the execution backend with `deepseek-v4-pro`.

| Benchmark | Agent | Model | Accuracy | Total Tokens | Evidence |
|---|---|---|---:|---:|---|
| RealFin-Bench | GA | deepseek-v4-pro | 60% | 3.72M | `results/realfin_deepseek_v4_pro_20260602` |
| RealFin-Bench | GA+Bayesian | deepseek-v4-pro | 65% | 3.70M | `results/realfin_deepseek_v4_pro_20260602` |
| RealFin-Bench | GA+BayesianIncremental | deepseek-v4-pro | 68% | 1.72M incremental | `results/realfin_deepseek_v4_pro_20260602` |

Compared with this historical GA-backed RealFin run, BA native reaches 77.5% final accuracy on `deepseek-v4-pro`, but spends more tokens because the minimal first-party harness lets the model inspect cached market data directly.

Published example artifacts are stored under `artifacts/`. New live runs write their own result and Skill evolution artifacts under each benchmark-specific result root.

The cross-harness path depends on the same evidence format: any agent framework that emits verified trajectories can feed the Bayesian Skill registry and receive model-facing Skill/SOP text through an adapter.
