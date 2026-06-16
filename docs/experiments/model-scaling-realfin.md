# Model Scaling on RealFin-Bench

This note records a focused model-size ablation for Bayesian-Agent on
RealFin-Bench. All full runs use the same experiment shape:

- Harness: `mini-swe-agent` compatibility backend.
- Evolution mode: `bayesian_full`.
- Benchmark: `realfin` with 40 tasks.
- Runtime controls: `--max-turns 16 --mini-swe-env-timeout 180`.
- Metric: task completion accuracy, token usage, and efficiency.

Efficiency is computed as successful tasks per one million tokens:

```text
efficiency = successes / total_tokens * 1,000,000
```

## Commands

Before the full run, a one-task smoke test was run for both DashScope models to
verify that the API, adapter, and verifier chain worked:

| Model | Smoke Result | Total Tokens | Evidence |
|---|---:|---:|---|
| `qwen3.5-35b-a3b` | 1/1 | 151,658 | `results/model_scaling_mini_swe_realfin_qwen_smoke_20260612_004502/qwen3_5_35b_a3b` |
| `qwen3.5-122b-a10b` | 0/1 | 65,267 | `results/model_scaling_mini_swe_realfin_qwen_smoke_20260612_004502/qwen3_5_122b_a10b` |

The smoke test is only an execution check, not a quality claim.

Full Qwen runs used the DashScope OpenAI-compatible endpoint:

```bash
cd /Users/wuxiaojun/code/My-Agent/Bayesian-Agent
set -a && . ./.env && set +a

.venv/bin/python experiments/run_benchmarks.py \
  --harness mini-swe-agent \
  --mini-swe-agent-root /Users/wuxiaojun/code/My-Agent/mini-swe-agent \
  --model qwen3.5-35b-a3b \
  --bench realfin \
  --mode bayesian-full \
  --out-root results/model_scaling_mini_swe_realfin_qwen_full_screen_20260612_004933/qwen3_5_35b_a3b \
  --api-key-env ALI_API_KEY \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --max-turns 16 \
  --mini-swe-env-timeout 180 \
  --limit 0
```

The `qwen3.5-122b-a10b` command was identical except for `--model` and
`--out-root`. In this local desktop shell, `nohup` exited immediately with empty
logs, so the full runs were submitted as detached `screen` sessions. The result
artifacts and logs are still written under the same run root.

## Results

The Qwen rows are newly run in this session. The DeepSeek rows are preexisting
full RealFin artifacts from the mini-swe-agent backend.

| Model | Provider | Size Label | Evidence Type | Success | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.5-35b-a3b` | DashScope | 35B / A3B | newly_run | 18/40 | 45.0% | 6.95M | 370k | 7.32M | 2.46 |
| `qwen3.5-122b-a10b` | DashScope | 122B / A10B | newly_run | 3/40 | 7.5% | 3.19M | 144k | 3.34M | 0.90 |
| `deepseek-v4-flash` | DeepSeek | 284B, user-provided | preexisting_artifact | 22/40 | 55.0% | 6.50M | 453k | 6.96M | 3.16 |
| `deepseek-v4-pro` | DeepSeek | 1.6T, user-provided | preexisting_artifact | 28/40 | 70.0% | 6.28M | 459k | 6.73M | 4.16 |

Evidence paths:

- `results/model_scaling_mini_swe_realfin_qwen_full_screen_20260612_004933/qwen3_5_35b_a3b/bayesian_full/results.json`
- `results/model_scaling_mini_swe_realfin_qwen_full_screen_20260612_004933/qwen3_5_122b_a10b/bayesian_full/results.json`
- `results/mini_swe_agent/deepseek_v4_flash/realfin/bayesian_full/results.json`
- `results/mini_swe_agent/deepseek_v4_pro/realfin/bayesian_full/results.json`

## Failure Shape

The failure distribution matters as much as the final score because
Bayesian-Agent can only evolve Skills from observable execution evidence.

| Model | Failed Tasks | Failures With No Output File | Failures With Output File But Invalid Details |
|---|---:|---:|---:|
| `qwen3.5-35b-a3b` | 22 | 10 | 12 |
| `qwen3.5-122b-a10b` | 37 | 37 | 0 |
| `deepseek-v4-flash` | 18 | 9 | 9 |
| `deepseek-v4-pro` | 12 | 7 | 5 |

`qwen3.5-122b-a10b` is the most important negative result. Its failures are
dominated by `file_created = 0`, which means the backend often did not produce
the required benchmark artifact. That is a harness-output-contract failure
shape, not merely a weak indicator-computation failure. It also explains why the
122B run spent fewer tokens: many tasks ended before producing a valid artifact,
so lower token usage here should not be read as better efficiency.

`qwen3.5-35b-a3b` is a more useful Skill-evolution target. Many failed tasks
created a file and missed specific verifier predicates. Those failures expose
actionable evidence, such as wrong format, incomplete metrics, or one missing
condition, so posterior-weighted Skill patches have something concrete to learn
from.

## Case Notes

`task_30_composite_scoring` shows that scaling is not monotonic across models.
Qwen 35B-A3B and DeepSeek V4 Flash passed it, while DeepSeek V4 Pro and Qwen
122B-A10B failed. The 122B run did not create the file; Pro created no valid
artifact for this task in the stored run.

`task_38_feature_engineering` shows the opposite pattern. Qwen 122B-A10B and
DeepSeek V4 Pro passed, while Qwen 35B-A3B created a file but missed
`feature_count_valid` and `signal_present`. Larger models can still help on
feature-design-heavy tasks, but the benefit only appears when the backend
successfully preserves the file-writing contract.

`task_22_bullish_sandwich` and `task_23_gentle_volume_rise` failed for all four
models. For Qwen 35B-A3B, DeepSeek Flash, and DeepSeek Pro, the output files
were created but missed a specific technical predicate such as MA trend,
valid return, or turnover validity. These are good candidates for targeted
Skill rewrite rules. For Qwen 122B-A10B, the same tasks produced no output file,
so the immediate fix should be adapter/prompt-contract stabilization before
domain-specific Skill evolution.

`task_39_interest_rate_sector_rotation` and
`task_40_commodity_equity_linkage` failed for all four models with no valid
output file in the compared runs. These failures likely need stronger data
access/tooling guidance in the harness, not only larger model size.

## Takeaways

Within the DeepSeek family, the larger Pro model improves RealFin Bayesian full
accuracy from 55.0% to 70.0% while using slightly fewer total tokens. That is the
cleanest size-scaling signal in this set.

Across providers and model families, the result is not monotonic. Qwen 35B-A3B
outperforms Qwen 122B-A10B by a wide margin under the current mini-swe-agent
adapter, and DeepSeek Flash also outperforms Qwen 122B-A10B. This suggests that
Bayesian-Agent performance depends on the full stack:

- base model capability,
- backend tool behavior,
- file-writing contract adherence,
- benchmark data access,
- Skill evolution evidence quality.

For Bayesian Skill evolution, a model that reliably produces imperfect but
verifiable artifacts may be more valuable than a larger model that frequently
fails to emit the required artifact. Posterior updates and Skill rewrites need
observations; no-output failures provide much weaker learning signal than
fine-grained verifier failures.

## Protocol Risks

- Single run per model; no variance estimate.
- Cross-provider comparison mixes model size with serving stack, tokenizer,
  decoding defaults, and adapter behavior.
- Qwen size labels include total and active parameter names from the model
  identifiers; DeepSeek size labels are user-provided.
- RealFin-Bench depends on cached market-data access and file-based grading, so
  failures may reflect harness/data behavior as well as model reasoning.
