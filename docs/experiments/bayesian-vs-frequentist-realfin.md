# Bayesian vs Frequentist on RealFin-Bench

This note compares Bayesian Skill evolution with a frequentist Skill-evolving control on RealFin-Bench. The goal is not to claim that Bayesian updating is universally better in every run. The narrower claim supported here is:

> When agent runs are expensive, samples are limited, and failures are sparse but informative, a Bayesian evolution layer gives a more stable way to use prior Skill belief, uncertainty, and failure-mode evidence than a pure empirical-frequency update.

RealFin-Bench is a useful stress test for this claim because each task requires multi-step market-data inspection, strict output contracts, and robust handling of missing or blank fields. A single failed case is expensive in tokens and latency, so the method cannot wait for large-sample frequency estimates to stabilize.

## Setup

All rows below are full RealFin-Bench runs with 40 tasks. `Efficiency` is the benchmark runner's success-per-million-token score.

| Field | Value |
|---|---|
| Benchmark | RealFin-Bench |
| Task count | 40 |
| Backends | GenericAgent compatibility backend; Bayesian-Agent native backend |
| Models | `deepseek-v4-flash`; `deepseek-v4-pro` |
| Bayesian method | Categorical Bayesian Skill evolution |
| Frequentist control | Empirical success-rate Skill evolution, no prior, no smoothing |
| Mode | Full self-evolving run from scratch |

Evidence types:

- `preexisting_artifact`: already available local result artifact.
- `newly_run`: produced in the 2026-06-11 rerun for the frequentist native backend comparison.

## Main Results

| Backend | Model | Method | Accuracy | Success | Input Tokens | Output Tokens | Total Tokens | Efficiency | Elapsed | Evidence |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| GA | `deepseek-v4-flash` | Bayesian | 52.5% | 21/40 | 2.90M | 244k | 3.15M | 6.67 | 2318s | `preexisting_artifact` |
| GA | `deepseek-v4-flash` | Frequentist | 52.5% | 21/40 | 3.06M | 244k | 3.31M | 6.35 | 2958s | `preexisting_artifact` |
| GA | `deepseek-v4-pro` | Bayesian | 65.0% | 26/40 | 3.38M | 323k | 3.70M | 7.02 | 5756s | `preexisting_artifact` |
| GA | `deepseek-v4-pro` | Frequentist | 57.5% | 23/40 | 3.46M | 309k | 3.77M | 6.10 | 6338s | `preexisting_artifact` |
| BA native | `deepseek-v4-flash` | Bayesian | 70.0% | 28/40 | 10.43M | 463k | 10.89M | 2.57 | 4485s | `preexisting_artifact` |
| BA native | `deepseek-v4-flash` | Frequentist | 37.5% | 15/40 | 3.79M | 236k | 4.03M | 3.72 | 3698s | `newly_run` |
| BA native | `deepseek-v4-pro` | Bayesian | 70.0% | 28/40 | 9.33M | 579k | 9.91M | 2.83 | 10573s | `preexisting_artifact` |
| BA native | `deepseek-v4-pro` | Frequentist | 45.0% | 18/40 | 3.86M | 299k | 4.16M | 4.32 | 5684s | `newly_run` |

## What The Table Shows

On the GA backend, Bayesian is never worse than the frequentist control. With `deepseek-v4-flash`, both methods solve 21 of 40 tasks, but Bayesian uses fewer tokens and finishes faster. With `deepseek-v4-pro`, Bayesian solves 26 tasks while the frequentist control solves 23.

On the BA native backend, the gap is much larger. Bayesian solves 28 of 40 tasks for both models. The frequentist control solves only 15 tasks with `deepseek-v4-flash` and 18 tasks with `deepseek-v4-pro`.

The native frequentist runs spend far fewer tokens, so their token efficiency can look higher. That number is misleading if read alone: the lower token count often comes from early failures such as missing requested output files or crashes on blank OHLCV fields. For RealFin, task completion is the primary metric; token efficiency is meaningful only after accuracy is held roughly constant.

## Pairwise Deltas

| Backend | Model | Bayesian Success | Frequentist Success | Accuracy Delta | Token Delta | Interpretation |
|---|---|---:|---:|---:|---:|---|
| GA | `deepseek-v4-flash` | 21/40 | 21/40 | 0.0 pts | Bayesian uses 161k fewer tokens | Accuracy tie; Bayesian wins efficiency. |
| GA | `deepseek-v4-pro` | 26/40 | 23/40 | +7.5 pts | Bayesian uses 68k fewer tokens | Bayesian wins both completion and token cost. |
| BA native | `deepseek-v4-flash` | 28/40 | 15/40 | +32.5 pts | Bayesian uses 6.87M more tokens | Bayesian spends more exploration budget but solves far more tasks. |
| BA native | `deepseek-v4-pro` | 28/40 | 18/40 | +25.0 pts | Bayesian uses 5.75M more tokens | Bayesian again wins completion; frequentist is cheaper but much less reliable. |

## Case Analysis

### Case 1: Frequentist Overreacts To Sparse Native Evidence

In the BA native `deepseek-v4-flash` run, Bayesian succeeds on several tasks where the frequentist control fails with either blank-field crashes or missing output files.

| Task | Bayesian | Frequentist | Frequentist failure mode |
|---|---:|---:|---|
| `task_04_consecutive_rise` | success | fail | `blank_ohlcv_field_crashed_calculation` |
| `task_05_triple_golden_cross` | success | fail | `blank_ohlcv_field_crashed_calculation` |
| `task_06_bollinger_squeeze` | success | fail | `blank_ohlcv_field_crashed_calculation` |
| `task_12_catl_correlation_kdj` | success | fail | `missing_requested_output_file` |
| `task_18_momentum_portfolio` | success | fail | `missing_requested_output_file` |
| `task_29_momentum_reversal_combo` | success | fail | `blank_ohlcv_field_crashed_calculation` |
| `task_30_composite_scoring` | success | fail | `missing_requested_output_file` |
| `task_31_triple_timeframe_macd` | success | fail | `missing_requested_output_file` |
| `task_36_intraday_anomaly` | success | fail | `missing_requested_output_file` |
| `task_38_feature_engineering` | success | fail | `blank_ohlcv_field_crashed_calculation` |

This is the most important qualitative pattern in the native comparison. The failure is not merely "the answer was numerically off"; many frequentist failures happen before the benchmark can even evaluate the requested artifact. Bayesian's advantage here is consistent with a posterior-driven Skill layer that preserves robust guardrails for output contracts, blank-field handling, and evidence extraction, instead of relying only on the empirical success ratio seen so far.

### Case 2: Bayesian Is Not Just Spending More Tokens Blindly

The BA native `deepseek-v4-flash` Bayesian run spends 10.89M tokens, while the frequentist run spends 4.03M. More token use alone would not be impressive if it only produced marginal gains. Here the completion gap is large: 28/40 vs 15/40.

Several high-complexity tasks show this pattern:

| Task | Bayesian scores | Frequentist scores |
|---|---|---|
| `task_29_momentum_reversal_combo` | all six checks pass | all six checks fail after blank-field crash |
| `task_30_composite_scoring` | all six checks pass | output file missing; all checks fail |
| `task_31_triple_timeframe_macd` | all five checks pass | output file missing; all checks fail |
| `task_38_feature_engineering` | all six checks pass | blank-field crash; all checks fail |

These tasks require combining multiple indicators, maintaining output format discipline, and handling dataset irregularities. The Bayesian run spends more context and tool budget, but it converts that budget into completed artifacts.

### Case 3: GA Flash Shows A Tie, Not A Universal Win

On GA with `deepseek-v4-flash`, both methods finish at 21/40. The task-level swaps are balanced:

| Direction | Tasks |
|---|---|
| Bayesian succeeds, frequentist fails | `task_10_macd_histogram_trend`, `task_12_catl_correlation_kdj`, `task_32_weekly_breakout_daily_pullback`, `task_36_intraday_anomaly` |
| Frequentist succeeds, Bayesian fails | `task_17_fund_flow_obv`, `task_19_52week_high_followthrough`, `task_27_atr_volatility_breakout`, `task_38_feature_engineering` |

This is a useful negative-control style result. Bayesian is not automatically better on every model/backend pairing. In this setting, the better claim is efficiency: same completion rate, fewer tokens, and shorter elapsed time.

### Case 4: GA Pro Shows Bayesian's Advantage With A Stronger Model

On GA with `deepseek-v4-pro`, Bayesian solves 26 tasks and the frequentist control solves 23. The Bayesian-only successes are:

```text
task_01_macd_rsi_filter
task_24_doji_to_surge
task_27_atr_volatility_breakout
task_31_triple_timeframe_macd
task_32_weekly_breakout_daily_pullback
task_38_feature_engineering
```

The frequentist-only successes are:

```text
task_12_catl_correlation_kdj
task_25_golden_valley
task_37_floor_ceiling_reversal
```

This suggests the Bayesian layer benefits more when the backend model is capable enough to use the evolved Skill context. The posterior evidence does not replace model capability; it conditions a capable model toward more reliable task behavior.

### Case 5: Frequentist Still Has Local Wins

The BA native `deepseek-v4-pro` frequentist run succeeds on `task_24_doji_to_surge`, while the Bayesian run fails with `invalid_realfin_output_format`. This matters because it prevents an overly clean story.

The evidence supports a practical claim, not a dogma: Bayesian evolution is more robust overall in these runs, but single-case outcomes can still favor the frequentist control because LLM execution is stochastic, task difficulty varies, and current policies are still heuristic.

## Why This Supports The Bayesian Claim

The frequentist control estimates Skill reliability from observed frequencies. In small samples, this is brittle:

```text
p_hat(success | skill, context) = successes / observations
```

Bayesian evolution keeps a belief state instead:

```text
P(skill quality | evidence) proportional to P(evidence | skill quality) P(skill quality)
```

In this implementation, that belief is operationalized through posterior-weighted Skill selection, failure-mode accumulation, and conservative rewrite/patch activation. The important distinction is not decorative math. It changes the control behavior:

| Issue | Frequentist control | Bayesian evolution |
|---|---|---|
| Very few observations | Empirical rate can swing sharply. | Prior and posterior uncertainty keep updates conservative. |
| Repeated failure modes | Treated mostly as counts. | Become evidence for targeted Skill patches. |
| Expensive cases | Needs more observations to stabilize. | Can act under uncertainty with fewer samples. |
| Cross-harness transfer | Frequency estimates are tied to observed local history. | Belief state can be conditioned by benchmark, model, harness, and failure metadata. |

RealFin demonstrates this difference because each task is costly, and failures such as missing output files or blank OHLCV crashes are sparse but highly diagnostic. Bayesian evolution can preserve and reuse those diagnostics as Skill evidence; a pure frequentist controller has less structure for deciding how much to trust or generalize early observations.

## Caveats

These results should be read as local experimental evidence, not as a universal theorem.

- Each row is a single full run, not a repeated-seed average.
- Some artifacts are from earlier local runs, while the native frequentist rows were newly run on 2026-06-11.
- The native Bayesian rows spend substantially more tokens, so they should be interpreted as stronger completion performance rather than cheaper execution.
- RealFin is data-heavy and output-contract-sensitive; results may differ on short text-only benchmarks.

## Artifact Inventory

| Backend | Model | Method | Evidence Type | Artifact |
|---|---|---|---|---|
| GA | `deepseek-v4-flash` | Bayesian | `preexisting_artifact` | `results/realfin_deepseek_v4_flash_full_20260602/bayesian_full/results.json` |
| GA | `deepseek-v4-flash` | Frequentist | `preexisting_artifact` | `results/frequentist_ga_deepseek_v4_flash_20260611/realfin/bayesian_full/results.json` |
| GA | `deepseek-v4-pro` | Bayesian | `preexisting_artifact` | `results/realfin_deepseek_v4_pro_20260602/bayesian_full/results.json` |
| GA | `deepseek-v4-pro` | Frequentist | `preexisting_artifact` | `results/frequentist_ga_deepseek_v4_pro_realfin_20260611/bayesian_full/results.json` |
| BA native | `deepseek-v4-flash` | Bayesian | `preexisting_artifact` | `results/native_harness_deepseek_v4_flash_full/realfin/bayesian_full/results.json` |
| BA native | `deepseek-v4-flash` | Frequentist | `newly_run` | `results/native_backend_frequentist_realfin_full_rerun_20260611_203737/deepseek_v4_flash/bayesian_full/results.json` |
| BA native | `deepseek-v4-pro` | Bayesian | `preexisting_artifact` | `results/native_harness_deepseek_v4_pro_full/realfin_retry/bayesian_full/results.json` |
| BA native | `deepseek-v4-pro` | Frequentist | `newly_run` | `results/native_backend_frequentist_realfin_full_rerun_20260611_203737/deepseek_v4_pro/bayesian_full/results.json` |

