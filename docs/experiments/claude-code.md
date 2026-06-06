# Claude Code Benchmark Results

This page records the Claude Code compatibility-backend runs available in the
local experiment artifacts. Bayesian-Agent handled benchmark orchestration,
grading, and result writing; Claude Code executed each task prompt through the
adapter boundary.

Evidence type: newly run / verified local artifact. Metrics were read from
`results/claude_code/**/baseline/results.json` and
`results/claude_code_smoke/**/baseline/results.json`.

## Commands

Full baseline runs used this harness shape:

```bash
python experiments/run_benchmarks.py \
  --harness claude-code \
  --model "$MODEL" \
  --bench "$BENCH" \
  --mode baseline \
  --out-root "results/claude_code/${MODEL_SLUG}/${BENCH}"
```

Smoke runs used the same backend with `--limit 1` and smoke-specific output
roots under `results/claude_code_smoke/`.

## Full Baseline Results

| Model | Benchmark | Accuracy | Total Tokens | Efficiency | Failed Tasks |
|---|---:|---:|---:|---:|---|
| deepseek-v4-flash | SOP-Bench | 18/20 (90%) | 5,887,709 | 3.06 | `sop_15`, `sop_16` |
| deepseek-v4-flash | Lifelong AgentBench | 20/20 (100%) | 1,552,970 | 12.88 | none |
| deepseek-v4-flash | RealFin-Bench | 31/40 (77.5%) | 49,414,353 | 0.63 | 9 tasks |
| deepseek-v4-pro[1m] | SOP-Bench | 13/20 (65%) | 2,759,895 | 4.71 | 7 tasks |
| deepseek-v4-pro[1m] | Lifelong AgentBench | 20/20 (100%) | 1,552,632 | 12.88 | none |
| deepseek-v4-pro[1m] | RealFin-Bench | 26/40 (65%) | 27,034,780 | 0.96 | 14 tasks |

## Bayesian-Agent Modes with Claude Code Backend

These runs use Claude Code as the task execution backend and Bayesian-Agent for
benchmark orchestration, verifier grading, Skill evidence updates, and
failure-mode patch injection.

Evidence type: newly run / verified local artifact, 2026-06-06 to 2026-06-07.
Metrics were read from:

```text
results/claude_code/{deepseek_v4_flash,deepseek_v4_pro_1m}/{sop,lifelong,realfin}/{baseline,bayesian_full,bayesian_incremental}/results.json
```

Commands used the same shape for all three benchmarks:

```bash
python experiments/run_benchmarks.py \
  --harness claude-code \
  --model "$MODEL" \
  --bench "$BENCH" \
  --mode "$MODE" \
  --out-root "results/claude_code/${MODEL_SLUG}/${BENCH}"
```

For `bayesian_incremental`, the runner received the matching Claude Code
baseline `results.json` and reran only failed tasks.

### deepseek-v4-flash

| Benchmark | Mode | Score | Repaired | Input Tokens | Output Tokens | Total Tokens | Efficiency | Cumulative Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SOP-Bench | baseline | 18/20 (90.0%) | - | 5.81M | 78k | 5.89M | 3.06 | - |
| SOP-Bench | bayesian_full | 20/20 (100.0%) | - | 3.39M | 55k | 3.44M | 5.81 | - |
| SOP-Bench | bayesian_incremental | 20/20 (100.0%) | 2/2 | 360k | 6k | 366k | 5.46 | 6.25M |
| Lifelong AgentBench | baseline | 20/20 (100.0%) | - | 1.53M | 25k | 1.55M | 12.88 | - |
| Lifelong AgentBench | bayesian_full | 20/20 (100.0%) | - | 1.54M | 24k | 1.57M | 12.77 | - |
| Lifelong AgentBench | bayesian_incremental | 20/20 (100.0%) | 0/0 | 0 | 0 | 0 | 0.00 | 1.55M |
| RealFin-Bench | baseline | 31/40 (77.5%) | - | 48.57M | 841k | 49.41M | 0.63 | - |
| RealFin-Bench | bayesian_full | 32/40 (80.0%) | - | 47.18M | 774k | 47.95M | 0.67 | - |
| RealFin-Bench | bayesian_incremental | 35/40 (87.5%) | 4/9 | 7.04M | 152k | 7.19M | 0.56 | 56.61M |

### deepseek-v4-pro[1m]

| Benchmark | Mode | Score | Repaired | Input Tokens | Output Tokens | Total Tokens | Efficiency | Cumulative Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SOP-Bench | baseline | 13/20 (65.0%) | - | 2.71M | 50k | 2.76M | 4.71 | - |
| SOP-Bench | bayesian_full | 19/20 (95.0%) | - | 2.78M | 45k | 2.82M | 6.74 | - |
| SOP-Bench | bayesian_incremental | 20/20 (100.0%) | 7/7 | 958k | 20k | 977k | 7.16 | 3.74M |
| Lifelong AgentBench | baseline | 20/20 (100.0%) | - | 1.53M | 24k | 1.55M | 12.88 | - |
| Lifelong AgentBench | bayesian_full | 20/20 (100.0%) | - | 1.55M | 24k | 1.57M | 12.74 | - |
| Lifelong AgentBench | bayesian_incremental | 20/20 (100.0%) | 0/0 | 0 | 0 | 0 | 0.00 | 1.55M |
| RealFin-Bench | baseline | 26/40 (65.0%) | - | 26.53M | 509k | 27.03M | 0.96 | - |
| RealFin-Bench | bayesian_full | 27/40 (67.5%) | - | 32.04M | 691k | 32.73M | 0.82 | - |
| RealFin-Bench | bayesian_incremental | 30/40 (75.0%) | 4/14 | 14.15M | 296k | 14.45M | 0.28 | 41.48M |

Notes:

- Incremental rows report repair-only token usage. `Cumulative Tokens` reports
  baseline cost plus the incremental repair run cost.
- Lifelong AgentBench had no failed baseline tasks, so the incremental run
  performed no model calls and records `0` repair tokens for both models.
- SOP-Bench shows the clearest benefit. With `deepseek-v4-flash`, Bayesian
  full self-evolution reaches 100% while using fewer total tokens than the
  Claude Code baseline. With `deepseek-v4-pro[1m]`, incremental repair fixes
  all 7 baseline failures and raises final accuracy from 65.0% to 100.0%.
- RealFin-Bench benefits from incremental repair on both models: `deepseek-v4-flash`
  rises from 77.5% to 87.5%, and `deepseek-v4-pro[1m]` rises from 65.0% to 75.0%.

## Smoke Results

| Model | Benchmark | Accuracy | Total Tokens | Efficiency | Failed Tasks |
|---|---:|---:|---:|---:|---|
| deepseek-v4-flash | SOP-Bench | 1/1 (100%) | 276,483 | 3.62 | none |
| deepseek-v4-flash | Lifelong AgentBench | 1/1 (100%) | 75,551 | 13.24 | none |
| deepseek-v4-flash | RealFin-Bench | 0/1 (0%) | 686,454 | 0.00 | `task_01_macd_rsi_filter` |
| deepseek-v4-pro[1m] | SOP-Bench | 1/1 (100%) | 149,332 | 6.70 | none |
| deepseek-v4-pro[1m] | Lifelong AgentBench | 1/1 (100%) | 75,565 | 13.23 | none |
| deepseek-v4-pro[1m] | RealFin-Bench | 0/1 (0%) | 1,040,695 | 0.00 | `task_01_macd_rsi_filter` |

## RealFin Failures

### deepseek-v4-flash

- `task_01_macd_rsi_filter`
- `task_11_semiconductor_macd`
- `task_15_volume_price_contraction_breakout`
- `task_16_pe_bollinger_reversal`
- `task_21_morning_star`
- `task_22_bullish_sandwich`
- `task_27_atr_volatility_breakout`
- `task_31_triple_timeframe_macd`
- `task_32_weekly_breakout_daily_pullback`

### deepseek-v4-pro[1m]

- `task_11_semiconductor_macd`
- `task_20_ultimate_multi_condition`
- `task_21_morning_star`
- `task_22_bullish_sandwich`
- `task_23_gentle_volume_rise`
- `task_24_doji_to_surge`
- `task_26_ma_convergence_divergence`
- `task_27_atr_volatility_breakout`
- `task_28_volatility_compression_explosion`
- `task_31_triple_timeframe_macd`
- `task_32_weekly_breakout_daily_pullback`
- `task_33_sector_leadership`
- `task_39_interest_rate_sector_rotation`
- `task_40_commodity_equity_linkage`

## Artifacts

The committed result artifacts intentionally include run summaries and
`results.json` files only. Large per-task workspaces, transcripts, and tool logs
remain under the ignored `results/` tree unless explicitly force-added later.

- Full runs: `results/claude_code/`
- Smoke runs: `results/claude_code_smoke/`
- Adapter: `bayesian_agent/adapters/claude_code.py`
- Runner: `experiments/run_benchmarks.py --harness claude-code`
