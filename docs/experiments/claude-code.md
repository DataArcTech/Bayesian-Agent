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
