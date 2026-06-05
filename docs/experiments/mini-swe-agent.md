# mini-swe-agent Benchmark Results

This page records the full mini-swe-agent compatibility-backend run from
2026-06-05. Bayesian-Agent handled benchmark orchestration, Bayesian full
self-evolution, incremental repair, grading, and artifact writing; mini-swe-agent
executed each task prompt through the adapter boundary.

Evidence type: newly run and verified from local result artifacts. Metrics were
read from `results/mini_swe_agent/*/*/*/results.json` after the runs completed.
RealFin final runs used `--max-turns 16 --mini-swe-env-timeout 180`.

## Commands

Baseline, Bayesian full, and Bayesian incremental runs used this harness shape:

```bash
python experiments/run_benchmarks.py \
  --harness mini-swe-agent \
  --mini-swe-agent-root ../mini-swe-agent \
  --model "$MODEL" \
  --bench "$BENCH" \
  --mode all \
  --out-root "results/mini_swe_agent/${MODEL_SLUG}/${BENCH}"
```

For Pro RealFin, the interrupted Bayesian full run was rerun to completion, and
Bayesian incremental was then run from the completed baseline file:

```bash
python experiments/run_benchmarks.py \
  --harness mini-swe-agent \
  --mini-swe-agent-root ../mini-swe-agent \
  --model deepseek-v4-pro \
  --bench realfin \
  --mode bayesian-incremental \
  --out-root results/mini_swe_agent/deepseek_v4_pro/realfin \
  --baseline-results results/mini_swe_agent/deepseek_v4_pro/realfin/baseline/results.json \
  --max-turns 16 \
  --mini-swe-env-timeout 180
```

## Summary

| Model | Benchmark | Baseline | Bayesian full | Bayesian incremental final | Incremental repairs |
|---|---:|---:|---:|---:|---:|
| deepseek-v4-flash | SOP-Bench | 20/20 (100%) | 19/20 (95%) | 20/20 (100%) | 0/0 |
| deepseek-v4-flash | Lifelong AgentBench | 17/20 (85%) | 19/20 (95%) | 20/20 (100%) | 3/3 |
| deepseek-v4-flash | RealFin-Bench | 24/40 (60%) | 22/40 (55%) | 28/40 (70%) | 4/16 |
| deepseek-v4-pro | SOP-Bench | 19/20 (95%) | 20/20 (100%) | 20/20 (100%) | 1/1 |
| deepseek-v4-pro | Lifelong AgentBench | 18/20 (90%) | 20/20 (100%) | 20/20 (100%) | 2/2 |
| deepseek-v4-pro | RealFin-Bench | 28/40 (70%) | 28/40 (70%) | 32/40 (80%) | 4/12 |

## Token Usage

| Model | Benchmark | Mode | Total Tokens | Efficiency |
|---|---:|---|---:|---:|
| deepseek-v4-flash | SOP-Bench | baseline | 1,017,462 | 19.66 |
| deepseek-v4-flash | SOP-Bench | Bayesian full | 1,012,377 | 18.77 |
| deepseek-v4-flash | SOP-Bench | incremental extra | 0 | 0.00 |
| deepseek-v4-flash | Lifelong AgentBench | baseline | 580,764 | 29.27 |
| deepseek-v4-flash | Lifelong AgentBench | Bayesian full | 497,914 | 38.16 |
| deepseek-v4-flash | Lifelong AgentBench | incremental extra | 36,456 | 82.29 |
| deepseek-v4-flash | RealFin-Bench | baseline | 5,728,953 | 4.19 |
| deepseek-v4-flash | RealFin-Bench | Bayesian full | 6,956,061 | 3.16 |
| deepseek-v4-flash | RealFin-Bench | incremental extra | 2,506,388 | 1.60 |
| deepseek-v4-pro | SOP-Bench | baseline | 907,207 | 20.94 |
| deepseek-v4-pro | SOP-Bench | Bayesian full | 953,207 | 20.98 |
| deepseek-v4-pro | SOP-Bench | incremental extra | 48,434 | 20.65 |
| deepseek-v4-pro | Lifelong AgentBench | baseline | 348,842 | 51.60 |
| deepseek-v4-pro | Lifelong AgentBench | Bayesian full | 329,718 | 60.66 |
| deepseek-v4-pro | Lifelong AgentBench | incremental extra | 27,302 | 73.25 |
| deepseek-v4-pro | RealFin-Bench | baseline | 6,100,563 | 4.59 |
| deepseek-v4-pro | RealFin-Bench | Bayesian full | 6,734,253 | 4.16 |
| deepseek-v4-pro | RealFin-Bench | incremental extra | 2,016,244 | 1.98 |

## RealFin Remaining Failures After Incremental

### deepseek-v4-flash

- `task_11_semiconductor_macd`
- `task_12_catl_correlation_kdj`
- `task_16_pe_bollinger_reversal`
- `task_20_ultimate_multi_condition`
- `task_22_bullish_sandwich`
- `task_23_gentle_volume_rise`
- `task_30_composite_scoring`
- `task_31_triple_timeframe_macd`
- `task_33_sector_leadership`
- `task_34_etf_constituent_arbitrage`
- `task_37_floor_ceiling_reversal`
- `task_40_commodity_equity_linkage`

### deepseek-v4-pro

- `task_11_semiconductor_macd`
- `task_16_pe_bollinger_reversal`
- `task_20_ultimate_multi_condition`
- `task_22_bullish_sandwich`
- `task_23_gentle_volume_rise`
- `task_33_sector_leadership`
- `task_39_interest_rate_sector_rotation`
- `task_40_commodity_equity_linkage`

## Artifacts

- Adapter: `bayesian_agent/adapters/mini_swe_agent.py`
- Runner: `experiments/run_benchmarks.py --harness mini-swe-agent`
- Results root: `results/mini_swe_agent/`
- Local ignored convenience summary: `results/mini_swe_agent/summary.md`
