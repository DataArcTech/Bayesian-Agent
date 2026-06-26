# Zero-Shot vs Catalog Skill Evolution

This note compares two BA native backend runs with `deepseek-v4-flash`:

- **Catalog-first BA**: historical run under `results/native_harness_deepseek_v4_flash_full`. These results were produced before the explicit `use_skill_catalog` flag was added. The runner behavior is equivalent to the current default catalog-first mode: benchmark guardrails and handwritten failure-mode patch catalogs are available.
- **Zero-shot BA without catalog skills**: new run under `results/zero_shot_native_flash_20260626`, explicitly executed with `--no-use-skill-catalog`. No benchmark guardrails or handwritten failure-to-patch catalog is injected. Failure modes and patches are discovered from trajectories.

The comparison is not a strict same-seed paired A/B test. The most important question here is narrower: **when we remove catalog skills, does BA still provide useful self-evolution signal?** The answer is yes, especially for recurring output-contract and missing-artifact failures.

## Result Summary

### Catalog-First BA

| Benchmark | Baseline | Bayesian Full | Bayesian Incremental | Main Observation |
|---|---:|---:|---:|---|
| SOP-Bench | 95% / 1.05M tokens | 100% / 870k tokens | 100% / 45k tokens | Catalog guardrails make the SOP failure modes easy to repair. |
| Lifelong AgentBench | 95% / 538k tokens | 100% / 514k tokens | 100% / 65k tokens | SQL-specific catalog rules repair the remaining failure. |
| RealFin-Bench | 62% / 10.29M tokens | 70% / 10.89M tokens | 72% / 3.76M tokens | Domain catalog helps with cache usage, file format, indicators, and sparse OHLCV rows. |

Source summaries:

- `results/native_harness_deepseek_v4_flash_full/sop/summary.md`
- `results/native_harness_deepseek_v4_flash_full/lifelong/summary.md`
- `results/native_harness_deepseek_v4_flash_full/realfin/summary.md`

### Zero-Shot BA without Catalog Skills

| Benchmark | Baseline | Bayesian Full | Bayesian Incremental | Lift vs Own Baseline |
|---|---:|---:|---:|---:|
| SOP-Bench | 70% / 1.23M tokens | 75% / 1.23M tokens | 100% / 366k tokens | Full: +5 pp; Incremental: +30 pp |
| Lifelong AgentBench | 95% / 468k tokens | 90% / 510k tokens | 95% / 39k tokens | Full: -5 pp; Incremental: +0 pp |
| RealFin-Bench | 38% / 3.77M tokens | 45% / 3.21M tokens | 45% / 1.73M tokens | Full: +7 pp; Incremental: +7 pp |

Source summaries:

- `results/zero_shot_native_flash_20260626/sop/summary.md`
- `results/zero_shot_native_flash_20260626/lifelong/summary.md`
- `results/zero_shot_native_flash_20260626/realfin/summary.md`

## What This Shows

### 1. Catalog skills are still stronger

Catalog-first BA is the upper-performance path in these experiments. That is expected. The catalog contains domain-specific prior knowledge:

- SOP-Bench: target row indexing, CSV writeback, raw category-only output.
- Lifelong AgentBench: one SQL statement only, no transcript text, no invented primary-key columns.
- RealFin-Bench: local cache usage, market-code normalization, indicator calculation guardrails, sparse OHLCV filtering.

Those rules are not just labels. They are high-precision executable guidance. Removing them makes the agent solve a harder problem: it must first discover the failure mode and then distill a useful patch from limited evidence.

### 2. No-catalog BA is still useful

The zero-shot run still improves over its own baseline on SOP-Bench and RealFin-Bench:

- **SOP-Bench incremental** improves from 70% to 100% while rerunning only 6 failed tasks, using 366k total tokens instead of another full 1.23M-token pass.
- **RealFin-Bench full and incremental** improve from 38% to 45%. The gain is smaller, but it happens without any benchmark-specific failure taxonomy.

This is the practical evidence that BA is not only a handwritten skill catalog wrapper. The Bayesian loop can still extract repeated failure evidence and turn it into prompt-time repair pressure.

### 3. The artifact trail confirms true zero-shot discovery

Artifact counts:

| Run | Skill Context Snapshots | Benchmark Guardrails | Auto Failure Patches |
|---|---:|---:|---:|
| Catalog-first BA | 97 | 97 | 0 |
| Zero-shot BA | 112 | 0 | 83 |

In the catalog run, a SOP context starts with handwritten guardrails:

```text
### Benchmark SOP Guardrails: sop_bench
- Read `sop.txt`, `tools.py`, and the target CSV row before acting.
- The requested row is one-indexed after the header ...
```

In the no-catalog run, the corresponding context contains only automatically discovered patches:

```text
### Bayesian Failure-Mode Patches: sop_bench
- failure_mode=auto_empty_output observed=2
  - After writing, re-read the required output and verify it is non-empty before finishing.
  - If the answer is intentionally empty, write the benchmark-accepted empty-result wording or header instead of leaving a blank artifact.
```

This matters: the no-catalog run is not secretly using `Benchmark SOP Guardrails`. It learns generic failure modes such as:

- `auto_empty_output`
- `auto_missing_requested_artifact`
- `auto_output_contract_violation`
- `auto_sql_execution_error`
- `auto_unclassified_failure`

## Case Analysis

### SOP-Bench: repeated empty output becomes a useful patch

In the zero-shot SOP run, early tasks repeatedly left the target CSV `expected_output` blank. After two verified failures, BA activated:

```text
failure_mode=auto_empty_output observed=2
```

The patch is generic, but it matches the real failure: re-read the required output and verify it is non-empty. In incremental mode, BA reran the six baseline failures and repaired all six, lifting the merged result to 100%.

Takeaway: for recurring output omissions, automatic discovery is enough. We do not need a handwritten SOP-specific `left_expected_output_blank` catalog entry to get useful behavior.

### RealFin-Bench: missing artifacts improve, domain reasoning remains hard

RealFin without catalog skills is much harder. Many failures are not merely "forgot to write a file"; they require finance-specific calculations, cache routing, indicator implementation, and exact output formats.

Still, the zero-shot run learns a repeated `auto_missing_requested_artifact` pattern. Some originally failed tasks are repaired in full mode, including examples such as:

- `task_05_triple_golden_cross`
- `task_13_dual_timeframe_macd`
- `task_26_ma_convergence_divergence`
- `task_28_volatility_compression_explosion`
- `task_32_weekly_breakout_daily_pullback`
- `task_35_low_drawdown_growth`
- `task_38_feature_engineering`

The overall lift is modest but real: 38% to 45%. This is exactly the expected shape for zero-shot discovery: it helps with reusable execution-contract failures, but it cannot fully replace a domain catalog for complex financial reasoning.

### Lifelong AgentBench: high baseline leaves little room

The no-catalog Lifelong baseline is already 95%. Incremental mode reruns only one failed task and does not repair it, so the merged accuracy remains 95%. Full mode drops to 90%, mainly because generic patches are weaker than SQL-specific catalog rules and can add prompt pressure without precisely targeting the SQL failure.

Takeaway: when baseline accuracy is already high and failures are sparse, the two-occurrence patch threshold intentionally avoids overfitting. That makes the zero-shot method conservative, but it also means single rare failures may not be fixed.

## Interpretation

The result supports a layered view of BA:

1. **Best path**: catalog-first BA, where domain experts or previous runs provide precise failure taxonomy and patch rules.
2. **Useful fallback path**: zero-shot BA, where no catalog exists and the system must discover recurring failure modes from verifier feedback.
3. **Practical deployment path**: start zero-shot on a new benchmark or product task, inspect the learned `auto_*` patches, then promote stable recurring patches into a benchmark-specific catalog.

So the catalog is not the whole method. The catalog is a high-quality prior. When that prior is absent, BA still has an evidence loop:

```text
trajectory -> verifier outcome -> auto failure mode -> repeated evidence -> distilled patch -> next run
```

The current evidence shows that this loop is useful for repeated, observable failures. It is weaker for rare semantic errors or domain-specific reasoning gaps. That is a good boundary to state clearly: **zero-shot BA is a bootstrap mechanism, not a replacement for all domain knowledge.**

## Conclusion

Without catalog skills, BA still improves:

- SOP-Bench incremental: 70% -> 100%
- RealFin-Bench full/incremental: 38% -> 45%

With catalog skills, BA performs better:

- SOP-Bench: up to 100%
- Lifelong AgentBench: up to 100%
- RealFin-Bench: up to 72%

This gives a clean story: **Bayesian-Agent works even without a handcrafted skill catalog, and catalog skills act as stronger priors that further improve reliability and efficiency.**
