# CLI

The `bayesian-agent` command provides trace ingestion, result summarization, and incremental repair utilities.

## Install the CLI

```bash
python -m pip install -e .
bayesian-agent --help
```

## `evolve`

Update a Bayesian Skill registry from one or more results JSON files.

```bash
bayesian-agent evolve \
  --results artifacts/ga_deepseek_baseline/sop_results.json \
  --results artifacts/ga_deepseek_baseline/lifelong_results.json \
  --registry temp/beliefs.json \
  --algorithm categorical_bayes \
  --context-out temp/skill_context.md
```

Arguments:

| Argument | Required | Description |
|---|---:|---|
| `--results` | yes | Path to a result JSON file. Can be repeated. |
| `--registry` | yes | Output registry JSON path. |
| `--algorithm` | no | Belief backend. Defaults to `categorical_bayes`; `naive_bayes` is accepted as a legacy alias, and `beta_bernoulli` enables the global success-rate baseline. |
| `--context-out` | no | Optional Markdown path for rendered Skill context. |

## `repair-plan`

List failed task ids for incremental repair.

```bash
bayesian-agent repair-plan \
  --baseline artifacts/ga_deepseek_baseline/sop_results.json \
  --out temp/failed_tasks.json
```

Output shape:

```json
{
  "sop_bench": ["sop_12", "sop_13"]
}
```

## `summarize`

Summarize accuracy and token usage for a result file.

```bash
bayesian-agent summarize \
  --results artifacts/bayesian_full/results.json \
  --out temp/summary.json
```

## `incremental-summary`

Summarize the lift from a baseline run plus repair traces.

```bash
bayesian-agent incremental-summary \
  --baseline artifacts/ga_deepseek_baseline/sop_results.json \
  --repairs artifacts/bayesian_incremental/results.json \
  --out temp/incremental_summary.json
```

This command is useful for measuring the additional inference cost required to repair failed tasks.

## `replay-skill-artifacts`

Rebuild per-task Skill evolution artifacts from an existing `results.json` without rerunning the model.

```bash
bayesian-agent replay-skill-artifacts \
  --results results/sop_lifelong_deepseek_v4_flash/bayesian_full/results.json
```

By default, artifacts are written next to the result file:

```text
<result-dir>/skill_evolution/
  index.json
  <benchmark>/<task_id>/
    skill_context_before.md
    skill_context_after.md
    posterior_context_before.md
    posterior_context_after.md
    belief_before.json
    belief_after.json
    snapshot_before.json
    snapshot_after.json
```

`skill_context_*.md` stores the exact model-facing Skill/SOP text. `posterior_context_*.md` stores posterior summaries for audit/debugging and is not part of the built-in benchmark prompt.

Use `--out-root` when the run root is different from the result file's parent directory.

## Result File Assumptions

The CLI accepts benchmark-style JSON result files that can be normalized into benchmark names and run lists. Each run should contain at least:

- `task_id`
- success signal such as `success`
- token fields such as `input_tokens`, `output_tokens`, and `total_tokens`

Extra fields are preserved in evidence metadata.
