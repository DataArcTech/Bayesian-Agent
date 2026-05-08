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
  --context-out temp/skill_context.md
```

Arguments:

| Argument | Required | Description |
|---|---:|---|
| `--results` | yes | Path to a result JSON file. Can be repeated. |
| `--registry` | yes | Output registry JSON path. |
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

## Result File Assumptions

The CLI accepts benchmark-style JSON result files that can be normalized into benchmark names and run lists. Each run should contain at least:

- `task_id`
- success signal such as `success`
- token fields such as `input_tokens`, `output_tokens`, and `total_tokens`

Extra fields are preserved in evidence metadata.
