# Schemas

Bayesian-Agent includes JSON schemas for portable evidence and belief exchange.

## Trajectory Evidence

File: [`schemas/trajectory.schema.json`](https://github.com/DataArcTech/Bayesian-Agent/blob/main/schemas/trajectory.schema.json)

Required fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Stable task identifier. |
| `skill_id` | string | Skill or SOP hypothesis id. |
| `context` | string | Benchmark, domain, or task context. |
| `outcome` | string | `success`, `failure`, `failed`, or `error`. |

Optional fields:

| Field | Type | Description |
|---|---|---|
| `input_tokens` | integer | Prompt/input token usage. |
| `output_tokens` | integer | Completion/output token usage. |
| `total_tokens` | integer | Total token usage. |
| `turns` | integer | Agent turns used by the run. |
| `elapsed_seconds` | number | Wall-clock runtime. |
| `failure_mode` | string | Normalized failure type. |
| `summary` | string | Short run summary. |
| `metadata` | object | Harness-specific fields. |
| `created_at` | string | Evidence timestamp. |

## Skill Belief

File: [`schemas/skill_belief.schema.json`](https://github.com/DataArcTech/Bayesian-Agent/blob/main/schemas/skill_belief.schema.json)

Important fields:

| Field | Type | Description |
|---|---|---|
| `skill_id` | string | Skill or SOP hypothesis id. |
| `alpha` | number | Success count plus prior. |
| `beta` | number | Failure count plus prior. |
| `posterior_success` | number | Expected success probability. |
| `contexts` | object | Context observation counts. |
| `failure_modes` | object | Failure mode counts. |
| `observations` | integer | Number of evidence updates. |
| `mean_tokens` | number | Average total token cost. |
| `evidence` | array | Recent evidence window. |

## Compatibility Guideline

Adapters should preserve extra fields in `metadata`. This keeps the common schema stable while allowing each harness to carry local diagnostics.
