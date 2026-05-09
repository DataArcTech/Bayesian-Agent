# OpenClaw / Hermes Workflow Log Example

This example shows how a persistent assistant harness can feed workflow outcomes into Bayesian-Agent without adopting a new runtime.

## JSONL Input

Each line is one workflow run. Field names are intentionally permissive.

```jsonl
{"id":"grade-001","workflow":"grading","sop_id":"openclaw/grading/rubric_feedback","success":true,"total_tokens":2400,"signals":["verified"]}
{"id":"grade-002","workflow":"grading","sop_id":"openclaw/grading/rubric_feedback","success":false,"failure_mode":"rubric_mismatch","total_tokens":2100,"signals":["failure_mode_recorded"]}
```

## Evolve a Registry

```bash
bayesian-agent evolve-workflow-log \
  --jsonl runs.jsonl \
  --registry temp/openclaw_beliefs.json \
  --context-out temp/openclaw_context.md \
  --strategy context_aware
```

## Why This Matters

OpenClaw and Hermes-style agents repeatedly execute workflows such as grading, coding repair, browser tasks, project reviews, and inbox triage. Bayesian-Agent can turn those runs into evidence-weighted Skills:

- successful workflows become reusable context;
- recurring failure modes become repair targets;
- uncertain workflows can be explored instead of trusted blindly;
- low-token/high-success workflows can be preferred when cost matters.
