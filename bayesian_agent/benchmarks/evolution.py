"""Bayesian-Agent owned benchmark Skill evolution helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from bayesian_agent.core.context import SkillContextBuilder
from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.core.registry import BayesianSkillRegistry


def classify_failure(benchmark: str, run: Mapping[str, Any]) -> str:
    """Classify common benchmark failures into reusable evidence labels."""

    if run.get("success"):
        return ""
    if benchmark == "sop_bench":
        got = str(run.get("got") or "")
        expected = str(run.get("expected") or "")
        if "<final_decision>" in got:
            return "wrote_xml_tags_in_csv_expected_output"
        if not got or got.lower() == "none":
            return "left_expected_output_blank"
        if expected and got != expected:
            return "computed_decision_for_wrong_or_unverified_target_row"
    if benchmark == "lifelong_agentbench":
        got = str(run.get("got_sql") or "")
        expected = str(run.get("expected_sql") or "")
        if "payment_id" in got and "payment_id" not in expected:
            return "invented_unrequested_column"
        if "Turn 1" in got or "🛠️" in got:
            return "wrote_transcript_instead_of_sql_after_workspace_confusion"
        if run.get("error"):
            return str(run.get("error"))[:160]
    return str(run.get("error") or "benchmark_failure")[:160]


def evidence_from_run(benchmark: str, run: Mapping[str, Any]) -> TrajectoryEvidence:
    failure_mode = str(run.get("failure_mode") or classify_failure(benchmark, run))
    return TrajectoryEvidence.from_run(
        run,
        skill_id=f"benchmark/{benchmark}",
        context=benchmark,
        failure_mode=failure_mode,
    )


def record_benchmark_run(registry: BayesianSkillRegistry, benchmark: str, run: Mapping[str, Any]) -> None:
    enriched = dict(run)
    enriched["failure_mode"] = str(run.get("failure_mode") or classify_failure(benchmark, run))
    registry.record(evidence_from_run(benchmark, enriched))


def seed_registry_from_results(registry: BayesianSkillRegistry, benchmark_runs: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    for benchmark, runs in benchmark_runs.items():
        for run in runs:
            record_benchmark_run(registry, benchmark, run)


def build_benchmark_skill_context(benchmark: str, registry: BayesianSkillRegistry) -> str:
    """Render posterior Skill context plus benchmark-specific guardrails."""

    posterior = SkillContextBuilder(registry).render(task_context=benchmark, limit=5)
    rules = _stable_rules(benchmark)
    if not posterior and not rules:
        return ""
    lines = []
    if posterior:
        lines.append(posterior)
    if rules:
        lines.extend(["", f"### Benchmark SOP Guardrails: {benchmark}"])
        lines.extend(f"- {rule}" for rule in rules)
    return "\n".join(line for line in lines if line is not None).strip()


def _stable_rules(benchmark: str):
    if benchmark == "sop_bench":
        return [
            "Read `sop.txt`, `tools.py`, and the target CSV row before acting.",
            "The requested row is one-indexed after the header; update `rows[row_index - 1]` when using `csv.DictReader`.",
            "Before calling tools, verify the target row's `order_id`, `product_id`, `quantity_requested`, `customer_id`, and `order_total`; never reuse inputs from another row.",
            "Compute only the target row and write only its `expected_output` cell.",
            "Use Python's `csv` module for writing; preserve all other rows and columns exactly.",
            "Write the raw category string only, for example `manual_review`; never write XML tags, Markdown, quotes, or explanations into the cell.",
            "Verify the target row's `expected_output` is non-empty before finishing.",
        ]
    if benchmark == "lifelong_agentbench":
        return [
            "Read `task.json` in the current workspace; do not inspect sibling benchmark runs.",
            "Write exactly one SQL statement to `answer.sql`; no Markdown and no explanation.",
            "Use only columns present in `task.json` unless the instruction explicitly asks for a new value in an existing column.",
            "For INSERT statements, do not include id or primary-key columns unless the instruction explicitly provides their values.",
            "For mutation tasks, write executable SQL that reproduces the expected table state.",
            "If SQL ranking is needed, express ranking inside a subquery and keep the final output to one SQL statement.",
        ]
    return []
