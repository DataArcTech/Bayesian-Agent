"""Command line interface for Bayesian-Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bayesian_agent.core.context import SkillContextBuilder
from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.core.registry import BayesianSkillRegistry
from bayesian_agent.core.repair import failed_task_ids, normalize_results, summarize, summarize_incremental_lift


def _read_json(path: str) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, data: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _events_from_results(raw: Mapping[str, Any]):
    results = normalize_results(raw)
    for benchmark, runs in results.items():
        for run in runs:
            yield TrajectoryEvidence.from_run(
                run,
                skill_id=f"benchmark/{benchmark}",
                context=benchmark,
                failure_mode=str(run.get("failure_mode") or run.get("error") or ""),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bayesian-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    evolve = sub.add_parser("evolve", help="Update a Bayesian Skill registry from result traces.")
    evolve.add_argument("--results", action="append", required=True, help="Path to a results JSON file.")
    evolve.add_argument("--registry", required=True, help="Output registry JSON path.")
    evolve.add_argument("--context-out", default="", help="Optional rendered Skill context path.")

    summarize_cmd = sub.add_parser("summarize", help="Summarize a results JSON file.")
    summarize_cmd.add_argument("--results", required=True)
    summarize_cmd.add_argument("--out", required=True)

    repair = sub.add_parser("repair-plan", help="List failed task ids for incremental repair.")
    repair.add_argument("--baseline", required=True)
    repair.add_argument("--out", required=True)

    lift = sub.add_parser("incremental-summary", help="Summarize baseline plus repair traces.")
    lift.add_argument("--baseline", required=True)
    lift.add_argument("--repairs", required=True)
    lift.add_argument("--out", required=True)

    return parser


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evolve":
        registry = BayesianSkillRegistry(args.registry)
        for result_path in args.results:
            registry.record_many(_events_from_results(_read_json(result_path)))
        registry.save()
        if args.context_out:
            Path(args.context_out).write_text(SkillContextBuilder(registry).render(), encoding="utf-8")
        return 0
    if args.command == "summarize":
        _write_json(args.out, summarize(normalize_results(_read_json(args.results))))
        return 0
    if args.command == "repair-plan":
        failures = {k: sorted(v) for k, v in failed_task_ids(normalize_results(_read_json(args.baseline))).items()}
        _write_json(args.out, failures)
        return 0
    if args.command == "incremental-summary":
        baseline = normalize_results(_read_json(args.baseline))
        repairs = normalize_results(_read_json(args.repairs))
        _write_json(args.out, summarize_incremental_lift(baseline, repairs))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
