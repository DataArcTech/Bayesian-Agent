"""Utilities for incremental Bayesian repair runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Set


BenchmarkResults = Dict[str, List[Dict[str, Any]]]


def normalize_results(raw: Mapping[str, Any]) -> BenchmarkResults:
    if "results" in raw and isinstance(raw["results"], dict):
        raw = raw["results"]
    return {str(bench): list(runs) for bench, runs in raw.items()}


def failed_task_ids(results: Mapping[str, Iterable[Mapping[str, Any]]]) -> Dict[str, Set[str]]:
    failed: Dict[str, Set[str]] = {}
    for benchmark, runs in results.items():
        ids = {str(run.get("task_id")) for run in runs if run.get("task_id") and not run.get("success")}
        if ids:
            failed[str(benchmark)] = ids
    return failed


def dedupe_by_task_id(runs: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    order: List[str] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        task_id = str(run.get("task_id"))
        if task_id not in by_id:
            order.append(task_id)
        by_id[task_id] = dict(run)
    return [by_id[task_id] for task_id in order]


def merge_repairs(baseline: BenchmarkResults, repairs: BenchmarkResults) -> BenchmarkResults:
    merged: BenchmarkResults = {}
    for benchmark, runs in baseline.items():
        repair_by_id = {str(run.get("task_id")): dict(run) for run in repairs.get(benchmark, [])}
        merged[benchmark] = [
            {
                **repair_by_id.get(str(run.get("task_id")), dict(run)),
                "incremental_repair": str(run.get("task_id")) in repair_by_id,
            }
            for run in runs
        ]
    for benchmark, runs in repairs.items():
        if benchmark not in merged:
            merged[benchmark] = [dict(run) for run in runs]
    return merged


def summarize(results: BenchmarkResults) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for benchmark, runs in results.items():
        runs = list(runs)
        successes = sum(1 for run in runs if run.get("success"))
        input_tokens = sum(int(run.get("input_tokens") or 0) for run in runs)
        output_tokens = sum(int(run.get("output_tokens") or 0) for run in runs)
        total_tokens = sum(int(run.get("total_tokens") or 0) for run in runs) or input_tokens + output_tokens
        summaries[benchmark] = {
            "tasks": len(runs),
            "successes": successes,
            "accuracy": successes / len(runs) if runs else 0.0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "efficiency": round((successes / total_tokens) * 1_000_000, 2) if total_tokens else 0.0,
        }
    return summaries


def summarize_incremental_lift(baseline: BenchmarkResults, repairs: BenchmarkResults) -> Dict[str, Dict[str, Any]]:
    baseline_summaries = summarize(baseline)
    repair_summaries = summarize(repairs)
    combined = merge_repairs(baseline, repairs)
    combined_summaries = summarize(combined)
    lift: Dict[str, Dict[str, Any]] = {}
    for benchmark, final in combined_summaries.items():
        base = baseline_summaries.get(benchmark, {"successes": 0, "accuracy": 0.0, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0})
        repair = repair_summaries.get(benchmark, {"tasks": 0, "successes": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        repaired_successes = int(final["successes"]) - int(base["successes"])
        repair_total = int(repair["total_tokens"])
        cumulative_total = int(base["total_tokens"]) + repair_total
        lift[benchmark] = {
            "tasks": final["tasks"],
            "successes": final["successes"],
            "accuracy": final["accuracy"],
            "baseline_successes": base["successes"],
            "baseline_accuracy": base["accuracy"],
            "repair_attempts": repair["tasks"],
            "repaired_successes": repaired_successes,
            "input_tokens": repair["input_tokens"],
            "output_tokens": repair["output_tokens"],
            "total_tokens": repair_total,
            "efficiency": round((repaired_successes / repair_total) * 1_000_000, 2) if repair_total else 0.0,
            "cumulative_total_tokens": cumulative_total,
            "cumulative_efficiency": round((final["successes"] / cumulative_total) * 1_000_000, 2) if cumulative_total else 0.0,
        }
    return lift
