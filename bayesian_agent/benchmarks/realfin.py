"""RealFin-Bench runner owned by Bayesian-Agent."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from bayesian_agent.benchmarks.evolution import (
    build_benchmark_skill_context,
    classify_failure,
    save_skill_evolution_snapshot,
    seed_registry_from_results,
)
from bayesian_agent.benchmarks.sop_lifelong import (
    DEFAULT_DATA_ROOT,
    agent_name_for_mode,
    format_efficiency,
    format_tokens,
    load_results_from_paths,
    prepare_belief_store,
    write_json,
)
from bayesian_agent.core.registry import BayesianSkillRegistry
from bayesian_agent.core.repair import failed_task_ids, merge_repairs, summarize, summarize_incremental_lift
from bayesian_agent.harness import HarnessTask, ensure_harness


BENCHMARK = "realfin_benchmark"


def run_realfin(
    adapter,
    *,
    out_root: Path,
    model: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    mode: str = "baseline",
    limit: int = 0,
    max_turns: int = 8,
    baseline_paths: Optional[Sequence[str]] = None,
    agent_name: str = "",
    evolution_algorithm: str = "categorical_bayes",
) -> Mapping[str, Any]:
    """Run RealFin-Bench with optional Bayesian Skill evolution."""

    out_root = Path(out_root).resolve()
    data_root = Path(data_root).resolve()
    mode = mode.replace("_", "-")
    bayesian_enabled = mode in {"bayesian-full", "bayesian-incremental"}
    out_root.mkdir(parents=True, exist_ok=True)
    registry = prepare_belief_store(out_root, mode, algorithm=evolution_algorithm)
    harness = ensure_harness(adapter, cortex_path=out_root / "cortical_memory.json", registry=registry)

    baseline_results = (
        load_results_from_paths(baseline_paths or [], {BENCHMARK})
        if mode == "bayesian-incremental"
        else {}
    )
    only_failed = failed_task_ids(baseline_results) if baseline_results else {}
    if baseline_results:
        seed_registry_from_results(registry, baseline_results)

    results: MutableMapping[str, Any] = {
        BENCHMARK: run_realfin_bench(
            harness,
            data_root=data_root,
            out_root=out_root,
            registry=registry,
            bayesian_enabled=bayesian_enabled,
            limit=limit,
            max_turns=max_turns,
            only_task_ids=incremental_task_filter(baseline_results, only_failed),
        )
    }

    repair_summaries = summarize(results)
    if baseline_results:
        combined_results = merge_repairs(baseline_results, results)
        combined_summaries = summarize(combined_results)
        summaries = summarize_incremental_lift(baseline_results, results)
    else:
        combined_results = {}
        combined_summaries = {}
        summaries = repair_summaries

    payload = {
        "model": model,
        "mode": mode,
        "bench": "realfin",
        "bayesian_evolution": bayesian_enabled,
        "baseline_paths": list(baseline_paths or []),
        "baseline_summaries": summarize(baseline_results) if baseline_results else {},
        "repair_summaries": repair_summaries,
        "combined_summaries": combined_summaries,
        "summaries": summaries,
        "results": results,
        "combined_results": combined_results,
        "belief_store": str(registry.path),
        "skill_evolution_artifacts": str(out_root / "skill_evolution") if bayesian_enabled else "",
        "cache_manifest": str(build_realfin_cache_manifest(data_root)["manifest_path"]),
    }
    write_json(out_root / "results.json", payload)
    write_table(out_root / "table.md", summaries, model=model, agent_name=agent_name or agent_name_for_mode(mode))
    return payload


def run_realfin_bench(
    harness,
    *,
    data_root: Path,
    out_root: Path,
    registry: BayesianSkillRegistry,
    bayesian_enabled: bool,
    limit: int,
    max_turns: int,
    only_task_ids: Optional[Iterable[str]] = None,
):
    tasks = load_realfin_tasks(data_root)
    if limit:
        tasks = tasks[:limit]
    if only_task_ids is not None:
        wanted = set(only_task_ids)
        tasks = [task for task in tasks if task["id"] in wanted]

    results = []
    for pos, task in enumerate(tasks, 1):
        task_id = str(task["id"])
        workspace = out_root / BENCHMARK / task_id
        run = load_existing_adapter_run(harness, workspace)
        if run is None:
            setup_realfin_workspace(task, data_root=data_root, workspace=workspace)
            prompt = build_realfin_prompt(task, workspace)
            skill_context = ""
            if bayesian_enabled:
                skill_context = build_benchmark_skill_context(BENCHMARK, registry)
                save_skill_evolution_snapshot(
                    out_root=out_root,
                    benchmark=BENCHMARK,
                    task_id=task_id,
                    stage="before",
                    registry=registry,
                    context=skill_context,
                )

            turns = max_turns or max(3, int(task.get("timeout_seconds", 300) or 300) // 60)
            run = harness.run_task(
                HarnessTask(
                    task_id=task_id,
                    prompt=prompt,
                    workspace=workspace,
                    max_turns=turns,
                    skill_context=skill_context,
                    memory_context=bayesian_enabled,
                    task_context=BENCHMARK,
                )
            )
        scores, success, error = grade_realfin_task(task, run, workspace)
        result = {
            **run,
            "task_id": task_id,
            "success": success,
            "scores": scores,
            "error": error,
            "grading_note": "automated_checks_only",
            "category": task.get("category", ""),
            "grading_type": task.get("grading_type", ""),
            "requested_output_files": requested_output_files(task),
            "output_contract": ",".join(requested_output_files(task)),
        }
        result["failure_mode"] = classify_failure(BENCHMARK, result)
        results.append(result)
        if bayesian_enabled:
            harness.record_outcome(
                result,
                skill_id=f"benchmark/{BENCHMARK}",
                context=BENCHMARK,
                success=bool(result["success"]),
                failure_mode=str(result["failure_mode"]),
                summary=task_id,
                metadata={"benchmark": BENCHMARK},
            )
            save_skill_evolution_snapshot(
                out_root=out_root,
                benchmark=BENCHMARK,
                task_id=task_id,
                stage="after",
                registry=registry,
                context=build_benchmark_skill_context(BENCHMARK, registry),
                result=result,
            )
        print(
            f"[realfin] {pos}/{len(tasks)} task={task_id} success={success} "
            f"scores={scores} error={error[:120]!r}",
            flush=True,
        )
    return results


def load_existing_adapter_run(adapter, workspace: Path):
    loader = getattr(adapter, "load_run_from_workspace", None)
    if not loader:
        return None
    return loader(workspace)


def load_realfin_tasks(data_root: Path):
    task_path = Path(data_root) / BENCHMARK / "tasks_finance.jsonl"
    return [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def setup_realfin_workspace(task: Mapping[str, Any], *, data_root: Path, workspace: Path) -> Mapping[str, Any]:
    workspace = Path(workspace)
    make_clean_dir(workspace)
    task_view = {
        key: value
        for key, value in dict(task).items()
        if key not in {"reference_ans", "automated_checks", "llm_judge_criteria", "grading_criteria"}
    }
    (workspace / "task.json").write_text(json.dumps(task_view, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = build_realfin_cache_manifest(Path(data_root))
    write_json(workspace / "realfin_cache_manifest.json", manifest)
    cache_link = workspace / "api_cache"
    cache_root = Path(manifest["cache_root"])
    if cache_root.exists():
        try:
            cache_link.symlink_to(cache_root, target_is_directory=True)
            manifest["workspace_cache_entry"] = str(cache_link)
        except OSError:
            manifest["workspace_cache_entry"] = str(cache_root)
            write_json(workspace / "realfin_cache_manifest.json", manifest)
    return manifest


def build_realfin_cache_manifest(data_root: Path) -> Mapping[str, Any]:
    benchmark_root = Path(data_root).resolve() / BENCHMARK
    cache_root = benchmark_root / "api_cache"
    manifest = {
        "benchmark_root": str(benchmark_root),
        "cache_root": str(cache_root),
        "manifest_path": str(cache_root / "realfin_cache_manifest.json"),
        "baostock_daily_manifest": str(cache_root / "baostock" / "manifest_daily_qfq_20230101_20260331.json"),
        "baostock_daily_dir": str(cache_root / "baostock" / "daily_qfq_20230101_20260331"),
        "baostock_aux_manifest": str(cache_root / "baostock" / "manifest_aux_daily_qfq_20230101_20260331.json"),
        "baostock_universe": str(cache_root / "baostock" / "universe_20260331_sz30.csv"),
        "tencent_etf_manifest": str(cache_root / "tencent" / "manifest_day_qfq_20240101_20260331.json"),
        "notes": [
            "A-share daily qfq CSV files use baostock codes such as sz.300131.csv.",
            "Index CSV files are available from the auxiliary baostock manifest, for example sh.000001 and sz.399006.",
            "Tencent ETF JSON files use market-prefixed codes such as sz159642_day_qfq_20240101_20260331.json.",
            "Tencent rows are arrays like [date, open, close, high, low, volume] under day or qfqday.",
        ],
    }
    return manifest


def build_realfin_prompt(task: Mapping[str, Any], workspace: Path) -> str:
    workspace = Path(workspace).resolve()
    outputs = requested_output_files(task)
    output_hint = ", ".join(outputs) if outputs else "the output file requested by the task prompt"
    return (
        "You are solving one RealFin-Bench task inside this exact workspace:\n"
        f"- workspace: {workspace}\n"
        f"- task JSON: {workspace / 'task.json'}\n"
        f"- local cache manifest: {workspace / 'realfin_cache_manifest.json'}\n"
        f"- local cache symlink: {workspace / 'api_cache'}\n"
        f"- requested output file(s): {output_hint}\n\n"
        "Use local cached market data first. Do not call EastMoney historical endpoints such as "
        "push2his.eastmoney.com; that endpoint may return ERR_EMPTY_RESPONSE. Do not depend on live web APIs.\n\n"
        "Cache usage guide:\n"
        "- 创业板股票 universe: api_cache/baostock/universe_20260331_sz30.csv\n"
        "- 创业板股票日线: api_cache/baostock/daily_qfq_20230101_20260331/sz.300131.csv for code 300131.\n"
        "- 指数日线: see api_cache/baostock/manifest_aux_daily_qfq_20230101_20260331.json; examples include sh.000001 and sz.399006.\n"
        "- ETF 日线: see api_cache/tencent/manifest_day_qfq_20240101_20260331.json; rows are [date, open, close, high, low, volume].\n"
        "- Baostock CSV columns include date, code, open, high, low, close, preclose, volume, amount, turn, pctChg, tradestatus, isST.\n\n"
        "Prefer a deterministic Python script for calculations. When parsing cached CSV files, filter to valid trading rows "
        "with non-empty numeric open/high/low/close/volume values and `tradestatus == 1`; skip rows with blank OHLCV fields. "
        "Print a short progress line such as `loaded history/kline OHLCV data` so the transcript records the data source. "
        "Write exactly the requested output file(s) "
        "in the workspace, with no Markdown wrapper. For stock-code output, strip cache market prefixes unless "
        "the task explicitly asks otherwise: write `300531`, not `sz.300531`. If the task asks for technical "
        "indicators, compute them from cached OHLCV kline/history data and mention the relevant indicator names "
        "in the analysis transcript naturally, for example MACD, RSI, KDJ, MA/均线, Bollinger, volume, ATR, or correlation.\n\n"
        f"{task['prompt']}"
    )


def requested_output_files(task: Mapping[str, Any]):
    text = f"{task.get('prompt', '')}\n{task.get('automated_checks', '')}"
    found = re.findall(r"`([^`]+\.txt)`", text)
    found.extend(re.findall(r'workspace / "([^"]+\.txt)"', text))
    found.extend(re.findall(r"workspace / '([^']+\.txt)'", text))
    result = []
    for name in found:
        clean = name.strip()
        if clean and clean not in result:
            result.append(clean)
    return result


def grade_realfin_task(task: Mapping[str, Any], run: Mapping[str, Any], workspace: Path):
    scores = {}
    success = False
    error = ""
    try:
        ns = {}
        exec(task.get("automated_checks") or "", ns)
        trace = realfin_trace_text(run, workspace)
        scores = ns["grade"]([trace], str(workspace))
        success = bool(scores) and all(float(value) >= 1.0 for value in scores.values())
    except Exception as exc:
        error = str(exc)
    return scores, success, error


def realfin_trace_text(run: Mapping[str, Any], workspace: Path) -> str:
    parts = [str(run.get("transcript") or "")]
    log_path = Path(workspace) / "model_response_log.txt"
    if log_path.exists():
        parts.append(log_path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(part for part in parts if part)


def incremental_task_filter(baseline_results: Mapping[str, Any], failed: Mapping[str, Any]):
    if not baseline_results:
        return None
    return set(failed.get(BENCHMARK, set()))


def write_table(path: Path, summaries: Mapping[str, Mapping[str, Any]], *, model: str, agent_name: str) -> None:
    lines = [
        "| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    if BENCHMARK in summaries:
        summary = summaries[BENCHMARK]
        lines.append(
            f"| RealFin-Bench | {agent_name} | {model} | {summary['accuracy']:.0%} | "
            f"{format_tokens(summary['input_tokens'])} | {format_tokens(summary['output_tokens'])} | "
            f"{format_tokens(summary['total_tokens'])} | {format_efficiency(summary['efficiency'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_clean_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
