"""SOP-Bench and Lifelong AgentBench runner owned by Bayesian-Agent."""

from __future__ import annotations

import csv
import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from bayesian_agent.benchmarks.evolution import (
    build_benchmark_skill_context,
    classify_failure,
    record_benchmark_run,
    seed_registry_from_results,
)
from bayesian_agent.core.registry import BayesianSkillRegistry
from bayesian_agent.core.repair import (
    failed_task_ids,
    merge_repairs,
    normalize_results,
    summarize,
    summarize_incremental_lift,
)


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "GA-Technical-Report" / "datasets"


def run_sop_lifelong(
    adapter,
    *,
    out_root: Path,
    model: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    bench: str = "core",
    mode: str = "baseline",
    limit: int = 0,
    max_turns: int = 8,
    baseline_paths: Optional[Sequence[str]] = None,
) -> Mapping[str, Any]:
    """Run SOP-Bench and/or Lifelong AgentBench."""

    out_root = Path(out_root).resolve()
    data_root = Path(data_root).resolve()
    selected = selected_benchmarks(bench)
    mode = mode.replace("_", "-")
    bayesian_enabled = mode in {"bayesian-full", "bayesian-incremental"}
    out_root.mkdir(parents=True, exist_ok=True)
    registry = BayesianSkillRegistry(out_root / "bayesian_skill_beliefs.json")

    baseline_results = load_results_from_paths(baseline_paths or [], selected) if mode == "bayesian-incremental" else {}
    only_failed = failed_task_ids(baseline_results) if baseline_results else {}
    if baseline_results:
        seed_registry_from_results(registry, baseline_results)

    results: MutableMapping[str, Any] = {}
    if "sop_bench" in selected:
        results["sop_bench"] = run_sop_bench(
            adapter,
            data_root=data_root,
            out_root=out_root,
            registry=registry,
            bayesian_enabled=bayesian_enabled,
            limit=limit,
            max_turns=max_turns,
            only_task_ids=incremental_task_filter(baseline_results, only_failed, "sop_bench"),
        )
    if "lifelong_agentbench" in selected:
        results["lifelong_agentbench"] = run_lifelong_bench(
            adapter,
            data_root=data_root,
            out_root=out_root,
            registry=registry,
            bayesian_enabled=bayesian_enabled,
            limit=limit,
            max_turns=max_turns,
            only_task_ids=incremental_task_filter(baseline_results, only_failed, "lifelong_agentbench"),
        )

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
        "bench": bench,
        "bayesian_evolution": bayesian_enabled,
        "baseline_paths": list(baseline_paths or []),
        "baseline_summaries": summarize(baseline_results) if baseline_results else {},
        "repair_summaries": repair_summaries,
        "combined_summaries": combined_summaries,
        "summaries": summaries,
        "results": results,
        "combined_results": combined_results,
        "belief_store": str(registry.path),
    }
    write_json(out_root / "results.json", payload)
    write_table(out_root / "table.md", summaries, model=model, agent_name=agent_name_for_mode(mode))
    return payload


def run_sop_bench(
    adapter,
    *,
    data_root: Path,
    out_root: Path,
    registry: BayesianSkillRegistry,
    bayesian_enabled: bool,
    limit: int,
    max_turns: int,
    only_task_ids: Optional[Iterable[str]] = None,
):
    bench_dir = data_root / "sop_bench"
    rows = list(csv.DictReader((bench_dir / "test_set_with_outputs.csv").open(encoding="utf-8")))
    indexed_rows = [(idx, row) for idx, row in enumerate(rows[: limit or None], 1)]
    if only_task_ids is not None:
        wanted = set(only_task_ids)
        indexed_rows = [(idx, row) for idx, row in indexed_rows if f"sop_{idx:02d}" in wanted]

    results = []
    for pos, (idx, row) in enumerate(indexed_rows, 1):
        workspace = out_root / "sop_bench" / f"task_{idx:02d}"
        setup_sop_workspace(bench_dir, workspace)
        prompt = build_sop_prompt(idx, workspace)
        if bayesian_enabled:
            context = build_benchmark_skill_context("sop_bench", registry)
            if context:
                prompt = f"{context}\n\n{prompt}"
        run = adapter.run_task(prompt=prompt, workspace=workspace, max_turns=max_turns)
        got = read_sop_answer(workspace, idx)
        expected = row["expected_output"].strip()
        result = {**run, "task_id": f"sop_{idx:02d}", "expected": expected, "got": got, "success": got == expected}
        result["failure_mode"] = classify_failure("sop_bench", result)
        results.append(result)
        if bayesian_enabled:
            record_benchmark_run(registry, "sop_bench", result)
        print(f"[sop] {pos}/{len(indexed_rows)} task={idx} success={result['success']} got={got!r} expected={expected!r}", flush=True)
    return results


def run_lifelong_bench(
    adapter,
    *,
    data_root: Path,
    out_root: Path,
    registry: BayesianSkillRegistry,
    bayesian_enabled: bool,
    limit: int,
    max_turns: int,
    only_task_ids: Optional[Iterable[str]] = None,
):
    entries = json.loads((data_root / "lifelong_agentbench" / "entry_dict.json").read_text(encoding="utf-8"))
    keys = sorted(entries, key=lambda value: int(value))
    if limit:
        keys = keys[:limit]
    if only_task_ids is not None:
        wanted = set(only_task_ids)
        keys = [key for key in keys if f"lifelong_{key}" in wanted]

    results = []
    for pos, key in enumerate(keys, 1):
        entry = entries[key]
        workspace = out_root / "lifelong_agentbench" / f"task_{int(key):02d}"
        setup_lifelong_workspace(entry, workspace)
        prompt = build_lifelong_prompt(entry, workspace)
        if bayesian_enabled:
            context = build_benchmark_skill_context("lifelong_agentbench", registry)
            if context:
                prompt = f"{context}\n\n{prompt}"
        run = adapter.run_task(prompt=prompt, workspace=workspace, max_turns=max_turns)
        got_sql = read_lifelong_answer(workspace, run)
        expected_sql = entry["answer_info"]["sql"]
        success = False
        error = ""
        try:
            success = execute_sql(entry, got_sql) == execute_sql(entry, expected_sql)
        except Exception as exc:
            error = str(exc)
        result = {
            **run,
            "task_id": f"lifelong_{key}",
            "expected_sql": expected_sql,
            "got_sql": got_sql,
            "success": success,
            "error": error,
        }
        result["failure_mode"] = classify_failure("lifelong_agentbench", result)
        results.append(result)
        if bayesian_enabled:
            record_benchmark_run(registry, "lifelong_agentbench", result)
        print(f"[lifelong] {pos}/{len(keys)} task={key} success={success} sql={got_sql!r} error={error[:120]!r}", flush=True)
    return results


def selected_benchmarks(bench: str):
    return {
        "sop": {"sop_bench"},
        "lifelong": {"lifelong_agentbench"},
        "core": {"sop_bench", "lifelong_agentbench"},
    }[bench]


def incremental_task_filter(baseline_results: Mapping[str, Any], failed: Mapping[str, Any], benchmark: str):
    if not baseline_results:
        return None
    return set(failed.get(benchmark, set()))


def setup_sop_workspace(bench_dir: Path, workspace: Path) -> None:
    make_clean_dir(workspace)
    for name in ("sop.txt", "tools.py", "toolspecs.json"):
        shutil.copy2(bench_dir / name, workspace / name)
    all_rows = list(csv.DictReader((bench_dir / "test_set_with_outputs.csv").open(encoding="utf-8")))
    blank_rows = []
    for row in all_rows:
        row = dict(row)
        row["expected_output"] = ""
        blank_rows.append(row)
    with (workspace / "test_set_with_outputs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blank_rows[0].keys()))
        writer.writeheader()
        writer.writerows(blank_rows)


def setup_lifelong_workspace(entry: Mapping[str, Any], workspace: Path) -> None:
    make_clean_dir(workspace)
    task_view = {key: value for key, value in entry.items() if key != "answer_info"}
    (workspace / "task.json").write_text(json.dumps(task_view, ensure_ascii=False, indent=2), encoding="utf-8")


def build_sop_prompt(row_index: int, workspace: Path) -> str:
    workspace = Path(workspace).resolve()
    return (
        "You are solving one SOP-Bench task inside this exact workspace:\n"
        f"- workspace: {workspace}\n"
        f"- SOP: {workspace / 'sop.txt'}\n"
        f"- tools: {workspace / 'tools.py'}\n"
        f"- CSV to edit: {workspace / 'test_set_with_outputs.csv'}\n\n"
        "Tool path rule: file_read/file_write/code_run already default to this task workspace; "
        "do not pass cwd. If you absolutely must pass cwd, it must be the exact workspace path above.\n\n"
        f"Read sop.txt and tools.py, then locate row {row_index} in test_set_with_outputs.csv "
        "(skip the header). Before calling tools, verify that row's order_id, product_id, "
        "quantity_requested, customer_id, and order_total; do not reuse inputs from any other row. "
        "Execute the order-fulfillment decision for only that row and write the "
        "raw category string into that row's expected_output cell. Use the Python csv module when "
        "editing the CSV. Do not use pandas for the write. Do not write XML tags, Markdown, quotes, "
        "or explanations into the cell. Preserve all other rows and columns exactly. Verify the cell "
        "is non-empty before finishing."
    )


def build_lifelong_prompt(entry: Mapping[str, Any], workspace: Path) -> str:
    workspace = Path(workspace).resolve()
    return (
        "You are solving one Lifelong AgentBench DB-Bench task inside this exact workspace:\n"
        f"- workspace: {workspace}\n"
        f"- task JSON: {workspace / 'task.json'}\n"
        f"- SQL output: {workspace / 'answer.sql'}\n\n"
        "Tool path rule: file_read/file_write/code_run already default to this task workspace; "
        "do not pass cwd. If you absolutely must pass cwd, it must be the exact workspace path above.\n\n"
        "Read task.json, build or reason over the provided table_info if useful, solve the instruction, "
        "and write exactly one SQL statement to answer.sql. Do not include Markdown or explanation in "
        "answer.sql. Use only columns present in task.json unless the instruction explicitly asks for a "
        "new value in an existing column. Do not include id or primary-key columns in INSERT statements "
        "unless the instruction explicitly provides their values. Then finish.\n\n"
        f"Instruction: {entry['instruction']}"
    )


def read_sop_answer(workspace: Path, row_index: int) -> str:
    try:
        rows = list(csv.DictReader((workspace / "test_set_with_outputs.csv").open(encoding="utf-8")))
        return rows[row_index - 1].get("expected_output", "").strip()
    except Exception as exc:
        return f"ERROR:{exc}"


def read_lifelong_answer(workspace: Path, run: Mapping[str, Any]) -> str:
    answer = workspace / "answer.sql"
    if answer.exists():
        return normalize_sql_text(answer.read_text(encoding="utf-8", errors="replace"))
    return normalize_sql_text(str(run.get("transcript") or ""))


def normalize_sql_text(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    lines = [line.strip() for line in text.strip().splitlines() if line.strip() and not line.strip().startswith("--")]
    text = "\n".join(lines).strip()
    if ";" in text:
        text = text[: text.find(";") + 1]
    return text


def setup_sqlite(entry: Mapping[str, Any]):
    conn = sqlite3.connect(":memory:")
    info = entry["table_info"]
    cols = [f'"{col["name"]}" {col["type"]}' for col in info["column_info_list"]]
    conn.execute(f'CREATE TABLE "{info["name"]}" ({", ".join(cols)})')
    names = [col["name"] for col in info["column_info_list"]]
    placeholders = ",".join(["?"] * len(names))
    for row in info["row_list"]:
        conn.execute(f'INSERT INTO "{info["name"]}" VALUES ({placeholders})', row)
    conn.commit()
    return conn


def execute_sql(entry: Mapping[str, Any], sql: str):
    conn = setup_sqlite(entry)
    cur = conn.cursor()
    cur.execute(sql)
    if sql.strip().lower().startswith("select"):
        return {"type": "select", "rows": cur.fetchall()}
    conn.commit()
    table = entry["table_info"]["name"]
    rows = conn.execute(f'SELECT * FROM "{table}"').fetchall()
    return {"type": "mutation", "rows": rows}


def load_results_from_paths(paths: Sequence[str], allowed_benches: Iterable[str]):
    allowed = set(allowed_benches)
    merged = {}
    for raw_path in paths:
        data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        for benchmark, runs in normalize_results(data).items():
            if benchmark in allowed:
                merged.setdefault(benchmark, []).extend(runs)
    return {benchmark: dedupe_by_task_id(runs) for benchmark, runs in merged.items()}


def dedupe_by_task_id(runs):
    order = []
    by_id = {}
    for run in runs:
        task_id = run.get("task_id")
        if task_id not in by_id:
            order.append(task_id)
        by_id[task_id] = compact_baseline_run(run)
    return [by_id[task_id] for task_id in order]


def compact_baseline_run(run: Mapping[str, Any]):
    return {key: value for key, value in dict(run).items() if key not in {"transcript", "usage_events", "exit_reason"}}


def write_table(path: Path, summaries: Mapping[str, Mapping[str, Any]], *, model: str, agent_name: str) -> None:
    lines = [
        "| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ("sop_bench", "lifelong_agentbench"):
        if benchmark not in summaries:
            continue
        summary = summaries[benchmark]
        lines.append(
            f"| {display_benchmark(benchmark)} | {agent_name} | {model} | {summary['accuracy']:.0%} | "
            f"{format_tokens(summary['input_tokens'])} | {format_tokens(summary['output_tokens'])} | "
            f"{format_tokens(summary['total_tokens'])} | {format_efficiency(summary['efficiency'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def make_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def display_benchmark(benchmark: str) -> str:
    if benchmark == "sop_bench":
        return "SOP-Bench"
    if benchmark == "lifelong_agentbench":
        return "Lifelong AgentBench"
    return benchmark


def agent_name_for_mode(mode: str) -> str:
    if mode == "bayesian-incremental":
        return "GA+BayesianIncremental"
    if mode == "bayesian-full":
        return "GA+Bayesian"
    return "GA"


def format_tokens(value: int) -> str:
    value = int(value or 0)
    if value == 0:
        return "0"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{round(value / 1000):.0f}k"


def format_efficiency(value: Any) -> str:
    return value if isinstance(value, str) else f"{float(value):.2f}"
