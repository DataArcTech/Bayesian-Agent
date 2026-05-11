#!/usr/bin/env python3
"""Run SOP-Bench and Lifelong AgentBench with GenericAgent as harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bayesian_agent.adapters.generic_agent import GenericAgentAdapter
from bayesian_agent.benchmarks.sop_lifelong import DEFAULT_DATA_ROOT, run_sop_lifelong


@dataclass
class ExperimentRun:
    name: str
    mode: str
    out: Path
    baseline_paths: List[str] = None

    def __post_init__(self) -> None:
        self.baseline_paths = list(self.baseline_paths or [])


def build_run_plan(mode: str, out_root: Path, baseline_paths: Sequence[str]) -> List[ExperimentRun]:
    mode = mode.replace("_", "-")
    out_root = Path(out_root)
    supplied_baseline = [str(path) for path in baseline_paths]
    fresh_baseline = str(out_root / "baseline" / "results.json")
    plan: List[ExperimentRun] = []
    if mode in {"all", "baseline"}:
        plan.append(ExperimentRun("baseline", "baseline", out_root / "baseline"))
    if mode in {"all", "bayesian-full"}:
        plan.append(ExperimentRun("bayesian_full", "bayesian-full", out_root / "bayesian_full"))
    if mode in {"all", "bayesian-incremental"}:
        plan.append(
            ExperimentRun(
                "bayesian_incremental",
                "bayesian-incremental",
                out_root / "bayesian_incremental",
                supplied_baseline or [fresh_baseline],
            )
        )
    if not plan:
        raise ValueError(f"Unsupported mode: {mode}")
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SOP-Bench and Lifelong AgentBench through Bayesian-Agent.")
    parser.add_argument("--mode", choices=["all", "baseline", "bayesian-full", "bayesian-incremental"], default="all")
    parser.add_argument("--bench", choices=["sop", "lifelong", "core"], default="core")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--genericagent-root", default="", help="Local GenericAgent checkout. Defaults to discovery.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-root", default="temp/sop_lifelong")
    parser.add_argument("--limit", type=int, default=0, help="Limit tasks for smoke tests. 0 means full benchmark.")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--anthropic-base-url", default="https://api.deepseek.com/anthropic")
    parser.add_argument("--protocol", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--baseline-results", action="append", default=[], help="Baseline results.json for incremental mode.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without calling the model.")
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file()
    args.api_key_env = args.api_key_env or "DEEPSEEK_API_KEY"
    out_root = Path(args.out_root).resolve()
    plan = build_run_plan(args.mode, out_root, args.baseline_results)
    adapter = GenericAgentAdapter(
        root=args.genericagent_root or None,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        anthropic_base_url=args.anthropic_base_url,
        protocol=args.protocol,
    )

    if args.dry_run:
        print_dry_run(adapter, args, plan)
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for spec in plan:
        if spec.mode == "bayesian-incremental":
            missing = [path for path in spec.baseline_paths if not Path(path).exists()]
            if missing:
                raise FileNotFoundError(
                    "Incremental mode needs baseline results. Missing: "
                    + ", ".join(missing)
                    + ". Run --mode all or pass --baseline-results."
                )
        print(f"[experiment] starting {spec.name} -> {spec.out}", flush=True)
        started = time.time()
        result = run_sop_lifelong(
            adapter,
            out_root=spec.out,
            model=args.model,
            data_root=Path(args.data_root),
            bench=args.bench,
            mode=spec.mode,
            limit=args.limit,
            max_turns=args.max_turns,
            baseline_paths=spec.baseline_paths,
        )
        completed.append(
            {
                "name": spec.name,
                "mode": spec.mode,
                "out": str(spec.out),
                "elapsed_seconds": round(time.time() - started, 2),
                "summaries": result.get("summaries", {}),
                "combined_summaries": result.get("combined_summaries", {}),
            }
        )
        print(f"[experiment] finished {spec.name}", flush=True)

    write_experiment_summary(out_root, args.model, completed)
    return 0


def print_dry_run(adapter: GenericAgentAdapter, args: argparse.Namespace, plan: Sequence[ExperimentRun]) -> None:
    header = {
        "genericagent_root": str(adapter.resolve_root()),
        "data_root": str(Path(args.data_root).resolve()),
        "model": args.model,
        "bench": args.bench,
    }
    print(json.dumps(header, indent=2))
    for spec in plan:
        print(f"\n[{spec.name}] mode={spec.mode} out={spec.out}")
        if spec.baseline_paths:
            print("baseline_results=" + ",".join(spec.baseline_paths))


def load_env_file(path: Path = None) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring python-dotenv."""

    path = Path(path or ".env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def write_experiment_summary(out_root: Path, model: str, completed: Sequence[Mapping[str, object]]) -> None:
    lines = [
        f"# SOP-Bench + Lifelong AgentBench: {model}",
        "",
        "This experiment uses GenericAgent as the execution harness and Bayesian-Agent for benchmark orchestration and Skill evolution.",
        "",
    ]
    for item in completed:
        lines.extend([f"## {item['name']}", "", f"- Output: `{item['out']}`", f"- Elapsed: `{item['elapsed_seconds']}s`", ""])
        table_path = Path(str(item["out"])) / "table.md"
        if table_path.exists():
            lines.append(table_path.read_text(encoding="utf-8").strip())
            lines.append("")
    manifest = {"model": model, "runs": list(completed)}
    (out_root / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
