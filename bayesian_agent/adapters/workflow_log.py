"""Utilities for converting generic assistant workflow logs into trajectory evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from bayesian_agent.core.evidence import TrajectoryEvidence


SUCCESS_VALUES = {"success", "succeeded", "ok", "passed", "complete", "completed", True}


def workflow_record_to_evidence(
    record: Mapping[str, Any],
    *,
    default_skill_id: str = "workflow/default",
    default_context: str = "workflow",
) -> TrajectoryEvidence:
    """Convert an OpenClaw/Hermes-like workflow record into `TrajectoryEvidence`.

    The function intentionally accepts several common field names so external
    harnesses can integrate without adopting Bayesian-Agent internals first.
    """

    task_id = str(record.get("task_id") or record.get("id") or record.get("run_id") or "")
    skill_id = str(record.get("skill_id") or record.get("sop_id") or record.get("workflow_id") or default_skill_id)
    context = str(record.get("context") or record.get("task_family") or record.get("workflow") or default_context)
    raw_outcome = record.get("outcome", record.get("status", record.get("success")))
    outcome = "success" if raw_outcome in SUCCESS_VALUES else "failure"
    return TrajectoryEvidence(
        task_id=task_id,
        skill_id=skill_id,
        context=context,
        outcome=outcome,
        input_tokens=int(record.get("input_tokens") or record.get("prompt_tokens") or 0),
        output_tokens=int(record.get("output_tokens") or record.get("completion_tokens") or 0),
        total_tokens=int(record.get("total_tokens") or 0),
        turns=int(record.get("turns") or record.get("steps") or 0),
        elapsed_seconds=float(record.get("elapsed_seconds") or record.get("duration_seconds") or 0.0),
        failure_mode=str(record.get("failure_mode") or record.get("error_type") or record.get("error") or ""),
        summary=str(record.get("summary") or record.get("title") or task_id),
        metadata={k: v for k, v in record.items() if k not in {"transcript", "messages"}},
    )


def iter_jsonl(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def evidence_from_jsonl(
    path: str | Path,
    *,
    default_skill_id: str = "workflow/default",
    default_context: str = "workflow",
) -> Iterable[TrajectoryEvidence]:
    """Read assistant workflow records from JSONL and yield trajectory evidence."""

    for record in iter_jsonl(path):
        yield workflow_record_to_evidence(record, default_skill_id=default_skill_id, default_context=default_context)
