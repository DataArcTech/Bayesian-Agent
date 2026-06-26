"""Automatic failure discovery and trajectory-to-Skill distillation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping


AUTO_FAILURE_PREFIX = "auto_"


@dataclass
class FailureDiscovery:
    """Normalized failure label plus distilled repair hints."""

    failure_mode: str
    source: str = "automatic"
    signals: Dict[str, Any] = field(default_factory=dict)
    patch_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_mode": self.failure_mode,
            "source": self.source,
            "signals": _json_safe(self.signals),
            "patch_rules": list(self.patch_rules),
        }


def discover_failure(
    benchmark: str,
    run: Mapping[str, Any],
    *,
    known_failure: str = "",
    source: str = "automatic",
) -> FailureDiscovery:
    """Infer or normalize a reusable failure mode from trajectory signals."""

    run = dict(run or {})
    if run.get("success"):
        return FailureDiscovery("", source="verifier", signals={})

    if known_failure and source != "automatic":
        return FailureDiscovery(
            failure_mode=str(known_failure),
            source=source,
            signals=_signals(benchmark, run),
            patch_rules=[],
        )

    signals = _signals(benchmark, run)
    error = str(run.get("error") or "")
    error_lower = error.lower()
    scores = dict(run.get("scores") or {})
    got = _first_nonempty(run, "got", "got_sql", "answer", "output")
    expected = _first_nonempty(run, "expected", "expected_sql", "reference", "reference_ans")

    if _score_is_zero(scores, "file_created") or _mentions_missing_artifact(error_lower):
        mode = "auto_missing_requested_artifact"
    elif not str(got or "").strip() and any(key in run for key in ("got", "got_sql", "answer", "output")):
        mode = "auto_empty_output"
    elif _looks_like_sql_error(error_lower):
        mode = "auto_sql_execution_error"
    elif _looks_like_numeric_parse_error(error_lower):
        mode = "auto_numeric_parse_error"
    elif _has_format_score_failure(scores):
        mode = "auto_output_contract_violation"
    elif expected and got and str(expected).strip() != str(got).strip():
        mode = "auto_expected_output_mismatch"
    elif error:
        mode = "auto_runtime_exception"
    else:
        mode = str(known_failure or "auto_unclassified_failure")
        if not mode.startswith(AUTO_FAILURE_PREFIX):
            mode = "auto_unclassified_failure"

    return FailureDiscovery(
        failure_mode=mode,
        source="automatic",
        signals=signals,
        patch_rules=distill_patch_rules(mode, [run]),
    )


def distill_patch_rules(failure_mode: str, evidence: Iterable[Mapping[str, Any]]) -> List[str]:
    """Convert repeated trajectory evidence into generic model-facing repair rules."""

    evidence = [dict(item or {}) for item in evidence]
    outputs = _collect_requested_outputs(evidence)
    output_hint = ", ".join(outputs[:4]) if outputs else "the requested output artifact(s)"

    if failure_mode == "auto_missing_requested_artifact":
        return [
            f"Before finishing, verify each requested output artifact exists in the workspace: {output_hint}.",
            "If the valid result is empty or no rows qualify, still create the requested artifact with the task-accepted empty-result wording, header, or placeholder.",
        ]
    if failure_mode == "auto_empty_output":
        return [
            "After writing, re-read the required output and verify it is non-empty before finishing.",
            "If the answer is intentionally empty, write the benchmark-accepted empty-result wording or header instead of leaving a blank artifact.",
        ]
    if failure_mode == "auto_output_contract_violation":
        return [
            "Before finishing, re-read the generated output and compare it against the prompt's exact format constraints.",
            "Remove Markdown wrappers, explanations, extra columns, and wrong code prefixes unless the task explicitly requests them.",
        ]
    if failure_mode == "auto_numeric_parse_error":
        return [
            "When parsing tabular or market data, skip rows with blank or non-numeric numeric fields before casting.",
            "Run the calculation script once after filtering sparse rows and only finish after it exits without traceback.",
        ]
    if failure_mode == "auto_sql_execution_error":
        return [
            "Validate the SQL against the provided schema before finishing; use only columns and tables present in the task input.",
            "Execute or mentally simulate the final SQL and fix syntax, missing-column, and missing-table errors before writing the final answer.",
        ]
    if failure_mode == "auto_expected_output_mismatch":
        return [
            "Recompute only the target entity requested by the task and verify its identifiers before writing the final output.",
            "Compare the final artifact against the task's expected output contract rather than copying a nearby or previously solved case.",
        ]
    if failure_mode == "auto_runtime_exception":
        return [
            "Inspect the verifier or traceback, fix the root runtime error, and rerun the deterministic script before finishing.",
            "Do not finish after a failed tool call; produce the requested artifact only after the final command succeeds.",
        ]
    if failure_mode.startswith(AUTO_FAILURE_PREFIX):
        return [
            "Inspect the verifier error and transcript, identify the missing contract, and add an explicit check for it before finishing.",
            "Re-run the final verification step and only finish after the requested artifact and format are both present.",
        ]
    return []


def is_auto_failure_mode(failure_mode: str) -> bool:
    return str(failure_mode or "").startswith(AUTO_FAILURE_PREFIX)


def _signals(benchmark: str, run: Mapping[str, Any]) -> Dict[str, Any]:
    scores = dict(run.get("scores") or {})
    score_failures = {
        str(key): value
        for key, value in scores.items()
        if isinstance(value, (int, float)) and float(value) < 1.0
    }
    return {
        "benchmark": str(benchmark or ""),
        "error": _compact_text(run.get("error") or ""),
        "requested_outputs": _requested_outputs_from_run(run),
        "output_contract": str(run.get("output_contract") or ""),
        "score_failures": score_failures,
        "got_empty": not bool(str(_first_nonempty(run, "got", "got_sql", "answer", "output") or "").strip()),
    }


def _collect_requested_outputs(evidence: Iterable[Mapping[str, Any]]) -> List[str]:
    outputs: List[str] = []
    for item in evidence:
        run = _event_to_run_like(item)
        for output in _requested_outputs_from_run(run):
            if output not in outputs:
                outputs.append(output)
    return outputs


def _requested_outputs_from_run(run: Mapping[str, Any]) -> List[str]:
    raw = run.get("requested_output_files") or run.get("requested_outputs") or []
    if isinstance(raw, str):
        outputs = [raw]
    else:
        outputs = [str(item) for item in raw if str(item or "").strip()]
    if outputs:
        return outputs

    output_contract = str(run.get("output_contract") or "")
    return [item for item in re.findall(r"[\w./-]+\.txt", output_contract) if item]


def _event_to_run_like(item: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = dict(item.get("metadata") or {})
    discovery = dict(metadata.get("failure_discovery") or item.get("failure_discovery") or {})
    signals = dict(discovery.get("signals") or {})
    merged = dict(item)
    for key, value in metadata.items():
        merged.setdefault(key, value)
    for key, value in signals.items():
        merged.setdefault(key, value)
    if "requested_outputs" in signals:
        merged.setdefault("requested_output_files", signals["requested_outputs"])
    return merged


def _first_nonempty(run: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = run.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _score_is_zero(scores: Mapping[str, Any], key: str) -> bool:
    try:
        return float(scores.get(key)) == 0.0
    except (TypeError, ValueError):
        return False


def _has_format_score_failure(scores: Mapping[str, Any]) -> bool:
    format_names = {
        "valid_format",
        "valid_values",
        "valid_codes",
        "reasonable_values",
        "consistency",
        "sorted_desc",
        "valid_dates",
        "count_limit",
        "file_format",
        "format",
    }
    for key, value in scores.items():
        if any(name in str(key) for name in format_names):
            try:
                if float(value) < 1.0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _mentions_missing_artifact(text: str) -> bool:
    patterns = (
        "no such file",
        "file not found",
        "filenotfounderror",
        "missing output",
        "missing requested output",
    )
    return any(pattern in text for pattern in patterns)


def _looks_like_sql_error(text: str) -> bool:
    patterns = (
        "sqlite",
        "sql error",
        "syntax error",
        "no such column",
        "no such table",
        "operationalerror",
    )
    return any(pattern in text for pattern in patterns)


def _looks_like_numeric_parse_error(text: str) -> bool:
    patterns = (
        "could not convert string to float",
        "invalid literal",
        "non-numeric",
        "nan",
        "valueerror",
    )
    return any(pattern in text for pattern in patterns)


def _compact_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
