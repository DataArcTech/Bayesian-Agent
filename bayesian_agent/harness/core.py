"""Core harness orchestration owned by Bayesian-Agent."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.core.registry import BayesianSkillRegistry
from bayesian_agent.harness.types import HarnessTask
from bayesian_agent.memory import ThreeLayerMemory


class AgentHarness:
    """Normalize task execution around an interchangeable backend adapter."""

    def __init__(
        self,
        adapter: Any,
        *,
        memory: Optional[ThreeLayerMemory] = None,
        cortex_path: Optional[Union[str, Path]] = None,
        registry: Optional[BayesianSkillRegistry] = None,
        registry_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.adapter = adapter
        self.memory = memory or ThreeLayerMemory(
            cortex_path=cortex_path,
            registry=registry,
            registry_path=registry_path,
        )

    def integration_note(self) -> str:
        note = getattr(self.adapter, "integration_note", None)
        if note is None:
            return "Bayesian-Agent harness executes normalized tasks through an adapter backend."
        return note()

    def resolve_root(self):
        resolver = getattr(self.adapter, "resolve_root", None)
        if resolver is None:
            return None
        return resolver()

    def load_run_from_workspace(self, workspace: Union[str, Path]):
        loader = getattr(self.adapter, "load_run_from_workspace", None)
        if loader is None:
            return None
        return loader(workspace)

    def run(self, task: Mapping[str, Any], skill_context: str = "") -> Mapping[str, Any]:
        prompt = str(task["prompt"])
        context = skill_context or str(task.get("skill_context") or "")
        envelope = HarnessTask(
            task_id=str(task.get("task_id") or ""),
            prompt=prompt,
            workspace=task["workspace"],
            max_turns=int(task.get("max_turns", 8) or 8),
            skill_context=context,
            memory_context=bool(task.get("memory_context", False)),
            task_context=str(task.get("task_context") or ""),
            metadata=dict(task.get("metadata") or {}),
        )
        return self.run_task(envelope)

    def run_task(self, task: HarnessTask) -> Mapping[str, Any]:
        workspace = task.workspace_path()
        workspace.mkdir(parents=True, exist_ok=True)
        prompt, memory_context = self.prepare_prompt(task)
        task_payload = task.to_dict()
        task_payload["prepared_prompt"] = prompt
        _write_json(workspace / "harness_task.json", task_payload)

        started = time.time()
        run = self.adapter.run_task(prompt=prompt, workspace=workspace, max_turns=int(task.max_turns or 8))
        elapsed = time.time() - started
        normalized = self._normalize_run(
            run,
            task=task,
            workspace=workspace,
            elapsed_seconds=elapsed,
            memory_context=memory_context,
        )
        _write_json(workspace / "harness_run.json", normalized)
        self.memory.hippocampus.remember(
            f"{task.task_id or workspace.name}: exit={normalized.get('exit_reason', '')} "
            f"tokens={normalized.get('total_tokens', 0)}"
        )
        self.memory.state.set("last_task_id", task.task_id)
        self.memory.state.set("last_workspace", str(workspace))
        return normalized

    def prepare_prompt(self, task: HarnessTask) -> tuple[str, str]:
        memory_context = self.memory.render_context() if task.memory_context else ""
        parts = [task.skill_context, memory_context, task.prompt]
        return "\n\n".join(str(part).strip() for part in parts if str(part or "").strip()), memory_context

    def record_outcome(
        self,
        run: Mapping[str, Any],
        *,
        skill_id: str,
        context: str,
        success: bool,
        failure_mode: str = "",
        summary: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        outcome = "success" if success else "failure"
        enriched = dict(run)
        enriched["success"] = bool(success)
        if failure_mode:
            enriched["failure_mode"] = failure_mode
        if summary:
            enriched["summary"] = summary
        if metadata:
            enriched["metadata"] = {**dict(enriched.get("metadata") or {}), **dict(metadata)}
        evidence = TrajectoryEvidence.from_run(
            enriched,
            skill_id=skill_id,
            context=context,
            failure_mode=failure_mode or str(enriched.get("failure_mode") or ""),
        )
        belief = self.memory.cortex.record_evidence(evidence)
        self.memory.hippocampus.remember(f"{evidence.task_id}: {outcome} for {skill_id}")
        self.memory.state.set("last_outcome", outcome)
        self.memory.state.set("last_failure_mode", failure_mode)
        return belief

    @staticmethod
    def _normalize_run(
        run: Mapping[str, Any],
        *,
        task: HarnessTask,
        workspace: Path,
        elapsed_seconds: float,
        memory_context: str,
    ) -> Mapping[str, Any]:
        normalized = dict(run or {})
        input_tokens = int(normalized.get("input_tokens") or 0)
        output_tokens = int(normalized.get("output_tokens") or 0)
        total_tokens = int(normalized.get("total_tokens") or 0)
        if not total_tokens:
            total_tokens = input_tokens + output_tokens
        normalized["task_id"] = str(task.task_id)
        normalized["workspace"] = str(workspace)
        normalized["input_tokens"] = input_tokens
        normalized["output_tokens"] = output_tokens
        normalized["total_tokens"] = total_tokens
        normalized.setdefault("elapsed_seconds", elapsed_seconds)
        normalized["harness_memory_context"] = bool(memory_context)
        normalized["harness_artifacts"] = {
            "task": str(workspace / "harness_task.json"),
            "run": str(workspace / "harness_run.json"),
        }
        return _json_safe(normalized)


def ensure_harness(
    runner: Any,
    *,
    cortex_path: Optional[Union[str, Path]] = None,
    registry: Optional[BayesianSkillRegistry] = None,
    registry_path: Optional[Union[str, Path]] = None,
) -> AgentHarness:
    """Return a harness, preserving an existing wrapper when supplied."""

    if isinstance(runner, AgentHarness):
        if registry is not None:
            runner.memory.cortex.registry = registry
        if cortex_path is not None and runner.memory.cortex.path is None:
            runner.memory.cortex.path = Path(cortex_path)
        return runner
    return AgentHarness(runner, cortex_path=cortex_path, registry=registry, registry_path=registry_path)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(data)), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
