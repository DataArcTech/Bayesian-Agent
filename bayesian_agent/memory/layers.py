"""Three focused memory layers inspired by hippocampus/state/cortex."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.core.registry import BayesianSkillRegistry


@dataclass
class HippocampusMemory:
    """Fast episodic memory for the current harness session."""

    items: List[str] = field(default_factory=list)
    max_items: int = 12

    def remember(self, text: Any) -> None:
        text = str(text or "").strip()
        if not text:
            return
        self.items.append(text)
        self.items = self.items[-int(self.max_items or 12) :]

    def clear(self) -> None:
        self.items.clear()

    def render(self) -> str:
        if not self.items:
            return ""
        lines = ["### Hippocampus"]
        lines.extend(f"- {item}" for item in self.items)
        return "\n".join(lines)


@dataclass
class StateMemory:
    """Intermediate state for the active task or benchmark run."""

    values: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        key = str(key or "").strip()
        if key:
            self.values[key] = _json_safe(value)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def clear(self) -> None:
        self.values.clear()

    def render(self) -> str:
        if not self.values:
            return ""
        lines = ["### Intermediate State"]
        for key in sorted(self.values):
            lines.append(f"- {key}: {self.values[key]}")
        return "\n".join(lines)


class CorticalMemory:
    """Durable memory: stable notes plus persistent Bayesian Skill beliefs."""

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        registry: Optional[BayesianSkillRegistry] = None,
        registry_path: Optional[Union[str, Path]] = None,
        max_items: int = 48,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.max_items = int(max_items or 48)
        self.registry = registry or (
            BayesianSkillRegistry(registry_path) if registry_path is not None else BayesianSkillRegistry.in_memory()
        )
        self.items: List[str] = []
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            self.items = []
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.items = []
            return
        self.items = [str(item) for item in raw.get("items", []) if str(item or "").strip()]
        self.items = self.items[-self.max_items :]

    def remember(self, text: Any) -> None:
        text = str(text or "").strip()
        if not text:
            return
        self.items.append(text)
        self.items = self.items[-self.max_items :]
        self.save()

    def record_evidence(self, evidence: TrajectoryEvidence):
        belief = self.registry.record(evidence)
        self.save()
        return belief

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "items": self.items}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def render(self) -> str:
        if not self.items:
            return ""
        lines = ["### Cortex"]
        lines.extend(f"- {item}" for item in self.items)
        return "\n".join(lines)


class ThreeLayerMemory:
    """BA memory with exactly three layers: hippocampus, state, cortex."""

    def __init__(
        self,
        *,
        hippocampus: Optional[HippocampusMemory] = None,
        state: Optional[StateMemory] = None,
        cortex: Optional[CorticalMemory] = None,
        cortex_path: Optional[Union[str, Path]] = None,
        registry: Optional[BayesianSkillRegistry] = None,
        registry_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.hippocampus = hippocampus or HippocampusMemory()
        self.state = state or StateMemory()
        self.cortex = cortex or CorticalMemory(cortex_path, registry=registry, registry_path=registry_path)

    def layer_names(self) -> List[str]:
        return ["hippocampus", "state", "cortex"]

    def render_context(self) -> str:
        sections = [self.hippocampus.render(), self.state.render(), self.cortex.render()]
        return "\n\n".join(section for section in sections if section).strip()

    def reset_fast_layers(self) -> None:
        self.hippocampus.clear()
        self.state.clear()

    def save(self) -> None:
        self.cortex.save()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
