"""Skill belief registry with JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from bayesian_agent.core.belief import SkillBelief
from bayesian_agent.core.evidence import TrajectoryEvidence, utc_now


class BayesianSkillRegistry:
    """Persistent registry of Bayesian Skill/SOP beliefs."""

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.path = Path(path) if path is not None else None
        self.data = self._load()

    @classmethod
    def in_memory(cls) -> "BayesianSkillRegistry":
        return cls(None)

    def _load(self) -> Dict[str, Any]:
        if self.path is None or not self.path.exists():
            return {"version": 1, "updated_at": utc_now(), "skills": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {"version": 1, "updated_at": utc_now(), "skills": {}}
        raw.setdefault("version", 1)
        raw.setdefault("updated_at", utc_now())
        raw.setdefault("skills", {})
        return raw

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, skill_id: str) -> SkillBelief:
        raw = self.data.get("skills", {}).get(skill_id, {})
        return SkillBelief.from_dict(skill_id, raw)

    def record(self, event: TrajectoryEvidence) -> SkillBelief:
        belief = self.get(event.skill_id)
        belief.update(event)
        self.data.setdefault("skills", {})[event.skill_id] = belief.to_dict()
        self.save()
        return belief

    def record_many(self, events: Iterable[TrajectoryEvidence]) -> List[SkillBelief]:
        beliefs = []
        for event in events:
            beliefs.append(self.record(event))
        return beliefs

    def beliefs(self) -> List[SkillBelief]:
        return [SkillBelief.from_dict(skill_id, raw) for skill_id, raw in self.data.get("skills", {}).items()]

    def top(self, limit: int = 5, context: str = "") -> List[SkillBelief]:
        beliefs = self.beliefs()

        def score(belief: SkillBelief):
            context_bonus = 1 if context and context in belief.contexts else 0
            return (context_bonus, belief.success_probability, belief.observations, -belief.mean_tokens)

        return sorted(beliefs, key=score, reverse=True)[:limit]
