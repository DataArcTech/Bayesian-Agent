"""Skill belief registry with JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from bayesian_agent.core.algorithms import DEFAULT_ALGORITHM, SUPPORTED_ALGORITHMS, normalize_algorithm
from bayesian_agent.core.belief import SkillBelief
from bayesian_agent.core.evidence import TrajectoryEvidence, utc_now


class BayesianSkillRegistry:
    """Persistent registry of Bayesian Skill/SOP beliefs."""

    def __init__(self, path: Optional[Union[str, Path]] = None, algorithm: str = DEFAULT_ALGORITHM):
        self.path = Path(path) if path is not None else None
        self.algorithm = normalize_algorithm(algorithm if algorithm in SUPPORTED_ALGORITHMS else DEFAULT_ALGORITHM)
        self.data = self._load()
        self.data.setdefault("algorithm", self.algorithm)
        self.algorithm = normalize_algorithm(self.data.get("algorithm") if self.data.get("algorithm") in SUPPORTED_ALGORITHMS else self.algorithm)
        self.data["algorithm"] = self.algorithm

    @classmethod
    def in_memory(cls, algorithm: str = DEFAULT_ALGORITHM) -> "BayesianSkillRegistry":
        return cls(None, algorithm=algorithm)

    def _load(self) -> Dict[str, Any]:
        if self.path is None or not self.path.exists():
            return {"version": 1, "algorithm": self.algorithm, "updated_at": utc_now(), "skills": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {"version": 1, "algorithm": self.algorithm, "updated_at": utc_now(), "skills": {}}
        raw.setdefault("version", 1)
        raw.setdefault("updated_at", utc_now())
        raw.setdefault("skills", {})
        if "algorithm" not in raw and raw["skills"]:
            raw["algorithm"] = "beta_bernoulli"
        return raw

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        self.data["algorithm"] = self.algorithm
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, skill_id: str) -> SkillBelief:
        raw = self.data.get("skills", {}).get(skill_id, {})
        return SkillBelief.from_dict(skill_id, raw, algorithm=self.algorithm)

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
        return [
            SkillBelief.from_dict(skill_id, raw, algorithm=self.algorithm)
            for skill_id, raw in self.data.get("skills", {}).items()
        ]

    def top(self, limit: int = 5, context: str = "") -> List[SkillBelief]:
        beliefs = self.beliefs()

        def score(belief: SkillBelief):
            context_bonus = 1 if context and context in belief.contexts else 0
            success = belief.predict_success_probability(context=context) if context else belief.success_probability
            return (context_bonus, success, belief.observations, -belief.mean_tokens)

        return sorted(beliefs, key=score, reverse=True)[:limit]
