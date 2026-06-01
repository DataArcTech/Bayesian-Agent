"""Categorical Bayesian evidence model for context-conditioned Skill reliability."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from bayesian_agent.core.evidence import TrajectoryEvidence


LABELS = ("success", "failure")
UNKNOWN = "__unknown__"
NONE = "__none__"


def _label(event: TrajectoryEvidence) -> str:
    return "success" if event.success else "failure"


def _bucket_number(value: float, buckets):
    for upper, name in buckets:
        if value <= upper:
            return name
    return buckets[-1][1]


def _token_bucket(tokens: int) -> str:
    return _bucket_number(
        float(tokens or 0),
        [
            (0, "0"),
            (1_000, "1_1k"),
            (10_000, "1k_10k"),
            (100_000, "10k_100k"),
            (1_000_000, "100k_1m"),
            (float("inf"), "1m_plus"),
        ],
    )


def _turn_bucket(turns: int) -> str:
    return _bucket_number(
        float(turns or 0),
        [
            (0, "0"),
            (2, "1_2"),
            (5, "3_5"),
            (10, "6_10"),
            (20, "11_20"),
            (float("inf"), "20_plus"),
        ],
    )


def _latency_bucket(seconds: float) -> str:
    return _bucket_number(
        float(seconds or 0.0),
        [
            (0, "0s"),
            (10, "0s_10s"),
            (60, "10s_60s"),
            (300, "1m_5m"),
            (1800, "5m_30m"),
            (float("inf"), "30m_plus"),
        ],
    )


def features_from_event(event: TrajectoryEvidence) -> Dict[str, str]:
    """Extract discrete evidence features from verified trajectory evidence."""

    features = {
        "context": str(event.context or UNKNOWN),
        "failure_mode": str(event.failure_mode or NONE),
        "token_bucket": _token_bucket(event.total_tokens),
        "turn_bucket": _turn_bucket(event.turns),
        "latency_bucket": _latency_bucket(event.elapsed_seconds),
    }
    for key, value in sorted((event.metadata or {}).items()):
        if isinstance(value, (str, int, float, bool)) and len(str(value)) <= 80:
            features[f"metadata.{key}"] = str(value)
    return features


@dataclass
class CategoricalBayesState:
    """Categorical likelihood state for binary success/failure evidence."""

    alpha: float = 1.0
    class_counts: Dict[str, int] = field(default_factory=lambda: {label: 0 for label in LABELS})
    feature_counts: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    feature_vocab: Dict[str, Dict[str, int]] = field(default_factory=dict)
    observations: int = 0

    def update(self, event: TrajectoryEvidence) -> "CategoricalBayesState":
        label = _label(event)
        self.class_counts[label] = int(self.class_counts.get(label, 0)) + 1
        self.observations += 1
        for name, value in features_from_event(event).items():
            value = str(value)
            self.feature_vocab.setdefault(name, {})
            self.feature_vocab[name][value] = int(self.feature_vocab[name].get(value, 0)) + 1
            self.feature_counts.setdefault(label, {}).setdefault(name, {})
            current = self.feature_counts[label][name].get(value, 0)
            self.feature_counts[label][name][value] = int(current) + 1
        return self

    def class_probability(self, label: str) -> float:
        count = int(self.class_counts.get(label, 0))
        total = sum(int(self.class_counts.get(item, 0)) for item in LABELS)
        return (count + self.alpha) / (total + self.alpha * len(LABELS))

    def feature_probability(self, name: str, value: Any, label: str) -> float:
        value = str(value)
        vocab = self.feature_vocab.get(name, {})
        vocab_size = len(vocab) + (0 if value in vocab else 1)
        vocab_size = max(vocab_size, 1)
        count = int(self.feature_counts.get(label, {}).get(name, {}).get(value, 0))
        feature_total = sum(int(item) for item in self.feature_counts.get(label, {}).get(name, {}).values())
        return (count + self.alpha) / (feature_total + self.alpha * vocab_size)

    def predict_proba(self, features: Mapping[str, Any] = None) -> Dict[str, float]:
        features = dict(features or {})
        logs = {}
        for label in LABELS:
            logs[label] = math.log(self.class_probability(label))
            for name, value in sorted(features.items()):
                logs[label] += math.log(self.feature_probability(name, value, label))
        max_log = max(logs.values())
        scores = {label: math.exp(value - max_log) for label, value in logs.items()}
        total = sum(scores.values())
        return {label: scores[label] / total if total else 0.0 for label in LABELS}

    def predict_success(self, features: Mapping[str, Any] = None) -> float:
        return self.predict_proba(features).get("success", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "class_counts": {label: int(self.class_counts.get(label, 0)) for label in LABELS},
            "feature_counts": self.feature_counts,
            "feature_vocab": self.feature_vocab,
            "observations": int(self.observations),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CategoricalBayesState":
        class_counts = {label: int((raw.get("class_counts") or {}).get(label, 0)) for label in LABELS}
        return cls(
            alpha=float(raw.get("alpha", 1.0)),
            class_counts=class_counts,
            feature_counts=_nested_ints(raw.get("feature_counts") or {}),
            feature_vocab=_nested_ints(raw.get("feature_vocab") or {}),
            observations=int(raw.get("observations") or sum(class_counts.values())),
        )


def _nested_ints(raw):
    if not isinstance(raw, dict):
        return {}
    result = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            result[str(key)] = _nested_ints(value)
        else:
            result[str(key)] = int(value)
    return result


NaiveBayesState = CategoricalBayesState
