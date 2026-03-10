"""
TS Self-Modifying Convergence — adaptive parameters.

Stores reasoning parameters as small floats with hard bounds.
Persists to config/adaptive_config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# Bounds (hard limits to prevent unstable behaviour)
PROPAGATION_STRENGTH_MIN, PROPAGATION_STRENGTH_MAX = 0.05, 0.20
CONVERGENCE_THRESHOLD_MIN, CONVERGENCE_THRESHOLD_MAX = 0.80, 0.90
DECISION_COLLAPSE_THRESHOLD_MIN, DECISION_COLLAPSE_THRESHOLD_MAX = 0.70, 0.85
FIELD_INFLUENCE_STRENGTH_MIN, FIELD_INFLUENCE_STRENGTH_MAX = 0.05, 0.12
ATTRACTOR_INFLUENCE_STRENGTH_MIN, ATTRACTOR_INFLUENCE_STRENGTH_MAX = 0.05, 0.15

DEFAULT_PROPAGATION_STRENGTH = 0.10
DEFAULT_CONVERGENCE_THRESHOLD = 0.85
DEFAULT_DECISION_COLLAPSE_THRESHOLD = 0.75
DEFAULT_FIELD_INFLUENCE_STRENGTH = 0.08
DEFAULT_ATTRACTOR_INFLUENCE_STRENGTH = 0.10


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class AdaptiveConfig:
    def __init__(
        self,
        propagation_strength: float = DEFAULT_PROPAGATION_STRENGTH,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        decision_collapse_threshold: float = DEFAULT_DECISION_COLLAPSE_THRESHOLD,
        field_influence_strength: float = DEFAULT_FIELD_INFLUENCE_STRENGTH,
        attractor_influence_strength: float = DEFAULT_ATTRACTOR_INFLUENCE_STRENGTH,
    ) -> None:
        self.propagation_strength = _clamp(propagation_strength, PROPAGATION_STRENGTH_MIN, PROPAGATION_STRENGTH_MAX)
        self.convergence_threshold = _clamp(convergence_threshold, CONVERGENCE_THRESHOLD_MIN, CONVERGENCE_THRESHOLD_MAX)
        self.decision_collapse_threshold = _clamp(
            decision_collapse_threshold,
            DECISION_COLLAPSE_THRESHOLD_MIN,
            DECISION_COLLAPSE_THRESHOLD_MAX,
        )
        self.field_influence_strength = _clamp(field_influence_strength, FIELD_INFLUENCE_STRENGTH_MIN, FIELD_INFLUENCE_STRENGTH_MAX)
        self.attractor_influence_strength = _clamp(
            attractor_influence_strength,
            ATTRACTOR_INFLUENCE_STRENGTH_MIN,
            ATTRACTOR_INFLUENCE_STRENGTH_MAX,
        )

    def apply_delta(self, param: str, delta: float) -> bool:
        """Apply a small adjustment (±0.01 typical). Returns True if value changed."""
        step = max(-0.01, min(0.01, float(delta)))
        if param == "propagation_strength":
            old = self.propagation_strength
            self.propagation_strength = _clamp(self.propagation_strength + step, PROPAGATION_STRENGTH_MIN, PROPAGATION_STRENGTH_MAX)
            return self.propagation_strength != old
        if param == "convergence_threshold":
            old = self.convergence_threshold
            self.convergence_threshold = _clamp(self.convergence_threshold + step, CONVERGENCE_THRESHOLD_MIN, CONVERGENCE_THRESHOLD_MAX)
            return self.convergence_threshold != old
        if param == "decision_collapse_threshold":
            old = self.decision_collapse_threshold
            self.decision_collapse_threshold = _clamp(
                self.decision_collapse_threshold + step,
                DECISION_COLLAPSE_THRESHOLD_MIN,
                DECISION_COLLAPSE_THRESHOLD_MAX,
            )
            return self.decision_collapse_threshold != old
        if param == "field_influence_strength":
            old = self.field_influence_strength
            self.field_influence_strength = _clamp(self.field_influence_strength + step, FIELD_INFLUENCE_STRENGTH_MIN, FIELD_INFLUENCE_STRENGTH_MAX)
            return self.field_influence_strength != old
        if param == "attractor_influence_strength":
            old = self.attractor_influence_strength
            self.attractor_influence_strength = _clamp(
                self.attractor_influence_strength + step,
                ATTRACTOR_INFLUENCE_STRENGTH_MIN,
                ATTRACTOR_INFLUENCE_STRENGTH_MAX,
            )
            return self.attractor_influence_strength != old
        return False

    def to_dict(self) -> Dict[str, float]:
        return {
            "propagation_strength": self.propagation_strength,
            "convergence_threshold": self.convergence_threshold,
            "decision_collapse_threshold": self.decision_collapse_threshold,
            "field_influence_strength": self.field_influence_strength,
            "attractor_influence_strength": self.attractor_influence_strength,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveConfig:
        return cls(
            propagation_strength=float(data.get("propagation_strength", DEFAULT_PROPAGATION_STRENGTH)),
            convergence_threshold=float(data.get("convergence_threshold", DEFAULT_CONVERGENCE_THRESHOLD)),
            decision_collapse_threshold=float(data.get("decision_collapse_threshold", DEFAULT_DECISION_COLLAPSE_THRESHOLD)),
            field_influence_strength=float(data.get("field_influence_strength", DEFAULT_FIELD_INFLUENCE_STRENGTH)),
            attractor_influence_strength=float(data.get("attractor_influence_strength", DEFAULT_ATTRACTOR_INFLUENCE_STRENGTH)),
        )


def get_config_path() -> Path:
    # Repo root config/ (next to graph.yaml)
    return Path(__file__).resolve().parents[2] / "config" / "adaptive_config.yaml"


def load_adaptive_config() -> AdaptiveConfig:
    path = get_config_path()
    if not path.exists():
        return AdaptiveConfig()
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return AdaptiveConfig.from_dict(data)
    except Exception:
        return AdaptiveConfig()


def save_adaptive_config(config: AdaptiveConfig) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False)
