"""
TS Self-Modifying Convergence — performance tracking and analysis.

Keeps last 20 reasoning cycles; suggests small parameter adjustments.
No ML, no heavy optimization — simple statistics only.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml

from .config import MAX_TICKS

MAX_RECENT_CYCLES = 20
ADJUSTMENT_STEP = 0.01


def _performance_history_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "performance_history.yaml"

# Heuristics
TICK_COUNT_SLOW = 4  # consider "too many ticks" if often >= this
CONVERGENCE_LOW = 0.75  # consider "too low" if mean convergence below this
STABILITY_RATIO_HIGH = 0.6  # if stability_ratio > this, may want to reduce propagation


class ReasoningPerformanceMonitor:
    def __init__(self) -> None:
        self.recent_tick_counts: deque = deque(maxlen=MAX_RECENT_CYCLES)
        self.recent_convergence_scores: deque = deque(maxlen=MAX_RECENT_CYCLES)
        self.recent_stability_events: deque = deque(maxlen=MAX_RECENT_CYCLES)

    def record_cycle(
        self,
        ticks_run: int,
        convergence: float,
        stopped_by: str,
    ) -> None:
        self.recent_tick_counts.append(ticks_run)
        self.recent_convergence_scores.append(convergence)
        self.recent_stability_events.append(stopped_by == "stability")

    def save_history(self) -> None:
        path = _performance_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tick_counts": list(self.recent_tick_counts),
            "convergence_scores": list(self.recent_convergence_scores),
            "stability_events": list(self.recent_stability_events),
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def load_history(self) -> None:
        path = _performance_history_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            self.recent_tick_counts = deque(data.get("tick_counts", [])[-MAX_RECENT_CYCLES:], maxlen=MAX_RECENT_CYCLES)
            self.recent_convergence_scores = deque(data.get("convergence_scores", [])[-MAX_RECENT_CYCLES:], maxlen=MAX_RECENT_CYCLES)
            self.recent_stability_events = deque(data.get("stability_events", [])[-MAX_RECENT_CYCLES:], maxlen=MAX_RECENT_CYCLES)
        except Exception:
            pass

    def analyze_performance(self) -> List[Tuple[str, float]]:
        """
        Detect patterns and suggest small parameter deltas.
        Returns list of (param_name, delta) with delta in [-0.01, 0.01].
        """
        suggestions: List[Tuple[str, float]] = []
        n = len(self.recent_tick_counts)
        if n < 3:
            return suggestions

        mean_ticks = sum(self.recent_tick_counts) / float(n)
        mean_convergence = sum(self.recent_convergence_scores) / float(n)
        stability_count = sum(1 for x in self.recent_stability_events if x)
        stability_ratio = stability_count / float(n)

        # At most one suggestion per parameter; prefer order: propagation, then decision threshold.
        # Reasoning taking too many ticks -> increase propagation_strength
        if mean_ticks >= TICK_COUNT_SLOW:
            suggestions.append(("propagation_strength", ADJUSTMENT_STEP))
        # Instability (often stopping by stability, low convergence) -> reduce propagation_strength
        elif stability_ratio > STABILITY_RATIO_HIGH and mean_convergence < CONVERGENCE_LOW:
            suggestions.append(("propagation_strength", -ADJUSTMENT_STEP))
        # Convergence scores too low and slow -> increase propagation
        elif mean_convergence < CONVERGENCE_LOW and mean_ticks >= 3:
            suggestions.append(("propagation_strength", ADJUSTMENT_STEP))

        # Decisions collapsing too early (low convergence proxy): require higher confidence
        if mean_convergence < 0.72 and n >= 5:
            suggestions.append(("decision_collapse_threshold", ADJUSTMENT_STEP))

        return suggestions


def apply_adjustments(
    config: Any,
    suggestions: List[Tuple[str, float]],
    logger: Optional[Any] = None,
) -> int:
    """
    Apply suggested deltas to AdaptiveConfig. Returns number of params changed.
    """
    changed = 0
    for param, delta in suggestions:
        if config.apply_delta(param, delta):
            changed += 1
            if logger:
                new_val = getattr(config, param, None)
                logger.info(
                    "Adaptive convergence update: %s %s to %.2f",
                    param,
                    "increased" if delta > 0 else "reduced",
                    new_val,
                )
    return changed


_GLOBAL_MONITOR: Optional[ReasoningPerformanceMonitor] = None


def get_global_performance_monitor() -> ReasoningPerformanceMonitor:
    global _GLOBAL_MONITOR
    if _GLOBAL_MONITOR is None:
        _GLOBAL_MONITOR = ReasoningPerformanceMonitor()
        _GLOBAL_MONITOR.load_history()
    return _GLOBAL_MONITOR
