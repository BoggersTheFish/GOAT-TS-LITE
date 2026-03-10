"""
TS decision nodes for Thinking Wave convergence.

DecisionNode models a node that can collapse to a discrete reasoning state
once its state probabilities exceed a collapse threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence

from .config import MAX_NODES


@dataclass
class DecisionNode:
    node_id: str
    possible_states: Sequence[Any]
    state_probabilities: List[float]
    collapse_threshold: float = 0.75
    _collapsed_state: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Hard limit: <= 5 possible states.
        if len(self.possible_states) > 5:
            self.possible_states = list(self.possible_states)[:5]

        probs = [float(p) for p in (self.state_probabilities or [])]
        if len(probs) != len(self.possible_states):
            probs = [1.0 for _ in self.possible_states]
        total = float(sum(probs))
        if total <= 0.0:
            probs = [1.0 / float(len(probs)) for _ in probs] if probs else [1.0]
        else:
            probs = [p / total for p in probs]
        self.state_probabilities = probs

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed_state is not None

    def evaluate(self) -> bool:
        """
        Return True when this node should collapse according to the current
        probability distribution.
        """
        if self.is_collapsed:
            return True
        max_p = max(self.state_probabilities) if self.state_probabilities else 0.0
        # Per spec: collapse when max_probability > 0.75
        return float(max_p) > float(self.collapse_threshold)

    def collapse(self) -> bool:
        """
        Collapse the node to its best state if threshold is exceeded.
        Returns True if a collapse happened in this call.
        """
        if self.is_collapsed:
            return False
        if not self.evaluate():
            return False
        idx = int(max(range(len(self.state_probabilities)), key=self.state_probabilities.__getitem__))
        self._collapsed_state = self.possible_states[idx]
        return True

    def get_best_state(self) -> Any:
        """
        Return the current best state (collapsed state if available,
        otherwise the argmax state).
        """
        if self.is_collapsed:
            return self._collapsed_state
        if not self.state_probabilities:
            return self.possible_states[0] if self.possible_states else None
        idx = int(max(range(len(self.state_probabilities)), key=self.state_probabilities.__getitem__))
        return self.possible_states[idx]

