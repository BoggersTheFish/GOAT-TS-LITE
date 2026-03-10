"""
Stability detection for TS waves.

detect_stability(nodes) returns True when average state-vector change
between ticks falls below a small threshold.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .wave_nodes import WaveNode
from .config import VECTOR_SIZE

_PREV_STATES: Dict[str, np.ndarray] = {}


def detect_stability(nodes: Dict[str, WaveNode], threshold: float = 0.01) -> bool:
    if not nodes:
        return True
    total_change = 0.0
    count = 0
    for node_id, n in nodes.items():
        cur = np.asarray(n.state_vector, dtype=np.float32)
        if cur.size != VECTOR_SIZE:
            cur = cur[:VECTOR_SIZE] if cur.size > VECTOR_SIZE else np.pad(cur, (0, VECTOR_SIZE - int(cur.size)))
        prev = _PREV_STATES.get(node_id)
        if prev is not None:
            total_change += float(np.mean(np.abs(cur - prev)))
            count += 1
        _PREV_STATES[node_id] = cur.copy()
    if count == 0:
        # No prior snapshot yet; don't declare stability on first tick.
        return False
    avg_change = total_change / float(count)
    return avg_change < threshold

