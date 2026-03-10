"""
Convergence metrics for TS wave reasoning.

compute_convergence(nodes) returns a 0–1 score:
score = 1 - normalized_state_change

Uses only:
- average state change
- energy variance (as a stabilizer)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .wave_nodes import WaveNode

from .config import VECTOR_SIZE

# Module-level tiny cache to avoid storing history in nodes.
_PREV_STATES: Dict[str, np.ndarray] = {}


def compute_convergence(
    nodes: Dict[str, WaveNode],
) -> float:
    if not nodes:
        return 0.0

    # Average state change vs previous tick.
    total_change = 0.0
    count = 0
    for node_id, n in nodes.items():
        prev = _PREV_STATES.get(node_id)
        cur = np.asarray(n.state_vector, dtype=np.float32)
        if cur.size != VECTOR_SIZE:
            cur = cur[:VECTOR_SIZE] if cur.size > VECTOR_SIZE else np.pad(cur, (0, VECTOR_SIZE - int(cur.size)))
        if prev is not None:
            total_change += float(np.mean(np.abs(cur - prev)))
            count += 1
        _PREV_STATES[node_id] = cur.copy()

    if count == 0:
        # No prior snapshot yet; start at low convergence.
        return 0.0
    avg_change = total_change / float(count)

    # Normalize state change into [0, 1] using a conservative cap.
    # With VECTOR_SIZE=8 and small nudges, avg_change should stay low.
    normalized_change = min(1.0, avg_change / 0.1)

    # Energy variance as a stabilizer (higher variance -> lower score).
    energies = np.array([n.energy for n in nodes.values()], dtype=np.float32)
    energy_var = float(energies.var()) if energies.size else 0.0
    energy_penalty = min(0.2, energy_var)  # small penalty cap

    score = (1.0 - normalized_change) - energy_penalty
    return float(max(0.0, min(1.0, score)))

