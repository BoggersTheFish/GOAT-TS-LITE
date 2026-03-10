"""
TSAgent: multi-agent TS reasoning waves.

Each agent has its own local view over a subset of nodes and a perspective
vector that nudges node energies slightly in a particular direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .wave_nodes import WaveNode
from .config import VECTOR_SIZE


@dataclass
class TSAgent:
    agent_id: str
    local_wave_nodes: Dict[str, WaveNode]
    perspective_vector: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.perspective_vector, dtype=np.float32)
        if v.size == 0:
            v = np.zeros(VECTOR_SIZE, dtype=np.float32)
        if v.size < VECTOR_SIZE:
            v = np.concatenate([v, np.zeros(VECTOR_SIZE - int(v.size), dtype=np.float32)])
        elif v.size > VECTOR_SIZE:
            v = v[:VECTOR_SIZE]
        self.perspective_vector = v

    def generate_wave(self, strength: float = 0.05) -> None:
        """
        Seed energies on local nodes to start a wave.
        Lightweight: only touches the local subset.
        """
        for node in self.local_wave_nodes.values():
            node.update_state(delta_energy=strength)

    def propagate(self) -> None:
        """
        Placeholder for agent-internal propagation.
        We keep this extremely lightweight; main propagation happens in the
        shared TSConvergenceEngine.
        """
        # No-op for now to avoid extra cost; reserved for future extensions.
        return

    def influence_nodes(self, nodes: Dict[str, WaveNode], strength: float = 0.03) -> None:
        """
        Gently adjust node state vectors based on the agent's perspective vector.
        Only acts on the intersection of local_wave_nodes and the global nodes.
        """
        for node_id, local_node in self.local_wave_nodes.items():
            node = nodes.get(node_id)
            if node is None:
                continue
            # Element-wise nudge (no dot products, no large ops).
            delta = self.perspective_vector * float(strength)
            node.update_state(delta=delta, delta_energy=0.0)

