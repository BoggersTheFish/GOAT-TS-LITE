"""
GOAT-TS Lite — TS wave-state nodes for minimal-cost reasoning.
Represents wave-like reasoning states instead of heavy graph updates.
Graph updates store node states rather than full object graphs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .config import VECTOR_SIZE

class WaveNode:
    """
    Lightweight node for wave-like reasoning states.
    Used for incremental propagation; only touched nodes update.
    """

    __slots__ = ("node_id", "state_vector", "energy", "connections")

    def __init__(
        self,
        node_id: str,
        state_vector: Optional[np.ndarray] = None,
        energy: float = 0.0,
        connections: Optional[List[str]] = None,
    ) -> None:
        self.node_id = node_id
        if state_vector is None:
            self.state_vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
        else:
            v = np.asarray(state_vector, dtype=np.float32)
            if v.size < VECTOR_SIZE:
                pad = np.zeros(VECTOR_SIZE - int(v.size), dtype=np.float32)
                v = np.concatenate([v, pad])
            elif v.size > VECTOR_SIZE:
                v = v[:VECTOR_SIZE]
            self.state_vector = v
        self.energy = float(energy)
        self.connections = list(connections) if connections is not None else []

    def propagate(self, damping: float = 0.9) -> Dict[str, float]:
        """
        Propagate energy to connected nodes. Returns dict of (node_id -> delta_energy).
        Lightweight: no full graph recompute.
        """
        if not self.connections or self.energy <= 0.0:
            return {}
        share = (self.energy * (1.0 - damping)) / max(len(self.connections), 1)
        return {nid: share for nid in self.connections}

    def update_state(self, delta: Optional[np.ndarray] = None, delta_energy: float = 0.0) -> None:
        """Update state vector and energy (in-place, minimal alloc)."""
        if delta is not None:
            d = np.asarray(delta, dtype=np.float32)
            if d.size != VECTOR_SIZE:
                if d.size < VECTOR_SIZE:
                    pad = np.zeros(VECTOR_SIZE - int(d.size), dtype=np.float32)
                    d = np.concatenate([d, pad])
                else:
                    d = d[:VECTOR_SIZE]
            self.state_vector = self.state_vector + d
        self.energy = max(0.0, self.energy + delta_energy)
        # Cheap normalization to keep values bounded.
        norm = float(np.linalg.norm(self.state_vector))
        if norm > 0.0 and norm > 10.0:
            self.state_vector = self.state_vector / norm

    def merge(self, other: WaveNode, weight: float = 0.5) -> None:
        """Merge another node's state into this one (in-place)."""
        if other.state_vector.shape == self.state_vector.shape:
            w = max(0.0, min(1.0, float(weight)))
            self.state_vector = (1.0 - w) * self.state_vector + w * other.state_vector
        self.energy = (1.0 - weight) * self.energy + weight * other.energy
        seen = {self.node_id}
        for c in other.connections:
            if c not in seen and c not in self.connections:
                self.connections.append(c)
                seen.add(c)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node state (for storage / graph backend)."""
        return {
            "node_id": self.node_id,
            "state_vector": self.state_vector.tolist(),
            "energy": self.energy,
            "connections": list(self.connections),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WaveNode:
        """Deserialize from dict."""
        return cls(
            node_id=data["node_id"],
            state_vector=np.array(data.get("state_vector", []), dtype=np.float32),
            energy=float(data.get("energy", 0.0)),
            connections=list(data.get("connections", [])),
        )
