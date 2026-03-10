"""
Temporal Attractor Memory for GOAT-TS Lite.

Stores reasoning trajectories (state₁ → state₂ → …) across ticks so future
cycles can match similar starting states and fast-forward along known paths.

Constraints:
- max 30 trajectories
- signature length <= 8 (VECTOR_SIZE)
- max 5 ticks stored per trajectory
- no tensors or large matrices
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import VECTOR_SIZE, MAX_TICKS
from .wave_nodes import WaveNode

# Max signatures stored per trajectory (performance rule).
MAX_TICKS_PER_TRAJECTORY = 5


def _normalize_signature(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    if v.size != VECTOR_SIZE:
        if v.size < VECTOR_SIZE:
            v = np.concatenate([v, np.zeros(VECTOR_SIZE - int(v.size), dtype=np.float32)])
        else:
            v = v[:VECTOR_SIZE]
    max_abs = float(np.max(np.abs(v))) if v.size else 1.0
    if max_abs > 0.0:
        v = v / max_abs
    v = np.clip(v, -1.0, 1.0)
    return v


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class TemporalTrajectory:
    trajectory_id: int
    state_sequence: List[np.ndarray]  # each shape (VECTOR_SIZE,), length <= MAX_TICKS_PER_TRAJECTORY
    length: int
    usage_count: int = 0


class TemporalAttractorMemory:
    def __init__(
        self,
        max_trajectories: int = 30,
        similarity_threshold: float = 0.75,
    ) -> None:
        self.max_trajectories = max_trajectories
        self.similarity_threshold = similarity_threshold
        self.trajectories: List[TemporalTrajectory] = []
        self._next_id = 1

    def compute_signature(self, nodes: Dict[str, WaveNode]) -> np.ndarray:
        if not nodes:
            return np.zeros(VECTOR_SIZE, dtype=np.float32)
        acc = np.zeros(VECTOR_SIZE, dtype=np.float32)
        count = 0
        for n in nodes.values():
            v = np.asarray(n.state_vector, dtype=np.float32)
            if v.size != VECTOR_SIZE:
                if v.size < VECTOR_SIZE:
                    v = np.concatenate([v, np.zeros(VECTOR_SIZE - int(v.size), dtype=np.float32)])
                else:
                    v = v[:VECTOR_SIZE]
            acc += v
            count += 1
        if count > 0:
            acc /= float(count)
        return _normalize_signature(acc)

    def find_best_match(self, signature: np.ndarray) -> Tuple[float, Optional[TemporalTrajectory]]:
        if not self.trajectories:
            return 0.0, None
        sig = _normalize_signature(signature)
        best_sim = 0.0
        best_traj: Optional[TemporalTrajectory] = None
        for traj in self.trajectories:
            if not traj.state_sequence:
                continue
            first_sig = traj.state_sequence[0]
            sim = _cosine_similarity(sig, first_sig)
            if sim > best_sim:
                best_sim = sim
                best_traj = traj
        return best_sim, best_traj

    def get_predicted_next_signature(self, traj: Optional[TemporalTrajectory]) -> np.ndarray:
        """Next expected state along the trajectory (for fast-forward nudge)."""
        if traj is None or not traj.state_sequence:
            return np.zeros(VECTOR_SIZE, dtype=np.float32)
        if len(traj.state_sequence) < 2:
            return traj.state_sequence[0].copy()
        return traj.state_sequence[1].copy()

    def apply_trajectory_influence(
        self,
        nodes: Dict[str, WaveNode],
        active_ids: Sequence[str],
        predicted_signature: np.ndarray,
        strength: float = 0.1,
    ) -> None:
        delta = _normalize_signature(predicted_signature) * float(strength)
        for nid in active_ids:
            node = nodes.get(nid)
            if node is None:
                continue
            node.update_state(delta=delta, delta_energy=0.0)

    def store_trajectory(
        self,
        state_sequence: List[np.ndarray],
        logger: Optional[Any] = None,
    ) -> Optional[TemporalTrajectory]:
        if not state_sequence:
            return None
        # Cap to max ticks per trajectory.
        seq = [_normalize_signature(s) for s in state_sequence[:MAX_TICKS_PER_TRAJECTORY]]
        if not seq:
            return None

        traj = TemporalTrajectory(
            trajectory_id=self._next_id,
            state_sequence=seq,
            length=len(seq),
            usage_count=1,
        )
        self._next_id += 1
        self.trajectories.append(traj)
        if logger:
            logger.info("Temporal trajectory stored (length %d).", traj.length)

        if len(self.trajectories) > self.max_trajectories:
            self._prune()
        return traj

    def _prune(self) -> None:
        if not self.trajectories:
            return
        self.trajectories.sort(key=lambda t: (t.usage_count, t.length))
        self.trajectories.pop(0)

    def summary(self) -> Dict[str, Any]:
        return {
            "count": len(self.trajectories),
            "max_length": max((t.length for t in self.trajectories), default=0),
        }


_GLOBAL_TEMPORAL: Optional[TemporalAttractorMemory] = None


def get_global_temporal_memory() -> TemporalAttractorMemory:
    global _GLOBAL_TEMPORAL
    if _GLOBAL_TEMPORAL is None:
        _GLOBAL_TEMPORAL = TemporalAttractorMemory()
    return _GLOBAL_TEMPORAL
