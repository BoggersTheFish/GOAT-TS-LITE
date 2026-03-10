"""
TS Attractor Memory for low-resource GOAT-TS Lite.

Stores a small number of converged reasoning signatures (attractors) and
lets future reasoning cycles gravitate toward similar signatures.

Constraints:
- max 50 attractors
- signature length <= VECTOR_SIZE (8)
- simple float operations only (no large tensors / embeddings)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import VECTOR_SIZE
from .wave_nodes import WaveNode


@dataclass
class TSAttractor:
    attractor_id: int
    state_signature: np.ndarray  # shape: (VECTOR_SIZE,)
    energy_level: float
    usage_count: int = 0


class TSAttractorMemory:
    def __init__(
        self,
        max_attractors: int = 50,
        similarity_threshold: float = 0.75,
    ) -> None:
        self.max_attractors = max_attractors
        self.similarity_threshold = similarity_threshold
        self.attractors: List[TSAttractor] = []
        self._next_id = 1

    def _normalize_signature(self, vec: np.ndarray) -> np.ndarray:
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
        return self._normalize_signature(acc)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        if denom <= 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def find_best_match(self, signature: np.ndarray) -> Tuple[float, Optional[TSAttractor]]:
        if not self.attractors:
            return 0.0, None
        sig = self._normalize_signature(signature)
        best_sim = 0.0
        best_attr: Optional[TSAttractor] = None
        for attr in self.attractors:
            sim = self._cosine_similarity(sig, attr.state_signature)
            if sim > best_sim:
                best_sim = sim
                best_attr = attr
        return best_sim, best_attr

    def apply_gravity(
        self,
        nodes: Dict[str, WaveNode],
        active_ids: Sequence[str],
        signature: np.ndarray,
        strength: float = 0.1,
    ) -> None:
        sig = self._normalize_signature(signature) * float(strength)
        for nid in active_ids:
            node = nodes.get(nid)
            if node is None:
                continue
            node.update_state(delta=sig, delta_energy=0.0)

    def store_attractor(
        self,
        signature: np.ndarray,
        energy_level: float,
        logger: Optional[Any] = None,
    ) -> TSAttractor:
        sig = self._normalize_signature(signature)
        # If similar to an existing attractor, just bump usage_count.
        best_sim, best_attr = self.find_best_match(sig)
        if best_attr is not None and best_sim >= self.similarity_threshold:
            best_attr.usage_count += 1
            if logger:
                logger.info("Existing attractor reused (id=%s, similarity=%.2f).", best_attr.attractor_id, best_sim)
            return best_attr

        # Create new attractor.
        attr = TSAttractor(
            attractor_id=self._next_id,
            state_signature=sig,
            energy_level=float(energy_level),
            usage_count=1,
        )
        self._next_id += 1
        self.attractors.append(attr)
        if logger:
            logger.info("New attractor stored (id=%s).", attr.attractor_id)

        # Prune if over limit: drop lowest usage_count.
        if len(self.attractors) > self.max_attractors:
            self._prune()
        return attr

    def _prune(self) -> None:
        if not self.attractors:
            return
        self.attractors.sort(key=lambda a: (a.usage_count, a.energy_level))
        # Drop one least-used attractor.
        self.attractors.pop(0)

    def summary(self) -> Dict[str, Any]:
        count = len(self.attractors)
        best_usage = max((a.usage_count for a in self.attractors), default=0)
        return {
            "count": count,
            "max_usage": best_usage,
        }


_GLOBAL_MEMORY: Optional[TSAttractorMemory] = None


def get_global_memory() -> TSAttractorMemory:
    global _GLOBAL_MEMORY
    if _GLOBAL_MEMORY is None:
        _GLOBAL_MEMORY = TSAttractorMemory()
    return _GLOBAL_MEMORY

