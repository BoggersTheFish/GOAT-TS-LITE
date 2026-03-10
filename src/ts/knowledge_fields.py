"""
TS Knowledge Attractor Fields for GOAT-TS Lite.

Stable regions of knowledge in the node graph; reasoning waves move toward
these fields. Lightweight, CPU-only: max 20 fields, signature length ≤ 8,
member lists ≤ 100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import VECTOR_SIZE
from .wave_nodes import WaveNode

MAX_FIELDS = 20
MAX_MEMBERS_PER_FIELD = 100
FIELD_SIMILARITY_THRESHOLD = 0.7
FIELD_GRAVITY_STRENGTH = 0.08
FIELD_ENERGY_DECAY = 0.95  # per cycle when not used
FIELD_ENERGY_BOOST = 1.0   # when field is activated


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
class KnowledgeField:
    field_id: int
    center_signature: np.ndarray  # (VECTOR_SIZE,), normalized -1..1
    member_nodes: List[str]  # node ids, max MAX_MEMBERS_PER_FIELD
    field_energy: float
    usage_count: int = 0


class KnowledgeFieldManager:
    def __init__(
        self,
        max_fields: int = MAX_FIELDS,
        field_creation_threshold: float = FIELD_SIMILARITY_THRESHOLD,
    ) -> None:
        self.max_fields = max_fields
        self.field_creation_threshold = field_creation_threshold
        self.fields: List[KnowledgeField] = []
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

    def find_best_match(self, signature: np.ndarray) -> Tuple[float, Optional[KnowledgeField]]:
        if not self.fields:
            return 0.0, None
        sig = _normalize_signature(signature)
        best_sim = 0.0
        best_field: Optional[KnowledgeField] = None
        for f in self.fields:
            sim = _cosine_similarity(sig, f.center_signature)
            if sim > best_sim:
                best_sim = sim
                best_field = f
        return best_sim, best_field

    def apply_field_gravity(
        self,
        nodes: Dict[str, WaveNode],
        target_node_ids: Sequence[str],
        center_signature: np.ndarray,
        strength: float = FIELD_GRAVITY_STRENGTH,
    ) -> int:
        delta = _normalize_signature(center_signature) * float(strength)
        count = 0
        for nid in target_node_ids:
            node = nodes.get(nid)
            if node is None:
                continue
            node.update_state(delta=delta, delta_energy=0.0)
            count += 1
        return count

    def decay_all(self) -> None:
        for f in self.fields:
            f.field_energy *= FIELD_ENERGY_DECAY

    def create_or_match_field(
        self,
        signature: np.ndarray,
        member_node_ids: Sequence[str],
        logger: Optional[Any] = None,
    ) -> Optional[KnowledgeField]:
        sig = _normalize_signature(signature)
        sim, existing = self.find_best_match(sig)
        if existing is not None and sim >= self.field_creation_threshold:
            existing.usage_count += 1
            existing.field_energy += FIELD_ENERGY_BOOST
            # Optionally add new members (cap at MAX_MEMBERS_PER_FIELD).
            for nid in member_node_ids:
                if nid not in existing.member_nodes and len(existing.member_nodes) < MAX_MEMBERS_PER_FIELD:
                    existing.member_nodes.append(nid)
            return existing

        # Create new field.
        members = list(member_node_ids)[:MAX_MEMBERS_PER_FIELD]
        field = KnowledgeField(
            field_id=self._next_id,
            center_signature=sig.copy(),
            member_nodes=members,
            field_energy=FIELD_ENERGY_BOOST,
            usage_count=1,
        )
        self._next_id += 1
        self.fields.append(field)
        if logger:
            logger.info("New knowledge field created (field_id=%s).", field.field_id)

        if len(self.fields) > self.max_fields:
            self._prune()
        return field

    def _prune(self) -> None:
        if not self.fields:
            return
        self.fields.sort(key=lambda f: f.field_energy)
        self.fields.pop(0)

    def summary(self) -> Dict[str, Any]:
        return {
            "count": len(self.fields),
            "max_energy": max((f.field_energy for f in self.fields), default=0.0),
        }


_GLOBAL_FIELD_MANAGER: Optional[KnowledgeFieldManager] = None


def get_global_field_manager() -> KnowledgeFieldManager:
    global _GLOBAL_FIELD_MANAGER
    if _GLOBAL_FIELD_MANAGER is None:
        _GLOBAL_FIELD_MANAGER = KnowledgeFieldManager()
    return _GLOBAL_FIELD_MANAGER
