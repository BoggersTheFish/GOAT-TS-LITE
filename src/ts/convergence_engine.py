"""
TSConvergenceEngine: Thinking Wave convergence dynamics.

Operates over WaveNode objects with an active wavefront:
- only nodes in the active wavefront are processed each tick
- incremental propagation (no full graph recomputes)
- multi-agent influences
- convergence scoring and stability detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .wave_nodes import WaveNode
from .agents import TSAgent
from .decision_nodes import DecisionNode
from .convergence_metrics import compute_convergence
from .stability import detect_stability
from .config import MAX_NODES, MAX_WAVEFRONT, MAX_TICKS, MAX_AGENTS
from .attractor_memory import TSAttractorMemory
from .temporal_attractors import TemporalAttractorMemory
from .knowledge_fields import KnowledgeFieldManager, MAX_MEMBERS_PER_FIELD
from .adaptive_config import AdaptiveConfig


@dataclass
class TSConvergenceEngine:
    convergence_threshold: float = 0.85
    max_ticks: int = MAX_TICKS
    nodes: Dict[str, WaveNode] = field(default_factory=dict)
    active_wavefront: Set[str] = field(default_factory=set)

    _last_deltas: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_convergence: float = field(default=0.0, init=False, repr=False)

    def initialize(
        self,
        nodes: Dict[str, WaveNode],
        active_wavefront: Optional[Set[str]] = None,
    ) -> None:
        self.nodes = nodes
        if len(self.nodes) > MAX_NODES:
            self._prune_nodes()
        self.active_wavefront = set(active_wavefront) if active_wavefront else set(nodes.keys())
        self._cap_wavefront()

    def _cap_wavefront(self) -> None:
        if len(self.active_wavefront) <= MAX_WAVEFRONT:
            return
        # Keep the highest-energy nodes in the wavefront.
        ranked = sorted(
            (nid for nid in self.active_wavefront if nid in self.nodes),
            key=lambda nid: self.nodes[nid].energy,
            reverse=True,
        )
        self.active_wavefront = set(ranked[:MAX_WAVEFRONT])

    def _prune_nodes(self, logger: Optional[Any] = None) -> None:
        if len(self.nodes) <= MAX_NODES:
            return
        if logger:
            logger.warning("Node limit reached — pruning oldest nodes.")
        # Prune least active (lowest energy) nodes first, prefer pruning nodes not in wavefront.
        protected = set(self.active_wavefront)
        candidates = [nid for nid in self.nodes.keys() if nid not in protected]
        if not candidates:
            candidates = list(self.nodes.keys())
        candidates.sort(key=lambda nid: self.nodes[nid].energy)
        while len(self.nodes) > MAX_NODES and candidates:
            nid = candidates.pop(0)
            self.nodes.pop(nid, None)
            self.active_wavefront.discard(nid)

    def propagate_wave(self, damping: float = 0.9) -> None:
        """
        Step 1: propagate energy from active nodes to their neighbors.
        Only nodes in active_wavefront are processed.
        damping: 1 - propagation_strength from AdaptiveConfig (default 0.9).
        """
        next_wavefront: Set[str] = set()
        deltas: Dict[str, float] = {}

        self._cap_wavefront()
        for node_id in list(self.active_wavefront):
            node = self.nodes.get(node_id)
            if node is None:
                continue
            local_deltas = node.propagate(damping=damping)
            for target_id, delta_energy in local_deltas.items():
                if delta_energy == 0.0:
                    continue
                deltas[target_id] = deltas.get(target_id, 0.0) + delta_energy
                next_wavefront.add(target_id)

        self._last_deltas = deltas
        self.active_wavefront = next_wavefront
        self._cap_wavefront()

    def update_node_states(self) -> None:
        """
        Step 2: apply energy deltas to target nodes (incremental update).
        """
        for node_id, delta_energy in self._last_deltas.items():
            node = self.nodes.get(node_id)
            if node is None:
                # Safety: create only if within node limit, otherwise prune.
                if len(self.nodes) >= MAX_NODES:
                    self._prune_nodes()
                    if len(self.nodes) >= MAX_NODES:
                        continue
                self.nodes[node_id] = WaveNode(node_id=node_id, energy=0.0)
                node = self.nodes[node_id]
            node.update_state(delta_energy=delta_energy)

    def calculate_convergence(
        self,
        decision_nodes: Optional[Dict[str, DecisionNode]] = None,
    ) -> float:
        """
        Step 3: compute global convergence score (0–1).
        """
        self._last_convergence = compute_convergence(self.nodes)
        return self._last_convergence

    def collapse_decisions(
        self,
        decision_nodes: Optional[Dict[str, DecisionNode]] = None,
    ) -> List[DecisionNode]:
        """
        Step 4: collapse any decision nodes whose probabilities exceed
        their collapse thresholds.
        """
        if not decision_nodes:
            return []
        collapsed: List[DecisionNode] = []
        for dn in decision_nodes.values():
            if dn.collapse():
                collapsed.append(dn)
        return collapsed

    def run_cycle(
        self,
        ticks: int,
        agents: Optional[List[TSAgent]] = None,
        decision_nodes: Optional[Dict[str, DecisionNode]] = None,
        dry_run: bool = False,
        logger: Optional[Any] = None,
        attractor_memory: Optional[TSAttractorMemory] = None,
        temporal_memory: Optional[TemporalAttractorMemory] = None,
        knowledge_field_manager: Optional[KnowledgeFieldManager] = None,
        adaptive_config: Optional[AdaptiveConfig] = None,
    ) -> Dict[str, Any]:
        """
        Run up to `ticks` reasoning steps (bounded by max_ticks).

        Each tick:
        - propagates wave from active wavefront
        - updates node states incrementally
        - lets agents influence nodes
        - computes convergence score
        - optionally collapses decision nodes
        - checks stability
        """
        ticks = max(1, min(MAX_TICKS, min(self.max_ticks, ticks)))
        agents = (agents or [])[:MAX_AGENTS]
        decision_nodes = decision_nodes or {}

        # Load parameters from adaptive config if provided.
        convergence_threshold = self.convergence_threshold
        propagation_damping = 0.9
        field_strength = 0.08
        attractor_strength = 0.10
        if adaptive_config is not None:
            convergence_threshold = adaptive_config.convergence_threshold
            propagation_damping = 1.0 - adaptive_config.propagation_strength
            field_strength = adaptive_config.field_influence_strength
            attractor_strength = adaptive_config.attractor_influence_strength

        stopped_by: str = "ticks"
        collapsed_total: List[DecisionNode] = []
        attractor_applied = False
        attractor_sim = 0.0
        temporal_applied = False
        temporal_sim = 0.0
        knowledge_field_applied = False
        knowledge_field_sim = 0.0
        active_field_id: Optional[int] = None
        nodes_influenced_by_field = 0
        tick_signatures: List[Any] = []

        # Shared signature: compute once when any acceleration system is present.
        current_sig = None
        if self.nodes and (
            attractor_memory is not None
            or temporal_memory is not None
            or knowledge_field_manager is not None
        ):
            src = attractor_memory or temporal_memory or knowledge_field_manager
            current_sig = src.compute_signature(self.nodes)

        # Order of reasoning acceleration: 1) temporal, 2) attractor, 3) knowledge field.
        # 1) Temporal trajectory matching.
        if temporal_memory is not None and current_sig is not None:
            temporal_sim, best_traj = temporal_memory.find_best_match(current_sig)
            if best_traj is not None and temporal_sim >= temporal_memory.similarity_threshold:
                if logger:
                    logger.info(
                        "Temporal attractor matched (similarity %.2f) — accelerating reasoning.",
                        temporal_sim,
                    )
                pred = temporal_memory.get_predicted_next_signature(best_traj)
                temporal_memory.apply_trajectory_influence(
                    self.nodes,
                    self.active_wavefront or list(self.nodes.keys()),
                    pred,
                    strength=attractor_strength,
                )
                temporal_applied = True

        # 2) Attractor memory matching.
        if attractor_memory is not None and current_sig is not None:
            attractor_sim, best_attr = attractor_memory.find_best_match(current_sig)
            if best_attr is not None and attractor_sim >= attractor_memory.similarity_threshold:
                if logger:
                    logger.info(
                        "Attractor match detected (similarity %.2f) — applying attractor gravity.",
                        attractor_sim,
                    )
                attractor_memory.apply_gravity(
                    self.nodes,
                    self.active_wavefront or self.nodes.keys(),
                    best_attr.state_signature,
                    strength=attractor_strength,
                )
                attractor_applied = True

        # 3) Knowledge field activation.
        if knowledge_field_manager is not None and current_sig is not None:
            knowledge_field_manager.decay_all()
            knowledge_field_sim, best_field = knowledge_field_manager.find_best_match(current_sig)
            if best_field is not None and knowledge_field_sim >= knowledge_field_manager.field_creation_threshold:
                if logger:
                    logger.info(
                        "Knowledge field activated (field_id=%s, similarity=%.2f).",
                        best_field.field_id,
                        knowledge_field_sim,
                    )
                target_ids = list(
                    self.active_wavefront or self.nodes.keys()
                )
                nodes_influenced_by_field = knowledge_field_manager.apply_field_gravity(
                    self.nodes,
                    target_ids,
                    best_field.center_signature,
                    strength=field_strength,
                )
                best_field.usage_count += 1
                best_field.field_energy += 1.0
                knowledge_field_applied = True
                active_field_id = best_field.field_id

        # Start trajectory buffer with initial state (for temporal storage).
        if temporal_memory is not None and current_sig is not None:
            tick_signatures.append(current_sig.copy())

        for t in range(ticks):
            self._prune_nodes(logger=logger)

            # Incremental wave propagation (damping from adaptive_config or default).
            self.propagate_wave(damping=propagation_damping)
            self.update_node_states()

            # Multi-agent influences (very lightweight).
            for agent in agents:
                agent.propagate()
                agent.influence_nodes(self.nodes)

            # Record state signature this tick for temporal trajectory (before convergence check).
            if temporal_memory is not None and self.nodes:
                tick_signatures.append(temporal_memory.compute_signature(self.nodes).copy())

            # Convergence and stability checks.
            score = self.calculate_convergence(decision_nodes)
            if logger:
                logger.info(
                    "Tick %d | Active Nodes: %d | Convergence: %.2f",
                    t + 1,
                    len(self.active_wavefront),
                    score,
                )

            collapsed_now = self.collapse_decisions(decision_nodes)
            collapsed_total.extend(collapsed_now)
            if collapsed_now and logger:
                logger.info("Collapsed decisions: %s", [d.node_id for d in collapsed_now])

            stable = detect_stability(self.nodes)
            if logger and stable:
                logger.info("Stability detected (avg state change < 0.01).")

            if score >= convergence_threshold:
                stopped_by = "convergence"
                # Store attractor on convergence.
                if attractor_memory is not None:
                    sig = attractor_memory.compute_signature(self.nodes)
                    avg_energy = sum(n.energy for n in self.nodes.values()) / float(len(self.nodes) or 1)
                    attractor_memory.store_attractor(sig, avg_energy, logger=logger)
                # Store temporal trajectory on convergence.
                if temporal_memory is not None and tick_signatures:
                    temporal_memory.store_trajectory(tick_signatures, logger=logger)
                # Create or update knowledge field on convergence.
                if knowledge_field_manager is not None:
                    sig = knowledge_field_manager.compute_signature(self.nodes)
                    member_ids = list(self.nodes.keys())[:MAX_MEMBERS_PER_FIELD]
                    knowledge_field_manager.create_or_match_field(sig, member_ids, logger=logger)
                break
            if stable:
                stopped_by = "stability"
                break

        return {
            "ticks_run": t + 1 if "t" in locals() else 0,
            "convergence": self._last_convergence,
            "stopped_by": stopped_by,
            "active_nodes": list(self.active_wavefront),
            "collapsed_nodes": [
                (d.node_id, d.get_best_state()) for d in decision_nodes.values() if d.is_collapsed
            ],
            "attractor_applied": attractor_applied,
            "attractor_similarity": attractor_sim,
            "attractor_count": len(attractor_memory.attractors) if attractor_memory is not None else 0,
            "temporal_applied": temporal_applied,
            "temporal_similarity": temporal_sim,
            "temporal_trajectory_count": len(temporal_memory.trajectories) if temporal_memory is not None else 0,
            "knowledge_field_applied": knowledge_field_applied,
            "knowledge_field_similarity": knowledge_field_sim,
            "active_field_id": active_field_id,
            "knowledge_field_energy": (
                next((f.field_energy for f in knowledge_field_manager.fields if f.field_id == active_field_id), None)
                if knowledge_field_manager and active_field_id is not None
                else None
            ),
            "nodes_influenced_by_field": nodes_influenced_by_field,
            "knowledge_field_count": len(knowledge_field_manager.fields) if knowledge_field_manager is not None else 0,
        }

