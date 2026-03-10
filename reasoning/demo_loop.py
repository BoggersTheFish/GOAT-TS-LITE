#!/usr/bin/env python3
"""
GOAT-TS Lite — minimal-cost reasoning loop.
Default 3 ticks, max 5; incremental wave propagation only on touched nodes.
Supports --ticks and --dry-run. RAM safety: under 4GB enforces ticks=3 and reduced batches.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger("demo_loop")

# --- RAM detection (cross-platform, best-effort) ---
def get_ram_gb() -> float:
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        elif sys.platform == "darwin":
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / (1024 ** 3)
        elif sys.platform == "win32":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                ]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    return 8.0  # assume safe if unknown


def _ram_effective_gb(ram_gb: float) -> float:
    """Treat 0 or negative as unknown (assume safe)."""
    return ram_gb if ram_gb > 0 else 8.0


def main() -> None:
    parser = argparse.ArgumentParser(description="GOAT-TS Lite TS convergence reasoning loop")
    parser.add_argument("--ticks", type=int, default=None, help="Reasoning ticks (default 3, max 5)")
    parser.add_argument("--dry-run", action="store_true", help="Skip model inference; simulate TS waves only")
    args = parser.parse_args()

    from src.ts.wave_nodes import WaveNode
    from src.ts.convergence_engine import TSConvergenceEngine
    from src.ts.agents import TSAgent
    from src.ts.decision_nodes import DecisionNode
    from src.ts.attractor_memory import get_global_memory
    from src.ts.temporal_attractors import get_global_temporal_memory
    from src.ts.knowledge_fields import get_global_field_manager
    from src.ts.adaptive_config import load_adaptive_config, save_adaptive_config
    from src.ts.performance_monitor import (
        get_global_performance_monitor,
        apply_adjustments,
    )
    from src.ts.config import DEFAULT_TICKS, MAX_TICKS, VECTOR_SIZE, MAX_AGENTS
    import numpy as np

    ram_gb = _ram_effective_gb(get_ram_gb())
    low_ram = ram_gb < 4.0
    if low_ram:
        log.warning("Low RAM detected (%.1f GB) — GOAT-TS Lite mode enabled.", ram_gb)
        ticks = DEFAULT_TICKS  # enforced
        batch_size = 4
    else:
        ticks = args.ticks if args.ticks is not None else DEFAULT_TICKS
        ticks = max(1, min(MAX_TICKS, int(ticks)))
        batch_size = 8

    dry_run = args.dry_run
    log.info("Running TS convergence loop: ticks=%s, dry_run=%s, low_ram=%s", ticks, dry_run, low_ram)

    # Minimal in-memory graph over WaveNodes.
    nodes: dict[str, WaveNode] = {}

    def ensure_node(nid: str) -> WaveNode:
        if nid not in nodes:
            nodes[nid] = WaveNode(node_id=nid, energy=1.0)
        return nodes[nid]

    # Simple star-shaped seed graph.
    ensure_node("root").connections.extend(["a", "b"])
    ensure_node("a").connections.append("root")
    ensure_node("b").connections.append("root")

    # Multi-agent TS reasoning: two lightweight perspectives.
    dim = VECTOR_SIZE
    perspective1 = np.random.randn(dim).astype(np.float32)
    perspective2 = -perspective1

    agent1 = TSAgent(agent_id="agent_1", local_wave_nodes={"root": nodes["root"]}, perspective_vector=perspective1)
    agent2 = TSAgent(agent_id="agent_2", local_wave_nodes={"a": nodes["a"], "b": nodes["b"]}, perspective_vector=perspective2)
    agents = [agent1, agent2][:MAX_AGENTS]

    adaptive_config = load_adaptive_config()
    engine = TSConvergenceEngine(convergence_threshold=adaptive_config.convergence_threshold, max_ticks=MAX_TICKS)
    engine.initialize(nodes, active_wavefront={"root"})

    # Decision node on root; use adaptive decision_collapse_threshold.
    decision_nodes = {
        "root": DecisionNode(
            node_id="root",
            possible_states=["low", "medium", "high"],
            state_probabilities=[0.33, 0.34, 0.33],
            collapse_threshold=adaptive_config.decision_collapse_threshold,
        )
    }

    attractor_memory = get_global_memory()
    temporal_memory = get_global_temporal_memory()
    knowledge_field_manager = get_global_field_manager()
    performance_monitor = get_global_performance_monitor()

    result = engine.run_cycle(
        ticks=ticks,
        agents=agents,
        decision_nodes=decision_nodes,
        dry_run=dry_run,
        logger=log,
        attractor_memory=attractor_memory,
        temporal_memory=temporal_memory,
        knowledge_field_manager=knowledge_field_manager,
        adaptive_config=adaptive_config,
    )

    # Record performance and optionally update adaptive parameters.
    performance_monitor.record_cycle(
        result["ticks_run"],
        result["convergence"],
        result["stopped_by"],
    )
    suggestions = performance_monitor.analyze_performance()
    if suggestions:
        n_changed = apply_adjustments(adaptive_config, suggestions, logger=log)
        if n_changed > 0:
            save_adaptive_config(adaptive_config)
    performance_monitor.save_history()

    # Optional minimal model step (kept extremely small and skipped in dry-run).
    if not dry_run:
        try:
            import torch
            from src.utils.low_spec import apply_low_spec

            d = VECTOR_SIZE
            model = torch.nn.Linear(d, d)
            model = apply_low_spec(model)
            with torch.no_grad():
                buf = torch.randn(min(batch_size, 4), d)
                _ = model(buf)
            log.info("Minimal model step OK (CPU).")
        except Exception as e:
            log.debug("Model step skipped or failed: %s", e)
    else:
        log.info("Dry-run: TS waves only (no model inference).")

    log.info(
        "Reasoning finished after %d ticks (stopped_by=%s, convergence=%.3f, nodes=%d)",
        result["ticks_run"],
        result["stopped_by"],
        result["convergence"],
        len(nodes),
    )
    if result.get("attractor_applied"):
        log.info(
            "Attractor gravity applied (similarity=%.2f, total_attractors=%d).",
            result.get("attractor_similarity", 0.0),
            result.get("attractor_count", 0),
        )
    if result.get("temporal_applied"):
        log.info(
            "Temporal trajectory applied (similarity=%.2f, trajectories=%d).",
            result.get("temporal_similarity", 0.0),
            result.get("temporal_trajectory_count", 0),
        )
    if result.get("knowledge_field_applied"):
        log.info(
            "Knowledge field activated (field_id=%s, similarity=%.2f, nodes_influenced=%d).",
            result.get("active_field_id"),
            result.get("knowledge_field_similarity", 0.0),
            result.get("nodes_influenced_by_field", 0),
        )
    if result["collapsed_nodes"]:
        log.info("Collapsed decision nodes: %s", result["collapsed_nodes"])


if __name__ == "__main__":
    main()
