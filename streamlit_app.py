"""
GOAT-TS Lite — Streamlit UI.
Lightweight dashboard for reasoning state and run controls.
"""

import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="GOAT-TS Lite", page_icon="🐐", layout="centered")

st.title("GOAT-TS Lite — TS Convergence")
st.caption("Thinking Wave convergence engine on low-end CPU-only hardware")

with st.sidebar:
    st.header("Config")
    ticks = st.slider("Ticks", min_value=1, max_value=5, value=3)
    dry_run = st.checkbox("Dry run (no model inference)", value=True)


def run_ts_preview(ticks: int) -> dict:
    """
    Lightweight preview of TS convergence for the UI.
    Uses a tiny in-memory graph and always runs in dry-run mode.
    """
    from src.ts.wave_nodes import WaveNode
    from src.ts.convergence_engine import TSConvergenceEngine
    from src.ts.decision_nodes import DecisionNode
    from src.ts.agents import TSAgent
    from src.ts.attractor_memory import get_global_memory
    from src.ts.temporal_attractors import get_global_temporal_memory
    from src.ts.knowledge_fields import get_global_field_manager
    from src.ts.adaptive_config import load_adaptive_config
    from src.ts.performance_monitor import get_global_performance_monitor
    from src.ts.config import VECTOR_SIZE, MAX_WAVEFRONT
    import numpy as np

    adaptive_config = load_adaptive_config()
    nodes: dict[str, WaveNode] = {}

    def ensure_node(nid: str) -> WaveNode:
        if nid not in nodes:
            nodes[nid] = WaveNode(node_id=nid, energy=1.0)
        return nodes[nid]

    ensure_node("root").connections.extend(["a", "b"])
    ensure_node("a").connections.append("root")
    ensure_node("b").connections.append("root")

    dim = VECTOR_SIZE
    perspective = np.random.randn(dim).astype(np.float32)

    agent = TSAgent(agent_id="ui_agent", local_wave_nodes=nodes, perspective_vector=perspective)
    agents = [agent]

    decision_nodes = {
        "root": DecisionNode(
            node_id="root",
            possible_states=["low", "medium", "high"],
            state_probabilities=[0.33, 0.34, 0.33],
            collapse_threshold=adaptive_config.decision_collapse_threshold,
        )
    }

    engine = TSConvergenceEngine(convergence_threshold=adaptive_config.convergence_threshold, max_ticks=5)
    engine.initialize(nodes, active_wavefront={"root"})
    attractor_memory = get_global_memory()
    temporal_memory = get_global_temporal_memory()
    knowledge_field_manager = get_global_field_manager()
    performance_monitor = get_global_performance_monitor()
    result = engine.run_cycle(
        ticks=ticks,
        agents=agents,
        decision_nodes=decision_nodes,
        dry_run=True,
        logger=None,
        attractor_memory=attractor_memory,
        temporal_memory=temporal_memory,
        knowledge_field_manager=knowledge_field_manager,
        adaptive_config=adaptive_config,
    )
    performance_monitor.record_cycle(result["ticks_run"], result["convergence"], result["stopped_by"])
    result["node_count"] = len(nodes)
    result["active_nodes"] = result.get("active_nodes", [])[:MAX_WAVEFRONT]
    result["attractor_count"] = result.get("attractor_count", len(attractor_memory.attractors))
    result["temporal_trajectory_count"] = result.get("temporal_trajectory_count", len(temporal_memory.trajectories))
    result["knowledge_field_count"] = result.get("knowledge_field_count", len(knowledge_field_manager.fields))
    return result


if st.button("Run TS convergence preview"):
    try:
        from reasoning.demo_loop import get_ram_gb

        ram = get_ram_gb()
        st.info(f"RAM: {ram:.1f} GB detected on this machine.")
        res = run_ts_preview(ticks)
        st.subheader("Wave convergence status")
        st.write(f"Ticks run: {res['ticks_run']} (stopped by: {res['stopped_by']})")
        st.write(f"Convergence score: {res['convergence']:.3f}")
        st.write(f"Active wavefront nodes (<=50): {', '.join(res['active_nodes']) or 'none'}")
        st.write(f"Collapsed decision nodes: {res['collapsed_nodes'] or 'none'}")
        st.write(f"Total nodes in graph: {res['node_count']}")
        st.subheader("TS Attractor Memory")
        st.write(f"Stored attractors: {res.get('attractor_count', 0)}")
        st.write(
            f"Last attractor similarity: {res.get('attractor_similarity', 0.0):.2f} "
            f"({'activated' if res.get('attractor_applied') else 'not activated'})"
        )
        st.subheader("Temporal Attractor Memory")
        st.write(f"Stored trajectories: {res.get('temporal_trajectory_count', 0)}")
        st.write(
            f"Matched trajectory similarity: {res.get('temporal_similarity', 0.0):.2f} "
            f"({'accelerated' if res.get('temporal_applied') else 'not accelerated'})"
        )
        st.subheader("Knowledge Attractor Fields")
        st.write(f"Number of knowledge fields: {res.get('knowledge_field_count', 0)}")
        st.write(f"Active field id: {res.get('active_field_id', 'none')}")
        st.write(
            f"Field energy level: {res.get('knowledge_field_energy') if res.get('knowledge_field_energy') is not None else 'n/a'}"
        )
        st.write(f"Nodes influenced by field: {res.get('nodes_influenced_by_field', 0)}")
        st.subheader("Adaptive convergence (self-modifying)")
        from src.ts.adaptive_config import load_adaptive_config
        from src.ts.performance_monitor import get_global_performance_monitor
        cfg = load_adaptive_config()
        mon = get_global_performance_monitor()
        st.write("**Current parameters:**")
        st.write(f"propagation_strength=%.2f, convergence_threshold=%.2f, decision_collapse_threshold=%.2f" % (cfg.propagation_strength, cfg.convergence_threshold, cfg.decision_collapse_threshold))
        st.write(f"field_influence_strength=%.2f, attractor_influence_strength=%.2f" % (cfg.field_influence_strength, cfg.attractor_influence_strength))
        if mon.recent_convergence_scores:
            recent = list(mon.recent_convergence_scores)[-10:]
            st.write("**Recent convergence scores (last 10):** %s" % [round(x, 2) for x in recent])
        if mon.recent_tick_counts:
            recent_ticks = list(mon.recent_tick_counts)[-10:]
            st.write("**Recent tick counts (last 10):** %s" % recent_ticks)
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.markdown("**CLI usage**")
st.code(
    "python reasoning/demo_loop.py --ticks 3 --dry-run\n"
    "python reasoning/demo_loop.py --ticks 3\n"
    "streamlit run streamlit_app.py",
    language="text",
)
