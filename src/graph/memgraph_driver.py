"""
GOAT-TS Lite — Memgraph graph backend driver.
Optional; used when config/graph.yaml sets graph_backend: memgraph.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("goatts.memgraph")

try:
    import pymgclient
    PYMGC_AVAILABLE = True
except ImportError:
    PYMGC_AVAILABLE = False
    pymgclient = None


def _load_config() -> Dict[str, Any]:
    import yaml
    config_path = Path(__file__).resolve().parents[2] / "config" / "graph.yaml"
    if not config_path.exists():
        return {"graph_backend": "memory", "host": "localhost", "port": 7687}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


class MemgraphDriver:
    """Thin driver for Memgraph: store/load wave node states (no full graph objects)."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        cfg = _load_config()
        self.host = host or cfg.get("host", "localhost")
        self.port = port or cfg.get("port", 7687)
        self._conn = None

    def connect(self) -> bool:
        if not PYMGC_AVAILABLE:
            logger.warning("pymgclient not installed; Memgraph disabled.")
            return False
        try:
            self._conn = pymgclient.connect(host=self.host, port=self.port)
            return True
        except Exception as e:
            logger.warning("Memgraph connection failed: %s", e)
            return False

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def save_node_state(self, node_id: str, state_vector: List[float], energy: float, connections: List[str]) -> bool:
        """Upsert a single wave node state (incremental update)."""
        if not self._conn:
            if not self.connect():
                return False
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "MERGE (n:WaveNode {id: $id}) SET n.state = $state, n.energy = $energy, n.connections = $conn",
                    {"id": node_id, "state": state_vector, "energy": energy, "conn": connections},
                )
            return True
        except Exception as e:
            logger.debug("save_node_state failed: %s", e)
            return False

    def load_node_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Load one node state by id."""
        if not self._conn and not self.connect():
            return None
        try:
            with self._conn.cursor() as cur:
                cur.execute("MATCH (n:WaveNode {id: $id}) RETURN n.state AS state, n.energy AS energy, n.connections AS conn", {"id": node_id})
                row = cur.fetchone()
            if not row:
                return None
            return {"state_vector": row[0] or [], "energy": row[1] or 0.0, "connections": row[2] or []}
        except Exception as e:
            logger.debug("load_node_state failed: %s", e)
            return None
