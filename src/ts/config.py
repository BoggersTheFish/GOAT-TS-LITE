"""
Hard limits for GOAT-TS Lite low-resource runtime.

These limits are mandatory for Pentium-class CPU-only laptops:
- Node graph <= 300 nodes
- Vector sizes <= 8 floats
- Tick cycles <= 5
- Agents <= 3
"""

MAX_NODES = 300
VECTOR_SIZE = 8
MAX_TICKS = 5
DEFAULT_TICKS = 3
MAX_AGENTS = 3

# Active wavefront is limited to avoid full-graph work.
MAX_WAVEFRONT = 50

