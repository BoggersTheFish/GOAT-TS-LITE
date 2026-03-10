"""
GOAT-TS Lite — low-spec CPU optimizations.
Reduces compute and memory for Pentium Silver / 4–8GB RAM.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("goatts.low_spec")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def apply_low_spec(model: Any) -> Any:
    """
    Apply CPU-friendly optimizations to a PyTorch model:
    - channels-last memory layout
    - torch.compile when supported (with safe fallback)
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available; skipping low_spec optimizations.")
        return model

    if not hasattr(model, "to"):
        logger.warning("Object has no .to(); skipping low_spec.")
        return model

    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception as e:
        logger.debug("channels_last not applied: %s", e)

    try:
        model = torch.compile(model)
    except Exception as e:
        logger.debug("torch.compile skipped (expected on some setups): %s", e)

    return model
