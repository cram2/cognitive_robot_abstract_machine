"""Internal shared utilities for the llmr workflows package."""

from __future__ import annotations

from typing_extensions import Optional, Tuple


def _pose_to_xyz(pose: object) -> Optional[Tuple[float, float, float]]:
    """Return ``(x, y, z)`` floats from a ``HomogeneousTransformationMatrix``, or ``None``."""
    try:
        pt = pose.to_position()
        return float(pt.x), float(pt.y), float(pt.z)
    except Exception:
        return None
