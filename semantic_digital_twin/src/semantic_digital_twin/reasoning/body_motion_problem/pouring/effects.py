"""
Effect types for the pouring domain (D_pour).
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.reasoning.body_motion_problem.types import (
    MonotoneDecreasingEffect,
)


@dataclass(eq=False, kw_only=True)
class PouringEffect(MonotoneDecreasingEffect):
    """
    Effect achieved when the fill level of a container drops to or below ``goal_value``.

    Inherits :meth:`is_achieved` from :class:`MonotoneDecreasingEffect`:
    ``current_value <= goal_value + tolerance``.
    """
