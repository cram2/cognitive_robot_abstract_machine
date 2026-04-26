"""
Effect types for the liquid pouring domain.

Defines the goal-state representations used when pouring liquids, tracking
fill levels of source and receiver containers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pycram.body_motion_problem.types import (
    MonotoneDecreasingEffect,
    MonotoneIncreasingEffect,
)

if TYPE_CHECKING:
    from semantic_digital_twin.physics.pouring_equations import HasFillLevel


@dataclass(eq=False, kw_only=True)
class PouringEffect(MonotoneDecreasingEffect):
    """
    Effect achieved when the fill level of a container drops to or below ``goal_value``.

    The target object must be a :class:`~semantic_digital_twin.physics.pouring_equations.HasFillLevel`.
    """

    target_object: HasFillLevel


@dataclass(eq=False, kw_only=True)
class ReceiverFillEffect(MonotoneIncreasingEffect):
    """
    Effect achieved when the fill level of a receiver container reaches ``goal_value``.

    The target object must be a :class:`~semantic_digital_twin.physics.pouring_equations.HasFillLevel`
    representing the receiving container.
    """

    target_object: HasFillLevel
