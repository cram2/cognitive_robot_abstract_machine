"""
Effect types for the pouring domain (D_pour).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from semantic_digital_twin.reasoning.body_motion_problem.types import (
    MonotoneDecreasingEffect,
    MonotoneIncreasingEffect,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel


@dataclass(eq=False, kw_only=True)
class PouringEffect(MonotoneDecreasingEffect):
    """
    Effect achieved when the fill level of a container drops to or below ``goal_value``.

    Inherits :meth:`is_achieved` from :class:`MonotoneDecreasingEffect`:
    ``current_value <= goal_value + tolerance``.

    The target object must be a :class:`~semantic_digital_twin.semantic_annotations.mixins.HasFillLevel`
    so that :class:`~semantic_digital_twin.reasoning.body_motion_problem.pouring.predicates.PouringCauses`
    can access the fill connection and its governing equation.
    """

    target_object: HasFillLevel


@dataclass(eq=False, kw_only=True)
class ReceiverFillEffect(MonotoneIncreasingEffect):
    """
    Effect achieved when the fill level of a receiver container reaches ``goal_value``.

    Inherits :meth:`is_achieved` from :class:`MonotoneIncreasingEffect`:
    ``current_value >= goal_value - tolerance``.

    The target object must be a :class:`~semantic_digital_twin.semantic_annotations.mixins.HasFillLevel`
    representing the receiving container.
    """

    target_object: HasFillLevel
