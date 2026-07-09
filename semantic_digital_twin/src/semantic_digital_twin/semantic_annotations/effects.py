"""
Effect types for manipulation domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from semantic_digital_twin.world_description.effects import (
    MonotoneDecreasingEffect,
    MonotoneIncreasingEffect,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel


@dataclass(eq=False, kw_only=True)
class OpenedEffect(MonotoneIncreasingEffect):
    """
    Effect achieved when an articulated container (drawer, door) is open.
    """


@dataclass(eq=False, kw_only=True)
class ClosedEffect(MonotoneDecreasingEffect):
    """
    Effect achieved when an articulated container (drawer, door) is closed.
    """


@dataclass(eq=False, kw_only=True)
class PouringEffect(MonotoneDecreasingEffect):
    """
    Effect achieved when the fill level of a source container drops to or below ``goal_value``.
    """

    target_object: HasFillLevel
