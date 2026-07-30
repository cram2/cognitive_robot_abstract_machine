"""
Effect types for manipulation domains.
"""

# %% imports

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.world_description.effects import (
    MonotoneDecreasingEffect,
    MonotoneIncreasingEffect,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel


# %% articulated-container effects


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


# %% liquid-transfer effects


def _read_fill_level(container: HasFillLevel) -> float:
    """
    Read the container's current fill level in ``[0, 1]``.

    :param container: The filled container whose fill level is read.
    :return: The current fill level.
    """
    return container.fill_level


@dataclass(eq=False, kw_only=True)
class PouringEffect(MonotoneDecreasingEffect):
    """
    Effect achieved when the fill level of a source container drops to or below
    ``goal_value``.
    """

    target_object: HasFillLevel
    """
    The filled container whose fill level this effect drains.
    """

    property_getter: Callable[[HasFillLevel], float] = field(default=_read_fill_level)
    """
    Reads the container's current fill level.
    """
