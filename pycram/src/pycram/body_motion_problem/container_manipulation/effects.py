"""
Effect types for articulated container manipulation.

Defines the goal-state representations used when opening or closing articulated
containers such as drawers and doors.
"""

from __future__ import annotations

from dataclasses import dataclass

from pycram.body_motion_problem.types import (
    MonotoneDecreasingEffect,
    MonotoneIncreasingEffect,
)


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
