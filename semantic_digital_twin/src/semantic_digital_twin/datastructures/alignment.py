from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.spatial_types import Vector3


@dataclass
class AlignmentPair:
    """
    A pair of normals that should be aligned during a tool motion: the normal of the
    tool tip and the normal of the goal it acts on.
    """

    tip_normal: Vector3
    """
    The normal of the tool tip, expressed in the tool's frame.
    """

    goal_normal: Vector3
    """
    The normal of the goal, expressed in the goal's frame.
    """
