from __future__ import annotations

from dataclasses import dataclass

from krrood_spatial_math.spatial_types import Vector3


@dataclass
class AlignmentPair:
    """
    A pair of normals that should be aligned during a tool motion: the normal of the
    tool tip and the normal of the target it acts on.
    """

    tip_normal: Vector3
    """
    The normal of the tool tip, expressed in the tool's frame.
    """

    reference_normal: Vector3
    """
    The normal of the target, expressed in the target's frame.
    """
