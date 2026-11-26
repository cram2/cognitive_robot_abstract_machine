"""
Sequences of poses that a motion passes through in order.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood_spatial_math.spatial_types import Pose


@dataclass(kw_only=True)
class PoseTrajectory:
    """
    Immutable wrapper for a sequence of waypoint poses.
    """

    poses: list[Pose]
    """
    Ordered waypoint poses.
    """
