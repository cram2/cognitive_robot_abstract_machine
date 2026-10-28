from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class PoseTrajectory:
    """
    Immutable wrapper for a sequence of waypoint poses.
    """

    poses: list[Pose]
    """
    Ordered waypoint poses.
    """
