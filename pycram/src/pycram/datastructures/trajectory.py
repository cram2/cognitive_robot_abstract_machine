from __future__ import annotations

from dataclasses import dataclass
from typing import List

from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class PoseTrajectory:
    """
    Immutable wrapper for a sequence of waypoint poses.
    """

    poses: List[Pose]
    """
    Ordered waypoint poses.
    """

