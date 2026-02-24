from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .pose import PoseStamped


@dataclass()
class PoseTrajectory:
    """
    Immutable wrapper for a sequence of waypoint poses.
    """

    poses: List[PoseStamped, ...]
    """
    Ordered waypoint poses.
    """

