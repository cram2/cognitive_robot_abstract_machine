from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Iterable, Tuple

from .pose import PoseStamped


@dataclass()
class PoseTrajectory:
    """
    Immutable wrapper for a sequence of waypoint poses.
    """

    poses: Tuple[PoseStamped, ...]
    """
    Ordered waypoint poses.
    """

    def __post_init__(self):
        self.poses = tuple(self.poses)