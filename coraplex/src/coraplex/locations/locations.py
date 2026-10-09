from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Iterator

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ReachFraction
from coraplex.locations.base import Location
from coraplex.locations.costmaps import (
    Costmap,
    OccupancyCostmap,
    RingCostmap,
    VisibilityCostmap,
)
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types.spatial_types import Pose, Pose2D


@dataclass
class CostmapLocation(Location, ABC):
    """
    A location whose candidates are sampled from a costmap of the world as it is when
    they are sampled.
    """

    context: Context = field(kw_only=True)
    """
    The context holding the robot and the world the costmap is built from.
    """

    map_resolution: float = field(default=0.02, kw_only=True)
    """
    Edge length, in meters, of a cell of the costmap the candidates are sampled from.
    """

    map_cells: int = field(default=200, kw_only=True)
    """
    Number of cells along each side of that costmap.
    """

    def __post_init__(self) -> None:
        """
        Take the plan's sampling seed unless this location has its own.
        """
        if self.seed is not None:
            return
        self.seed = self.context.sampling_seed

    @abstractmethod
    def costmap(self) -> Costmap:
        """
        :return: The costmap the candidates are sampled from, built from the world as it
            is now.
        """

    def candidates(self) -> Iterator[Pose2D]:
        return self.costmap().sample(self.number_of_samples, self.seed)

    def _in_world(self, pose: Pose) -> Pose:
        """
        :param pose: A pose in any frame of the world.
        :return: Where `pose` is in the world frame now, so a pose given relative to a
            body follows that body.
        """
        return self.context.world.transform(pose, self.context.world.root)


@dataclass
class ReachabilityLocation(CostmapLocation):
    """
    Where the robot can stand to reach a target with one arm.
    """

    target_pose: Pose
    """
    The pose the arm is to reach.

    Given relative to a body, it is where that body is when the candidates are sampled.
    """

    arm: Arm
    """
    The arm with which to reach the target.
    """

    reach_fraction: float = ReachFraction.GRASPING
    """
    The fraction of the arm's length the robot stands off the target by.
    """

    ring_standard_deviation: int = field(default=15, kw_only=True)
    """
    How far, in cells, the standing poses spread around the stand-off distance.
    """

    def costmap(self) -> Costmap:
        """
        :return: Standing poses clear of the surroundings, at the arm's reach distance
            around the target.
        """
        target_pose = self._in_world(self.target_pose)
        occupancy = OccupancyCostmap.default_map(
            context=self.context,
            target=target_pose,
            resolution=self.map_resolution,
            cells=self.map_cells,
        )
        ring = RingCostmap.from_arm_reach_distance(
            context=self.context,
            arm=self.arm,
            origin=target_pose,
            reach_fraction=self.reach_fraction,
            resolution=self.map_resolution,
            cells=self.map_cells,
            standard_deviation=self.ring_standard_deviation,
        )
        return occupancy & ring

    def candidates(self) -> Iterator[Pose2D]:
        """
        :return: The poses sampled from the costmap, in the order they were sampled,
            leaving out those farther from the target along the floor than the arm is
            long, since the target cannot be reached from there.
        """
        target = self._in_world(self.target_pose).position
        arm_length = float(self.arm.approximate_length())
        for candidate in super().candidates():
            offset = candidate.position.to_np()[:2] - target.to_np()[:2]
            if np.linalg.norm(offset) <= arm_length:
                yield candidate


@dataclass
class VisibilityLocation(CostmapLocation):
    """
    Where the robot can stand to see a target with its camera.
    """

    target_pose: Pose
    """
    The pose that should be visible.

    Given relative to a body, it is where that body is when the candidates are sampled.
    """

    def costmap(self) -> Costmap:
        """
        :return: Standing poses clear of the surroundings from which the target is in
            view.
        """
        target_pose = self._in_world(self.target_pose)
        camera = self.context.robot.get_default_camera()
        occupancy = OccupancyCostmap.default_map(
            context=self.context,
            target=target_pose,
            resolution=self.map_resolution,
            cells=self.map_cells,
        )
        visibility = VisibilityCostmap(
            minimum_height=camera.minimal_height,
            maximum_height=camera.maximal_height,
            world=self.context.world,
            width=self.map_cells,
            height=self.map_cells,
            resolution=self.map_resolution,
            origin=target_pose,
        )
        return occupancy & visibility
