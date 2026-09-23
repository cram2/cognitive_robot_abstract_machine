from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, Self, TYPE_CHECKING

from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.lidar_reading import LidarReading
from semantic_digital_twin.datastructures.scan_pattern import ScanPattern
from semantic_digital_twin.robots.robot_parts import Sensor
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

if TYPE_CHECKING:
    from rclpy.node import Node


# %% where readings come from


@dataclass
class LidarSource(ABC):
    """
    Where the readings of a lidar come from, either the world or a real scanner.
    """

    @abstractmethod
    def get_lidar_reading(self, lidar: Lidar) -> LidarReading:
        """
        :param lidar: The lidar whose beams are read.
        :return: The most recent sweep of that lidar.
        """


@dataclass
class SimulatedLidarSource(LidarSource):
    """
    A source that measures the world's collision geometry by casting a ray along every
    beam of the lidar's scan pattern.
    """

    def get_lidar_reading(self, lidar: Lidar) -> LidarReading:
        world_T_lidar = lidar.root.global_transform.to_np()
        world_V_beams = lidar.scan_pattern.beam_directions @ world_T_lidar[:3, :3].T
        world_P_lidar = np.tile(
            world_T_lidar[:3, 3], (lidar.scan_pattern.beam_count, 1)
        )

        points, index_ray, _ = lidar.root._world.ray_tracer.ray_test(
            world_P_lidar,
            world_P_lidar + world_V_beams * lidar.scan_pattern.maximum_range,
            multiple_hits=True,
            min_distance=lidar.scan_pattern.minimum_range,
            max_distance=lidar.scan_pattern.maximum_range,
        )

        return LidarReading(
            reference_frame=lidar.root,
            scan_pattern=lidar.scan_pattern,
            ranges=self._nearest_hit_per_beam(
                points, index_ray, world_P_lidar, lidar.scan_pattern.beam_count
            ),
        )

    def _nearest_hit_per_beam(
        self,
        points: np.ndarray,
        index_ray: np.ndarray,
        world_P_lidar: np.ndarray,
        beam_count: int,
    ) -> np.ndarray:
        """
        Reduces the hits of a ray test to the one distance each beam measures.

        :param points: The positions where the beams met a surface.
        :param index_ray: The beam each of those positions belongs to.
        :param world_P_lidar: The origin of every beam.
        :param beam_count: How many beams were cast.
        :return: The distance of the closest hit per beam, and ``math.inf`` for beams
            that hit nothing.

        ..note:: A beam can meet several surfaces, and the ray test does not order its
            hits, so the closest one is picked explicitly.
        """
        distances = np.full(beam_count, math.inf)
        if len(index_ray) == 0:
            return distances

        hit_distances = np.linalg.norm(points - world_P_lidar[index_ray], axis=1)
        farthest_first = np.argsort(hit_distances)[::-1]
        distances[index_ray[farthest_first]] = hit_distances[farthest_first]
        return distances


# %% the lidar a robot carries


@dataclass(eq=False)
class Lidar(Sensor, ABC):
    """
    A lidar is a sensor that measures the distance to the surfaces around it along a fan
    of beams.

    Subclasses state where they are mounted, and a source decides whether the readings
    are measured in the world or received from a real scanner.
    """

    scan_pattern: ScanPattern = field(kw_only=True)
    """
    The directions this lidar sweeps and the distances it can measure.
    """

    source: LidarSource = field(kw_only=True)
    """
    Where the readings of this lidar come from.
    """

    def get_lidar_reading(self) -> LidarReading:
        """
        :return: The most recent sweep of this lidar.
        """
        return self.source.get_lidar_reading(self)

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    @abstractmethod
    def with_source(
        cls, robot_root: KinematicStructureEntity, source: LidarSource
    ) -> Self:
        """
        :param robot_root: The root of the robot carrying this lidar.
        :param source: Where the readings come from, such as a real scanner on a topic.
        :return: This lidar, mounted on the body and sweeping the pattern its robot
            description declares.
        """

    @classmethod
    def with_simulated_source(cls, robot_root: KinematicStructureEntity) -> Self:
        """
        :param robot_root: The root of the robot carrying this lidar.
        :return: This lidar, measuring the world it stands in.
        """
        return cls.with_source(robot_root, SimulatedLidarSource())

    @classmethod
    def with_real_source(
        cls, robot_root: KinematicStructureEntity, node: Node, topic: str
    ) -> Self:
        """
        Creates a Lidar sensor attached to a real source.

        :param robot_root: The root of the robot carrying this lidar.
        :param node: The ros node, used for subscribing
        :param topic: The topic name of the real source.
        :return: A lidar sensor, measuring the real  world it stands in.
        """
        from semantic_digital_twin.adapters.ros.lidar import SubscribedLidarSource

        return cls.with_source(
            robot_root, SubscribedLidarSource(node=node, topic_name=topic)
        )

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls.with_simulated_source(robot_root)
