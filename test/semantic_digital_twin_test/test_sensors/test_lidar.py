from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest
from typing_extensions import List, Self

from semantic_digital_twin.adapters.sensors.lidar import SimulatedLaser
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.laser_reading import LaserReading
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.scan_pattern import ScanPattern
from semantic_digital_twin.exceptions import InvalidScanPattern
from semantic_digital_twin.robots.robot_parts import Laser, Sensor
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% scan patterns and scene geometry shared by the cases below

WALL_THICKNESS = 0.5
"""
Extent of every wall box along the axis the beams travel.
"""

NEAR_WALL_DISTANCE = 0.5
"""
Distance from the laser to the center of the wall used to exercise the minimum range.
"""

FAR_WALL_DISTANCE = 2.0
"""
Distance from the laser to the center of the wall the beams are meant to find.
"""


def wall_surface_distance(wall_center_distance: float) -> float:
    """
    :param wall_center_distance: Distance from the laser to the center of the wall.
    :return: Distance from the laser to the wall surface facing it.
    """
    return wall_center_distance - WALL_THICKNESS / 2


def forward_beam_pattern(
    minimum_range: float = 0.0, maximum_range: float = 10.0
) -> ScanPattern:
    """
    :return: A pattern of a single beam along the laser's forward axis.
    """
    return ScanPattern(
        minimum_angle=0.0,
        maximum_angle=0.0,
        angle_increment=np.pi / 4,
        minimum_range=minimum_range,
        maximum_range=maximum_range,
    )


@dataclass(eq=False)
class BodyMountedLaser(SimulatedLaser):
    """
    A simulated laser mounted on a body that is already present in the world.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(root=robot_root, scan_pattern=forward_beam_pattern())


@dataclass(eq=False)
class ConstantLaser(Laser):
    """
    A laser that answers every request with the same prepared reading.
    """

    reading: LaserReading = field(default_factory=LaserReading, kw_only=True)
    """
    The reading handed back on every call.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(root=robot_root, scan_pattern=forward_beam_pattern())

    def get_laser_reading(self) -> LaserReading:
        return self.reading


def world_with_walls(*wall_center_distances: float) -> tuple[World, Body]:
    """
    Builds a world holding the laser's mount body at the origin and a wall box centered
    on the positive x axis at each given distance.

    :return: The world and the body the laser is mounted on.
    """
    world = World.create_with_root_body("map")
    mount = Body(name=PrefixedName("laser_mount"))
    with world.modify_world():
        world.add_body(mount)
        world.add_connection(FixedConnection(parent=world.root, child=mount))
        for index, distance in enumerate(wall_center_distances):
            wall = Body(
                name=PrefixedName(f"wall_{index}"),
                collision=ShapeCollection([Box(scale=Scale(WALL_THICKNESS, 2.0, 2.0))]),
            )
            world.add_body(wall)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=wall,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=distance
                    ),
                )
            )
    return world, mount


# %% scan pattern


def test_beam_directions_cover_the_pattern_from_its_minimum_to_its_maximum_angle():
    pattern = ScanPattern(
        minimum_angle=-np.pi / 2,
        maximum_angle=np.pi / 2,
        angle_increment=np.pi / 4,
        minimum_range=0.0,
        maximum_range=10.0,
    )

    directions = pattern.beam_directions_in_frame(None)

    assert len(directions) == pattern.beam_count == 5
    assert np.allclose(directions[0].to_np(), [0.0, -1.0, 0.0, 0.0])
    assert np.allclose(directions[-1].to_np(), [0.0, 1.0, 0.0, 0.0])


def test_beam_directions_are_expressed_in_the_given_reference_frame():
    _, mount = world_with_walls()

    [direction] = forward_beam_pattern().beam_directions_in_frame(mount)

    assert direction.reference_frame is mount


def test_scan_pattern_rejects_a_non_positive_angle_increment():
    with pytest.raises(InvalidScanPattern):
        ScanPattern(
            minimum_angle=-np.pi / 2,
            maximum_angle=np.pi / 2,
            angle_increment=0.0,
            minimum_range=0.0,
            maximum_range=10.0,
        )


def test_scan_pattern_rejects_a_maximum_range_below_its_minimum_range():
    with pytest.raises(InvalidScanPattern):
        ScanPattern(
            minimum_angle=0.0,
            maximum_angle=0.0,
            angle_increment=np.pi / 4,
            minimum_range=5.0,
            maximum_range=1.0,
        )


# %% simulated laser


def test_simulated_laser_reports_the_distance_to_the_wall_surface():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    laser = BodyMountedLaser(root=mount, scan_pattern=forward_beam_pattern())

    reading = laser.get_laser_reading()

    [distance] = reading.distance
    assert distance == pytest.approx(wall_surface_distance(FAR_WALL_DISTANCE))


def test_simulated_laser_reports_infinity_for_a_beam_that_hits_nothing():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    backward_beam = ScanPattern(
        minimum_angle=np.pi,
        maximum_angle=np.pi,
        angle_increment=np.pi / 4,
        minimum_range=0.0,
        maximum_range=10.0,
    )
    laser = BodyMountedLaser(root=mount, scan_pattern=backward_beam)

    reading = laser.get_laser_reading()

    assert reading.distance == [math.inf]


def test_simulated_laser_reports_the_surface_behind_a_wall_closer_than_its_minimum_range():
    _, mount = world_with_walls(NEAR_WALL_DISTANCE, FAR_WALL_DISTANCE)
    minimum_range = wall_surface_distance(NEAR_WALL_DISTANCE) + WALL_THICKNESS + 0.1
    laser = BodyMountedLaser(
        root=mount, scan_pattern=forward_beam_pattern(minimum_range=minimum_range)
    )

    reading = laser.get_laser_reading()

    [distance] = reading.distance
    assert distance == pytest.approx(wall_surface_distance(FAR_WALL_DISTANCE))


def test_simulated_laser_reports_infinity_beyond_its_maximum_range():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    laser = BodyMountedLaser(
        root=mount,
        scan_pattern=forward_beam_pattern(
            maximum_range=wall_surface_distance(FAR_WALL_DISTANCE) / 2
        ),
    )

    reading = laser.get_laser_reading()

    assert reading.distance == [math.inf]


def test_simulated_laser_returns_one_direction_and_one_distance_per_beam():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    pattern = ScanPattern(
        minimum_angle=-np.pi / 2,
        maximum_angle=np.pi / 2,
        angle_increment=np.pi / 8,
        minimum_range=0.0,
        maximum_range=10.0,
    )
    laser = BodyMountedLaser(root=mount, scan_pattern=pattern)

    reading = laser.get_laser_reading()

    assert len(reading.direction) == len(reading.distance) == pattern.beam_count


def test_simulated_laser_expresses_its_beams_in_its_own_root():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    laser = BodyMountedLaser(root=mount, scan_pattern=forward_beam_pattern())

    reading = laser.get_laser_reading()

    assert {direction.reference_frame for direction in reading.direction} == {mount}


# %% laser robot part


def test_a_laser_is_a_sensor():
    _, mount = world_with_walls()

    laser = ConstantLaser(root=mount, scan_pattern=forward_beam_pattern())

    assert isinstance(laser, Sensor)


def test_a_laser_hands_back_the_reading_it_takes():
    _, mount = world_with_walls()
    reading = LaserReading()
    laser = ConstantLaser(
        root=mount, scan_pattern=forward_beam_pattern(), reading=reading
    )

    assert laser.get_laser_reading() is reading


def test_a_laser_keeps_the_scan_pattern_it_was_built_with():
    _, mount = world_with_walls()
    pattern = forward_beam_pattern()

    laser = ConstantLaser(root=mount, scan_pattern=pattern)

    assert laser.scan_pattern is pattern
