from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import Self

from semantic_digital_twin.adapters.sensors.lidar import (
    Lidar,
    LidarSource,
    SimulatedLidarSource,
)
from semantic_digital_twin.datastructures.lidar_reading import LidarReading
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.scan_pattern import ScanPattern
from semantic_digital_twin.exceptions import InvalidScanPattern
from semantic_digital_twin.robots.robot_parts import Sensor
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
Distance from the lidar to the center of the wall used to exercise the minimum range.
"""

FAR_WALL_DISTANCE = 2.0
"""
Distance from the lidar to the center of the wall the beams are meant to find.
"""


def wall_surface_distance(wall_center_distance: float) -> float:
    """
    :param wall_center_distance: Distance from the lidar to the center of the wall.
    :return: Distance from the lidar to the wall surface facing it.
    """
    return wall_center_distance - WALL_THICKNESS / 2


def forward_beam_pattern(
    minimum_range: float = 0.0, maximum_range: float = 10.0
) -> ScanPattern:
    """
    :return: A pattern of a single beam along the lidar's forward axis.
    """
    return ScanPattern(
        minimum_angle=0.0,
        maximum_angle=0.0,
        angle_increment=np.pi / 4,
        minimum_range=minimum_range,
        maximum_range=maximum_range,
    )


@dataclass(eq=False)
class BodyMountedLidar(Lidar):
    """
    A lidar mounted on a body that is already present in the world.
    """

    @classmethod
    def with_source(
        cls, robot_root: KinematicStructureEntity, source: LidarSource
    ) -> Self:
        return cls(root=robot_root, scan_pattern=forward_beam_pattern(), source=source)


@dataclass
class ConstantLidarSource(LidarSource):
    """
    A source that answers every request with the same prepared reading.
    """

    reading: LidarReading
    """
    The reading handed back on every call.
    """

    def get_lidar_reading(self, lidar: Lidar) -> LidarReading:
        return self.reading


def forward_beam_reading(mount: Body) -> LidarReading:
    """
    :return: A reading of :func:`forward_beam_pattern` taken by a lidar on the given body.
    """
    return LidarReading(
        reference_frame=mount,
        scan_pattern=forward_beam_pattern(),
        ranges=np.array([FAR_WALL_DISTANCE]),
    )


def simulated_lidar(mount: Body, scan_pattern: ScanPattern) -> BodyMountedLidar:
    """
    :return: A lidar on the given body that measures the world it stands in.
    """
    return BodyMountedLidar(
        root=mount, scan_pattern=scan_pattern, source=SimulatedLidarSource()
    )


def world_with_walls(*wall_center_distances: float) -> tuple[World, Body]:
    """
    Builds a world holding the lidar's mount body at the origin and a wall box centered
    on the positive x axis at each given distance.

    :return: The world and the body the lidar is mounted on.
    """
    world = World.create_with_root_body("map")
    mount = Body(name=PrefixedName("lidar_mount"))
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


# %% simulated source


def test_simulated_source_reports_the_distance_to_the_wall_surface():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    lidar = simulated_lidar(mount, forward_beam_pattern())

    reading = lidar.get_lidar_reading()

    [distance] = reading.ranges
    assert distance == pytest.approx(wall_surface_distance(FAR_WALL_DISTANCE))


def test_simulated_source_reports_infinity_for_a_beam_that_hits_nothing():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    backward_beam = ScanPattern(
        minimum_angle=np.pi,
        maximum_angle=np.pi,
        angle_increment=np.pi / 4,
        minimum_range=0.0,
        maximum_range=10.0,
    )
    lidar = simulated_lidar(mount, backward_beam)

    reading = lidar.get_lidar_reading()

    assert reading.ranges.tolist() == [math.inf]


def test_simulated_source_reports_the_surface_behind_a_wall_closer_than_its_minimum_range():
    _, mount = world_with_walls(NEAR_WALL_DISTANCE, FAR_WALL_DISTANCE)
    minimum_range = wall_surface_distance(NEAR_WALL_DISTANCE) + WALL_THICKNESS + 0.1
    lidar = simulated_lidar(mount, forward_beam_pattern(minimum_range=minimum_range))

    reading = lidar.get_lidar_reading()

    [distance] = reading.ranges
    assert distance == pytest.approx(wall_surface_distance(FAR_WALL_DISTANCE))


def test_simulated_source_reports_infinity_beyond_its_maximum_range():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    lidar = simulated_lidar(
        mount,
        forward_beam_pattern(
            maximum_range=wall_surface_distance(FAR_WALL_DISTANCE) / 2
        ),
    )

    reading = lidar.get_lidar_reading()

    assert reading.ranges.tolist() == [math.inf]


def test_simulated_source_returns_one_range_per_beam():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    pattern = ScanPattern(
        minimum_angle=-np.pi / 2,
        maximum_angle=np.pi / 2,
        angle_increment=np.pi / 8,
        minimum_range=0.0,
        maximum_range=10.0,
    )
    lidar = simulated_lidar(mount, pattern)

    reading = lidar.get_lidar_reading()

    assert len(reading.ranges) == pattern.beam_count


def test_simulated_source_expresses_its_beams_in_the_lidars_own_root():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    lidar = simulated_lidar(mount, forward_beam_pattern())

    reading = lidar.get_lidar_reading()

    assert reading.reference_frame is mount


def test_simulated_source_reading_carries_the_lidars_scan_pattern():
    _, mount = world_with_walls(FAR_WALL_DISTANCE)
    lidar = simulated_lidar(mount, forward_beam_pattern())

    reading = lidar.get_lidar_reading()

    assert reading.scan_pattern is lidar.scan_pattern


# %% lidar robot part


def test_a_lidar_is_a_sensor():
    _, mount = world_with_walls()

    lidar = BodyMountedLidar(
        root=mount,
        scan_pattern=forward_beam_pattern(),
        source=ConstantLidarSource(reading=forward_beam_reading(mount)),
    )

    assert isinstance(lidar, Sensor)


def test_a_lidar_hands_back_the_reading_its_source_takes():
    _, mount = world_with_walls()
    reading = forward_beam_reading(mount)
    lidar = BodyMountedLidar(
        root=mount,
        scan_pattern=forward_beam_pattern(),
        source=ConstantLidarSource(reading=reading),
    )

    assert lidar.get_lidar_reading() is reading


def test_a_lidar_keeps_the_scan_pattern_it_was_built_with():
    _, mount = world_with_walls()
    pattern = forward_beam_pattern()

    lidar = BodyMountedLidar(
        root=mount,
        scan_pattern=pattern,
        source=ConstantLidarSource(reading=forward_beam_reading(mount)),
    )

    assert lidar.scan_pattern is pattern


# %% choosing where the readings come from


def test_a_lidar_is_built_where_its_own_class_mounts_it():
    _, mount = world_with_walls()

    lidar = BodyMountedLidar.with_simulated_source(mount)

    assert lidar.root is mount
    assert lidar.scan_pattern == forward_beam_pattern()


def test_a_lidar_built_with_a_simulated_source_measures_the_world():
    _, mount = world_with_walls()

    lidar = BodyMountedLidar.with_simulated_source(mount)

    assert isinstance(lidar.source, SimulatedLidarSource)


def test_a_lidar_annotated_from_a_robot_description_measures_the_world():
    _, mount = world_with_walls()

    lidar = BodyMountedLidar.setup_default_configuration_in_world_below_robot_root(
        mount
    )

    assert isinstance(lidar.source, SimulatedLidarSource)
