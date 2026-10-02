from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from typing_extensions import Type

from semantic_digital_twin.adapters.sensors.lidar import Lidar, SimulatedLidarSource
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.hsrb import HSRB, HSRBBaseLidar
from semantic_digital_twin.robots.pr2 import PR2, PR2BaseLidar
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch, StretchBaseLidar
from semantic_digital_twin.robots.tiago import Tiago, TiagoBaseLidar

# %% the robots carrying a base lidar


@dataclass(frozen=True)
class LidarCase:
    """
    One robot's base lidar, as its description mounts it.
    """

    robot: Type[AbstractRobot]
    """
    The robot the lidar is mounted on.
    """

    lidar: Type[Lidar]
    """
    The lidar the robot's mobile base is expected to carry.
    """

    lidar_link: str
    """
    The body the robot's description mounts the lidar on.
    """

    def annotate_own_world(self) -> AbstractRobot:
        """
        :return: This robot, annotated in a world parsed from its own description.
        """
        return self.robot.from_world(
            URDFParser.from_file(self.robot.get_ros_file_path()).parse()
        )


LIDAR_CASES = [
    LidarCase(PR2, PR2BaseLidar, "base_laser_link"),
    LidarCase(HSRB, HSRBBaseLidar, "base_range_sensor_link"),
    LidarCase(Tiago, TiagoBaseLidar, "base_laser_link"),
    LidarCase(Stretch, StretchBaseLidar, "laser"),
]


@pytest.fixture(
    scope="module", params=LIDAR_CASES, ids=lambda case: case.robot.__name__
)
def lidar_case(request) -> tuple[LidarCase, AbstractRobot]:
    """
    Annotates one robot in a world of its own, kept only for as long as this module
    runs.
    """
    case: LidarCase = request.param
    return case, case.annotate_own_world()


# %% the lidar the mobile base carries


def test_the_mobile_base_carries_the_robots_lidar(lidar_case):
    case, robot = lidar_case

    assert isinstance(robot.mobile_base.lidar, case.lidar)


def test_the_lidar_sits_on_the_link_its_description_names(lidar_case):
    case, robot = lidar_case

    assert robot.mobile_base.lidar.root.name.name == case.lidar_link


def test_the_lidar_is_one_of_the_robots_sensors(lidar_case):
    _, robot = lidar_case

    assert robot.mobile_base.lidar in robot.get_sensors()


def test_the_lidar_sweeps_the_pattern_its_description_declares(lidar_case):
    case, robot = lidar_case
    declared = case.lidar.with_simulated_source(robot.root).scan_pattern

    assert robot.mobile_base.lidar.scan_pattern == declared


def test_an_annotated_lidar_measures_the_world_it_stands_in(lidar_case):
    _, robot = lidar_case

    assert isinstance(robot.mobile_base.lidar.source, SimulatedLidarSource)


# %% the readings the mobile base hands back


def test_the_mobile_base_reports_one_measurement_per_beam(lidar_case):
    _, robot = lidar_case
    beam_count = robot.mobile_base.lidar.scan_pattern.beam_count

    reading = robot.mobile_base.get_lidar_reading()

    assert len(reading.ranges) == beam_count


def test_the_beams_are_expressed_in_the_lidars_own_frame(lidar_case):
    _, robot = lidar_case
    lidar = robot.mobile_base.lidar

    reading = robot.mobile_base.get_lidar_reading()

    assert reading.reference_frame is lidar.root


# %% a lidar reading a world it stands in


def test_a_lidar_in_a_furnished_world_measures_the_surfaces_around_it(
    pr2_apartment_world,
):
    robot = pr2_apartment_world.get_semantic_annotations_by_type(PR2)[0]

    reading = robot.mobile_base.get_lidar_reading()

    assert any(math.isfinite(distance) for distance in reading.ranges)
