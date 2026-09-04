from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from typing_extensions import Type

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.hsrb import HSRB, HSRBBaseLaser
from semantic_digital_twin.robots.pr2 import PR2, PR2BaseLaser
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Laser
from semantic_digital_twin.robots.stretch import Stretch, StretchBaseLaser
from semantic_digital_twin.robots.tiago import Tiago, TiagoBaseLaser

# %% the robots carrying a base laser


@dataclass(frozen=True)
class LaserCase:
    """
    One robot's base laser, as its description mounts it.
    """

    robot: Type[AbstractRobot]
    """
    The robot the laser is mounted on.
    """

    laser: Type[Laser]
    """
    The laser the robot's mobile base is expected to carry.
    """

    laser_link: str
    """
    The body the robot's description mounts the laser on.
    """

    def annotate_own_world(self) -> AbstractRobot:
        """
        :return: This robot, annotated in a world parsed from its own description.
        """
        return self.robot.from_world(
            URDFParser.from_file(self.robot.get_ros_file_path()).parse()
        )


LASER_CASES = [
    LaserCase(PR2, PR2BaseLaser, "base_laser_link"),
    LaserCase(HSRB, HSRBBaseLaser, "base_range_sensor_link"),
    LaserCase(Tiago, TiagoBaseLaser, "base_laser_link"),
    LaserCase(Stretch, StretchBaseLaser, "laser"),
]


@pytest.fixture(
    scope="module", params=LASER_CASES, ids=lambda case: case.robot.__name__
)
def laser_case(request) -> tuple[LaserCase, AbstractRobot]:
    """
    Annotates one robot in a world of its own, kept only for as long as this module
    runs.
    """
    case: LaserCase = request.param
    return case, case.annotate_own_world()


# %% the laser the mobile base carries


def test_the_mobile_base_carries_the_robots_laser(laser_case):
    case, robot = laser_case

    assert isinstance(robot.mobile_base.laser, case.laser)


def test_the_laser_sits_on_the_link_its_description_names(laser_case):
    case, robot = laser_case

    assert robot.mobile_base.laser.root.name.name == case.laser_link


def test_the_laser_is_one_of_the_robots_sensors(laser_case):
    _, robot = laser_case

    assert robot.mobile_base.laser in robot.get_sensors()


def test_the_laser_sweeps_the_pattern_its_description_declares(laser_case):
    case, robot = laser_case
    declared = case.laser.setup_default_configuration_in_world_below_robot_root(
        robot.root
    ).scan_pattern

    assert robot.mobile_base.laser.scan_pattern == declared


# %% the readings the mobile base hands back


def test_the_mobile_base_reports_one_measurement_per_beam(laser_case):
    _, robot = laser_case
    beam_count = robot.mobile_base.laser.scan_pattern.beam_count

    reading = robot.mobile_base.get_laser_reading()

    assert len(reading.direction) == len(reading.distance) == beam_count


def test_the_beams_are_expressed_in_the_lasers_own_frame(laser_case):
    _, robot = laser_case
    laser = robot.mobile_base.laser

    reading = robot.mobile_base.get_laser_reading()

    assert {direction.reference_frame for direction in reading.direction} == {
        laser.root
    }


# %% a laser reading a world it stands in


def test_a_laser_in_a_furnished_world_measures_the_surfaces_around_it(
    pr2_apartment_world,
):
    robot = pr2_apartment_world.get_semantic_annotations_by_type(PR2)[0]

    reading = robot.mobile_base.get_laser_reading()

    assert any(math.isfinite(distance) for distance in reading.distance)
