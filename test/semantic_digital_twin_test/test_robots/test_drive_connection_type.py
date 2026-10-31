import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.exceptions import MissingDriveConnectionError
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2, PR2MobileBase
from semantic_digital_twin.robots.stretch import Stretch, StretchMobileBase
from semantic_digital_twin.robots.tiago import Tiago, TiagoMujoco
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from ...dataset.robots_on_a_drive import RobotOnADifferentialDrive

# %% drive type resolved from the robot's mobile base


def test_omni_drive_robot_reports_omni_drive():
    assert PR2.get_drive_connection_type() is OmniDrive


def test_differential_drive_robots_report_differential_drive():
    assert Tiago.get_drive_connection_type() is DifferentialDrive
    assert TiagoMujoco.get_drive_connection_type() is DifferentialDrive
    assert Stretch.get_drive_connection_type() is DifferentialDrive


# %% robot without a mobile base


def test_robot_without_mobile_base_uses_fixed_connection():
    assert MinimalRobot.get_drive_connection_type() is FixedConnection


# %% mobile base exposes its bound drive


def test_mobile_base_reports_its_bound_drive():
    assert PR2MobileBase.get_drive_connection_type() is OmniDrive
    assert StretchMobileBase.get_drive_connection_type() is DifferentialDrive


# %% the drive a mobile base is moved by


def test_a_mobile_base_is_moved_by_the_drive_of_its_robot(
    cylinder_bot_diff_world: World,
):
    robot = cylinder_bot_diff_world.get_semantic_annotations_by_type(
        RobotOnADifferentialDrive
    )[0]

    assert robot.mobile_base.drive is robot.drive


def test_a_mobile_base_of_a_robot_attached_without_a_drive_has_no_drive():
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("bot")))
    robot = RobotOnADifferentialDrive.from_world(world)

    with pytest.raises(MissingDriveConnectionError) as raised:
        robot.mobile_base.drive

    assert raised.value.robot_part is robot.mobile_base
