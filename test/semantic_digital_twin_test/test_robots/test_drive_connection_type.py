import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.exceptions import MissingDriveConnectionError
from semantic_digital_twin.robots.pr2 import PR2MobileBase
from semantic_digital_twin.robots.stretch import StretchMobileBase
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    OmniDrive,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from ...dataset.robots_on_a_drive import RobotOnADifferentialDrive

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
