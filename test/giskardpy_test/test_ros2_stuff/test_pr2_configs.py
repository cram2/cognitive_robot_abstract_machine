from __future__ import annotations

import pytest

from giskardpy.middleware.ros2.command_publishing import (
    DriveVelocityCommandPublisher,
    JointGroupVelocityCommandPublisher,
    MultiDOFCommandFormat,
)
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import (
    PR2CommandTopic,
    PR2LocalizationFrame,
    PR2VelocityInterface,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.pr2 import PR2Joint, PR2MobileBase
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    OmniDrive,
)

# %% fixtures


@pytest.fixture()
def pr2_velocity_interface() -> PR2VelocityInterface:
    """
    The interface the real PR2 runs.
    """
    return PR2VelocityInterface()


@pytest.fixture()
def set_up_pr2_velocity_interface(
    init_background_node,
    pr2_world_specification: WorldSpecification,
    pr2_velocity_interface: PR2VelocityInterface,
) -> PR2VelocityInterface:
    """
    The interface the real PR2 runs, set up by a Giskard against the robot description.

    Closed-loop control needs live joint states, so this covers initialisation rather
    than motion.
    """
    Giskard(
        world_source=pr2_world_specification,
        robot_interface_config=pr2_velocity_interface,
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    ).setup()
    return pr2_velocity_interface


def joint_group_publishers(
    *, interface: PR2VelocityInterface
) -> dict[str, JointGroupVelocityCommandPublisher]:
    """
    :param interface: An interface that is set up.
    :return: The joint group publishers of the interface, keyed by their topic.
    """
    return {
        publisher.command_topic: publisher
        for publisher in interface.control_loop.command_publishers
        if isinstance(publisher, JointGroupVelocityCommandPublisher)
    }


# %% controlled joints


def test_velocity_controlled_joints_exist_in_the_world(
    pr2_world_copy: World, pr2_velocity_interface: PR2VelocityInterface
) -> None:
    """
    The velocity controllers address joints by name in a fixed order, so a joint renamed
    in the robot description would leave a controller driving nothing.
    """
    for joint_names in (
        pr2_velocity_interface.right_arm_velocity_joint_names(),
        pr2_velocity_interface.left_arm_velocity_joint_names(),
        pr2_velocity_interface.torso_velocity_joint_names(),
        pr2_velocity_interface.head_velocity_joint_names(),
    ):
        for joint_name in joint_names:
            connection = pr2_world_copy.get_connection_by_name(joint_name)
            assert isinstance(connection, ActiveConnection1DOF)


# %% velocity interface


def test_velocity_interface_commands_every_controller_of_the_pr2(
    set_up_pr2_velocity_interface: PR2VelocityInterface,
) -> None:
    """
    Every controller topic gets exactly one publisher, the base its drive and the joint
    groups their joints in the order the controllers expect.
    """
    interface = set_up_pr2_velocity_interface
    [drive_publisher] = [
        publisher
        for publisher in interface.control_loop.command_publishers
        if isinstance(publisher, DriveVelocityCommandPublisher)
    ]
    assert drive_publisher.command_topic == PR2CommandTopic.BASE
    assert (
        drive_publisher.connection
        == interface.world.get_connections_by_type(OmniDrive)[0]
    )

    expected_joint_names = {
        PR2CommandTopic.RIGHT_ARM: interface.right_arm_velocity_joint_names(),
        PR2CommandTopic.LEFT_ARM: interface.left_arm_velocity_joint_names(),
        PR2CommandTopic.TORSO: interface.torso_velocity_joint_names(),
        PR2CommandTopic.HEAD: interface.head_velocity_joint_names(),
    }
    publishers = joint_group_publishers(interface=interface)
    assert set(publishers) == set(expected_joint_names)
    for command_topic, joint_names in expected_joint_names.items():
        publisher = publishers[command_topic]
        assert [
            connection.name.name for connection in publisher.connections
        ] == joint_names
        assert isinstance(publisher.command_format, MultiDOFCommandFormat)


def test_velocity_interface_leaves_the_grippers_uncommanded(
    set_up_pr2_velocity_interface: PR2VelocityInterface,
) -> None:
    """
    The grippers have no velocity controller, so none of their joints may be commanded.
    """
    commanded_joint_names = {
        connection.name.name
        for publisher in joint_group_publishers(
            interface=set_up_pr2_velocity_interface
        ).values()
        for connection in publisher.connections
    }
    for gripper_joint_name in (
        PR2Joint.LEFT_GRIPPER_LEFT_FINGER,
        PR2Joint.LEFT_GRIPPER_RIGHT_FINGER,
        PR2Joint.RIGHT_GRIPPER_LEFT_FINGER,
        PR2Joint.RIGHT_GRIPPER_RIGHT_FINGER,
    ):
        assert gripper_joint_name not in commanded_joint_names


def test_velocity_interface_localizes_the_odometry_frame_in_the_map(
    set_up_pr2_velocity_interface: PR2VelocityInterface,
) -> None:
    """
    The localization is read from tf as the transform from the map to the frame the
    mobile base reports its odometry relative to.
    """
    [frames] = (
        set_up_pr2_velocity_interface.tf_frame_synchronizer.connection_to_frames.values()
    )
    assert frames == (PR2LocalizationFrame.MAP, PR2MobileBase.odometry_body_name)
