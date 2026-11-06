from datetime import timedelta

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchVelocityInterface,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.stretch import StretchJoint, StretchMobileBase
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
)

# %% controlled joints


def test_stretch_declares_both_wheels_as_controlled(stretch_world_copy) -> None:
    """
    The wheels are driven by the same velocity controller as the arm, so the robot
    declares them next to the base drive.
    """
    for joint_name in (StretchJoint.RIGHT_WHEEL, StretchJoint.LEFT_WHEEL):
        assert stretch_world_copy.get_connection_by_name(joint_name).is_controlled


# %% velocity interface


def test_velocity_controlled_joints_exist_in_the_world(stretch_world_copy):
    """
    The velocity group controller addresses joints by name in a fixed order, so a joint
    renamed in the robot description would leave the controller driving nothing.
    """
    interface = StretchVelocityInterface()

    for joint_name in interface.velocity_controlled_joint_names():
        connection = stretch_world_copy.get_connection_by_name(joint_name)
        assert isinstance(connection, ActiveConnection1DOF)


def test_velocity_interface_sets_up_against_the_robot_description(
    init_rospy, stretch_world_specification
):
    """
    The configuration the robot actually runs comes up against the robot description:
    the hardware topics are wired and every controlled joint resolves.

    Closed-loop control needs live joint states, so this covers initialisation rather
    than motion.
    """
    giskard = Giskard(
        world_source=stretch_world_specification,
        robot_interface_config=StretchVelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=25, braking_time=timedelta(seconds=1.15)
        ),
    )

    giskard.setup()

    world = giskard.executor.context.world
    assert world.get_connections_by_type(DifferentialDrive)
    for joint_name in StretchVelocityInterface().velocity_controlled_joint_names():
        assert world.get_connection_by_name(joint_name) is not None


def test_velocity_interface_reads_the_localization_relative_to_the_odometry_frame_of_its_mobile_base(
    init_rospy, stretch_world_specification, monkeypatch
):
    """
    The localization is read from tf as the frame the mobile base reports its odometry
    relative to, so a base naming another frame is still localized.
    """
    monkeypatch.setattr(StretchMobileBase, "odometry_body_name", "base_odometry")
    interface = StretchVelocityInterface()
    giskard = Giskard(
        world_source=stretch_world_specification,
        robot_interface_config=interface,
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=25, braking_time=timedelta(seconds=1.15)
        ),
    )

    giskard.setup()

    [(_, tf_child_frame)] = (
        interface.tf_frame_synchronizer.connection_to_frames.values()
    )
    assert tf_child_frame == StretchMobileBase.odometry_body_name
