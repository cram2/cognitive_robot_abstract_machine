from datetime import timedelta

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchVelocityInterface,
    WorldWithStretchConfigDiffDrive,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.stretch import Stretch, StretchJoint
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
)

# %% controlled joints


def test_stretch_declares_both_wheels_as_controlled(stretch_world_copy):
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


def test_velocity_interface_sets_up_against_the_robot_description(init_rospy):
    """
    The configuration the robot actually runs comes up against the robot description:
    the hardware topics are wired and every controlled joint resolves.

    Closed-loop control needs live joint states, so this covers initialisation rather
    than motion.
    """
    giskard = Giskard(
        world_config=WorldWithStretchConfigDiffDrive(
            urdf=load_xacro(Stretch.get_ros_file_path())
        ),
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
