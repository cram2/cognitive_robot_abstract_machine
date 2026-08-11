from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchStandaloneInterface,
    StretchVelocityInterface,
    WorldWithStretchConfigDiffDrive,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
)

# %% controlled joints


def test_controlled_joints_resolve_the_drive_from_the_world(stretch_world_copy):
    """
    The base drive is looked up in the world instead of being named by a literal, so the
    interface controls the base whatever the drive connection ended up being called.
    """
    drive = stretch_world_copy.get_connections_by_type(DifferentialDrive)[0]
    interface = StretchStandaloneInterface()

    controlled_joint_names = interface.controlled_joint_names(stretch_world_copy)

    assert controlled_joint_names[0] == drive.name
    assert controlled_joint_names[1:] == interface.joint_names


def test_every_controlled_joint_exists_in_the_world(stretch_world_copy):
    """
    Registering a joint the world does not know about fails at setup time, so every
    declared name must resolve to a connection.
    """
    interface = StretchStandaloneInterface()

    for joint_name in interface.controlled_joint_names(stretch_world_copy):
        assert stretch_world_copy.get_connection_by_name(joint_name) is not None


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
        behavior_tree_config=ClosedLoopBTConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=25, prediction_horizon=30
        ),
    )

    giskard.setup()

    world = GiskardBlackboard().executor.context.world
    assert world.get_connections_by_type(DifferentialDrive)
    for joint_name in StretchVelocityInterface().velocity_controlled_joint_names():
        assert world.get_connection_by_name(joint_name) is not None
