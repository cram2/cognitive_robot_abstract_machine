from dataclasses import dataclass, fields, is_dataclass

import pytest

from giskardpy.data_types.exceptions import NoControlledJointsError
from giskardpy.executor import NoPacing, RealTimePacer
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    WorldWithDaisyConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.hsr.configs import (
    WorldWithHSRConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import (
    PR2VelocityMujocoInterface,
    WorldWithPR2Config,
)
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    WorldWithStretchConfigDiffDrive,
)
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    WorldWithTracyConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.model.world_config import EmptyWorld, WorldConfig
from semantic_digital_twin.adapters.ros.input_synchronization import (
    LatestJointPositionSource,
    PendingJointPositionSource,
    SubscribedBasePoseSource,
)
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.robots.input_source import SimulatedJointPositionSource
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
    OmniDrive,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig

from ...semantic_digital_twin_test.test_ros.test_input_synchronization import (
    joint_state_message,
)

# %% the interface hierarchy is built from dataclasses


def test_the_robot_interface_base_is_a_dataclass():
    assert is_dataclass(RobotInterfaceConfig)


def test_the_interface_state_bound_after_construction_is_no_constructor_argument():
    assert [field.name for field in fields(RobotInterfaceConfig) if field.init] == []


def test_the_mujoco_interface_defaults_name_the_frames_its_localization_uses():
    interface = PR2VelocityMujocoInterface()

    assert (
        interface.map_name,
        interface.localization_joint_name,
        interface.odom_link_name,
    ) == ("map", "localization", "odom_combined")


# %% the tf frame synchronizer is created on demand


def test_an_interface_starts_without_a_tf_frame_synchronizer():
    assert GroupCommandingInterface(joint_names=[]).tf_frame_synchronizer is None


# %% attaching binds the giskard instance the accessors read from


def test_attaching_lets_the_interface_reach_the_server_config():
    server_config = GiskardServerConfig()
    interface = StandAloneRobotInterfaceConfig()
    giskard = Giskard(
        world_config=EmptyWorld(),
        server_config=server_config,
        robot_interface_config=interface,
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    interface.attach(giskard)

    assert interface.server_config is server_config


# %% the interface decides how the control loop is paced


def test_a_standalone_interface_is_not_paced():
    assert isinstance(StandAloneRobotInterfaceConfig().create_pacer(), NoPacing)


def test_a_closed_loop_interface_runs_in_real_time():
    interface = GroupCommandingInterface(joint_names=[])

    assert isinstance(interface.create_pacer(), RealTimePacer)


# %% standalone controls the joints the robot class declares


def controlled_joint_names(world: World) -> set[str]:
    """
    The names of the single degree of freedom connections of ``world`` that are flagged
    as controlled.
    """
    return {
        connection.name.name
        for connection in world.get_connections_by_type(ActiveConnection1DOF)
        if connection.is_controlled
    }


def set_up_standalone(world_config: WorldConfig) -> Giskard:
    """
    A Giskard that simulates the robot of ``world_config`` and is ready for goals.
    """
    giskard = Giskard(
        world_config=world_config,
        robot_interface_config=StandAloneRobotInterfaceConfig(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )
    giskard.setup()
    return giskard


def test_standalone_controls_the_joints_tracy_declares(init_rospy, tracy_world):
    giskard = set_up_standalone(
        WorldWithTracyConfig(urdf=load_xacro(Tracy.get_ros_file_path()))
    )

    assert controlled_joint_names(giskard.world_config.world) == (
        controlled_joint_names(tracy_world)
    )


def test_standalone_controls_the_joints_daisy_declares(init_rospy, daisy_world):
    giskard = set_up_standalone(
        WorldWithDaisyConfig(urdf=load_xacro(DAiSy.get_ros_file_path()))
    )

    assert controlled_joint_names(giskard.world_config.world) == (
        controlled_joint_names(daisy_world)
    )


def test_standalone_controls_the_joints_the_pr2_declares(init_rospy, pr2_world_copy):
    giskard = set_up_standalone(
        WorldWithPR2Config(urdf=load_xacro(PR2.get_ros_file_path()))
    )

    assert controlled_joint_names(giskard.world_config.world) == (
        controlled_joint_names(pr2_world_copy)
    )


def test_standalone_controls_the_joints_the_hsr_declares(init_rospy, _hsr_world_setup):
    giskard = set_up_standalone(
        WorldWithHSRConfig(urdf=load_xacro(HSRB.get_ros_file_path()))
    )

    assert controlled_joint_names(giskard.world_config.world) == (
        controlled_joint_names(_hsr_world_setup)
    )


def test_standalone_controls_the_joints_stretch_declares(
    init_rospy, stretch_world_copy
):
    giskard = set_up_standalone(
        WorldWithStretchConfigDiffDrive(urdf=load_xacro(Stretch.get_ros_file_path()))
    )

    assert controlled_joint_names(giskard.world_config.world) == (
        controlled_joint_names(stretch_world_copy)
    )


# %% the world config flags the drive it creates


def test_a_world_with_an_omni_drive_robot_controls_its_drive():
    world_config = WorldWithPR2Config(urdf=load_xacro(PR2.get_ros_file_path()))
    with world_config.world.modify_world():
        world_config.setup_world()

    [drive] = world_config.world.get_connections_by_type(OmniDrive)
    assert drive.is_controlled


def test_a_world_with_a_differential_drive_robot_controls_its_drive():
    world_config = WorldWithStretchConfigDiffDrive(
        urdf=load_xacro(Stretch.get_ros_file_path())
    )
    with world_config.world.modify_world():
        world_config.setup_world()

    [drive] = world_config.world.get_connections_by_type(DifferentialDrive)
    assert drive.is_controlled


def test_a_world_with_a_drive_limits_the_velocity_of_its_drive():
    world_config = WorldWithPR2Config(
        urdf=load_xacro(PR2.get_ros_file_path()),
        translation_velocity_limit=0.4,
        rotation_velocity_limit=0.7,
    )
    with world_config.world.modify_world():
        world_config.setup_world()

    [drive] = world_config.world.get_connections_by_type(OmniDrive)
    assert drive.x_velocity.limits.upper.velocity == (
        world_config.translation_velocity_limit
    )
    assert drive.yaw.limits.upper.velocity == world_config.rotation_velocity_limit


# %% in closed loop only commanded joints are controlled


@dataclass
class GroupCommandingInterface(ClosedLoopRobotInterfaceConfig):
    """
    An interface that commands one group of joints.
    """

    joint_names: list[PR2Joint]
    """
    The joints the group controller drives.
    """

    def setup(self):
        self.add_joint_velocity_group_controller(
            cmd_topic="group_velocity_commands", connections=self.joint_names
        )


def closed_loop_pr2(interface: ClosedLoopRobotInterfaceConfig) -> Giskard:
    """
    A Giskard that commands a PR2 through ``interface``, not set up yet.
    """
    return Giskard(
        world_config=WorldWithPR2Config(urdf=load_xacro(PR2.get_ros_file_path())),
        robot_interface_config=interface,
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )


def test_a_closed_loop_interface_that_commands_nothing_controls_no_joint(init_rospy):
    giskard = closed_loop_pr2(GroupCommandingInterface(joint_names=[]))

    with pytest.raises(NoControlledJointsError):
        giskard.setup()


def test_closed_loop_controls_exactly_the_commanded_joints(init_rospy):
    interface = GroupCommandingInterface(
        joint_names=[PR2Joint.HEAD_PAN, PR2Joint.HEAD_TILT]
    )
    giskard = closed_loop_pr2(interface)

    giskard.setup()

    assert controlled_joint_names(giskard.world_config.world) == set(
        interface.joint_names
    )


def test_closed_loop_does_not_control_a_drive_nothing_commands(init_rospy):
    giskard = closed_loop_pr2(GroupCommandingInterface(joint_names=[PR2Joint.HEAD_PAN]))

    giskard.setup()

    [drive] = giskard.world_config.world.get_connections_by_type(OmniDrive)
    assert not drive.is_controlled


# %% reading the robot's own parts from the robot


@dataclass
class PartReadingInterface(GroupCommandingInterface):
    """
    An interface that reads every part of the robot from the robot itself, on the topics
    the parts declare.
    """

    def setup(self):
        super().setup()
        self.sync_robot_parts()


@pytest.fixture()
def pr2_reading_its_own_parts(init_rospy) -> Giskard:
    """
    A closed-loop Giskard whose PR2 is read from what the robot publishes.
    """
    giskard = Giskard(
        world_config=WorldWithPR2Config(urdf=load_xacro(PR2.get_ros_file_path())),
        robot_interface_config=PartReadingInterface(joint_names=[PR2Joint.HEAD_PAN]),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )
    giskard.setup()
    return giskard


def test_a_chain_read_from_the_robot_is_applied_by_the_motion_server(
    pr2_reading_its_own_parts,
):
    arm = pr2_reading_its_own_parts.robot.left_arm

    assert isinstance(arm.source, PendingJointPositionSource)
    assert any(
        synchronizer is arm.source
        for synchronizer in pr2_reading_its_own_parts.motion_server.inputs.synchronizers
    )


def test_the_closed_control_loop_rewrites_the_joint_positions_every_cycle(
    pr2_reading_its_own_parts,
):
    """
    A closed-loop cycle integrates the commanded velocities, so it has to be pulled back
    onto the last measurement in every cycle rather than reading each message once.
    """
    loop_inputs = (
        pr2_reading_its_own_parts.motion_server.control_loop.inputs.synchronizers
    )

    assert any(
        isinstance(synchronizer, LatestJointPositionSource)
        for synchronizer in loop_inputs
    )
    assert not any(
        isinstance(synchronizer, PendingJointPositionSource)
        for synchronizer in loop_inputs
    )


def test_the_base_pose_is_read_by_both_loops_from_one_source(
    pr2_reading_its_own_parts,
):
    """
    Odometry reports a pose rather than integrating one, so the same source serves both
    loops and nothing has to be read twice.
    """
    base_source = pr2_reading_its_own_parts.robot.mobile_base.source
    motion_server = pr2_reading_its_own_parts.motion_server

    assert isinstance(base_source, SubscribedBasePoseSource)
    assert any(
        synchronizer is base_source
        for synchronizer in motion_server.inputs.synchronizers
    )
    assert any(
        synchronizer is base_source
        for synchronizer in motion_server.control_loop.inputs.synchronizers
    )


@dataclass
class BaseReadingInterface(GroupCommandingInterface):
    """
    An interface that reads only the robot's base from the robot itself, leaving its
    joints to whatever else reads them.
    """

    def setup(self):
        super().setup()
        self.sync_robot_part(self.robot.mobile_base)


@pytest.fixture()
def pr2_reading_only_its_base(init_rospy) -> Giskard:
    """
    A closed-loop Giskard whose PR2 reads only its base from what the robot publishes.
    """
    giskard = Giskard(
        world_config=WorldWithPR2Config(urdf=load_xacro(PR2.get_ros_file_path())),
        robot_interface_config=BaseReadingInterface(joint_names=[PR2Joint.HEAD_PAN]),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )
    giskard.setup()
    return giskard


def test_syncing_one_part_applies_that_part(pr2_reading_only_its_base):
    base_source = pr2_reading_only_its_base.robot.mobile_base.source
    motion_server = pr2_reading_only_its_base.motion_server

    assert isinstance(base_source, SubscribedBasePoseSource)
    assert [
        synchronizer
        for synchronizer in motion_server.inputs.synchronizers
        if synchronizer is base_source
    ] == [base_source]


def test_syncing_one_part_leaves_the_other_parts_reading_the_world(
    pr2_reading_only_its_base,
):
    arm = pr2_reading_only_its_base.robot.left_arm

    assert isinstance(arm.source, SimulatedJointPositionSource)


# %% reading the robot from a joint state topic


@dataclass
class ControlLoopMimic:
    """
    Holds the inputs a control loop reads in every cycle.
    """

    inputs: WorldStateInputs
    """
    The inputs read before every tick of the controller.
    """


@dataclass
class MotionServerMimic:
    """
    Holds the inputs read between goals and the control loop that executes a goal.
    """

    inputs: WorldStateInputs
    """
    The inputs read while no goal is executed.
    """

    control_loop: ControlLoopMimic
    """
    The loop that executes a goal.
    """


@dataclass
class ContextMimic:
    """
    Holds the world a motion is executed in.
    """

    world: World
    """
    The world whose state the inputs write.
    """


@dataclass
class ExecutorMimic:
    """
    Holds the context of the motion.
    """

    context: ContextMimic
    """
    The context whose world the inputs write.
    """


@dataclass
class RobotMimic:
    """
    A robot that is only known by its name.
    """

    name: str
    """
    The name that joint state topics are grouped under.
    """


@dataclass
class GiskardMimic:
    """
    Offers the parts of Giskard that a robot interface configures.
    """

    executor: ExecutorMimic
    """
    The executor whose world the inputs write.
    """

    robot: RobotMimic
    """
    The robot the interface talks to.
    """

    motion_server: MotionServerMimic
    """
    The server whose inputs the interface registers.
    """


@dataclass
class JointStateTopicInterface(ClosedLoopRobotInterfaceConfig):
    """
    Reads the state of the robot from a single joint state topic.
    """

    def setup(self):
        self.sync_joint_state_topic("joint_states")


def test_closed_loop_does_not_rewrite_a_joint_state_it_already_wrote(
    init_rospy, mini_world: World
):
    [connection] = mini_world.connections
    control_loop_inputs = WorldStateInputs(world=mini_world)
    interface = JointStateTopicInterface()
    interface.attach(
        GiskardMimic(
            executor=ExecutorMimic(context=ContextMimic(world=mini_world)),
            robot=RobotMimic(name="robot"),
            motion_server=MotionServerMimic(
                inputs=WorldStateInputs(world=mini_world),
                control_loop=ControlLoopMimic(inputs=control_loop_inputs),
            ),
        )
    )
    interface.setup()
    [synchronizer] = control_loop_inputs.synchronizers
    synchronizer.latest_message = joint_state_message(connection.name.name, 0.42)

    control_loop_inputs.apply_inputs()
    mini_world.state[connection.raw_dof.id].position = 1.0
    control_loop_inputs.apply_inputs()

    assert mini_world.state[connection.raw_dof.id].position == 1.0
