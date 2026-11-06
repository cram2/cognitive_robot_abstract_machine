from dataclasses import dataclass, fields, is_dataclass

import pytest

from giskardpy.data_types.exceptions import NoControlledJointsError
from cramph.executor import NoPacing, RealTimePacer
from giskardpy.middleware.ros2.exceptions import NoUniqueDriveError
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from semantic_digital_twin.adapters.ros.input_synchronization import (
    LatestJointPositionSource,
    PendingJointPositionSource,
    SubscribedBasePoseSource,
)
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.robots.input_source import SimulatedJointPositionSource
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_source import WorldSource
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


# %% the tf frame synchronizer is created on demand


def test_an_interface_starts_without_a_tf_frame_synchronizer():
    assert GroupCommandingInterface(joint_names=[]).tf_frame_synchronizer is None


# %% attaching binds the giskard instance the accessors read from


def test_attaching_lets_the_interface_reach_the_server_config():
    server_config = GiskardServerConfig()
    interface = StandAloneRobotInterfaceConfig()
    giskard = Giskard(
        world_source=WorldSpecification(),
        server_config=server_config,
        robot_interface_config=interface,
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    interface.attach(giskard)

    assert interface.server_config is server_config


# %% the interface decides how the control loop is paced


def test_a_standalone_interface_is_not_paced() -> None:
    assert isinstance(StandAloneRobotInterfaceConfig().create_pacer(), NoPacing)


def test_a_closed_loop_interface_runs_in_real_time() -> None:
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


def set_up_standalone(world_source: WorldSource) -> Giskard:
    """
    A Giskard that simulates the robot of ``world_source`` and is ready for goals.
    """
    giskard = Giskard(
        world_source=world_source,
        robot_interface_config=StandAloneRobotInterfaceConfig(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )
    giskard.setup()
    return giskard


def test_standalone_controls_the_joints_tracy_declares(
    init_rospy, tracy_world, tracy_world_specification
) -> None:
    giskard = set_up_standalone(tracy_world_specification)

    assert controlled_joint_names(giskard.world) == (
        controlled_joint_names(tracy_world)
    )


def test_standalone_controls_the_joints_daisy_declares(
    init_rospy, daisy_world, daisy_world_specification
) -> None:
    giskard = set_up_standalone(daisy_world_specification)

    assert controlled_joint_names(giskard.world) == (
        controlled_joint_names(daisy_world)
    )


def test_standalone_controls_the_joints_the_pr2_declares(
    init_rospy, pr2_world_copy, pr2_world_specification
) -> None:
    giskard = set_up_standalone(pr2_world_specification)

    assert controlled_joint_names(giskard.world) == (
        controlled_joint_names(pr2_world_copy)
    )


def test_standalone_controls_the_joints_the_hsr_declares(
    init_rospy, _hsr_world_setup, hsr_world_specification
) -> None:
    giskard = set_up_standalone(hsr_world_specification)

    assert controlled_joint_names(giskard.world) == (
        controlled_joint_names(_hsr_world_setup)
    )


def test_standalone_controls_the_joints_stretch_declares(
    init_rospy, stretch_world_copy, stretch_world_specification
) -> None:
    giskard = set_up_standalone(stretch_world_specification)

    assert controlled_joint_names(giskard.world) == (
        controlled_joint_names(stretch_world_copy)
    )


# %% the world specification flags the drive it creates


def test_a_world_with_an_omni_drive_robot_controls_its_drive(
    pr2_world_specification,
) -> None:
    world = pr2_world_specification.create_world()

    [drive] = world.get_connections_by_type(OmniDrive)
    assert drive.is_controlled


def test_a_world_with_a_differential_drive_robot_controls_its_drive(
    stretch_world_specification,
) -> None:
    world = stretch_world_specification.create_world()

    [drive] = world.get_connections_by_type(DifferentialDrive)
    assert drive.is_controlled


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

    def setup(self) -> None:
        self.add_joint_group_velocity_command_topic(
            command_topic="group_velocity_commands", connections=self.joint_names
        )


def closed_loop_pr2(
    interface: ClosedLoopRobotInterfaceConfig,
    pr2_world_specification: WorldSpecification,
) -> Giskard:
    """
    A Giskard that commands a PR2 through ``interface``, not set up yet.
    """
    return Giskard(
        world_source=pr2_world_specification,
        robot_interface_config=interface,
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )


def test_a_closed_loop_interface_that_commands_nothing_controls_no_joint(
    init_rospy,
    pr2_world_specification,
) -> None:
    giskard = closed_loop_pr2(
        GroupCommandingInterface(joint_names=[]), pr2_world_specification
    )

    with pytest.raises(NoControlledJointsError):
        giskard.setup()


def test_closed_loop_controls_exactly_the_commanded_joints(
    init_rospy, pr2_world_specification
) -> None:
    interface = GroupCommandingInterface(
        joint_names=[PR2Joint.HEAD_PAN, PR2Joint.HEAD_TILT]
    )
    giskard = closed_loop_pr2(interface, pr2_world_specification)

    giskard.setup()

    assert controlled_joint_names(giskard.world) == set(interface.joint_names)


def test_closed_loop_does_not_control_a_drive_nothing_commands(
    init_rospy, pr2_world_specification
) -> None:
    giskard = closed_loop_pr2(
        GroupCommandingInterface(joint_names=[PR2Joint.HEAD_PAN]),
        pr2_world_specification,
    )

    giskard.setup()

    [drive] = giskard.world.get_connections_by_type(OmniDrive)
    assert not drive.is_controlled


# %% commanding the drive


@dataclass
class DriveCommandingInterface(ClosedLoopRobotInterfaceConfig):
    """
    An interface that commands the drive of the robot without naming the drive.
    """

    def setup(self) -> None:
        self.add_drive_velocity_command_topic(command_topic="drive_velocity_commands")


def test_a_drive_command_without_a_drive_commands_the_only_drive_of_the_world(
    init_rospy,
    pr2_world_specification,
) -> None:
    giskard = closed_loop_pr2(DriveCommandingInterface(), pr2_world_specification)

    giskard.setup()

    [drive] = giskard.world.get_connections_by_type(OmniDrive)
    [publisher] = giskard.motion_server.control_loop.command_publishers
    assert publisher.connection is drive


def test_a_drive_command_without_a_drive_in_a_world_without_a_drive_raises(
    init_rospy,
    tracy_world_specification,
) -> None:
    giskard = Giskard(
        world_source=tracy_world_specification,
        robot_interface_config=DriveCommandingInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )

    with pytest.raises(NoUniqueDriveError):
        giskard.setup()


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
def pr2_reading_its_own_parts(init_rospy, pr2_world_specification) -> Giskard:
    """
    A closed-loop Giskard whose PR2 is read from what the robot publishes.
    """
    giskard = Giskard(
        world_source=pr2_world_specification,
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
def pr2_reading_only_its_base(init_rospy, pr2_world_specification) -> Giskard:
    """
    A closed-loop Giskard whose PR2 reads only its base from what the robot publishes.
    """
    giskard = Giskard(
        world_source=pr2_world_specification,
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
