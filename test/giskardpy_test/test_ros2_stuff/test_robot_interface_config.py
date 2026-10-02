from dataclasses import dataclass, fields, is_dataclass

import pytest

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    DaisyStandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import (
    PR2VelocityMujocoInterface,
    WorldWithPR2Config,
)
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchStandaloneInterface,
)
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    TracyStandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.model.world_config import EmptyWorld
from semantic_digital_twin.adapters.ros.input_synchronization import (
    LatestJointPositionSource,
    PendingJointPositionSource,
    SubscribedBasePoseSource,
)
from semantic_digital_twin.robots.input_source import SimulatedJointPositionSource
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.daisy import DAiSyJoint
from semantic_digital_twin.robots.stretch import StretchJoint
from semantic_digital_twin.robots.tracy import TracyJoint
from giskardpy.qp.qp_controller_config import QPControllerConfig

# %% the interface hierarchy is built from dataclasses


def test_the_robot_interface_base_is_a_dataclass():
    assert is_dataclass(RobotInterfaceConfig)


def test_the_interface_state_bound_after_construction_is_no_constructor_argument():
    assert [field.name for field in fields(RobotInterfaceConfig) if field.init] == []


def test_the_standalone_interface_takes_the_joint_names_positionally():
    interface = StandAloneRobotInterfaceConfig(["torso_lift_joint", "head_pan_joint"])

    assert interface.joint_names == ["torso_lift_joint", "head_pan_joint"]


def test_two_interfaces_with_the_same_joint_names_are_equal():
    assert StandAloneRobotInterfaceConfig(["head_pan_joint"]) == (
        StandAloneRobotInterfaceConfig(["head_pan_joint"])
    )


def test_the_mujoco_interface_defaults_name_the_frames_its_localization_uses():
    interface = PR2VelocityMujocoInterface()

    assert (
        interface.map_name,
        interface.localization_joint_name,
        interface.odom_link_name,
    ) == ("map", "localization", "odom_combined")


# %% the tf frame synchronizer is created on demand


def test_an_interface_starts_without_a_tf_frame_synchronizer():
    assert StandAloneRobotInterfaceConfig([]).tf_frame_synchronizer is None


# %% attaching binds the giskard instance the accessors read from


def test_attaching_lets_the_interface_reach_the_server_config():
    server_config = GiskardServerConfig()
    interface = StandAloneRobotInterfaceConfig([])
    giskard = Giskard(
        world_config=EmptyWorld(),
        server_config=server_config,
        robot_interface_config=interface,
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    interface.attach(giskard)

    assert interface.server_config is server_config


# %% robot specific interfaces declare their controlled joints


def test_the_tracy_interface_controls_both_arms():
    assert TracyStandAloneRobotInterfaceConfig().joint_names == [
        TracyJoint.LEFT_SHOULDER_PAN,
        TracyJoint.LEFT_SHOULDER_LIFT,
        TracyJoint.LEFT_ELBOW,
        TracyJoint.LEFT_WRIST_1,
        TracyJoint.LEFT_WRIST_2,
        TracyJoint.LEFT_WRIST_3,
        TracyJoint.RIGHT_SHOULDER_PAN,
        TracyJoint.RIGHT_SHOULDER_LIFT,
        TracyJoint.RIGHT_ELBOW,
        TracyJoint.RIGHT_WRIST_1,
        TracyJoint.RIGHT_WRIST_2,
        TracyJoint.RIGHT_WRIST_3,
    ]


def test_the_daisy_interface_controls_both_arms_and_both_grippers():
    assert DaisyStandAloneRobotInterfaceConfig().joint_names == [
        DAiSyJoint.LEFT_SHOULDER_PAN,
        DAiSyJoint.LEFT_SHOULDER_LIFT,
        DAiSyJoint.LEFT_ELBOW,
        DAiSyJoint.LEFT_WRIST_1,
        DAiSyJoint.LEFT_WRIST_2,
        DAiSyJoint.LEFT_WRIST_3,
        DAiSyJoint.RIGHT_SHOULDER_PAN,
        DAiSyJoint.RIGHT_SHOULDER_LIFT,
        DAiSyJoint.RIGHT_ELBOW,
        DAiSyJoint.RIGHT_WRIST_1,
        DAiSyJoint.RIGHT_WRIST_2,
        DAiSyJoint.RIGHT_WRIST_3,
        DAiSyJoint.LEFT_GRIPPER_FINGER,
        DAiSyJoint.LEFT_GRIPPER_RIGHT_FINGER,
        DAiSyJoint.RIGHT_GRIPPER_FINGER,
        DAiSyJoint.RIGHT_GRIPPER_RIGHT_FINGER,
    ]


def test_two_daisy_interfaces_do_not_share_their_joint_name_list():
    first = DaisyStandAloneRobotInterfaceConfig()
    second = DaisyStandAloneRobotInterfaceConfig()

    assert first.joint_names is not second.joint_names


def test_the_stretch_interface_controls_every_joint_except_the_drive():
    assert StretchStandaloneInterface().joint_names == [
        StretchJoint.GRIPPER_LEFT_FINGER,
        StretchJoint.GRIPPER_RIGHT_FINGER,
        StretchJoint.RIGHT_WHEEL,
        StretchJoint.LEFT_WHEEL,
        StretchJoint.LIFT,
        StretchJoint.ARM_L3,
        StretchJoint.ARM_L2,
        StretchJoint.ARM_L1,
        StretchJoint.ARM_L0,
        StretchJoint.WRIST_YAW,
        StretchJoint.HEAD_PAN,
        StretchJoint.HEAD_TILT,
    ]


# %% reading the robot's own parts from the robot


@dataclass
class PartReadingInterface(RobotInterfaceConfig):
    """
    An interface that reads every part of the robot from the robot itself, on the topics
    the parts declare.
    """

    def setup(self):
        self.sync_robot_parts()


@pytest.fixture()
def pr2_reading_its_own_parts(init_rospy) -> Giskard:
    """
    A closed-loop Giskard whose PR2 is read from what the robot publishes.
    """
    giskard = Giskard(
        world_config=WorldWithPR2Config(urdf=load_xacro(PR2.get_ros_file_path())),
        robot_interface_config=PartReadingInterface(),
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
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
class BaseReadingInterface(RobotInterfaceConfig):
    """
    An interface that reads only the robot's base from the robot itself, leaving its
    joints to whatever else reads them.
    """

    def setup(self):
        self.sync_robot_part(self.robot.mobile_base)


@pytest.fixture()
def pr2_reading_only_its_base(init_rospy) -> Giskard:
    """
    A closed-loop Giskard whose PR2 reads only its base from what the robot publishes.
    """
    giskard = Giskard(
        world_config=WorldWithPR2Config(urdf=load_xacro(PR2.get_ros_file_path())),
        robot_interface_config=BaseReadingInterface(),
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
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
