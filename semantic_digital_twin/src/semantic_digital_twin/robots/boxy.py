from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
    Neck,
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
    Torso,
    AbstractRobot,
    JointState,
)
from ..spatial_types import Quaternion, Vector3
from ..world import World


@dataclass(eq=False)
class Boxy(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Class that describes the Boxy Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the Boxy robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a Boxy robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A Boxy robot view.
        """

        with world.modify_world():
            boxy = cls(
                name=PrefixedName(name="boxy", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            # Create left arm
            left_gripper_thumb = Finger(
                name=PrefixedName("left_gripper_thumb", prefix=boxy.name.name),
                root=world.get_body_by_name("left_gripper_gripper_left_link"),
                tip=world.get_body_by_name("left_gripper_finger_right_link"),
                _world=world,
            )

            left_gripper_finger = Finger(
                name=PrefixedName("left_gripper_finger", prefix=boxy.name.name),
                root=world.get_body_by_name("left_gripper_gripper_left_link"),
                tip=world.get_body_by_name("left_gripper_finger_left_link"),
                _world=world,
            )

            left_gripper = ParallelGripper(
                name=PrefixedName("left_gripper", prefix=boxy.name.name),
                root=world.get_body_by_name("left_gripper_base_link"),
                tool_frame=world.get_body_by_name("left_gripper_tool_frame"),
                front_facing_orientation=Quaternion(1, 0, 1, 0),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=left_gripper_thumb,
                finger=left_gripper_finger,
                _world=world,
            )
            left_arm = Arm(
                name=PrefixedName("left_arm", prefix=boxy.name.name),
                root=world.get_body_by_name("calib_left_arm_base_link"),
                tip=world.get_body_by_name("left_arm_7_link"),
                manipulator=left_gripper,
                _world=world,
            )

            boxy.add_arm(left_arm)

            # Create right arm
            right_gripper_thumb = Finger(
                name=PrefixedName("right_gripper_thumb", prefix=boxy.name.name),
                root=world.get_body_by_name("right_gripper_gripper_left_link"),
                tip=world.get_body_by_name("right_gripper_finger_right_link"),
                _world=world,
            )
            right_gripper_finger = Finger(
                name=PrefixedName("right_gripper_finger", prefix=boxy.name.name),
                root=world.get_body_by_name("right_gripper_gripper_left_link"),
                tip=world.get_body_by_name("right_gripper_finger_left_link"),
                _world=world,
            )
            right_gripper = ParallelGripper(
                name=PrefixedName("right_gripper", prefix=boxy.name.name),
                root=world.get_body_by_name("right_gripper_base_link"),
                tool_frame=world.get_body_by_name("right_gripper_tool_frame"),
                front_facing_orientation=Quaternion(1, 0, 1, 0),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=right_gripper_thumb,
                finger=right_gripper_finger,
                _world=world,
            )
            right_arm = Arm(
                name=PrefixedName("right_arm", prefix=boxy.name.name),
                root=world.get_body_by_name("calib_right_arm_base_link"),
                tip=world.get_body_by_name("right_arm_7_link"),
                manipulator=right_gripper,
                _world=world,
            )

            boxy.add_arm(right_arm)

            # Create camera and neck
            camera = Camera(
                name=PrefixedName("head_mount_kinect2_rgb_optical_frame", prefix=boxy.name.name),
                root=world.get_body_by_name("head_mount_kinect2_rgb_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=2.5,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=boxy.name.name),
                sensors={camera},
                root=world.get_body_by_name("neck_base_link"),
                tip=world.get_body_by_name("neck_wrist_3_link"),
                _world=world,
            )
            boxy.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=boxy.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("triangle_base_link"),
                _world=world,
            )
            boxy.add_torso(torso)

            #Create states
            left_arm_park = JointState(
                name=PrefixedName("left_arm_park", prefix=boxy.name.name),
                joints=[world.get_connection_by_name("left_arm_0_joint"), world.get_connection_by_name("left_arm_1_joint"),
                             world.get_connection_by_name("left_arm_2_joint"), world.get_connection_by_name("left_arm_3_joint"),
                             world.get_connection_by_name("left_arm_4_joint"), world.get_connection_by_name("left_arm_5_joint"),
                             world.get_connection_by_name("left_arm_6_joint")],
                joint_positions=[-1.858, 0.70571, 0.9614, -0.602, -2.5922, -1.94065, -1.28735],
                state_type=StaticJointState.PARK,
                kinematic_chains=[left_arm],
                _world=world,
            )

            right_arm_park = JointState(
                name=PrefixedName("right_arm_park", prefix=boxy.name.name),
                joints=[world.get_connection_by_name("right_arm_0_joint"), world.get_connection_by_name("right_arm_1_joint"),
                             world.get_connection_by_name("right_arm_2_joint"), world.get_connection_by_name("right_arm_3_joint"),
                             world.get_connection_by_name("right_arm_4_joint"), world.get_connection_by_name("right_arm_5_joint"),
                             world.get_connection_by_name("right_arm_6_joint")],
                joint_positions=[1.858, -0.70571, -0.9614, 0.602, 2.5922, 1.94065, 1.28735],
                state_type=StaticJointState.PARK,
                kinematic_chains=[right_arm],
                _world=world,
            )

            left_gripper_joint = [world.get_connection_by_name("left_gripper_joint")]

            left_gripper_open = JointState(
                name=PrefixedName("left_gripper_open", prefix=boxy.name.name),
                joints=left_gripper_joint,
                joint_positions=[0.548],
                state_type=GripperState.OPEN,
                kinematic_chains=[left_gripper],
                _world=world,
            )

            left_gripper_close = JointState(
                name=PrefixedName("left_gripper_close", prefix=boxy.name.name),
                joints=left_gripper_joint,
                joint_positions=[0.0],
                state_type=GripperState.CLOSE,
                kinematic_chains=[left_gripper],
                _world=world,
            )

            right_gripper_joint = [world.get_connection_by_name("right_gripper_joint")]

            right_gripper_open = JointState(
                name=PrefixedName("right_gripper_open", prefix=boxy.name.name),
                joints=right_gripper_joint,
                joint_positions=[0.548],
                state_type=GripperState.OPEN,
                kinematic_chains=[right_gripper],
                _world=world,
            )

            right_gripper_close = JointState(
                name=PrefixedName("right_gripper_close", prefix=boxy.name.name),
                joints=right_gripper_joint,
                joint_positions=[0.0],
                state_type=GripperState.CLOSE,
                kinematic_chains=[right_gripper],
                _world=world,
            )

            torso_joint = [world.get_connection_by_name("triangle_base_joint")]

            torso_low = JointState(
                name=PrefixedName("torso_low", prefix=boxy.name.name),
                joints=torso_joint,
                joint_positions=[-0.58],
                state_type=TorsoState.LOW,
                kinematic_chains=[torso],
                _world=world,
            )

            torso_mid = JointState(
                name=PrefixedName("torso_mid", prefix=boxy.name.name),
                joints=torso_joint,
                joint_positions=[0.29],
                state_type=TorsoState.MID,
                kinematic_chains=[torso],
                _world=world,
            )

            torso_high = JointState(
                name=PrefixedName("torso_high", prefix=boxy.name.name),
                joints=torso_joint,
                joint_positions=[0.0],
                state_type=TorsoState.HIGH,
                kinematic_chains=[torso],
                _world=world,
            )

            boxy.add_joint_states([left_arm_park, right_arm_park, left_gripper_open, left_gripper_close,
                                   right_gripper_open, right_gripper_close, torso_low, torso_mid, torso_high])

            world.add_semantic_annotation(boxy)

        return boxy
