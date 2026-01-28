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
class Armar(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Class that describes the Armar Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the Armar robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates an Armar robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: An Armar robot view.
        """

        with world.modify_world():
            armar = cls(
                name=PrefixedName(name="armar", prefix=world.name),
                root=world.get_body_by_name("platform"),
                _world=world,
            )

            # Create left arm
            left_gripper_thumb = Finger(
                name=PrefixedName("left_gripper_thumb", prefix=armar.name.name),
                root=world.get_body_by_name("Thumb L 1"),
                tip=world.get_body_by_name("Thumb L 2"),
                _world=world,
            )

            left_gripper_finger = Finger(
                name=PrefixedName("left_gripper_finger", prefix=armar.name.name),
                root=world.get_body_by_name("Index L 1"),
                tip=world.get_body_by_name("Index L 3"),
                _world=world,
            )

            left_gripper = ParallelGripper(
                name=PrefixedName("left_gripper", prefix=armar.name.name),
                root=world.get_body_by_name("arm_t8_r0"),
                tool_frame=world.get_body_by_name("left_tool_frame"),
                front_facing_orientation=Quaternion(0.707, 0.707, 0.707, 0.707),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=left_gripper_thumb,
                finger=left_gripper_finger,
                _world=world,
            )
            left_arm = Arm(
                name=PrefixedName("left_arm", prefix=armar.name.name),
                root=world.get_body_by_name("torso"),
                tip=world.get_body_by_name("arm_t8_r0"),
                manipulator=left_gripper,
                _world=world,
            )

            armar.add_arm(left_arm)

            # Create right arm
            right_gripper_thumb = Finger(
                name=PrefixedName("right_gripper_thumb", prefix=armar.name.name),
                root=world.get_body_by_name("Thumb R 1"),
                tip=world.get_body_by_name("Thumb R 2"),
                _world=world,
            )
            right_gripper_finger = Finger(
                name=PrefixedName("right_gripper_finger", prefix=armar.name.name),
                root=world.get_body_by_name("Index R 1"),
                tip=world.get_body_by_name("Index R 3"),
                _world=world,
            )
            right_gripper = ParallelGripper(
                name=PrefixedName("right_gripper", prefix=armar.name.name),
                root=world.get_body_by_name("arm_t8_r1"),
                tool_frame=world.get_body_by_name("right_tool_frame"),
                front_facing_orientation=Quaternion(0.707, 0.707, 0.707, 0.707),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=right_gripper_thumb,
                finger=right_gripper_finger,
                _world=world,
            )
            right_arm = Arm(
                name=PrefixedName("right_arm", prefix=armar.name.name),
                root=world.get_body_by_name("torso"),
                tip=world.get_body_by_name("arm_t8_r1"),
                manipulator=right_gripper,
                _world=world,
            )

            armar.add_arm(right_arm)

            # Create camera and neck
            camera = Camera(
                name=PrefixedName("Roboception", prefix=armar.name.name),
                root=world.get_body_by_name("Roboception"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=1.371500015258789,
                maximal_height=1.7365000247955322,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=armar.name.name),
                sensors={camera},
                root=world.get_body_by_name("lower_neck_link"),
                tip=world.get_body_by_name("upper_neck_link"),
                pitch_body=world.get_body_by_name("neck_2_pitch_link"),
                yaw_body=world.get_body_by_name("neck_1_yaw_link"),
                _world=world,
            )
            armar.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=armar.name.name),
                root=world.get_body_by_name("platform"),
                tip=world.get_body_by_name("torso"),
                _world=world,
            )
            armar.add_torso(torso)

            left_arm_park = JointState(
                name=PrefixedName("left_arm_park", prefix=armar.name.name),
                joints=[world.get_connection_by_name("torso_joint"), world.get_connection_by_name("arm_t12_joint_r0"),
                             world.get_connection_by_name("arm_t23_joint_r0"), world.get_connection_by_name("arm_t34_joint_r0"),
                             world.get_connection_by_name("arm_t45_joint_r0"), world.get_connection_by_name("arm_t56_joint_r0"),
                             world.get_connection_by_name("arm_t67_joint_r0"), world.get_connection_by_name("arm_t78_joint_r0"),
                             world.get_connection_by_name("arm_t8_joint_r0")],
                joint_positions=[-0.15, 0.0, 0.0, 1.5, 0.5, 2.0, 1.5, 0.0, 0.0],
                state_type=StaticJointState.PARK,
                kinematic_chains=[left_arm],
                _world=world,
            )

            right_arm_park = JointState(
                name=PrefixedName("right_arm_park", prefix=armar.name.name),
                joints=[world.get_connection_by_name("torso_joint"), world.get_connection_by_name("arm_t12_joint_r1"),
                             world.get_connection_by_name("arm_t23_joint_r1"), world.get_connection_by_name("arm_t34_joint_r1"),
                             world.get_connection_by_name("arm_t45_joint_r1"), world.get_connection_by_name("arm_t56_joint_r1"),
                             world.get_connection_by_name("arm_t67_joint_r1"), world.get_connection_by_name("arm_t78_joint_r1"),
                             world.get_connection_by_name("arm_t8_joint_r1")],
                joint_positions=[-0.15, 0.0, 0.0, 1.5, 2.64, 2.0, 1.6415, 0.0, 0.0],
                state_type=StaticJointState.PARK,
                kinematic_chains=[left_arm],
                _world=world,
            )

            left_gripper_joints = [world.get_connection_by_name("Thumb L 1 Joint"), world.get_connection_by_name("Thumb L 2 Joint"),
                             world.get_connection_by_name("Index L 1 Joint"), world.get_connection_by_name("Index L 2 Joint"),
                             world.get_connection_by_name("Index L 3 Joint"), world.get_connection_by_name("Middle L 1 Joint"),
                             world.get_connection_by_name("Middle L 2 Joint"), world.get_connection_by_name("Middle L 3 Joint"),
                             world.get_connection_by_name("Ring L 1 Joint"), world.get_connection_by_name("Ring L 2 Joint"),
                             world.get_connection_by_name("Ring L 3 Joint"), world.get_connection_by_name("Pinky L 1 Joint"),
                             world.get_connection_by_name("Pinky L 2 Joint"), world.get_connection_by_name("Pinky L 3 Joint")]

            left_gripper_open = JointState(
                name=PrefixedName("left_gripper_open", prefix=armar.name.name),
                joints=left_gripper_joints,
                joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                state_type=GripperState.OPEN,
                kinematic_chains=[left_gripper],
                _world=world,
            )

            left_gripper_close = JointState(
                name=PrefixedName("left_gripper_close", prefix=armar.name.name),
                joints=left_gripper_joints,
                joint_positions=[1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57],
                state_type=GripperState.CLOSE,
                kinematic_chains=[left_gripper],
                _world=world,
            )

            right_gripper_joints = [world.get_connection_by_name("Thumb R 1 Joint"), world.get_connection_by_name("Thumb R 2 Joint"),
                             world.get_connection_by_name("Index R 1 Joint"), world.get_connection_by_name("Index R 2 Joint"),
                             world.get_connection_by_name("Index R 3 Joint"), world.get_connection_by_name("Middle R 1 Joint"),
                             world.get_connection_by_name("Middle R 2 Joint"), world.get_connection_by_name("Middle R 3 Joint"),
                             world.get_connection_by_name("Ring R 1 Joint"), world.get_connection_by_name("Ring R 2 Joint"),
                             world.get_connection_by_name("Ring R 3 Joint"), world.get_connection_by_name("Pinky R 1 Joint"),
                             world.get_connection_by_name("Pinky R 2 Joint"), world.get_connection_by_name("Pinky R 3 Joint")]

            right_gripper_open = JointState(
                name=PrefixedName("right_gripper_open", prefix=armar.name.name),
                joints=right_gripper_joints,
                joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                state_type=GripperState.OPEN,
                kinematic_chains=[right_gripper],
                _world=world,
            )

            right_gripper_close = JointState(
                name=PrefixedName("right_gripper_close", prefix=armar.name.name),
                joints=right_gripper_joints,
                joint_positions=[1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57],
                state_type=GripperState.CLOSE,
                kinematic_chains=[right_gripper],
                _world=world,
            )

            torso_joint = [world.get_connection_by_name("torso_joint")]

            torso_low = JointState(
                name=PrefixedName("torso_low", prefix=armar.name.name),
                joints=torso_joint,
                joint_positions=[-0.365],
                state_type=TorsoState.LOW,
                kinematic_chains=[torso],
                _world=world,
            )

            torso_mid = JointState(
                name=PrefixedName("torso_mid", prefix=armar.name.name),
                joints=torso_joint,
                joint_positions=[-0.185],
                state_type=TorsoState.MID,
                kinematic_chains=[torso],
                _world=world,
            )

            torso_high = JointState(
                name=PrefixedName("torso_high", prefix=armar.name.name),
                joints=torso_joint,
                joint_positions=[0.0],
                state_type=TorsoState.HIGH,
                kinematic_chains=[torso],
                _world=world,
            )

            armar.add_joint_states([left_arm_park, right_arm_park, left_gripper_open, left_gripper_close,
                                    right_gripper_open, right_gripper_close, torso_low, torso_mid, torso_high])

            world.add_semantic_annotation(armar)

        return armar
