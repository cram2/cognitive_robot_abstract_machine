from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.joint_state import JointState
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
)
from ..spatial_types import Quaternion, Vector3
from ..world import World
from ..world_description.connections import FixedConnection


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
                sensors=[camera],
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

            left_arm_park = JointState.from_mapping(
                name=PrefixedName("left_arm_park", prefix=armar.name.name),
                mapping=dict(
                    zip(
                        [c for c in left_arm.connections if type(c) != FixedConnection],
                        [-0.15, 0.0, 0.0, 1.5, 0.5, 2.0, 1.5, 0.0, 0.0],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            left_arm.add_joint_state(left_arm_park)

            right_arm_park = JointState.from_mapping(
                name=PrefixedName("right_arm_park", prefix=armar.name.name),
                mapping=dict(
                    zip(
                        [
                            c
                            for c in right_arm.connections
                            if type(c) != FixedConnection
                        ],
                        [-0.15, 0.0, 0.0, 1.5, 2.64, 2.0, 1.6415, 0.0, 0.0],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            right_arm.add_joint_state(right_arm_park)

            left_gripper_joints = [
                c for c in left_gripper.connections if type(c) != FixedConnection
            ]

            left_gripper_open = JointState.from_mapping(
                name=PrefixedName("left_gripper_open", prefix=armar.name.name),
                mapping=dict(
                    zip(
                        left_gripper_joints,
                        [0] * len(left_gripper_joints),
                    )
                ),
                state_type=GripperState.OPEN,
            )

            left_gripper_close = JointState.from_mapping(
                name=PrefixedName("left_gripper_close", prefix=armar.name.name),
                mapping=dict(
                    zip(
                        left_gripper_joints,
                        [1.57] * len(left_gripper_joints),
                    )
                ),
                state_type=GripperState.CLOSE,
            )

            left_gripper.add_joint_state(left_gripper_close)
            left_gripper.add_joint_state(left_gripper_open)

            right_gripper_joints = [
                c for c in right_gripper.connections if type(c) != FixedConnection
            ]

            right_gripper_open = JointState.from_mapping(
                name=PrefixedName("right_gripper_open", prefix=armar.name.name),
                mapping=dict(
                    zip(
                        right_gripper_joints,
                        [0] * len(right_gripper_joints),
                    )
                ),
                state_type=GripperState.OPEN,
            )

            right_gripper_close = JointState.from_mapping(
                name=PrefixedName("right_gripper_close", prefix=armar.name.name),
                mapping=dict(
                    zip(
                        right_gripper_joints,
                        [1.57] * len(right_gripper_joints),
                    )
                ),
                state_type=GripperState.CLOSE,
            )

            right_gripper.add_joint_state(right_gripper_close)
            right_gripper.add_joint_state(right_gripper_open)

            torso_joint = [world.get_connection_by_name("torso_joint")]

            torso_low = JointState.from_mapping(
                name=PrefixedName("torso_low", prefix=armar.name.name),
                mapping=dict(zip(torso_joint, [-0.365])),
                state_type=TorsoState.LOW,
            )

            torso_mid = JointState.from_mapping(
                name=PrefixedName("torso_mid", prefix=armar.name.name),
                mapping=dict(zip(torso_joint, [-0.185])),
                state_type=TorsoState.MID,
            )

            torso_high = JointState.from_mapping(
                name=PrefixedName("torso_high", prefix=armar.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.HIGH,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            world.add_semantic_annotation(armar)

        return armar
