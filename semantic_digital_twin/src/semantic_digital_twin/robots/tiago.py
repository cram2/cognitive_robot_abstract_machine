from dataclasses import dataclass
from typing import Self

from .abstract_robot import (
    AbstractRobot,
    Arm,
    Neck,
    Finger,
    ParallelGripper,
    Camera,
    Torso,
    FieldOfView,
    Base,
)
from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World
from ..world_description.connections import FixedConnection


@dataclass(eq=False)
class Tiago(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Class that describes the Take It And Go Robot (TIAGo).
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the TIAGo robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a TIAGo robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A TIAGo robot view.
        """

        with world.modify_world():
            tiago = cls(
                name=PrefixedName("tiago", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            # Create left arm
            left_gripper_thumb = Finger(
                name=PrefixedName("left_gripper_thumb", prefix=tiago.name.name),
                root=world.get_body_by_name("gripper_left_link"),
                tip=world.get_body_by_name("gripper_left_left_finger_link"),
                _world=world,
            )

            left_gripper_finger = Finger(
                name=PrefixedName("left_gripper_finger", prefix=tiago.name.name),
                root=world.get_body_by_name("gripper_left_link"),
                tip=world.get_body_by_name("gripper_left_right_finger_link"),
                _world=world,
            )

            left_gripper = ParallelGripper(
                name=PrefixedName("left_gripper", prefix=tiago.name.name),
                root=world.get_body_by_name("gripper_left_link"),
                tool_frame=world.get_body_by_name("gripper_left_grasping_frame"),
                front_facing_orientation=Quaternion(0, 0, 0, 1),
                front_facing_axis=Vector3(1, 0, 0),
                thumb=left_gripper_thumb,
                finger=left_gripper_finger,
                _world=world,
            )
            left_arm = Arm(
                name=PrefixedName("left_arm", prefix=tiago.name.name),
                root=world.get_body_by_name("torso_lift_link"),
                tip=world.get_body_by_name("arm_left_7_link"),
                manipulator=left_gripper,
                _world=world,
            )

            tiago.add_arm(left_arm)

            # Create right arm
            right_gripper_thumb = Finger(
                name=PrefixedName("right_gripper_thumb", prefix=tiago.name.name),
                root=world.get_body_by_name("gripper_right_link"),
                tip=world.get_body_by_name("gripper_right_left_finger_link"),
                _world=world,
            )
            right_gripper_finger = Finger(
                name=PrefixedName("right_gripper_finger", prefix=tiago.name.name),
                root=world.get_body_by_name("gripper_right_link"),
                tip=world.get_body_by_name("gripper_right_right_finger_link"),
                _world=world,
            )
            right_gripper = ParallelGripper(
                name=PrefixedName("right_gripper", prefix=tiago.name.name),
                root=world.get_body_by_name("gripper_right_link"),
                tool_frame=world.get_body_by_name("gripper_right_grasping_frame"),
                front_facing_orientation=Quaternion(0, 0, 0, 1),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=right_gripper_thumb,
                finger=right_gripper_finger,
                _world=world,
            )
            right_arm = Arm(
                name=PrefixedName("right_arm", prefix=tiago.name.name),
                root=world.get_body_by_name("torso_lift_link"),
                tip=world.get_body_by_name("arm_right_7_link"),
                manipulator=right_gripper,
                _world=world,
            )

            tiago.add_arm(right_arm)

            # Create camera and neck
            camera = Camera(
                name=PrefixedName("xtion_optical_frame", prefix=tiago.name.name),
                root=world.get_body_by_name("xtion_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=1.0665,
                maximal_height=1.4165,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=tiago.name.name),
                sensors=[camera],
                root=world.get_body_by_name("torso_lift_link"),
                tip=world.get_body_by_name("head_2_link"),
                pitch_body=world.get_body_by_name("head_2_link"),
                yaw_body=world.get_body_by_name("head_1_link"),
                _world=world,
            )
            tiago.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=tiago.name.name),
                root=world.get_body_by_name("torso_fixed_link"),
                tip=world.get_body_by_name("torso_lift_link"),
                _world=world,
            )
            tiago.add_torso(torso)

            # Create states
            left_arm_park = JointState.from_mapping(
                name=PrefixedName("left_arm_park", prefix=tiago.name.name),
                mapping=dict(
                    zip(
                        [c for c in left_arm.connections if type(c) != FixedConnection],
                        [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            left_arm.add_joint_state(left_arm_park)

            right_arm_park = JointState.from_mapping(
                name=PrefixedName("right_arm_park", prefix=tiago.name.name),
                mapping=dict(
                    zip(
                        [
                            c
                            for c in right_arm.connections
                            if type(c) != FixedConnection
                        ],
                        [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            right_arm.add_joint_state(right_arm_park)

            left_gripper_joints = [
                world.get_connection_by_name("gripper_left_left_finger_joint"),
                world.get_connection_by_name("gripper_left_right_finger_joint"),
            ]

            left_gripper_open = JointState.from_mapping(
                name=PrefixedName("left_gripper_open", prefix=tiago.name.name),
                mapping=dict(zip(left_gripper_joints, [0.044, 0.044])),
                state_type=GripperState.OPEN,
            )

            left_gripper_close = JointState.from_mapping(
                name=PrefixedName("left_gripper_close", prefix=tiago.name.name),
                mapping=dict(zip(left_gripper_joints, [0.0, 0.0])),
                state_type=GripperState.CLOSE,
            )

            left_gripper.add_joint_state(left_gripper_close)
            left_gripper.add_joint_state(left_gripper_open)

            right_gripper_joints = [
                world.get_connection_by_name("gripper_right_left_finger_joint"),
                world.get_connection_by_name("gripper_right_right_finger_joint"),
            ]

            right_gripper_open = JointState.from_mapping(
                name=PrefixedName("right_gripper_open", prefix=tiago.name.name),
                mapping=dict(zip(right_gripper_joints, [0.044, 0.044])),
                state_type=GripperState.OPEN,
            )

            right_gripper_close = JointState.from_mapping(
                name=PrefixedName("right_gripper_close", prefix=tiago.name.name),
                mapping=dict(zip(right_gripper_joints, [0.0, 0.0])),
                state_type=GripperState.CLOSE,
            )

            right_gripper.add_joint_state(right_gripper_close)
            right_gripper.add_joint_state(right_gripper_open)

            torso_joint = [world.get_connection_by_name("torso_lift_joint")]

            torso_low = JointState.from_mapping(
                name=PrefixedName("torso_low", prefix=tiago.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.LOW,
            )

            torso_mid = JointState.from_mapping(
                name=PrefixedName("torso_mid", prefix=tiago.name.name),
                mapping=dict(zip(torso_joint, [0.15])),
                state_type=TorsoState.MID,
            )

            torso_high = JointState.from_mapping(
                name=PrefixedName("torso_high", prefix=tiago.name.name),
                mapping=dict(zip(torso_joint, [0.35])),
                state_type=TorsoState.HIGH,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            # Create the robot base
            base = Base(
                name=PrefixedName("base", prefix=tiago.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("base_link"),
                _world=world,
            )

            tiago.add_base(base)

            world.add_semantic_annotation(tiago)

        return tiago
