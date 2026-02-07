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
                name=PrefixedName(
                    "head_mount_kinect2_rgb_optical_frame", prefix=boxy.name.name
                ),
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
                sensors=[camera],
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

            # Create states
            left_arm_park = JointState.from_mapping(
                name=PrefixedName("left_arm_park", prefix=boxy.name.name),
                mapping=dict(
                    zip(
                        [c for c in left_arm.connections if type(c) != FixedConnection],
                        [
                            -1.858,
                            0.70571,
                            0.9614,
                            -0.602,
                            -2.5922,
                            -1.94065,
                            -1.28735,
                        ],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            left_arm.add_joint_state(left_arm_park)

            right_arm_park = JointState.from_mapping(
                name=PrefixedName("right_arm_park", prefix=boxy.name.name),
                mapping=dict(
                    zip(
                        [
                            c
                            for c in right_arm.connections
                            if type(c) != FixedConnection
                        ],
                        [
                            1.858,
                            -0.70571,
                            -0.9614,
                            0.602,
                            2.5922,
                            1.94065,
                            1.28735,
                        ],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            right_arm.add_joint_state(right_arm_park)

            left_gripper_joint = [world.get_connection_by_name("left_gripper_joint")]

            left_gripper_open = JointState.from_mapping(
                name=PrefixedName("left_gripper_open", prefix=boxy.name.name),
                mapping=dict(zip(left_gripper_joint, [0.548])),
                state_type=GripperState.OPEN,
            )

            left_gripper_close = JointState.from_mapping(
                name=PrefixedName("left_gripper_close", prefix=boxy.name.name),
                mapping=dict(zip(left_gripper_joint, [0.0])),
                state_type=GripperState.CLOSE,
            )

            left_gripper.add_joint_state(left_gripper_close)
            left_gripper.add_joint_state(left_gripper_open)

            right_gripper_joint = [world.get_connection_by_name("right_gripper_joint")]

            right_gripper_open = JointState.from_mapping(
                name=PrefixedName("right_gripper_open", prefix=boxy.name.name),
                mapping=dict(zip(right_gripper_joint, [0.548])),
                state_type=GripperState.OPEN,
            )

            right_gripper_close = JointState.from_mapping(
                name=PrefixedName("right_gripper_close", prefix=boxy.name.name),
                mapping=dict(zip(right_gripper_joint, [0.0])),
                state_type=GripperState.CLOSE,
            )

            right_gripper.add_joint_state(right_gripper_close)
            right_gripper.add_joint_state(right_gripper_open)

            torso_joint = [world.get_connection_by_name("triangle_base_joint")]

            torso_low = JointState.from_mapping(
                name=PrefixedName("torso_low", prefix=boxy.name.name),
                mapping=dict(zip(torso_joint, [-0.58])),
                state_type=TorsoState.LOW,
            )

            torso_mid = JointState.from_mapping(
                name=PrefixedName("torso_mid", prefix=boxy.name.name),
                mapping=dict(zip(torso_joint, [0.29])),
                state_type=TorsoState.MID,
            )

            torso_high = JointState.from_mapping(
                name=PrefixedName("torso_high", prefix=boxy.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.HIGH,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            world.add_semantic_annotation(boxy)

        return boxy
