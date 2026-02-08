from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

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
class ICub3(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Class that describes the iCub3 Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the iCub3 robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a iCub3 robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A iCub3 robot view.
        """

        with world.modify_world():
            icub3 = cls(
                name=PrefixedName(name="icub3", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            # Create left arm
            left_gripper_thumb = Finger(
                name=PrefixedName("left_gripper_thumb", prefix=icub3.name.name),
                root=world.get_body_by_name("l_hand_thumb_0"),
                tip=world.get_body_by_name("l_hand_thumb_tip"),
                _world=world,
            )

            left_gripper_index_finger = Finger(
                name=PrefixedName("left_gripper_index_finger", prefix=icub3.name.name),
                root=world.get_body_by_name("l_hand_index_0"),
                tip=world.get_body_by_name("l_hand_index_tip"),
                _world=world,
            )

            left_gripper_middle_finger = Finger(
                name=PrefixedName("left_gripper_middle_finger", prefix=icub3.name.name),
                root=world.get_body_by_name("l_hand_middle_0"),
                tip=world.get_body_by_name("l_hand_middle_tip"),
                _world=world,
            )

            left_gripper_ring_finger = Finger(
                name=PrefixedName("left_gripper_ring_finger", prefix=icub3.name.name),
                root=world.get_body_by_name("l_hand_ring_0"),
                tip=world.get_body_by_name("l_hand_ring_tip"),
                _world=world,
            )

            left_gripper_little_finger = Finger(
                name=PrefixedName("left_gripper_little_finger", prefix=icub3.name.name),
                root=world.get_body_by_name("l_hand_little_0"),
                tip=world.get_body_by_name("l_hand_little_tip"),
                _world=world,
            )

            left_gripper = ParallelGripper(
                name=PrefixedName("left_gripper", prefix=icub3.name.name),
                root=world.get_body_by_name("l_hand"),
                tool_frame=world.get_body_by_name("l_gripper_tool_frame"),
                front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=left_gripper_thumb,
                finger=left_gripper_index_finger,
                _world=world,
            )
            left_arm = Arm(
                name=PrefixedName("left_arm", prefix=icub3.name.name),
                root=world.get_body_by_name("root_link"),
                tip=world.get_body_by_name("l_hand"),
                manipulator=left_gripper,
                _world=world,
            )

            icub3.add_arm(left_arm)

            # Create right arm
            right_gripper_thumb = Finger(
                name=PrefixedName("right_gripper_thumb", prefix=icub3.name.name),
                root=world.get_body_by_name("r_hand_thumb_0"),
                tip=world.get_body_by_name("r_hand_thumb_tip"),
                _world=world,
            )
            right_gripper_index_finger = Finger(
                name=PrefixedName("right_gripper_index_finger", prefix=icub3.name.name),
                root=world.get_body_by_name("r_hand_index_0"),
                tip=world.get_body_by_name("r_hand_index_tip"),
                _world=world,
            )
            right_gripper_middle_finger = Finger(
                name=PrefixedName(
                    "right_gripper_middle_finger", prefix=icub3.name.name
                ),
                root=world.get_body_by_name("r_hand_middle_0"),
                tip=world.get_body_by_name("r_hand_middle_tip"),
                _world=world,
            )
            right_gripper_ring_finger = Finger(
                name=PrefixedName("right_gripper_ring_finger", prefix=icub3.name.name),
                root=world.get_body_by_name("r_hand_ring_0"),
                tip=world.get_body_by_name("r_hand_ring_tip"),
                _world=world,
            )
            right_gripper_little_finger = Finger(
                name=PrefixedName(
                    "right_gripper_little_finger", prefix=icub3.name.name
                ),
                root=world.get_body_by_name("r_hand_little_0"),
                tip=world.get_body_by_name("r_hand_little_tip"),
                _world=world,
            )

            right_gripper = ParallelGripper(
                name=PrefixedName("right_gripper", prefix=icub3.name.name),
                root=world.get_body_by_name("r_hand"),
                tool_frame=world.get_body_by_name("r_gripper_tool_frame"),
                front_facing_orientation=Quaternion(0, 0, -0.707, 0.707),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=right_gripper_thumb,
                finger=right_gripper_index_finger,
                _world=world,
            )
            right_arm = Arm(
                name=PrefixedName("right_arm", prefix=icub3.name.name),
                root=world.get_body_by_name("root_link"),
                tip=world.get_body_by_name("r_hand"),
                manipulator=right_gripper,
                _world=world,
            )

            icub3.add_arm(right_arm)

            # Create camera and neck
            camera = Camera(
                name=PrefixedName("eye_camera", prefix=icub3.name.name),
                root=world.get_body_by_name("head"),
                forward_facing_axis=Vector3(1, 0, 0),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=1.27,
                maximal_height=1.85,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=icub3.name.name),
                sensors=[camera],
                root=world.get_body_by_name("chest"),
                tip=world.get_body_by_name("head"),
                pitch_body=world.get_body_by_name("neck_pitch"),
                yaw_body=world.get_body_by_name("neck_yaw"),
                _world=world,
            )
            icub3.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=icub3.name.name),
                root=world.get_body_by_name("root_link"),
                tip=world.get_body_by_name("chest"),
                _world=world,
            )
            icub3.add_torso(torso)

            # Create states
            left_arm_park = JointState.from_mapping(
                name=PrefixedName("left_arm_park", prefix=icub3.name.name),
                mapping=dict(
                    zip(
                        [c for c in left_arm.connections if type(c) != FixedConnection],
                        [0.0] * len(list(left_arm.connections)),
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            left_arm.add_joint_state(left_arm_park)

            right_arm_park = JointState.from_mapping(
                name=PrefixedName("right_arm_park", prefix=icub3.name.name),
                mapping=dict(
                    zip(
                        [
                            c
                            for c in right_arm.connections
                            if type(c) != FixedConnection
                        ],
                        [0.0] * len(list(right_arm.connections)),
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            right_arm.add_joint_state(right_arm_park)

            left_gripper_joints = [
                c for c in left_gripper.connections if type(c) != FixedConnection
            ]

            left_gripper_open = JointState.from_mapping(
                name=PrefixedName("left_gripper_open", prefix=icub3.name.name),
                mapping=dict(
                    zip(
                        left_gripper_joints,
                        [0.0] * len(list(left_gripper_joints)),
                    )
                ),
                state_type=GripperState.OPEN,
            )

            left_gripper_close = JointState.from_mapping(
                name=PrefixedName("left_gripper_close", prefix=icub3.name.name),
                mapping=dict(
                    zip(
                        left_gripper_joints,
                        [
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            -0.3490658503988659,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            0.3490658503988659,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            0.3490658503988659,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            0.3490658503988659,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                        ],
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
                name=PrefixedName("right_gripper_open", prefix=icub3.name.name),
                mapping=dict(
                    zip(
                        right_gripper_joints,
                        [0.0] * len(list(right_gripper_joints)),
                    )
                ),
                state_type=GripperState.OPEN,
            )

            right_gripper_close = JointState.from_mapping(
                name=PrefixedName("right_gripper_close", prefix=icub3.name.name),
                mapping=dict(
                    zip(
                        right_gripper_joints,
                        [
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            -0.3490658503988659,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            0.3490658503988659,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            0.3490658503988659,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                            0.3490658503988659,
                            np.pi / 2,
                            np.pi / 2,
                            np.pi / 2,
                        ],
                    )
                ),
                state_type=GripperState.CLOSE,
            )

            right_gripper.add_joint_state(right_gripper_close)
            right_gripper.add_joint_state(right_gripper_open)

            torso_joint = [world.get_connection_by_name("torso_roll")]

            torso_low = JointState.from_mapping(
                name=PrefixedName("torso_low", prefix=icub3.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.LOW,
            )

            torso_mid = JointState.from_mapping(
                name=PrefixedName("torso_mid", prefix=icub3.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.MID,
            )

            torso_high = JointState.from_mapping(
                name=PrefixedName("torso_high", prefix=icub3.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.HIGH,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            world.add_semantic_annotation(icub3)

        return icub3
