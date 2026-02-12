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
class Justin(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Class that describes the Justin Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the Justin robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a Justin robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A Justin robot view.
        """

        with world.modify_world():
            justin = cls(
                name=PrefixedName(name="rollin_justin", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            # Create left arm
            left_gripper_thumb = Finger(
                name=PrefixedName("left_gripper_thumb", prefix=justin.name.name),
                root=world.get_body_by_name("left_1thumb_base"),
                tip=world.get_body_by_name("left_1thumb4"),
                _world=world,
            )

            left_gripper_tip_finger = Finger(
                name=PrefixedName("left_gripper_tip_finger", prefix=justin.name.name),
                root=world.get_body_by_name("left_2tip_base"),
                tip=world.get_body_by_name("left_2tip4"),
                _world=world,
            )

            left_gripper_middle_finger = Finger(
                name=PrefixedName(
                    "left_gripper_middle_finger", prefix=justin.name.name
                ),
                root=world.get_body_by_name("left_3middle_base"),
                tip=world.get_body_by_name("left_3middle4"),
                _world=world,
            )

            left_gripper_ring_finger = Finger(
                name=PrefixedName("left_gripper_ring_finger", prefix=justin.name.name),
                root=world.get_body_by_name("left_4ring_base"),
                tip=world.get_body_by_name("left_4ring4"),
                _world=world,
            )

            left_gripper = ParallelGripper(
                name=PrefixedName("left_gripper", prefix=justin.name.name),
                root=world.get_body_by_name("left_arm7"),
                tool_frame=world.get_body_by_name("l_gripper_tool_frame"),
                front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=left_gripper_thumb,
                finger=left_gripper_tip_finger,
                _world=world,
            )
            left_arm = Arm(
                name=PrefixedName("left_arm", prefix=justin.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("left_arm7"),
                manipulator=left_gripper,
                _world=world,
            )

            justin.add_arm(left_arm)

            # Create right arm
            right_gripper_thumb = Finger(
                name=PrefixedName("right_gripper_thumb", prefix=justin.name.name),
                root=world.get_body_by_name("right_1thumb_base"),
                tip=world.get_body_by_name("right_1thumb4"),
                _world=world,
            )

            right_gripper_tip_finger = Finger(
                name=PrefixedName("right_gripper_tip_finger", prefix=justin.name.name),
                root=world.get_body_by_name("right_2tip_base"),
                tip=world.get_body_by_name("right_2tip4"),
                _world=world,
            )

            right_gripper_middle_finger = Finger(
                name=PrefixedName(
                    "right_gripper_middle_finger", prefix=justin.name.name
                ),
                root=world.get_body_by_name("right_3middle_base"),
                tip=world.get_body_by_name("right_3middle4"),
                _world=world,
            )

            right_gripper_ring_finger = Finger(
                name=PrefixedName("right_gripper_ring_finger", prefix=justin.name.name),
                root=world.get_body_by_name("right_4ring_base"),
                tip=world.get_body_by_name("right_4ring4"),
                _world=world,
            )

            right_gripper_finger_4 = Finger(
                name=PrefixedName("right_gripper_finger_4", prefix=justin.name.name),
                root=world.get_body_by_name("right_1thumb4"),
                tip=world.get_body_by_name("right_1thumb4_tip"),
                _world=world,
            )

            right_gripper = ParallelGripper(
                name=PrefixedName("right_gripper", prefix=justin.name.name),
                root=world.get_body_by_name("right_arm7"),
                tool_frame=world.get_body_by_name("r_gripper_tool_frame"),
                front_facing_orientation=Quaternion(0.707, 0.707, 0.707, 0.707),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=right_gripper_thumb,
                finger=right_gripper_tip_finger,
                _world=world,
            )
            right_arm = Arm(
                name=PrefixedName("right_arm", prefix=justin.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("right_arm7"),
                manipulator=right_gripper,
                _world=world,
            )

            justin.add_arm(right_arm)

            # Create camera and neck

            # real camera unknown at the moment of writing (also missing in urdf), so using dummy camera for now
            camera = Camera(
                name=PrefixedName("dummy_camera", prefix=justin.name.name),
                root=world.get_body_by_name("head2"),
                forward_facing_axis=Vector3(1, 0, 0),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=1.27,
                maximal_height=1.85,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=justin.name.name),
                sensors=[camera],
                root=world.get_body_by_name("torso4"),
                tip=world.get_body_by_name("head2"),
                pitch_body=world.get_body_by_name("head1"),
                yaw_body=world.get_body_by_name("head2"),
                _world=world,
            )
            justin.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=justin.name.name),
                root=world.get_body_by_name("torso1"),
                tip=world.get_body_by_name("torso4"),
                _world=world,
            )
            justin.add_torso(torso)

            # Create states
            left_arm_park = JointState.from_mapping(
                name=PrefixedName("left_arm_park", prefix=justin.name.name),
                mapping=dict(
                    zip(
                        [c for c in left_arm.connections if type(c) != FixedConnection],
                        [
                            0.0,
                            0.0,
                            0.174533,
                            0.0,
                            0.0,
                            -1.9,
                            0.0,
                            1.0,
                            0.0,
                            -1.0,
                            0.0,
                        ],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            left_arm.add_joint_state(left_arm_park)

            right_arm_park = JointState.from_mapping(
                name=PrefixedName("right_arm_park", prefix=justin.name.name),
                mapping=dict(
                    zip(
                        [
                            c
                            for c in right_arm.connections
                            if type(c) != FixedConnection
                        ],
                        [
                            0.0,
                            0.0,
                            0.174533,
                            0.0,
                            0.0,
                            -1.9,
                            0.0,
                            1.0,
                            0.0,
                            -1.0,
                            0.0,
                        ],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            right_arm.add_joint_state(right_arm_park)

            left_gripper_joints = [
                c for c in left_gripper.connections if type(c) != FixedConnection
            ]

            left_gripper_open = JointState.from_mapping(
                name=PrefixedName("left_gripper_open", prefix=justin.name.name),
                mapping=dict(
                    zip(
                        left_gripper_joints,
                        [0.0] * len(left_gripper_joints),
                    )
                ),
                state_type=GripperState.OPEN,
            )

            left_gripper_close = JointState.from_mapping(
                name=PrefixedName("left_gripper_close", prefix=justin.name.name),
                mapping=dict(
                    zip(
                        left_gripper_joints,
                        [
                            0.0,
                            0.523599,
                            1.50098,
                            1.76278,
                            1.76278,
                            0.0,
                            0.523599,
                            1.50098,
                            1.76278,
                            1.76278,
                            0.0,
                            0.523599,
                            1.50098,
                            1.76278,
                            1.76278,
                            0.0,
                            0.523599,
                            1.50098,
                            1.76278,
                            1.76278,
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
                name=PrefixedName("right_gripper_open", prefix=justin.name.name),
                mapping=dict(
                    zip(
                        right_gripper_joints,
                        [0.0] * len(right_gripper_joints),
                    )
                ),
                state_type=GripperState.OPEN,
            )

            right_gripper_close = JointState.from_mapping(
                name=PrefixedName("right_gripper_close", prefix=justin.name.name),
                mapping=dict(
                    zip(
                        right_gripper_joints,
                        [
                            0.0,
                            0.523599,
                            1.50098,
                            1.76278,
                            1.76278,
                            0.0,
                            0.523599,
                            1.50098,
                            1.76278,
                            1.76278,
                            0.0,
                            0.523599,
                            1.50098,
                            1.76278,
                            1.76278,
                            0.0,
                            0.523599,
                            1.50098,
                            1.76278,
                            1.76278,
                        ],
                    )
                ),
                state_type=GripperState.CLOSE,
            )

            right_gripper.add_joint_state(right_gripper_close)
            right_gripper.add_joint_state(right_gripper_open)

            torso_joints = [
                world.get_connection_by_name("torso2_joint"),
                world.get_connection_by_name("torso3_joint"),
                world.get_connection_by_name("torso4_joint"),
            ]

            torso_low = JointState.from_mapping(
                name=PrefixedName("torso_low", prefix=justin.name.name),
                mapping=dict(zip(torso_joints, [-0.9, 2.33874, -1.57])),
                state_type=TorsoState.LOW,
            )

            torso_mid = JointState.from_mapping(
                name=PrefixedName("torso_mid", prefix=justin.name.name),
                mapping=dict(zip(torso_joints, [-0.8, 1.57, -0.77])),
                state_type=TorsoState.MID,
            )

            torso_high = JointState.from_mapping(
                name=PrefixedName("torso_high", prefix=justin.name.name),
                mapping=dict(zip(torso_joints, [0.0, 0.174533, 0.0])),
                state_type=TorsoState.HIGH,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            world.add_semantic_annotation(justin)

        return justin
