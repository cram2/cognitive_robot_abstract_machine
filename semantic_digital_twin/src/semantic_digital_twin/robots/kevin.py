from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from .robot_mixins import HasArms
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
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
class Kevin(AbstractRobot, HasArms):
    """
    Class that describes the Kevin Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the Kevin robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a Kevin robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A Kevin robot view.
        """

        with world.modify_world():
            kevin = cls(
                name=PrefixedName(name="kevin", prefix=world.name),
                root=world.get_body_by_name("robot_base_footprint"),
                _world=world,
            )

            # Create arm
            gripper_thumb = Finger(
                name=PrefixedName("left_gripper_thumb", prefix=kevin.name.name),
                root=world.get_body_by_name("robot_arm_wrist_link"),
                tip=world.get_body_by_name("robot_arm_left_finger_link"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("left_gripper_finger", prefix=kevin.name.name),
                root=world.get_body_by_name("robot_arm_wrist_link"),
                tip=world.get_body_by_name("robot_arm_right_finger_link"),
                _world=world,
            )

            gripper = ParallelGripper(
                name=PrefixedName("left_gripper", prefix=kevin.name.name),
                root=world.get_body_by_name("robot_arm_wrist_link"),
                tool_frame=world.get_body_by_name("robot_arm_tool_link"),
                front_facing_orientation=Quaternion(0, 0, 0, 1),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=gripper_thumb,
                finger=gripper_finger,
                _world=world,
            )
            arm = Arm(
                name=PrefixedName("left_arm", prefix=kevin.name.name),
                root=world.get_body_by_name("robot_arm_base_link"),
                tip=world.get_body_by_name("robot_arm_wrist_link"),
                manipulator=gripper,
                _world=world,
            )

            kevin.add_arm(arm)

            # Create camera
            camera = Camera(
                name=PrefixedName("top_cam", prefix=kevin.name.name),
                root=world.get_body_by_name("robot_top_3d_laser_link"),
                forward_facing_axis=Vector3(1, 0, 0),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=1.2,
                maximal_height=1.21,
                _world=world,
            )

            kevin.add_sensor(camera)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=kevin.name.name),
                root=world.get_body_by_name("robot_base_link"),
                tip=world.get_body_by_name("robot_arm_column_link"),
                _world=world,
            )
            kevin.add_torso(torso)

            # Create states
            arm_park = JointState.from_mapping(
                name=PrefixedName("arm_park", prefix=kevin.name.name),
                mapping=dict(
                    zip(
                        [c for c in arm.connections if type(c) != FixedConnection],
                        [0.63, 0.03, 4.70, -1.63],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            arm.add_joint_state(arm_park)

            gripper_joints = [
                world.get_connection_by_name("robot_arm_gripper_joint"),
                world.get_connection_by_name("robot_arm_gripper_mirror_joint"),
            ]

            gripper_open = JointState.from_mapping(
                name=PrefixedName("gripper_open", prefix=kevin.name.name),
                mapping=dict(zip(gripper_joints, [0.066, 0.066])),
                state_type=GripperState.OPEN,
            )

            gripper_close = JointState.from_mapping(
                name=PrefixedName("gripper_close", prefix=kevin.name.name),
                mapping=dict(zip(gripper_joints, [0.0, 0.0])),
                state_type=GripperState.CLOSE,
            )

            gripper.add_joint_state(gripper_close)
            gripper.add_joint_state(gripper_open)

            torso_joint = [world.get_connection_by_name("robot_arm_column_joint")]

            torso_low = JointState.from_mapping(
                name=PrefixedName("torso_low", prefix=kevin.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.LOW,
            )

            torso_mid = JointState.from_mapping(
                name=PrefixedName("torso_mid", prefix=kevin.name.name),
                mapping=dict(zip(torso_joint, [0.3])),
                state_type=TorsoState.MID,
            )

            torso_high = JointState.from_mapping(
                name=PrefixedName("torso_high", prefix=kevin.name.name),
                mapping=dict(zip(torso_joint, [0.69])),
                state_type=TorsoState.HIGH,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            world.add_semantic_annotation(kevin)

        return kevin
