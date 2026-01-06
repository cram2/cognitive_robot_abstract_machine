from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from .robot_mixins import HasArms
from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
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


@dataclass
class Kevin(AbstractRobot, HasArms):
    """
    Class that describes the Kevin Robot.
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

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
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            # Create arm
            gripper_thumb = Finger(
                name=PrefixedName("left_gripper_thumb", prefix=kevin.name.name),
                root=world.get_body_by_name("robot_arm_left_finger_link"),
                tip=world.get_body_by_name("robot_arm_left_finger_tip_link"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("left_gripper_finger", prefix=kevin.name.name),
                root=world.get_body_by_name("robot_arm_right_finger_link"),
                tip=world.get_body_by_name("robot_arm_right_finger_tip_link"),
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
            arm_park = JointState(
                name=PrefixedName("arm_park", prefix=kevin.name.name),
                joint_names=[world.get_body_by_name("robot_arm_column_joint"),
                             world.get_body_by_name("robot_arm_inner_joint"),
                             world.get_body_by_name("robot_arm_outer_joint"),
                             world.get_body_by_name("robot_arm_wrist_joint")],
                joint_positions=[0.63, 0.03, 4.70, -1.63],
                state_type="Park",
                kinematic_chains=[arm],
                _world=world,
            )

            gripper_joints = [world.get_body_by_name("robot_arm_gripper_joint"),
                             world.get_body_by_name("robot_arm_gripper_mirror_joint")]

            gripper_open = JointState(
                name=PrefixedName("gripper_open", prefix=kevin.name.name),
                joint_names=gripper_joints,
                joint_positions=[0.066, 0.066],
                state_type="Open",
                kinematic_chains=[gripper],
                _world=world,
            )

            gripper_close = JointState(
                name=PrefixedName("gripper_close", prefix=kevin.name.name),
                joint_names=gripper_joints,
                joint_positions=[0.0, 0.0],
                state_type="Close",
                kinematic_chains=[gripper],
                _world=world,
            )

            torso_joint = [world.get_body_by_name("robot_arm_column_joint")]

            torso_low = JointState(
                name=PrefixedName("torso_low", prefix=kevin.name.name),
                joint_names=torso_joint,
                joint_positions=[0.0],
                state_type="Low",
                kinematic_chains=[torso],
                _world=world,
            )

            torso_mid = JointState(
                name=PrefixedName("torso_mid", prefix=kevin.name.name),
                joint_names=torso_joint,
                joint_positions=[0.3],
                state_type="Mid",
                kinematic_chains=[torso],
                _world=world,
            )

            torso_high = JointState(
                name=PrefixedName("torso_high", prefix=kevin.name.name),
                joint_names=torso_joint,
                joint_positions=[0.69],
                state_type="High",
                kinematic_chains=[torso],
                _world=world,
            )

            kevin.add_joint_states([arm_park, gripper_open, gripper_close, torso_low, torso_mid, torso_high])

            world.add_semantic_annotation(kevin)

        return kevin
