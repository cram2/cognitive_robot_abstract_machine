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
    JointState,
)
from .robot_mixins import HasNeck, SpecifiesLeftRightArm, HasArms
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World


@dataclass
class Stretch(AbstractRobot, HasArms, HasNeck):
    """
    Class that describes the Stretch Robot.
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
        Loads the SRDF file for the Stretch robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a Stretch robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A Stretch robot view.
        """

        with world.modify_world():
            stretch = cls(
                name=PrefixedName("stretch", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            # Create arm
            gripper_thumb = Finger(
                name=PrefixedName("gripper_thumb", prefix=stretch.name.name),
                root=world.get_body_by_name("joint_gripper_finger_left_link"),
                tip=world.get_body_by_name("joint_gripper_finger_left_tip_link"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("gripper_finger", prefix=stretch.name.name),
                root=world.get_body_by_name("joint_gripper_finger_right_link"),
                tip=world.get_body_by_name("joint_gripper_finger_right_tip_link"),
                _world=world,
            )

            gripper = ParallelGripper(
                name=PrefixedName("gripper", prefix=stretch.name.name),
                root=world.get_body_by_name("link_straight_gripper"),
                tool_frame=world.get_body_by_name("link_grasp_center"),
                front_facing_orientation=Quaternion(0, 0, 0, 1),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=gripper_thumb,
                finger=gripper_finger,
                _world=world,
            )

            arm = Arm(
                name=PrefixedName("arm", prefix=stretch.name.name),
                root=world.get_body_by_name("link_mast"),
                tip=world.get_body_by_name("link_wrist_roll"),
                manipulator=gripper,
                _world=world,
            )

            stretch.add_arm(arm)

            # Create camera and neck
            camera_color = Camera(
                name=PrefixedName("camera_color_optical_frame", prefix=stretch.name.name),
                root=world.get_body_by_name("camera_color_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                minimal_height=1.322,
                maximal_height=1.322,
                _world=world,
            )

            camera_depth = Camera(
                name=PrefixedName("camera_depth_optical_frame", prefix=stretch.name.name),
                root=world.get_body_by_name("camera_depth_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                minimal_height=1.307,
                maximal_height=1.307,
                _world=world,
            )

            camera_infra1 = Camera(
                name=PrefixedName("camera_infra1_optical_frame", prefix=stretch.name.name),
                root=world.get_body_by_name("camera_infra1_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                minimal_height=1.307,
                maximal_height=1.307,
                _world=world,
            )

            camera_infra2 = Camera(
                name=PrefixedName("camera_infra2_optical_frame", prefix=stretch.name.name),
                root=world.get_body_by_name("camera_infra2_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                minimal_height=1.257,
                maximal_height=1.257,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=stretch.name.name),
                sensors={camera_color, camera_depth, camera_infra1, camera_infra2},
                root=world.get_body_by_name("link_head"),
                tip=world.get_body_by_name("link_head_tilt"),
                pitch_body=world.get_body_by_name("joint_head_tilt"),
                yaw_body=world.get_body_by_name("joint_head_pan"),
                _world=world,
            )
            stretch.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=stretch.name.name),
                root=world.get_body_by_name("torso_fixed_link"),
                tip=world.get_body_by_name("torso_lift_link"),
                _world=world,
            )
            stretch.add_torso(torso)

            # Create states
            arm_park = JointState(
                name=PrefixedName("arm_park", prefix=stretch.name.name),
                joint_names=[world.get_body_by_name("joint_lift"), world.get_body_by_name("joint_arm_l3"),
                             world.get_body_by_name("joint_arm_l2"), world.get_body_by_name("joint_arm_l1"),
                             world.get_body_by_name("joint_arm_l0"), world.get_body_by_name("joint_wrist_yaw"),
                             world.get_body_by_name("joint_wrist_pitch"), world.get_body_by_name("joint_wrist_roll")],
                joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                state_type="Park",
                kinematic_chains=[arm],
                _world=world,
            )

            gripper_joints = [world.get_body_by_name("joint_gripper_finger_left"),
                              world.get_body_by_name("joint_gripper_finger_right")]

            gripper_open = JointState(
                name=PrefixedName("gripper_open", prefix=stretch.name.name),
                joint_names=gripper_joints,
                joint_positions=[0.59, 0.59],
                state_type="Open",
                kinematic_chains=[gripper],
                _world=world,
            )

            gripper_close = JointState(
                name=PrefixedName("gripper_close", prefix=stretch.name.name),
                joint_names=gripper_joints,
                joint_positions=[0.0, 0.0],
                state_type="Close",
                kinematic_chains=[gripper],
                _world=world,
            )

            torso_joint = [world.get_body_by_name("joint_lift")]

            torso_low = JointState(
                name=PrefixedName("torso_low", prefix=stretch.name.name),
                joint_names=torso_joint,
                joint_positions=[0.0],
                state_type="Low",
                kinematic_chains=[torso],
                _world=world,
            )

            torso_mid = JointState(
                name=PrefixedName("torso_mid", prefix=stretch.name.name),
                joint_names=torso_joint,
                joint_positions=[0.5],
                state_type="Mid",
                kinematic_chains=[torso],
                _world=world,
            )

            torso_high = JointState(
                name=PrefixedName("torso_high", prefix=stretch.name.name),
                joint_names=torso_joint,
                joint_positions=[1.0],
                state_type="High",
                kinematic_chains=[torso],
                _world=world,
            )

            stretch.add_joint_states([arm_park, gripper_open, gripper_close, torso_low, torso_mid, torso_high])

            world.add_semantic_annotation(stretch)

        return stretch

