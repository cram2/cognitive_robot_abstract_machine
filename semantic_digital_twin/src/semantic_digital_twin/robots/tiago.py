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
    JointState,
)
from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World


@dataclass
class Tiago(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Class that describes the Take It And Go Robot (TIAGo).
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
                root=world.get_body_by_name("gripper_left_left_finger_link"),
                tip=world.get_body_by_name("gripper_left_left_finger_tip_link"),
                _world=world,
            )

            left_gripper_finger = Finger(
                name=PrefixedName("left_gripper_finger", prefix=tiago.name.name),
                root=world.get_body_by_name("gripper_left_right_finger_link"),
                tip=world.get_body_by_name("gripper_left_right_finger_tip_link"),
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
                root=world.get_body_by_name("gripper_right_left_finger_link"),
                tip=world.get_body_by_name("gripper_right_left_finger_tip_link"),
                _world=world,
            )
            right_gripper_finger = Finger(
                name=PrefixedName("right_gripper_finger", prefix=tiago.name.name),
                root=world.get_body_by_name("gripper_right_right_finger_link"),
                tip=world.get_body_by_name("gripper_right_right_finger_tip_link"),
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
                    horizontal_angle=1.0665, vertical_angle=1.4165
                ),
                minimal_height=0.99483,
                maximal_height=0.75049,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=tiago.name.name),
                sensors={camera},
                root=world.get_body_by_name("torso_lift_link"),
                tip=world.get_body_by_name("head_2_link"),
                pitch_body=world.get_body_by_name("head_2_joint"),
                yaw_body=world.get_body_by_name("head_1_joint"),
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
            left_arm_park = JointState(
                name=PrefixedName("left_arm_park", prefix=tiago.name.name),
                joint_names=[world.get_body_by_name("arm_left_1_joint"), world.get_body_by_name("arm_left_2_joint"),
                             world.get_body_by_name("arm_left_3_joint"), world.get_body_by_name("arm_left_4_joint"),
                             world.get_body_by_name("arm_left_5_joint"), world.get_body_by_name("arm_left_6_joint"),
                             world.get_body_by_name("arm_left_7_joint")],
                joint_positions=[0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                state_type="Park",
                kinematic_chains=[left_arm],
                _world=world,
            )

            right_arm_park = JointState(
                name=PrefixedName("right_arm_park", prefix=tiago.name.name),
                joint_names=[world.get_body_by_name("arm_right_1_joint"), world.get_body_by_name("arm_right_2_joint"),
                             world.get_body_by_name("arm_right_3_joint"), world.get_body_by_name("arm_right_4_joint"),
                             world.get_body_by_name("arm_right_5_joint"), world.get_body_by_name("arm_right_6_joint"),
                             world.get_body_by_name("arm_right_7_joint")],
                joint_positions=[0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                state_type="Park",
                kinematic_chains=[right_arm],
                _world=world,
            )

            left_gripper_joints = [world.get_body_by_name("gripper_left_left_finger_joint"),
                                   world.get_body_by_name("gripper_left_right_finger_joint")]

            left_gripper_open = JointState(
                name=PrefixedName("left_gripper_open", prefix=tiago.name.name),
                joint_names=left_gripper_joints,
                joint_positions=[0.048, 0.048],
                state_type="Open",
                kinematic_chains=[left_gripper],
                _world=world,
            )

            left_gripper_close = JointState(
                name=PrefixedName("left_gripper_close", prefix=tiago.name.name),
                joint_names=left_gripper_joints,
                joint_positions=[0.0, 0.0],
                state_type="Close",
                kinematic_chains=[left_gripper],
                _world=world,
            )

            right_gripper_joints = [world.get_body_by_name("gripper_right_left_finger_joint"),
                                    world.get_body_by_name("gripper_right_right_finger_joint")]

            right_gripper_open = JointState(
                name=PrefixedName("right_gripper_open", prefix=tiago.name.name),
                joint_names=right_gripper_joints,
                joint_positions=[0.048, 0.048],
                state_type="Open",
                kinematic_chains=[right_gripper],
                _world=world,
            )

            right_gripper_close = JointState(
                name=PrefixedName("right_gripper_close", prefix=tiago.name.name),
                joint_names=right_gripper_joints,
                joint_positions=[0.0, 0.0],
                state_type="Close",
                kinematic_chains=[right_gripper],
                _world=world,
            )

            torso_joint = [world.get_body_by_name("torso_lift_joint")]

            torso_low = JointState(
                name=PrefixedName("torso_low", prefix=tiago.name.name),
                joint_names=torso_joint,
                joint_positions=[0.3],
                state_type="Low",
                kinematic_chains=[torso],
                _world=world,
            )

            torso_mid = JointState(
                name=PrefixedName("torso_mid", prefix=tiago.name.name),
                joint_names=torso_joint,
                joint_positions=[0.15],
                state_type="Mid",
                kinematic_chains=[torso],
                _world=world,
            )

            torso_high = JointState(
                name=PrefixedName("torso_high", prefix=tiago.name.name),
                joint_names=torso_joint,
                joint_positions=[0.0],
                state_type="High",
                kinematic_chains=[torso],
                _world=world,
            )

            tiago.add_joint_states([left_arm_park, right_arm_park, left_gripper_open, left_gripper_close,
                                    right_gripper_open, right_gripper_close, torso_low, torso_mid, torso_high])

            world.add_semantic_annotation(tiago)

        return tiago

