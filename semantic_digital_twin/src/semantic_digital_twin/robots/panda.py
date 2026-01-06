from dataclasses import dataclass
from typing import Self

from .abstract_robot import (
    AbstractRobot,
    Arm,
    Finger,
    ParallelGripper,
    JointState
)
from .robot_mixins import HasArms
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World


@dataclass
class Panda(AbstractRobot, HasArms):
    """
    Class that describes the Panda Robot.
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
        Loads the SRDF file for the Panda robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a Panda robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A Panda robot view.
        """

        with world.modify_world():
            panda = cls(
                name=PrefixedName("panda", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            # Create arm
            gripper_thumb = Finger(
                name=PrefixedName("gripper_thumb", prefix=panda.name.name),
                root=world.get_body_by_name("left_finger_link"),
                tip=world.get_body_by_name("left_finger_tip_link"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("gripper_finger", prefix=panda.name.name),
                root=world.get_body_by_name("right_finger_link"),
                tip=world.get_body_by_name("right_finger_tip_link"),
                _world=world,
            )

            gripper = ParallelGripper(
                name=PrefixedName("gripper", prefix=panda.name.name),
                root=world.get_body_by_name("panda_link"),
                tool_frame=world.get_body_by_name("right_finger_link"),
                front_facing_orientation=Quaternion(0, 0, 0, 1),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=gripper_thumb,
                finger=gripper_finger,
                _world=world,
            )

            arm = Arm(
                name=PrefixedName("arm", prefix=panda.name.name),
                root=world.get_body_by_name("panda"),
                tip=world.get_body_by_name("link7"),
                manipulator=gripper,
                _world=world,
            )

            panda.add_arm(arm)

            arm_park = JointState(
                name=PrefixedName("arm_park", prefix=panda.name.name),
                joint_names=[world.get_body_by_name("joint1"), world.get_body_by_name("joint2"),
                             world.get_body_by_name("joint3"), world.get_body_by_name("joint4"),
                             world.get_body_by_name("joint5"), world.get_body_by_name("joint6"),
                             world.get_body_by_name("joint7"), world.get_body_by_name("joint8")],
                joint_positions=[0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
                state_type="Park",
                kinematic_chains=[arm],
                _world=world,
            )

            gripper_joints = [world.get_body_by_name("finger_joint1"), world.get_body_by_name("finger_joint2")]

            gripper_open = JointState(
                name=PrefixedName("gripper_open", prefix=panda.name.name),
                joint_names=gripper_joints,
                joint_positions=[0.04, 0.04],
                state_type="Open",
                kinematic_chains=[gripper],
                _world=world,
            )

            gripper_close = JointState(
                name=PrefixedName("gripper_close", prefix=panda.name.name),
                joint_names=gripper_joints,
                joint_positions=[0.0, 0.0],
                state_type="Close",
                kinematic_chains=[gripper],
                _world=world,
            )

            panda.add_joint_states([arm_park, gripper_open, gripper_close])

            world.add_semantic_annotation(panda)

        return panda

