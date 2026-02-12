from dataclasses import dataclass
from typing import Self

from .abstract_robot import AbstractRobot, Arm, Finger, ParallelGripper
from .robot_mixins import HasArms
from ..datastructures.definitions import StaticJointState, GripperState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World
from ..world_description.connections import FixedConnection


@dataclass(eq=False)
class Panda(AbstractRobot, HasArms):
    """
    Class that describes the Panda Robot.
    """

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

            arm_park = JointState.from_mapping(
                name=PrefixedName("arm_park", prefix=panda.name.name),
                mapping=dict(
                    zip(
                        [c for c in arm.connections if type(c) != FixedConnection],
                        [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            arm.add_joint_state(arm_park)

            gripper_joints = [
                world.get_connection_by_name("finger_joint1"),
                world.get_connection_by_name("finger_joint2"),
            ]

            gripper_open = JointState.from_mapping(
                name=PrefixedName("gripper_open", prefix=panda.name.name),
                mapping=dict(zip(gripper_joints, [0.04, 0.04])),
                state_type=GripperState.OPEN,
            )

            gripper_close = JointState.from_mapping(
                name=PrefixedName("gripper_close", prefix=panda.name.name),
                mapping=dict(zip(gripper_joints, [0.0, 0.0])),
                state_type=GripperState.CLOSE,
            )

            gripper.add_joint_state(gripper_open)
            gripper.add_joint_state(gripper_close)

            world.add_semantic_annotation(panda)

        return panda
