from dataclasses import dataclass
from typing import Self

from .abstract_robot import (
    AbstractRobot,
    Arm,
    Finger,
    ParallelGripper,
)
from .robot_mixins import HasArms
from ..datastructures.definitions import StaticJointState, GripperState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World
from ..world_description.connections import FixedConnection


@dataclass(eq=False)
class UR5Controlled(AbstractRobot, HasArms):
    """
    Class that describes the UR5 controlled Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the UR5 controlled robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a UR5 controlled robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A UR5 controlled robot view.
        """

        with world.modify_world():
            ur5_controlled = cls(
                name=PrefixedName("ur5_controlled", prefix=world.name),
                root=world.get_body_by_name("base_link"),
                _world=world,
            )

            # Create arm
            gripper_thumb = Finger(
                name=PrefixedName("gripper_thumb", prefix=ur5_controlled.name.name),
                root=world.get_body_by_name("robotiq_85_left_finger_link"),
                tip=world.get_body_by_name("robotiq_85_left_finger_tip_link"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("gripper_finger", prefix=ur5_controlled.name.name),
                root=world.get_body_by_name("robotiq_85_right_finger_link"),
                tip=world.get_body_by_name("robotiq_85_right_finger_tip_link"),
                _world=world,
            )

            gripper = ParallelGripper(
                name=PrefixedName("gripper", prefix=ur5_controlled.name.name),
                root=world.get_body_by_name("robotiq_gripper-2F-85_link"),
                tool_frame=world.get_body_by_name("right_pad"),
                front_facing_orientation=Quaternion(0, 0, 0, 1),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=gripper_thumb,
                finger=gripper_finger,
                _world=world,
            )

            arm = Arm(
                name=PrefixedName("arm", prefix=ur5_controlled.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("wrist_3_link"),
                manipulator=gripper,
                _world=world,
            )

            ur5_controlled.add_arm(arm)

            # Create states
            arm_park = JointState.from_mapping(
                name=PrefixedName("arm_park", prefix=ur5_controlled.name.name),
                mapping=dict(
                    zip(
                        [c for c in arm.connections if type(c) != FixedConnection],
                        [3.14, -1.56, 1.58, -1.57, -1.57, 0.0],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            arm.add_joint_state(arm_park)

            gripper_joints = [
                c for c in gripper.connections if type(c) != FixedConnection
            ]

            gripper_open = JointState.from_mapping(
                name=PrefixedName("gripper_open", prefix=ur5_controlled.name.name),
                mapping=dict(
                    zip(
                        gripper_joints,
                        [
                            0.798,
                            0.00366,
                            0.796,
                            -0.793,
                            0.798,
                            0.00366,
                            0.796,
                            -0.793,
                        ],
                    )
                ),
                state_type=GripperState.OPEN,
            )

            gripper_close = JointState.from_mapping(
                name=PrefixedName("gripper_close", prefix=ur5_controlled.name.name),
                mapping=dict(
                    zip(gripper_joints, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                ),
                state_type=GripperState.CLOSE,
            )

            gripper.add_joint_state(gripper_open)
            gripper.add_joint_state(gripper_close)

            world.add_semantic_annotation(ur5_controlled)

        return ur5_controlled
