from __future__ import annotations

from collections import defaultdict
from dataclasses import field, dataclass
from typing import Self

from ..datastructures.definitions import StaticJointState, GripperState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
    Neck,
    AbstractRobot,
)
from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..spatial_types import Quaternion, Vector3
from ..world import World
from ..world_description.connections import FixedConnection


@dataclass(eq=False)
class Tracy(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Represents two UR10e Arms on a table, with a pole between them holding a small camera.
     Example can be found at: https://vib.ai.uni-bremen.de/page/comingsoon/the-tracebot-laboratory/
    """

    def load_srdf(self): ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a Tracy robot semantic annotation from the given world.

        :param world: The world from which to create the robot semantic annotation.

        :return: A Tracy robot semantic annotation.
        """
        with world.modify_world():
            robot = cls(
                name=PrefixedName(name="tracy", prefix=world.name),
                root=world.get_body_by_name("table"),
                _world=world,
            )

            # Create left arm
            left_gripper_thumb = Finger(
                name=PrefixedName("left_gripper_thumb", prefix=robot.name.name),
                root=world.get_body_by_name("left_robotiq_85_left_knuckle_link"),
                tip=world.get_body_by_name("left_robotiq_85_left_finger_tip_link"),
                _world=world,
            )

            left_gripper_finger = Finger(
                name=PrefixedName("left_gripper_finger", prefix=robot.name.name),
                root=world.get_body_by_name("left_robotiq_85_right_knuckle_link"),
                tip=world.get_body_by_name("left_robotiq_85_right_finger_tip_link"),
                _world=world,
            )

            left_gripper = ParallelGripper(
                name=PrefixedName("left_gripper", prefix=robot.name.name),
                root=world.get_body_by_name("left_robotiq_85_base_link"),
                tool_frame=world.get_body_by_name("l_gripper_tool_frame"),
                front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=left_gripper_thumb,
                finger=left_gripper_finger,
                _world=world,
            )
            left_arm = Arm(
                name=PrefixedName("left_arm", prefix=robot.name.name),
                root=world.get_body_by_name("table"),
                tip=world.get_body_by_name("left_wrist_3_link"),
                manipulator=left_gripper,
                _world=world,
            )

            robot.add_arm(left_arm)

            right_gripper_thumb = Finger(
                name=PrefixedName("right_gripper_thumb", prefix=robot.name.name),
                root=world.get_body_by_name("right_robotiq_85_left_knuckle_link"),
                tip=world.get_body_by_name("right_robotiq_85_left_finger_tip_link"),
                _world=world,
            )
            right_gripper_finger = Finger(
                name=PrefixedName("right_gripper_finger", prefix=robot.name.name),
                root=world.get_body_by_name("right_robotiq_85_right_knuckle_link"),
                tip=world.get_body_by_name("right_robotiq_85_right_finger_tip_link"),
                _world=world,
            )
            right_gripper = ParallelGripper(
                name=PrefixedName("right_gripper", prefix=robot.name.name),
                root=world.get_body_by_name("right_robotiq_85_base_link"),
                tool_frame=world.get_body_by_name("r_gripper_tool_frame"),
                front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=right_gripper_thumb,
                finger=right_gripper_finger,
                _world=world,
            )
            right_arm = Arm(
                name=PrefixedName("right_arm", prefix=robot.name.name),
                root=world.get_body_by_name("table"),
                tip=world.get_body_by_name("right_wrist_3_link"),
                manipulator=right_gripper,
                _world=world,
            )
            robot.add_arm(right_arm)

            camera = Camera(
                name=PrefixedName("camera", prefix=robot.name.name),
                root=world.get_body_by_name("camera_link"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
                minimal_height=0.8,
                maximal_height=1.7,
                _world=world,
            )

            # Probably should be classified as "Neck", as that implies that i can move.
            neck = Neck(
                name=PrefixedName("neck", prefix=robot.name.name),
                sensors=[camera],
                root=world.get_body_by_name("camera_pole"),
                tip=world.get_body_by_name("camera_link"),
                _world=world,
            )

            robot.add_kinematic_chain(neck)

            # Create states
            left_arm_park = JointState.from_mapping(
                name=PrefixedName("left_arm_park", prefix=robot.name.name),
                mapping=dict(
                    zip(
                        [c for c in left_arm.connections if type(c) != FixedConnection],
                        [2.62, -1.035, 1.13, -0.966, -0.88, 2.07],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            left_arm.add_joint_state(left_arm_park)

            right_arm_park = JointState.from_mapping(
                name=PrefixedName("right_arm_park", prefix=robot.name.name),
                mapping=dict(
                    zip(
                        [
                            c
                            for c in right_arm.connections
                            if type(c) != FixedConnection
                        ],
                        [3.72, -2.07, -1.17, 4.0, 0.82, 0.75],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            right_arm.add_joint_state(right_arm_park)

            left_gripper_joints = [
                world.get_connection_by_name("left_robotiq_85_left_knuckle_joint"),
                world.get_connection_by_name("left_robotiq_85_right_knuckle_joint"),
            ]

            left_gripper_open = JointState.from_mapping(
                name=PrefixedName("left_gripper_open", prefix=robot.name.name),
                mapping=dict(zip(left_gripper_joints, [0.0, 0.0])),
                state_type=GripperState.OPEN,
            )

            left_gripper_close = JointState.from_mapping(
                name=PrefixedName("left_gripper_close", prefix=robot.name.name),
                mapping=dict(
                    zip(
                        left_gripper_joints,
                        [
                            0.8,
                            -0.8,
                        ],
                    )
                ),
                state_type=GripperState.CLOSE,
            )

            left_gripper.add_joint_state(left_gripper_close)
            left_gripper.add_joint_state(left_gripper_open)

            right_gripper_joints = [
                world.get_connection_by_name("right_robotiq_85_left_knuckle_joint"),
                world.get_connection_by_name("right_robotiq_85_right_knuckle_joint"),
            ]

            right_gripper_open = JointState.from_mapping(
                name=PrefixedName("right_gripper_open", prefix=robot.name.name),
                mapping=dict(zip(right_gripper_joints, [0.0, 0.0])),
                state_type=GripperState.OPEN,
            )

            right_gripper_close = JointState.from_mapping(
                name=PrefixedName("right_gripper_close", prefix=robot.name.name),
                mapping=dict(zip(right_gripper_joints, [0.8, -0.8])),
                state_type=GripperState.CLOSE,
            )

            right_gripper.add_joint_state(right_gripper_close)
            right_gripper.add_joint_state(right_gripper_open)

            world.add_semantic_annotation(robot)

            vel_limits = defaultdict(lambda: 0.2)
            robot.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

            return robot
