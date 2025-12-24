from __future__ import annotations

from collections import defaultdict
from dataclasses import field, dataclass
from typing import Self

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


@dataclass
class Tracy(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Represents two UR10e Arms on a table, with a pole between them holding a small camera.
     Example can be found at: https://vib.ai.uni-bremen.de/page/comingsoon/the-tracebot-laboratory/
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def setup_collision_config(self): ...

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
                sensors={camera},
                root=world.get_body_by_name("camera_pole"),
                tip=world.get_body_by_name("camera_link"),
                _world=world,
            )

            robot.add_kinematic_chain(neck)
            world.add_semantic_annotation(robot)

            vel_limits = defaultdict(lambda: 0.2)
            robot.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

            return robot
