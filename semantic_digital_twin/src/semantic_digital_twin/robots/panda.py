from abc import ABC
from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.robots.robot_parts import (
    Arm,
    Finger,
    ParallelGripper,
    AbstractRobot,
)
from semantic_digital_twin.robots.robot_part_mixins import (
    HasOneArm,
    HasParallelGripper,
)
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world import World


@dataclass(eq=False)
class PandaFinger(Finger, ABC): ...


@dataclass(eq=False)
class PandaLeftFinger(PandaFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("left_finger_link"),
            tip=world.get_body_by_name("left_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PandaRightFinger(PandaFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("right_finger_link"),
            tip=world.get_body_by_name("right_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PandaGripper(ParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        gripper = cls(
            root=world.get_body_by_name("panda_link8"),
            tool_frame=world.get_body_by_name("panda_hand_tcp"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = PandaLeftFinger.setup_default_configuration_in_world(self._world)
        self.add_thumb(thumb)
        finger = PandaRightFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(finger)


@dataclass(eq=False)
class PandaArm(Arm, HasParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("base_footprint"),
            tip=world.get_body_by_name("panda_link8"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = PandaGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class Panda(AbstractRobot, HasOneArm):
    """
    Class that describes the Panda Robot.
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_arm_semantic_annotations(self):
        arm = PandaArm.setup_default_configuration_in_world(self._world)
        self.add_arm(arm)
        arm.setup_end_effector_semantic_annotation()

    def setup_robot_part_semantic_annotations(self):
        self.setup_arm_semantic_annotations()
