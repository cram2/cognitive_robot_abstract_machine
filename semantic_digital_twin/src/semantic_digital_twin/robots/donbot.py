from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from semantic_digital_twin.robots.robot_part_mixins import (
    HasMobileBase,
    HasOneArm,
    HasParallelGripper,
    HasCameras,
)
from semantic_digital_twin.robots.robot_parts import (
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
    MobileBase,
    AbstractRobot,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World


@dataclass(eq=False)
class DonbotFinger(Finger, ABC): ...


@dataclass(eq=False)
class DonbotLeftFinger(DonbotFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("gripper_gripper_left_link"),
            tip=world.get_body_by_name("gripper_finger_right_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class DonbotRightFinger(DonbotFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("gripper_gripper_left_link"),
            tip=world.get_body_by_name("gripper_finger_left_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class DonbotCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("camera_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.5,
            maximal_height=1.2,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class DonbotGripper(ParallelGripper, HasCameras):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        gripper = cls(
            root=world.get_body_by_name("gripper_base_link"),
            tool_frame=world.get_body_by_name("gripper_tool_frame"),
            front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = DonbotLeftFinger.setup_default_configuration_in_world(self._world)
        self.add_thumb(thumb)
        finger = DonbotRightFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(finger)

    def setup_sensor_semantic_annotations(self):
        camera = DonbotCamera.setup_default_configuration_in_world(self._world)
        self.add_camera(camera)


@dataclass(eq=False)
class DonbotArm(Arm, HasParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("ur5_base_link"),
            tip=world.get_body_by_name("ur5_wrist_3_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = DonbotGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()
        gripper.setup_sensor_semantic_annotations()


@dataclass(eq=False)
class DonbotMobileBase(MobileBase, HasOneArm):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        mobile_base = cls(
            root=world.get_body_by_name("base_footprint"),
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_arm_semantic_annotations(self):
        arm = DonbotArm.setup_default_configuration_in_world(self._world)
        self.add_arm(arm)
        arm.setup_end_effector_semantic_annotation()


@dataclass(eq=False)
class Donbot(AbstractRobot, HasMobileBase):
    """
    Class that describes the Donbot Robot.
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = DonbotMobileBase.setup_default_configuration_in_world(self._world)
        self.add_mobile_base(mobile_base)
        mobile_base.setup_arm_semantic_annotations()

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()
