from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from semantic_digital_twin.robots.robot_part_mixins import (
    HasMobileBase,
    HasTorso,
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
    Torso,
    MobileBase,
    AbstractRobot,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World


@dataclass(eq=False)
class StretchFinger(Finger, ABC): ...


@dataclass(eq=False)
class StretchLeftFinger(StretchFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("link_gripper_finger_left"),
            tip=world.get_body_by_name("link_gripper_fingertip_left"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class StretchRightFinger(StretchFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("link_gripper_finger_right"),
            tip=world.get_body_by_name("link_gripper_fingertip_right"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class StretchGripper(ParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        gripper = cls(
            root=world.get_body_by_name("link_straight_gripper"),
            tool_frame=world.get_body_by_name("link_grasp_center"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = StretchLeftFinger.setup_default_configuration_in_world(self._world)
        self.add_thumb(thumb)
        finger = StretchRightFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(finger)


@dataclass(eq=False)
class StretchArm(Arm, HasParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("link_mast"),
            tip=world.get_body_by_name("link_wrist_roll"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = StretchGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class StretchCameraColor(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("camera_color_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.322,
            maximal_height=1.322,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class StretchCameraDepth(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("camera_depth_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.307,
            maximal_height=1.307,
            default_camera=False,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class StretchCameraInfra1(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("camera_infra1_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.307,
            maximal_height=1.307,
            default_camera=False,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class StretchCameraInfra2(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("camera_infra2_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.257,
            maximal_height=1.257,
            default_camera=False,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class StretchTorso(Torso, HasOneArm, HasCameras):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        torso = cls(
            root=world.get_body_by_name("link_mast"),
            tip=world.get_body_by_name("link_lift"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        arm = StretchArm.setup_default_configuration_in_world(self._world)
        self.add_arm(arm)
        arm.setup_end_effector_semantic_annotation()

    def setup_sensor_semantic_annotations(self):
        camera_color = StretchCameraColor.setup_default_configuration_in_world(
            self._world
        )
        self.add_camera(camera_color)

        camera_depth = StretchCameraDepth.setup_default_configuration_in_world(
            self._world
        )
        self.add_camera(camera_depth)

        camera_infra1 = StretchCameraInfra1.setup_default_configuration_in_world(
            self._world
        )
        self.add_camera(camera_infra1)

        camera_infra2 = StretchCameraInfra2.setup_default_configuration_in_world(
            self._world
        )
        self.add_camera(camera_infra2)


@dataclass(eq=False)
class StretchMobileBase(MobileBase, HasTorso):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        mobile_base = cls(
            root=world.get_body_by_name("base_link"),
            forward_axis=Vector3(0, -1, 0),
            full_body_controlled=True,
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_default_torso_semantic_annotation(self):
        torso = StretchTorso.setup_default_configuration_in_world(self._world)
        self.add_torso(torso)
        torso.setup_arm_semantic_annotations()
        torso.setup_sensor_semantic_annotations()


@dataclass(eq=False)
class Stretch(AbstractRobot, HasMobileBase):
    """
    Class that describes the Stretch Robot.
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = StretchMobileBase.setup_default_configuration_in_world(
            self._world
        )
        mobile_base.setup_default_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()
