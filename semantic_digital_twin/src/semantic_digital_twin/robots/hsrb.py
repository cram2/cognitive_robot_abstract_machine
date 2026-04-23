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
class HSRBFinger(Finger, ABC): ...


@dataclass(eq=False)
class HSRBLeftFinger(HSRBFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("hand_l_proximal_link"),
            tip=world.get_body_by_name("hand_l_distal_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class HSRBRightFinger(HSRBFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("hand_r_proximal_link"),
            tip=world.get_body_by_name("hand_r_finger_tip_frame"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class HSRBHandCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("hand_camera_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=False,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBGripper(ParallelGripper, HasCameras):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        gripper = cls(
            root=world.get_body_by_name("hand_palm_link"),
            tool_frame=world.get_body_by_name("hand_gripper_tool_frame"),
            front_facing_orientation=Quaternion(-0.70710678, 0.0, -0.70710678, 0.0),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = HSRBLeftFinger.setup_default_configuration_in_world(self._world)
        self.add_thumb(thumb)
        finger = HSRBRightFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(finger)

    def setup_sensor_semantic_annotations(self):
        camera = HSRBHandCamera.setup_default_configuration_in_world(self._world)
        self.add_camera(camera)


@dataclass(eq=False)
class HSRBArm(Arm, HasParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("arm_lift_link"),
            tip=world.get_body_by_name("hand_palm_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = HSRBGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()
        gripper.setup_sensor_semantic_annotations()


@dataclass(eq=False)
class HSRBHeadCenterCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("head_center_camera_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBHeadRightCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("head_r_stereo_camera_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=False,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBHeadLeftCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("head_l_stereo_camera_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=False,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBHeadRGBDCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("head_rgbd_sensor_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=False,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBTorso(Torso, HasOneArm, HasCameras):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        torso = cls(
            root=world.get_body_by_name("base_link"),
            tip=world.get_body_by_name("head_tilt_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        arm = HSRBArm.setup_default_configuration_in_world(self._world)
        self.add_arm(arm)
        arm.setup_end_effector_semantic_annotation()

    def setup_sensor_semantic_annotations(self):
        head_center = HSRBHeadCenterCamera.setup_default_configuration_in_world(
            self._world
        )
        self.add_camera(head_center)

        head_right = HSRBHeadRightCamera.setup_default_configuration_in_world(
            self._world
        )
        self.add_camera(head_right)

        head_left = HSRBHeadLeftCamera.setup_default_configuration_in_world(self._world)
        self.add_camera(head_left)

        head_rgbd = HSRBHeadRGBDCamera.setup_default_configuration_in_world(self._world)
        self.add_camera(head_rgbd)


@dataclass(eq=False)
class HSRBMobileBase(MobileBase, HasTorso):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        mobile_base = cls(
            root=world.get_body_by_name("base_link"),
            forward_axis=Vector3.X(),
            full_body_controlled=True,
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_default_torso_semantic_annotation(self):
        torso = HSRBTorso.setup_default_configuration_in_world(self._world)
        self.add_torso(torso)
        torso.setup_arm_semantic_annotations()
        torso.setup_sensor_semantic_annotations()


@dataclass(eq=False)
class HSRB(AbstractRobot, HasMobileBase):
    """
    Class that describes the Human Support Robot variant B (https://upmroboticclub.wordpress.com/robot/).
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = HSRBMobileBase.setup_default_configuration_in_world(self._world)
        mobile_base.setup_default_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()
