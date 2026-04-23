from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from semantic_digital_twin.robots.robot_part_mixins import (
    HasMobileBase,
    HasTorso,
    HasLeftRightArm,
    HasHumanoidHand,
    HasCameras,
)
from semantic_digital_twin.robots.robot_parts import (
    Finger,
    HumanoidHand,
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
class ICub3Finger(Finger, ABC): ...


@dataclass(eq=False)
class ICub3LeftThumb(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("l_hand_thumb_0"),
            tip=world.get_body_by_name("l_hand_thumb_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3LeftIndexFinger(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("l_hand_index_0"),
            tip=world.get_body_by_name("l_hand_index_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3LeftMiddleFinger(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("l_hand_middle_0"),
            tip=world.get_body_by_name("l_hand_middle_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3LeftRingFinger(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("l_hand_ring_0"),
            tip=world.get_body_by_name("l_hand_ring_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3LeftLittleFinger(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("l_hand_little_0"),
            tip=world.get_body_by_name("l_hand_little_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightThumb(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("r_hand_thumb_0"),
            tip=world.get_body_by_name("r_hand_thumb_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightIndexFinger(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("r_hand_index_0"),
            tip=world.get_body_by_name("r_hand_index_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightMiddleFinger(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("r_hand_middle_0"),
            tip=world.get_body_by_name("r_hand_middle_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightRingFinger(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("r_hand_ring_0"),
            tip=world.get_body_by_name("r_hand_ring_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightLittleFinger(ICub3Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("r_hand_little_0"),
            tip=world.get_body_by_name("r_hand_little_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3LeftHand(HumanoidHand):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        hand = cls(
            root=world.get_body_by_name("l_hand"),
            tool_frame=world.get_body_by_name("l_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )
        world.add_semantic_annotation(hand)
        return hand

    def setup_finger_semantic_annotations(self):
        thumb = ICub3LeftThumb.setup_default_configuration_in_world(self._world)
        self.add_thumb(thumb)

        index = ICub3LeftIndexFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(index)

        middle = ICub3LeftMiddleFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(middle)

        ring = ICub3LeftRingFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(ring)

        little = ICub3LeftLittleFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(little)


@dataclass(eq=False)
class ICub3RightHand(HumanoidHand):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        hand = cls(
            root=world.get_body_by_name("r_hand"),
            tool_frame=world.get_body_by_name("r_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0, 0, -0.707, 0.707),
        )
        world.add_semantic_annotation(hand)
        return hand

    def setup_finger_semantic_annotations(self):
        thumb = ICub3RightThumb.setup_default_configuration_in_world(self._world)
        self.add_thumb(thumb)

        index = ICub3RightIndexFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(index)

        middle = ICub3RightMiddleFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(middle)

        ring = ICub3RightRingFinger.setup_default_configuration_in_world(self._world)
        self.add_finger(ring)

        little = ICub3RightLittleFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(little)


@dataclass(eq=False)
class ICub3LeftArm(Arm, HasHumanoidHand):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("root_link"),
            tip=world.get_body_by_name("l_hand"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        hand = ICub3LeftHand.setup_default_configuration_in_world(self._world)
        self.add_end_effector(hand)
        hand.setup_finger_semantic_annotations()


@dataclass(eq=False)
class ICub3RightArm(Arm, HasHumanoidHand):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("root_link"),
            tip=world.get_body_by_name("r_hand"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        hand = ICub3RightHand.setup_default_configuration_in_world(self._world)
        self.add_end_effector(hand)
        hand.setup_finger_semantic_annotations()


@dataclass(eq=False)
class ICub3Camera(Camera):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("head"),
            forward_facing_axis=Vector3(1, 0, 0),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.85,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class ICub3Torso(Torso, HasLeftRightArm, HasCameras):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        torso = cls(
            root=world.get_body_by_name("root_link"),
            tip=world.get_body_by_name("chest"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        left_arm = ICub3LeftArm.setup_default_configuration_in_world(self._world)
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = ICub3RightArm.setup_default_configuration_in_world(self._world)
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_sensor_semantic_annotations(self):
        camera = ICub3Camera.setup_default_configuration_in_world(self._world)
        self.add_camera(camera)


@dataclass(eq=False)
class ICub3MobileBase(MobileBase, HasTorso):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        mobile_base = cls(
            root=world.get_body_by_name("base_footprint"),
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_default_torso_semantic_annotation(self):
        torso = ICub3Torso.setup_default_configuration_in_world(self._world)
        self.add_torso(torso)
        torso.setup_arm_semantic_annotations()
        torso.setup_sensor_semantic_annotations()


@dataclass(eq=False)
class ICub3(AbstractRobot, HasMobileBase):
    """
    Class that describes the iCub3 Robot.
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = ICub3MobileBase.setup_default_configuration_in_world(self._world)
        mobile_base.setup_default_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()
