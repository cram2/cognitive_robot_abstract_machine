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
class UnitreeG1Finger(Finger, ABC): ...


@dataclass(eq=False)
class UnitreeG1LeftThumb(UnitreeG1Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("left_hand_thumb_0_link"),
            tip=world.get_body_by_name("left_hand_thumb_2_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1LeftMiddleFinger(UnitreeG1Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("left_hand_middle_0_link"),
            tip=world.get_body_by_name("left_hand_middle_1_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1LeftIndexFinger(UnitreeG1Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("left_hand_index_0_link"),
            tip=world.get_body_by_name("left_hand_index_1_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1RightThumb(UnitreeG1Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("right_hand_thumb_0_link"),
            tip=world.get_body_by_name("right_hand_thumb_2_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1RightMiddleFinger(UnitreeG1Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("right_hand_middle_0_link"),
            tip=world.get_body_by_name("right_hand_middle_1_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1RightIndexFinger(UnitreeG1Finger):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("right_hand_index_0_link"),
            tip=world.get_body_by_name("right_hand_index_1_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1LeftHand(HumanoidHand):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        hand = cls(
            root=world.get_body_by_name("left_hand_palm_link"),
            tool_frame=world.get_body_by_name("left_hand_tool_frame"),
            front_facing_orientation=Quaternion(),
        )
        world.add_semantic_annotation(hand)
        return hand

    def setup_finger_semantic_annotations(self):
        thumb = UnitreeG1LeftThumb.setup_default_configuration_in_world(self._world)
        self.add_thumb(thumb)

        middle = UnitreeG1LeftMiddleFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(middle)

        index = UnitreeG1LeftIndexFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(index)


@dataclass(eq=False)
class UnitreeG1RightHand(HumanoidHand):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        hand = cls(
            root=world.get_body_by_name("right_hand_palm_link"),
            tool_frame=world.get_body_by_name("right_hand_tool_frame"),
            front_facing_orientation=Quaternion(),
        )
        world.add_semantic_annotation(hand)
        return hand

    def setup_finger_semantic_annotations(self):
        thumb = UnitreeG1RightThumb.setup_default_configuration_in_world(self._world)
        self.add_thumb(thumb)

        middle = UnitreeG1RightMiddleFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(middle)

        index = UnitreeG1RightIndexFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(index)


@dataclass(eq=False)
class UnitreeG1LeftArm(Arm, HasHumanoidHand):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("left_shoulder_pitch_link"),
            tip=world.get_body_by_name("left_wrist_yaw_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        hand = UnitreeG1LeftHand.setup_default_configuration_in_world(self._world)
        self.add_end_effector(hand)
        hand.setup_finger_semantic_annotations()


@dataclass(eq=False)
class UnitreeG1RightArm(Arm, HasHumanoidHand):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("right_shoulder_pitch_link"),
            tip=world.get_body_by_name("right_wrist_yaw_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        hand = UnitreeG1RightHand.setup_default_configuration_in_world(self._world)
        self.add_end_effector(hand)
        hand.setup_finger_semantic_annotations()


@dataclass(eq=False)
class UnitreeG1Camera(Camera):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("d435_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.60,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class UnitreeG1Torso(Torso, HasLeftRightArm, HasCameras):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        torso = cls(
            root=world.get_body_by_name("pelvis"),
            tip=world.get_body_by_name("torso_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        left_arm = UnitreeG1LeftArm.setup_default_configuration_in_world(self._world)
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = UnitreeG1RightArm.setup_default_configuration_in_world(self._world)
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_sensor_semantic_annotations(self):
        camera = UnitreeG1Camera.setup_default_configuration_in_world(self._world)
        self.add_camera(camera)


@dataclass(eq=False)
class UnitreeG1MobileBase(MobileBase, HasTorso):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        mobile_base = cls(
            root=world.get_body_by_name("pelvis"),
            forward_axis=Vector3.X(),
            full_body_controlled=False,
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_default_torso_semantic_annotation(self):
        torso = UnitreeG1Torso.setup_default_configuration_in_world(self._world)
        self.add_torso(torso)
        torso.setup_arm_semantic_annotations()
        torso.setup_sensor_semantic_annotations()


@dataclass(eq=False)
class UnitreeG1(AbstractRobot, HasMobileBase):
    """
    A class representing the Unitree G1 robot by the Unitree Robotics team.
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "pelvis"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = UnitreeG1MobileBase.setup_default_configuration_in_world(
            self._world
        )
        mobile_base.setup_default_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()
