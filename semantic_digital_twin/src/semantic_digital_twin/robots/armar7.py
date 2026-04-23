from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasMobileBase,
    HasTorso,
    HasHumanoidHand,
    HasCameras,
)
from semantic_digital_twin.robots.robot_parts import (
    Finger,
    Arm,
    Camera,
    FieldOfView,
    Torso,
    HumanoidHand,
    MobileBase,
    AbstractRobot,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World


@dataclass(eq=False)
class Armar7Camera(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("AzureKinect_RGB_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.371500015258789,
            maximal_height=1.7365000247955322,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class Armar7Finger(Finger, ABC): ...


@dataclass(eq=False)
class Armar7LeftGripperThumb(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand L Palm_link"),
            tip=world.get_body_by_name("Thumb L Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7LeftGripperRing(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand L Palm_link"),
            tip=world.get_body_by_name("Ring L Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7LeftGripperPinky(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand L Palm_link"),
            tip=world.get_body_by_name("Pinky L Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7LeftGripperMiddle(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand L Palm_link"),
            tip=world.get_body_by_name("Middle L Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7LeftGripperIndex(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand L Palm_link"),
            tip=world.get_body_by_name("Index L Tip_link"),
            finger_tip_frame=world.get_body_by_name("Hand L Index TCP_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightGripperThumb(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand R Palm_link"),
            tip=world.get_body_by_name("Thumb R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightGripperRing(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand R Palm_link"),
            tip=world.get_body_by_name("Ring R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightGripperPinky(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand R Palm_link"),
            tip=world.get_body_by_name("Pinky R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightGripperMiddle(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand R Palm_link"),
            tip=world.get_body_by_name("Middle R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightGripperIndex(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("Hand R Palm_link"),
            tip=world.get_body_by_name("Index R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7LeftGripper(HumanoidHand):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        left_gripper = cls(
            root=world.get_body_by_name("ArmL8_Wrist_Hemisphere_B_link"),
            tool_frame=world.get_body_by_name("Hand L TCP_link"),
            front_facing_orientation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        )
        world.add_semantic_annotation(left_gripper)
        return left_gripper

    def setup_finger_semantic_annotations(self):
        thumb = Armar7LeftGripperThumb.setup_default_configuration_in_world(self._world)
        self.add_thumb(thumb)

        ring = Armar7LeftGripperRing.setup_default_configuration_in_world(self._world)
        self.add_finger(ring)

        pinky = Armar7LeftGripperPinky.setup_default_configuration_in_world(self._world)
        self.add_finger(pinky)

        middle = Armar7LeftGripperMiddle.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(middle)

        index = Armar7LeftGripperIndex.setup_default_configuration_in_world(self._world)
        self.add_finger(index)


@dataclass(eq=False)
class Armar7RightGripper(HumanoidHand):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        right_gripper = cls(
            root=world.get_body_by_name("ArmR8_Wrist_Hemisphere_B_link"),
            tool_frame=world.get_body_by_name("Hand R TCP_link"),
            front_facing_orientation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        )
        world.add_semantic_annotation(right_gripper)
        return right_gripper

    def setup_finger_semantic_annotations(self):
        thumb = Armar7RightGripperThumb.setup_default_configuration_in_world(
            self._world
        )
        self.add_thumb(thumb)

        ring = Armar7RightGripperRing.setup_default_configuration_in_world(self._world)
        self.add_finger(ring)

        pinky = Armar7RightGripperPinky.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(pinky)

        middle = Armar7RightGripperMiddle.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(middle)

        index = Armar7RightGripperIndex.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(index)


@dataclass(eq=False)
class Armar7LeftArm(Arm, HasHumanoidHand):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        left_arm = cls(
            root=world.get_body_by_name("CenterArms_fixed_link"),
            tip=world.get_body_by_name("ArmL8_Wrist_Hemisphere_B_link"),
        )
        world.add_semantic_annotation(left_arm)
        return left_arm

    def setup_end_effector_semantic_annotation(self):
        gripper = Armar7LeftGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class Armar7RightArm(Arm, HasHumanoidHand):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        right_arm = cls(
            root=world.get_body_by_name("CenterArms_fixed_link"),
            tip=world.get_body_by_name("ArmR8_Wrist_Hemisphere_B_link"),
        )
        world.add_semantic_annotation(right_arm)
        return right_arm

    def setup_end_effector_semantic_annotation(self):
        gripper = Armar7RightGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class Armar7Torso(Torso, HasLeftRightArm, HasCameras):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        torso = cls(
            root=world.get_body_by_name("Platform_link"),
            tip=world.get_body_by_name("CenterArms_fixed_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        left_arm = Armar7LeftArm.setup_default_configuration_in_world(self._world)
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = Armar7RightArm.setup_default_configuration_in_world(self._world)
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_sensor_semantic_annotations(self):
        camera = Armar7Camera.setup_default_configuration_in_world(self._world)
        self.add_camera(camera)


@dataclass(eq=False)
class Armar7MobileBase(MobileBase, HasTorso):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        mobile_base = cls(
            root=world.get_body_by_name("Platform_body_link"),
            forward_axis=Vector3.Y(),
            full_body_controlled=False,
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_default_torso_semantic_annotation(self):
        torso = Armar7Torso.setup_default_configuration_in_world(self._world)
        self.add_torso(torso)
        torso.setup_arm_semantic_annotations()
        torso.setup_sensor_semantic_annotations()


@dataclass(eq=False)
class Armar7(AbstractRobot, HasMobileBase):
    """
    Class that describes the Armar7 Robot.
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "Dummy_Platform_link"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = Armar7MobileBase.setup_default_configuration_in_world(self._world)
        mobile_base.setup_default_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()
