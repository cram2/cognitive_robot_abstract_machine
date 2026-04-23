from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Union, Set
from uuid import UUID

from typing_extensions import TYPE_CHECKING, Type

from krrood.adapters.json_serializer import list_like_classes
from krrood.class_diagrams.attribute_introspector import (
    DataclassOnlyIntrospector,
    DiscoveredAttribute,
)
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from semantic_digital_twin.reasoning.predicates import LeftOf, RightOf
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import (
        Finger,
        Sensor,
        Camera,
        EndEffector,
        MechanicalGripper,
        ParallelGripper,
        Arm,
        MobileBase,
        Torso,
        Neck,
        HumanoidHand,
    )

logger = logging.getLogger("semantic_digital_twin")


@dataclass(eq=False)
class HasFingers(ABC):
    """
    Mixin class for robots or robot parts that have fingers as their direct children.
    """

    fingers: list[Finger] = field(default_factory=list)
    """
    The list of fingers attached to the robot.
    """

    thumb: Finger = field(default=None)
    """
    The thumb is a finger that always needs to be involved in the manipulation of objects.
    """

    @synchronized_attribute_modification
    def add_finger(self, finger: Finger):
        if finger == self.thumb:
            raise Exception(f"This finger is already part of the robot {self}.")
        self.fingers.append(finger)

    @synchronized_attribute_modification
    def add_thumb(self, thumb: Finger):
        if thumb in self.fingers:
            raise Exception(f"This finger is already part of the robot {self}.")
        self.thumb = thumb

    @abstractmethod
    def setup_finger_semantic_annotations(self):
        """
        Sets up the semantic annotations for the fingers of this robot part.
        """


@dataclass(eq=False)
class HasTwoFingers(HasFingers, ABC):
    """
    Mixin class for robots or robot parts that have exactly two fingers, one of which is a thumb.
    """

    @property
    def finger(self):
        if len(self.fingers) != 1 or self.thumb is None:
            raise Exception(
                f"This robot has {len(self.fingers)} fingers and {int(bool(self.thumb))} thumbs. Should have exactly one finger and one thumb"
            )
        return self.fingers[0]


@dataclass(eq=False)
class HasSensors(ABC):
    """
    Mixin class for robots or robot parts that have sensors
    """

    sensors: list[Sensor] = field(default_factory=list)
    """
    The list of sensors associated with the robot part.
    """

    @synchronized_attribute_modification
    def add_sensor(self, sensor: Sensor):
        self.sensors.append(sensor)

    @abstractmethod
    def setup_sensor_semantic_annotations(self):
        """
        Sets up the semantic annotations for the sensors of this robot part.
        """


@dataclass(eq=False)
class HasCameras(HasSensors, ABC):
    """
    Mixin class for robots or robot parts that have cameras as sensors.
    """

    cameras: list[Camera] = field(default_factory=list)
    """
    The list of cameras associated with the robot part.
    """

    @synchronized_attribute_modification
    def add_camera(self, camera: Camera):
        self.cameras.append(camera)
        self.sensors.append(camera)


@dataclass(eq=False)
class HasEndEffector(ABC):
    """
    Mixin class for robots or robot parts that have an end effector as their direct child.
    """

    end_effector: EndEffector = field(default=None)
    """
    The end effector attached to the robot part.
    """

    @synchronized_attribute_modification
    def add_end_effector(self, end_effector: EndEffector):
        self.end_effector = end_effector

    @abstractmethod
    def setup_end_effector_semantic_annotation(self):
        """
        Sets up the semantic annotation for the end effector of this robot part.
        """


@dataclass(eq=False)
class HasParallelGripper(HasEndEffector, ABC):
    """
    Mixin class for robots or robot parts that have a parallel gripper as their end effector.
    """

    end_effector: ParallelGripper = field(default=None)
    """
    The parallel gripper attached to the robot part.
    """


@dataclass(eq=False)
class HasHumanoidHand(HasEndEffector, ABC):
    """
    Mixin class for robots or robot parts that have a humanoid hand as their end effector.
    """

    end_effector: HumanoidHand = field(default=None)
    """
    The humanoid hand attached to the robot part.
    """


@dataclass(eq=False)
class HasArms(ABC):
    """
    Mixin class for robots or robot parts that have arms as their direct children.
    """

    arms: list[Arm] = field(default_factory=list)
    """
    The list of arms attached to the robot part.
    """

    @synchronized_attribute_modification
    def add_arm(self, arm: Arm):
        self.arms.append(arm)

    @abstractmethod
    def setup_arm_semantic_annotations(self):
        """
        Sets up the semantic annotations for the arms of this robot part.
        """


@dataclass(eq=False)
class HasOneArm(HasArms, ABC):
    """
    Mixin class for robots or robot parts that have exactly one arm.
    """

    @synchronized_attribute_modification
    def add_arm(self, arm: Arm):
        if len(self.arms) != 0:
            raise Exception(f"This robot already has an arm {self.arms}")
        self.arms.append(arm)

    @property
    def arm(self) -> Arm:
        return self.arms[0]


@dataclass(eq=False)
class HasLeftRightArm(HasArms, ABC):
    """
    Mixin class for robots or robot parts that have two arms and can specify which is the left and which is the right arm.
    """

    @cached_property
    def left_arm(self):
        from semantic_digital_twin.reasoning.predicates import LeftOf

        return self._assign_left_right_arms(LeftOf)

    @cached_property
    def right_arm(self):
        from semantic_digital_twin.reasoning.predicates import RightOf

        return self._assign_left_right_arms(RightOf)

    def _assign_left_right_arms(self, relation: Type[Union[LeftOf, RightOf]]) -> Arm:
        """
        Assigns the left and right arms based on their position relative to the robot's root body.
        :param relation: The relation to use for determining left or right (LeftOf or RightOf).
        :return: The arm that is on the left or right side of the robot.
        """
        assert (
            len(self.arms) == 2
        ), f"Must have exactly two arms to specify left and right arm, but found {len(self.arms)}."
        pov = self.root.global_transform
        first_arm = self.arms[0]
        second_arm = self.arms[1]
        # the arms may share a root, but the first body after the root should be different
        world_P_first_body = first_arm.bodies[1].global_transform.to_position()
        world_P_second_body = second_arm.bodies[1].global_transform.to_position()

        return (
            first_arm
            if relation(
                world_P_first_body,
                world_P_second_body,
                pov,
            )()
            else second_arm
        )


@dataclass(eq=False)
class HasMobileBase(ABC):
    """
    Mixin class for robots that have a mobile base.
    """

    mobile_base: MobileBase = field(default=None)
    """
    The mobile base attached to the robot part.
    """

    @synchronized_attribute_modification
    def add_mobile_base(self, mobile_base: MobileBase):
        self.mobile_base = mobile_base

    @abstractmethod
    def setup_mobile_base_semantic_annotation(self):
        """
        Sets up the semantic annotation for the mobile base of this robot.
        """


@dataclass(eq=False)
class HasTorso(ABC):
    """
    Mixin class for robots or robot parts that have a torso as their direct child.
    """

    torso: Torso = field(default=None)
    """
    The torso attached to the robot part.
    """

    @synchronized_attribute_modification
    def add_torso(self, torso: Torso):
        self.torso = torso

    @abstractmethod
    def setup_torso_semantic_annotation(self):
        """
        Sets up the semantic annotation for the torso of this robot part.
        """


@dataclass(eq=False)
class HasNeck(ABC):
    """
    Mixin class for robots or robot parts that have a neck as their direct child.
    """

    neck: Neck = field(default=None)
    """
    The neck attached to the robot part.
    """

    @synchronized_attribute_modification
    def add_neck(self, neck: Neck):
        self.neck = neck

    @abstractmethod
    def setup_neck_semantic_annotation(self):
        """
        Sets up the semantic annotation for the neck of this robot part.
        """
