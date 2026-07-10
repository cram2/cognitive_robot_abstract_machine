from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Union

from typing_extensions import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    Generic,
    TypeVarTuple,
    Unpack,
)

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.patterns.subclass_safe_generic import (
    SubClassSafeGeneric,
)
from krrood.utils import get_generic_type_parameters
from semantic_digital_twin.reasoning.predicates import LeftOf, RightOf
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

logger = logging.getLogger("semantic_digital_twin")

TGenericFingerOtherThanThumb = TypeVar("TGenericFingerOtherThanThumb")
TGenericThumb = TypeVar("TGenericThumb")
TGenericCamera = TypeVar("TGenericCamera")
TGenericEndEffector = TypeVar("TGenericEndEffector")
TGenericArm = TypeVar("TGenericArm")
TGenericMobileBase = TypeVar("TGenericMobileBase")
TGenericTorso = TypeVar("TGenericTorso")
TGenericNeck = TypeVar("TGenericNeck")
TGenericLeftArm = TypeVar("TGenericLeftArm")
TGenericRightArm = TypeVar("TGenericRightArm")
TGenericLeftFinger = TypeVar("TGenericLeftFinger")
TGenericRightFinger = TypeVar("TGenericRightFinger")

TGenericFingers = TypeVarTuple("TGenericFingers")
TGenericArms = TypeVarTuple("TGenericArms")
TGenericSensors = TypeVarTuple("TGenericSensors")


@dataclass(eq=False)
class RobotPartMixin(ABC):
    """
    Base mixin class for robot parts.
    """

    @abstractmethod
    def validate(self):
        """
        Validation method that describes assumptions made about the robot part.
        """


@dataclass(eq=False)
class HasFingers(
    Generic[TGenericThumb, Unpack[TGenericFingers]],
    SubClassSafeGeneric,
    RobotPartMixin,
    ABC,
):
    """
    Mixin class for robots or robot parts that have fingers as their direct children.
    """

    fingers: list[Union[TGenericThumb, Unpack[TGenericFingers]]] = field(
        default_factory=list, kw_only=True
    )
    """
    The list of fingers attached to the robot.
    """

    def validate(self):
        """
        Validation method that checks that there is exactly one thumb in the fingers list.
        """
        assert (
            len(self.fingers) >= 3
        ), f"Expected at least 3 fingers, got {len(self.fingers)}. If this RobotPart is supposed to only have two use HasTwoFingers instead."

    @property
    def thumb(self) -> TGenericThumb:
        concrete_thumb_class = get_generic_type_parameters(self, HasFingers)[0]
        [thumb] = [
            finger
            for finger in self.fingers
            if isinstance(finger, concrete_thumb_class)
        ]
        return thumb


@dataclass(eq=False)
class HasTwoFingers(
    Generic[TGenericLeftFinger, TGenericRightFinger],
    HasFingers[TGenericLeftFinger, TGenericRightFinger],
    SubClassSafeGeneric,
    ABC,
):
    """
    Mixin class for robots or robot parts that have exactly two fingers, one of which is a thumb.
    """

    def validate(self):
        assert (
            len(self.fingers) == 2
        ), f"Expected exactly 2 fingers, got {len(self.fingers)}"

    @property
    def finger(self) -> Union[TGenericLeftFinger, TGenericRightFinger]:
        concrete_thumb_class = get_generic_type_parameters(self, HasFingers)[0]

        [finger] = [
            finger
            for finger in self.fingers
            if not isinstance(finger, concrete_thumb_class)
        ]
        return finger


@dataclass(eq=False)
class HasSensors(
    Generic[Unpack[TGenericSensors]], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robots or robot parts that have sensors
    """

    sensors: list[Union[Unpack[TGenericSensors]]] = field(
        default_factory=list, kw_only=True
    )
    """
    The list of sensors associated with the robot part.
    """

    def validate(self):
        assert len(self.sensors) > 0, f"Expected at least one sensor, got 0"


@dataclass(eq=False)
class HasEndEffector(
    Generic[TGenericEndEffector], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robots or robot parts that have an end effector as their direct child.
    """

    end_effector: TGenericEndEffector = field(default=None, kw_only=True)
    """
    The end effector attached to the robot part.
    """

    def validate(self):
        assert self.end_effector is not None, f"Expected end effector, got None"


@dataclass(eq=False)
class HasArms(
    Generic[Unpack[TGenericArms]], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robots or robot parts that have arms as their direct children.
    """

    arms: list[Union[Unpack[TGenericArms]]] = field(default_factory=list, kw_only=True)
    """
    The list of arms attached to the robot part.
    """

    def validate(self):
        assert (
            len(self.arms) > 2
        ), f"Expected at least three arms, got {len(self.arms)}. If your robot only has one arm, use HasOneArm instead. If it has two arms, consider using HasLeftRightArm instead."


@dataclass(eq=False)
class HasOneArm(HasArms[TGenericArm], RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have exactly one arm.
    """

    def validate(self):
        assert len(self.arms) == 1, f"Expected exactly one arm, got {len(self.arms)}"

    @property
    def arm(self) -> TGenericArm:
        [arm] = self.arms
        return arm


@dataclass(eq=False)
class HasLeftRightArm(
    HasArms[TGenericLeftArm, TGenericRightArm],
    SubClassSafeGeneric,
    RobotPartMixin,
    ABC,
):
    """
    Mixin class for robots or robot parts that have two arms and can specify which is the left and which is the right arm.
    """

    def validate(self):
        assert len(self.arms) == 2, f"Expected exactly two arms, got {len(self.arms)}"

    @cached_property
    def left_arm(self) -> TGenericLeftArm:
        from semantic_digital_twin.reasoning.predicates import LeftOf

        return self._assign_left_right_arms(LeftOf)

    @cached_property
    def right_arm(self) -> TGenericRightArm:
        from semantic_digital_twin.reasoning.predicates import RightOf

        return self._assign_left_right_arms(RightOf)

    def _assign_left_right_arms(
        self, relation: Type[Union[LeftOf, RightOf]]
    ) -> Union[TGenericLeftArm, TGenericRightArm]:
        """
        Assigns the left and right arms based on their position relative to the robot's root body.
        :param relation: The relation to use for determining left or right (LeftOf or RightOf).
        :return: The arm that is on the left or right side of the robot.
        """
        assert (
            len(self.arms) == 2
        ), f"Must have exactly two arms to specify left and right arm, but found {len(self.arms)}."
        pov = self.root.global_transform
        [first_arm, second_arm] = self.arms
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
class HasMobileBase(
    Generic[TGenericMobileBase], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robots that have a mobile base.
    """

    mobile_base: TGenericMobileBase = field(default=None, kw_only=True)
    """
    The mobile base attached to the robot part.
    """

    def validate(self):
        assert self.mobile_base is not None, "Expected mobile base, got None"


@dataclass(eq=False)
class HasTorso(
    Generic[TGenericTorso], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robots or robot parts that have a torso as their direct child.
    """

    torso: TGenericTorso = field(default=None, kw_only=True)
    """
    The torso attached to the robot part.
    """

    def validate(self):
        assert self.torso is not None, f"Expected torso, got None"


@dataclass(eq=False)
class HasNeck(Generic[TGenericNeck], SubClassSafeGeneric, RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have a neck as their direct child.
    """

    neck: TGenericNeck = field(default=None, kw_only=True)
    """
    The neck attached to the robot part.
    """

    def validate(self):
        assert self.neck is not None, f"Expected neck, got None"
