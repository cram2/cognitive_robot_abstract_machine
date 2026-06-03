from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Union

from typing_extensions import TYPE_CHECKING, Type, TypeVar, Generic

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.patterns.subclass_safe_generic import (
    AbstractSubClassSafeGeneric,
)
from semantic_digital_twin.reasoning.predicates import LeftOf, RightOf
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

logger = logging.getLogger("semantic_digital_twin")

GenericFinger = TypeVar("GenericFinger")
GenericSensor = TypeVar("GenericSensor")
GenericCamera = TypeVar("GenericCamera")
GenericEndEffector = TypeVar("GenericEndEffector")
GenericArm = TypeVar("GenericArm")
GenericMobileBase = TypeVar("GenericMobileBase")
GenericTorso = TypeVar("GenericTorso")
GenericNeck = TypeVar("GenericNeck")
GenericLeftArm = TypeVar("GenericLeftArm")
GenericRightArm = TypeVar("GenericRightArm")
GenericLeftFinger = TypeVar("GenericLeftFinger")
GenericRightFinger = TypeVar("GenericRightFinger")


@dataclass(eq=False)
class RobotPartMixin(HasRootBody, ABC):
    pass


@dataclass(eq=False)
class HasFingers(Generic[GenericFinger], AbstractSubClassSafeGeneric, ABC):
    """
    Mixin class for robots or robot parts that have fingers as their direct children.
    """

    fingers: list[GenericFinger] = field(default_factory=list, kw_only=True)
    """
    The list of fingers attached to the robot.
    """

    @property
    def thumb(self) -> GenericFinger:
        [thumb] = [finger for finger in self.fingers if finger.is_thumb]
        return thumb


@dataclass(eq=False)
class HasTwoFingers(
    Generic[GenericLeftFinger, GenericRightFinger],
    HasFingers[Union[GenericLeftFinger, GenericRightFinger]],
    AbstractSubClassSafeGeneric,
    ABC,
):
    """
    Mixin class for robots or robot parts that have exactly two fingers, one of which is a thumb.
    """

    @property
    def finger(self) -> Union[GenericLeftFinger, GenericRightFinger]:
        [finger] = [finger for finger in self.fingers if not finger.is_thumb]
        return finger


@dataclass(eq=False)
class HasSensors(Generic[GenericSensor], AbstractSubClassSafeGeneric, ABC):
    """
    Mixin class for robots or robot parts that have sensors
    """

    sensors: list[GenericSensor] = field(default_factory=list, kw_only=True)
    """
    The list of sensors associated with the robot part.GenericFinger
    """


@dataclass(eq=False)
class HasEndEffector(Generic[GenericEndEffector], AbstractSubClassSafeGeneric, ABC):
    """
    Mixin class for robots or robot parts that have an end effector as their direct child.
    """

    end_effector: GenericEndEffector = field(default=None, kw_only=True)
    """
    The end effector attached to the robot part.
    """


@dataclass(eq=False)
class HasArms(Generic[GenericArm], AbstractSubClassSafeGeneric, ABC):
    """
    Mixin class for robots or robot parts that have arms as their direct children.
    """

    arms: list[GenericArm] = field(default_factory=list, kw_only=True)
    """
    The list of arms attached to the robot part.
    """


@dataclass(eq=False)
class HasOneArm(HasArms[GenericArm], ABC):
    """
    Mixin class for robots or robot parts that have exactly one arm.
    """

    @property
    def arm(self) -> GenericArm:
        [arm] = self.arms
        return arm


@dataclass(eq=False)
class HasLeftRightArm(
    Generic[GenericLeftArm, GenericRightArm],
    HasArms[Union[GenericLeftArm, GenericRightArm]],
    AbstractSubClassSafeGeneric,
    ABC,
):
    """
    Mixin class for robots or robot parts that have two arms and can specify which is the left and which is the right arm.
    """

    @cached_property
    def left_arm(self) -> GenericLeftArm:
        from semantic_digital_twin.reasoning.predicates import LeftOf

        return self._assign_left_right_arms(LeftOf)

    @cached_property
    def right_arm(self) -> GenericRightArm:
        from semantic_digital_twin.reasoning.predicates import RightOf

        return self._assign_left_right_arms(RightOf)

    def _assign_left_right_arms(
        self, relation: Type[Union[LeftOf, RightOf]]
    ) -> Union[GenericLeftArm, GenericRightArm]:
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
class HasMobileBase(Generic[GenericMobileBase], AbstractSubClassSafeGeneric, ABC):
    """
    Mixin class for robots that have a mobile base.
    """

    mobile_base: GenericMobileBase = field(default=None, kw_only=True)
    """
    The mobile base attached to the robot part.
    """


@dataclass(eq=False)
class HasTorso(Generic[GenericTorso], AbstractSubClassSafeGeneric, ABC):
    """
    Mixin class for robots or robot parts that have a torso as their direct child.
    """

    torso: GenericTorso = field(default=None, kw_only=True)
    """
    The torso attached to the robot part.
    """


@dataclass(eq=False)
class HasNeck(Generic[GenericNeck], AbstractSubClassSafeGeneric, ABC):
    """
    Mixin class for robots or robot parts that have a neck as their direct child.
    """

    neck: GenericNeck = field(default=None, kw_only=True)
    """
    The neck attached to the robot part.
    """
