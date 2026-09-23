from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Union

from typing_extensions import (
    ClassVar,
    Type,
    TypeVar,
    Generic,
    TypeVarTuple,
    Unpack,
)

from krrood.patterns.subclass_safe_generic import (
    SubClassSafeGeneric,
)
from krrood.utils import get_generic_type_parameters
from semantic_digital_twin.datastructures.lidar_reading import LidarReading
from semantic_digital_twin.reasoning.predicates import LeftOf, RightOf
from semantic_digital_twin.robots.exceptions import (
    MissingEndEffectorError,
    MissingLidarError,
    MissingMobileBaseError,
    MissingNeckError,
    MissingSensorsError,
    MissingTorsoError,
    TooFewArmsError,
    TooFewFingersError,
    UnexpectedArmCountError,
    UnexpectedFingerCountError,
)

logger = logging.getLogger("semantic_digital_twin")

TGenericFingerOtherThanThumb = TypeVar("TGenericFingerOtherThanThumb")
TGenericThumb = TypeVar("TGenericThumb")
TGenericCamera = TypeVar("TGenericCamera")
TGenericEndEffector = TypeVar("TGenericEndEffector")
TGenericArm = TypeVar("TGenericArm")
TGenericMobileBase = TypeVar("TGenericMobileBase")
TGenericMountingTable = TypeVar("TGenericMountingTable")
TGenericTorso = TypeVar("TGenericTorso")
TGenericNeck = TypeVar("TGenericNeck")
TGenericLeftArm = TypeVar("TGenericLeftArm")
TGenericRightArm = TypeVar("TGenericRightArm")
TGenericLeftFinger = TypeVar("TGenericLeftFinger")
TGenericRightFinger = TypeVar("TGenericRightFinger")

TGenericFingers = TypeVarTuple("TGenericFingers")
TGenericArms = TypeVarTuple("TGenericArms")
TGenericSensors = TypeVarTuple("TGenericSensors")
TGenericLidar = TypeVar("TGenericLidar")


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

    def validate_assumptions(self):
        """
        Checks the assumptions of every mixin this robot part combines.

        ..note:: Calling :meth:`validate` would reach only one mixin, since a part
            combining several of them resolves the name to the first.
        """
        for mixin in self._narrowest_mixins():
            mixin.validate(self)

    def _narrowest_mixins(self) -> list[Type[RobotPartMixin]]:
        """
        :return: The mixins stating this part's assumptions, leaving out every mixin
            another one of them narrows.
        """
        mixins = [
            ancestor
            for ancestor in type(self).__mro__
            if issubclass(ancestor, RobotPartMixin) and "validate" in vars(ancestor)
        ]
        return [
            mixin
            for mixin in mixins
            if not any(
                other is not mixin and issubclass(other, mixin) for other in mixins
            )
        ]


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

    minimum_finger_count: ClassVar[int] = 3
    """
    How many fingers a part combining this mixin has at least.
    """

    fingers: list[Union[TGenericThumb, Unpack[TGenericFingers]]] = field(
        default_factory=list, kw_only=True
    )
    """
    The list of fingers attached to the robot.
    """

    def validate(self):
        """
        :raises TooFewFingersError: If fewer fingers are attached than this mixin
            allows.
        """
        if len(self.fingers) < self.minimum_finger_count:
            raise TooFewFingersError(
                robot_part=self,
                minimum_count=self.minimum_finger_count,
                actual_count=len(self.fingers),
            )

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
    Mixin class for robots or robot parts that have exactly two fingers, one of which is
    a thumb.
    """

    finger_count: ClassVar[int] = 2
    """
    How many fingers a part combining this mixin has.
    """

    def validate(self):
        """
        :raises UnexpectedFingerCountError: If a different number of fingers is attached
            than this mixin allows.
        """
        if len(self.fingers) != self.finger_count:
            raise UnexpectedFingerCountError(
                robot_part=self,
                expected_count=self.finger_count,
                actual_count=len(self.fingers),
            )

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
    Mixin class for robots or robot parts that have sensors.
    """

    sensors: list[Union[Unpack[TGenericSensors]]] = field(
        default_factory=list, kw_only=True
    )
    """
    The list of sensors associated with the robot part.
    """

    def validate(self):
        """
        :raises MissingSensorsError: If no sensor is attached.
        """
        if not self.sensors:
            raise MissingSensorsError(robot_part=self)


@dataclass(eq=False)
class HasEndEffector(
    Generic[TGenericEndEffector], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robots or robot parts that have an end effector as their direct
    child.
    """

    end_effector: TGenericEndEffector = field(default=None, kw_only=True)
    """
    The end effector attached to the robot part.
    """

    def validate(self):
        """
        :raises MissingEndEffectorError: If no end effector is attached.
        """
        if self.end_effector is None:
            raise MissingEndEffectorError(robot_part=self)


@dataclass(eq=False)
class HasArms(Generic[Unpack[TGenericArms]], SubClassSafeGeneric, RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have arms as their direct children.
    """

    minimum_arm_count: ClassVar[int] = 3
    """
    How many arms a part combining this mixin has at least.
    """

    arms: list[Union[Unpack[TGenericArms]]] = field(default_factory=list, kw_only=True)
    """
    The list of arms attached to the robot part.
    """

    def validate(self):
        """
        :raises TooFewArmsError: If fewer arms are attached than this mixin allows.
        """
        if len(self.arms) < self.minimum_arm_count:
            raise TooFewArmsError(
                robot_part=self,
                minimum_count=self.minimum_arm_count,
                actual_count=len(self.arms),
            )


@dataclass(eq=False)
class HasOneArm(HasArms[TGenericArm], RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have exactly one arm.
    """

    arm_count: ClassVar[int] = 1
    """
    How many arms a part combining this mixin has.
    """

    def validate(self):
        """
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        if len(self.arms) != self.arm_count:
            raise UnexpectedArmCountError(
                robot_part=self,
                expected_count=self.arm_count,
                actual_count=len(self.arms),
            )

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
    Mixin class for robots or robot parts that have two arms and can specify which is
    the left and which is the right arm.
    """

    arm_count: ClassVar[int] = 2
    """
    How many arms a part combining this mixin has.
    """

    def validate(self):
        """
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        if len(self.arms) != self.arm_count:
            raise UnexpectedArmCountError(
                robot_part=self,
                expected_count=self.arm_count,
                actual_count=len(self.arms),
            )

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
        Assigns the left and right arms based on their position relative to the robot's
        root body.

        :param relation: The relation to use for determining left or right (LeftOf or
            RightOf).
        :return: The arm that is on the left or right side of the robot.
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        HasLeftRightArm.validate(self)
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
        """
        :raises MissingMobileBaseError: If no mobile base is attached.
        """
        if self.mobile_base is None:
            raise MissingMobileBaseError(robot_part=self)


@dataclass(eq=False)
class HasMountingTable(
    Generic[TGenericMountingTable], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for stationary robots bolted onto a table.
    """

    table: TGenericMountingTable = field(default=None, kw_only=True)
    """
    The table the robot is mounted on.
    """

    def validate(self):
        assert self.table is not None, "Expected table, got None"


@dataclass(eq=False)
class HasTorso(Generic[TGenericTorso], SubClassSafeGeneric, RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have a torso as their direct child.
    """

    torso: TGenericTorso = field(default=None, kw_only=True)
    """
    The torso attached to the robot part.
    """

    def validate(self):
        """
        :raises MissingTorsoError: If no torso is attached.
        """
        if self.torso is None:
            raise MissingTorsoError(robot_part=self)


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
        """
        :raises MissingNeckError: If no neck is attached.
        """
        if self.neck is None:
            raise MissingNeckError(robot_part=self)


@dataclass(eq=False)
class HasLidar(Generic[TGenericLidar], SubClassSafeGeneric, RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have a lidar as their direct child.
    """

    lidar: TGenericLidar = field(default=None, kw_only=True)
    """
    The lidar attached to the robot part.
    """

    def validate(self):
        """
        :raises MissingLidarError: If no lidar is attached.
        """
        if self.lidar is None:
            raise MissingLidarError(robot_part=self)

    def get_lidar_reading(self) -> LidarReading:
        """
        :return: The most recent sweep of the attached lidar.
        """
        return self.lidar.get_lidar_reading()
