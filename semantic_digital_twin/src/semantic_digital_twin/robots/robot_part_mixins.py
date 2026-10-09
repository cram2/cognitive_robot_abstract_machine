from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from types import NoneType
from typing import Union, get_args, get_origin

from typing_extensions import (
    Optional,
    TYPE_CHECKING,
    Type,
    TypeVar,
    Generic,
    TypeVarTuple,
    Unpack,
)

from krrood.ormatic.utils import classproperty
from krrood.patterns.subclass_safe_generic import (
    SubClassSafeGeneric,
)
from krrood.utils import get_existing_field_by_name, get_generic_type_parameters
from semantic_digital_twin.datastructures.lidar_reading import LidarReading
from semantic_digital_twin.reasoning.predicates import LeftOf, RightOf
from semantic_digital_twin.robots.exceptions import (
    MissingEndEffectorError,
    MissingInputSourceError,
    MissingLidarError,
    MissingMobileBaseError,
    MissingNeckError,
    MissingSensorsError,
    MissingTorsoError,
    TooFewArmsError,
    TooFewFingersError,
    UnexpectedArmCountError,
    UnexpectedFingerCountError,
    UnexpectedInputSourceError,
    UndeclaredTopicError,
)
from semantic_digital_twin.robots.input_source import InputSource

if TYPE_CHECKING:
    from rclpy.node import Node

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
TGenericInputSource = TypeVar("TGenericInputSource", bound=InputSource)


@dataclass(eq=False)
class RobotPartMixin(ABC):
    """
    Base mixin class for robot parts.

    Every mixin states its own assumption in :meth:`validate` and then hands the check
    on to the next mixin of the part, so that a part combining several of them has all
    of their assumptions checked rather than only the first one's.
    """

    def validate(self):
        """
        Checks the assumptions this mixin makes about the robot part.

        Ends the chain of checks a part's mixins hand along, so a mixin that makes no
        assumption of its own needs no implementation.
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
        :raises TooFewFingersError: If fewer fingers are attached than this mixin
            allows.
        """
        if len(self.fingers) < 2:
            raise TooFewFingersError(
                robot_part=self,
                minimum_count=2,
                actual_count=len(self.fingers),
            )
        super().validate()

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

    def validate(self):
        """
        :raises UnexpectedFingerCountError: If a different number of fingers is attached
            than this mixin allows.
        """
        if len(self.fingers) != 2:
            raise UnexpectedFingerCountError(
                robot_part=self,
                expected_count=2,
                actual_count=len(self.fingers),
            )
        super().validate()

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
        super().validate()


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
        super().validate()


@dataclass(eq=False)
class HasArms(Generic[Unpack[TGenericArms]], SubClassSafeGeneric, RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have arms as their direct children.
    """

    arms: list[Union[Unpack[TGenericArms]]] = field(default_factory=list, kw_only=True)
    """
    The list of arms attached to the robot part.
    """

    def validate(self):
        """
        :raises TooFewArmsError: If fewer arms are attached than this mixin allows.
        """
        if len(self.arms) < 1:
            raise TooFewArmsError(
                robot_part=self,
                minimum_count=1,
                actual_count=len(self.arms),
            )
        super().validate()


@dataclass(eq=False)
class HasOneArm(HasArms[TGenericArm], RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have exactly one arm.
    """

    def validate(self):
        """
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        if len(self.arms) != 1:
            raise UnexpectedArmCountError(
                robot_part=self,
                expected_count=1,
                actual_count=len(self.arms),
            )
        super().validate()

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

    def validate(self):
        """
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        self._validate_arm_count()
        super().validate()

    def _validate_arm_count(self):
        """
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        if len(self.arms) != 2:
            raise UnexpectedArmCountError(
                robot_part=self,
                expected_count=2,
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
        self._validate_arm_count()
        pov = self.root.global_transform
        [first_arm, second_arm] = self.arms
        # the arms may share a root, but the first body after the root should be different
        world_P_first_body = first_arm.bodies[1].global_transform.position
        world_P_second_body = second_arm.bodies[1].global_transform.position

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
        super().validate()


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
        super().validate()


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
        super().validate()


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
        super().validate()

    def get_lidar_reading(self) -> LidarReading:
        """
        :return: The most recent sweep of the attached lidar.
        """
        return self.lidar.get_lidar_reading()


# %% where a part is read from


@dataclass(eq=False)
class HasInputSource(
    Generic[TGenericInputSource], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robot parts that can be read either from the world they stand in or
    from the robot they stand for.

    The kind of source a part can be read from is bound as the generic parameter, so a
    part cannot be handed a source meant for another kind of part.
    """

    source: Optional[TGenericInputSource] = field(default=None, kw_only=True)
    """
    Where this part is read from.

    ..note:: A family of parts re-declares this field under the same type variable, to
        give it the default its own kind of source has.
    """

    @classproperty
    def topic_name(cls) -> Optional[str]:
        """
        The topic the real robot publishes this part's state on, if its description
        names one.
        """
        return None

    def validate(self):
        """
        :raises MissingInputSourceError: If nothing says where this part is read from.
        """
        if self.source is None:
            raise MissingInputSourceError(robot_part=self)
        super().validate()

    @classmethod
    def source_family(cls) -> Type[TGenericInputSource]:
        """
        :return: The kind of source this part can be read from.

        ..note:: Read off :attr:`source`, which :class:`SubClassSafeGeneric` narrows to
            the type the part binds, so the binding stays the only place it is stated.
        """
        source_type = get_existing_field_by_name(cls, "source").type
        if get_origin(source_type) is not Union:
            return source_type
        [source_family] = [
            member for member in get_args(source_type) if member is not NoneType
        ]
        return source_family

    @classmethod
    @abstractmethod
    def simulated_source(cls) -> TGenericInputSource:
        """
        :return: The source reading this part from the world it stands in.
        """

    @abstractmethod
    def real_source(self, node: Node) -> TGenericInputSource:
        """
        :param node: The ros node the messages are received on.
        :return: The source reading this part from the robot itself, on the topic this
            part declares.
        """

    def use_simulated_source(self) -> None:
        """
        Read this part from the world it stands in.
        """
        self.use_source(self.simulated_source())

    def use_real_source(self, node: Node) -> None:
        """
        Read this part from the robot itself, on the topic it declares.

        :param node: The ros node the messages are received on.
        :raises UndeclaredTopicError: If this part declares no topic.
        """
        if self.topic_name is None:
            raise UndeclaredTopicError(robot_part=self)
        self.use_source(self.real_source(node))

    def use_source(self, source: TGenericInputSource) -> None:
        """
        Read this part from the given source from now on, releasing the one it was read
        from before.

        :param source: Where this part is read from.
        :raises UnexpectedInputSourceError: If the source is not one this part can be
            read from.
        """
        if not isinstance(source, self.source_family()):
            raise UnexpectedInputSourceError(
                robot_part=self,
                source=source,
                expected_source_family=self.source_family(),
            )
        if self.source is not None and self.source is not source:
            self.source.close()
        self.source = source
