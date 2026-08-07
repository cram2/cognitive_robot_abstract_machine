from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from inspect import isabstract

from typing_extensions import (
    TypeVar,
    ClassVar,
    TYPE_CHECKING,
    List,
    Optional,
    Type,
    Iterable,
    Union,
)

from krrood.adapters.json_serializer import list_like_classes
from krrood.ormatic.data_access_objects.base import HasGeneric
from krrood.ormatic.utils import classes_of_package
from krrood.utils import recursive_subclasses
from .datastructures.enums import ExecutionType
from .plans.executables import GiskardExecutable
from semantic_digital_twin.robots.robot_parts import AbstractRobot

if TYPE_CHECKING:
    from .robot_plans import BaseMotion
else:
    BaseMotion = TypeVar("BaseMotion")

AbstractRobotType = TypeVar("AbstractRobotType", bound=AbstractRobot)
BaseMotionType = TypeVar("BaseMotionType", bound=BaseMotion)


@dataclass
class AlternativeMotion(HasGeneric[AbstractRobotType], ABC):
    execution_type: ClassVar[Union[ExecutionType, Iterable[ExecutionType]]]
    """
    Execution type(s) for which this alternative motion applies.

    A single execution type or an iterable of them; the alternative is selected when the
    active execution type is among these.
    """

    def perform(self):
        pass

    @staticmethod
    def check_for_alternative(
        alternatives: Iterable[Type[AlternativeMotion]],
        robot_view: AbstractRobot,
        motion: Type[BaseMotionType],
    ) -> Optional[Type[BaseMotionType]]:
        """
        Checks if there is an alternative motion for the given robot view, motion and
        execution type among the provided alternatives.

        :param alternatives: The alternative motion mappings to search through (e.g.
            from the context)
        :param robot_view: The robot for which the alternative motion should be found
        :param motion: The motion class for which an alternative should be found
        :return: The alternative motion class if found, None otherwise
        """
        for alternative in alternatives:
            if (
                issubclass(alternative, motion)
                and alternative.original_class() == robot_view.__class__
                and GiskardExecutable.execution_type
                in (
                    alternative.execution_type
                    if isinstance(alternative.execution_type, list_like_classes)
                    else [alternative.execution_type]
                )
            ):
                return alternative
        return None

    @classmethod
    def discover_all(cls) -> List[Type[AlternativeMotion]]:
        """
        Discover every concrete :class:`AlternativeMotion` for every robot.

        Importing ``coraplex.alternative_motion_mappings`` walks and imports its
        submodules, registering their :class:`AlternativeMotion` subclasses.
        Mainly a helper to pass to the context of a demo to make it robot agnostic with regards to the alternative
        motion mappings.

        :return: Every concrete alternative motion known to coraplex.
        """
        # Local import as module-level import would be circular and fail while this module is still initializing.
        import coraplex.alternative_motion_mappings as mappings_package

        classes_of_package(mappings_package)
        return [
            subclass
            for subclass in recursive_subclasses(cls)
            if not isabstract(subclass)
        ]
