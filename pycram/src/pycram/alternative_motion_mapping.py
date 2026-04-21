from abc import ABC
from dataclasses import dataclass

from typing_extensions import TypeVar, ClassVar, TYPE_CHECKING, Optional, Type

from krrood.ormatic.data_access_objects.base import HasGeneric
from .datastructures.enums import ExecutionType
from .motion_executor import MotionExecutor
from semantic_digital_twin.robots.abstract_robot import AbstractRobot

if TYPE_CHECKING:
    from .robot_plans import BaseMotion
else:
    BaseMotion = TypeVar("BaseMotion")

AbstractRobotType = TypeVar("AbstractRobotType", bound=AbstractRobot)
BaseMotionType = TypeVar("BaseMotionType", bound=BaseMotion)


@dataclass
class AlternativeMotion(HasGeneric[AbstractRobotType], ABC):
    execution_type: ClassVar[ExecutionType]

    def perform(self):
        pass

    @staticmethod
    def check_for_alternative(
        robot_view: AbstractRobot, motion: Type[BaseMotionType]
    ) -> Optional[Type[BaseMotionType]]:
        """
        Checks if there is an alternative motion for the given robot view, motion and execution type.

        :return: The alternative motion class if found, None otherwise
        """
        for alternative in AlternativeMotion.__subclasses__():
            if (
                issubclass(alternative, motion)
                and alternative.original_class() == robot_view.__class__
                and MotionExecutor.execution_type == alternative.execution_type
            ):
                return alternative
        return None
