from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass
from inspect import signature
from typing import Optional

from typing_extensions import TypeVar, ClassVar, Type

from giskardpy.motion_statechart.graph_node import Task
from krrood.ormatic.dao import HasGeneric
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from ...datastructures.enums import ExecutionType
from typing_extensions import TypeVar

from ...designator import DesignatorDescription
from ...process_module import ProcessModuleManager

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=AbstractRobot)


@dataclass
class AlternativeMotion(HasGeneric[T], ABC):
    execution_type: ClassVar[ExecutionType]

    def perform(self):
        pass

    @staticmethod
    def check_for_alternative(
        robot_view: AbstractRobot, motion: BaseMotion
    ) -> Optional[Type[BaseMotion]]:
        """
        Checks if there is an alternative motion for the given robot view, motion and execution type.

        :return: The alternative motion class if found, None otherwise
        """
        for alternative in AlternativeMotion.__subclasses__():
            if (
                issubclass(alternative, motion.__class__)
                and alternative.original_class() == robot_view.__class__
                and ProcessModuleManager.execution_type == alternative.execution_type
            ):
                return alternative
        return None


@dataclass
class BaseMotion(DesignatorDescription):

    @abstractmethod
    def perform(self):
        """
        Passes this designator to the process module for execution. Will be overwritten by each motion.
        """
        pass

    @property
    def motion_chart(self) -> Task:
        """
        Returns the mapped motion chart for this motion or the alternative motion if there is one.

        :return: The motion chart for this motion in this context
        """
        alternative = self.get_alternative_motion()
        if alternative:
            parameter = signature(self.__init__).parameters
            # Initialize alternative motion with the same parameters as the current motion
            alternative_instance = alternative(
                **{param: getattr(self, param) for param in parameter}
            )
            alternative_instance.plan_node = self.plan_node
            return alternative_instance._motion_chart
        return self._motion_chart

    @property
    @abstractmethod
    def _motion_chart(self) -> Task:
        pass

    def get_alternative_motion(self) -> Optional[Type[AlternativeMotion]]:
        return AlternativeMotion.check_for_alternative(self.robot_view, self)


MotionType = TypeVar("MotionType", bound=BaseMotion)
