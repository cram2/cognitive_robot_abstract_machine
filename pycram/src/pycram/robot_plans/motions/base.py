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
from ...alternative_motion_mapping import AlternativeMotion
from pycram.datastructures.enums import ExecutionType
from typing_extensions import TypeVar

from pycram.designator import DesignatorDescription
from pycram.motion_executor import MotionExecutor

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=AbstractRobot)


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
        return AlternativeMotion.check_for_alternative(self.robot_view, self.__class__)


MotionType = TypeVar("MotionType", bound=BaseMotion)
