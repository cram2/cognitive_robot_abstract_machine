from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass
from inspect import signature
from typing import Optional

from typing_extensions import ClassVar, Type
from typing_extensions import TypeVar

from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.graph_node import Task, MotionStatechartNode
from krrood.ormatic.dao import HasGeneric
from pycram.datastructures.enums import ExecutionType, Arms
from pycram.designator import DesignatorDescription
from pycram.motion_executor import MotionExecutor
from pycram.view_manager import ViewManager
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot

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
                and MotionExecutor.execution_type == alternative.execution_type
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
        """
        Returns the motion chart for this motion. Will be overwritten by each motion.
        """

    def get_alternative_motion(self) -> Optional[Type[AlternativeMotion]]:
        return AlternativeMotion.check_for_alternative(self.robot_view, self)

    def _only_allow_gripper_collision_rules(
        self, arm: Arms
    ) -> list[MotionStatechartNode]:
        """
        Returns collision rules that only allow collisions between the manipulator of an arm and the environment.

        :param arm: The arm for which to get the collision rules
        """
        manipulator_bodies = (
            ViewManager()
            .get_end_effector_view(arm, self.robot_view)
            .bodies_with_collision
        )
        rules = [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AllowCollisionBetweenGroups(
                        self.world.bodies_with_collision, manipulator_bodies
                    )
                ]
            )
        ]
        rules.extend(self.robot_view.special_constraints)
        return rules


MotionType = TypeVar("MotionType", bound=BaseMotion)
