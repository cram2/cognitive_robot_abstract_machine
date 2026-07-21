from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from inspect import signature
from typing_extensions import TypeVar, Type, Optional

from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.graph_node import Task, MotionStatechartNode
from coraplex.datastructures.enums import Arms
from coraplex.plans.designator import Designator
from coraplex.view_manager import ViewManager
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from coraplex.alternative_motion_mapping import AlternativeMotion

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=AbstractRobot)


@dataclass
class BaseMotion(Designator):
    """
    Base class for all motions.

    Motions are like builders for Motion State Charts. Motions never create any other
    motions or actions. Motions create exactly one goal.
    """

    def perform(self):
        """
        Passes this designator to the process module for execution.

        Will be overwritten by each motion.
        """
        pass

    @property
    def motion_chart(self) -> Task:
        """
        Returns the mapped motion chart for this motion or the alternative motion if
        there is one.

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
        return AlternativeMotion.check_for_alternative(
            self.context.alternative_motion_mappings, self.robot, self.__class__
        )

    def _only_allow_gripper_collision_rules(
        self, arm: Arms
    ) -> list[MotionStatechartNode]:
        """
        :param arm: The arm whose manipulator may collide with the environment.
        :return: Collision rules that only allow collisions between the manipulator of
            the given arm and the environment.
        """
        manipulator_bodies = (
            ViewManager().get_end_effector_view(arm, self.robot).bodies_with_collision
        )
        return [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AllowCollisionBetweenGroups(
                        self.world.bodies_with_collision, manipulator_bodies
                    )
                ]
            )
        ]


MotionType = TypeVar("MotionType", bound=BaseMotion)
