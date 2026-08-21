from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import signature
from typing_extensions import ClassVar, TypeVar, Type, Optional

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
from semantic_digital_twin.world_description.world_entity import Body
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

    holds_its_goal_until_the_motion_ends: ClassVar[bool] = False
    """
    Whether this motion keeps its goal in force after reaching it.

    A motion normally retires once its goal is observed, which hands the robot to the
    motion after it. The last motion of a chart hands it to nobody, and the chart keeps
    ticking until the world settles, so a motion that has to leave the robot exactly
    where it put it says so here.
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
            the given arm and the environment. A body held in that manipulator hangs
            below its tool frame and is one of its bodies, so it is freed along with it.
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


@dataclass
class StandaloneMotion(BaseMotion, ABC):
    """
    A motion that is executed in a motion statechart of its own.

    The motions around it are still merged with each other, never with this one.
    """


MotionType = TypeVar("MotionType", bound=BaseMotion)
