from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from inspect import signature
from typing_extensions import List, TypeVar, Type, Optional

from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.graph_node import Task, MotionStatechartNode
from giskardpy.motion_statechart.tasks.cartesian_tasks import HoldPose
from coraplex.plans.designator import Designator
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot, EndEffector
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

    def keep_base_still(self) -> List[MotionStatechartNode]:
        """
        :return: The task holding the robot's base where it stands, empty when the robot
            may move its whole body.

        A base that is not full body controlled stands still while the rest of the robot
        moves, so it is held rather than left for another task to command. A motion's
        goal is bound relative to the robot when the motion starts, so a base that moves
        afterwards carries the goal with it. Collision avoidance is what otherwise moves
        it, buying clearance by drifting the base.
        """
        robot = self.robot
        if (
            not isinstance(robot, HasMobileBase)
            or robot.mobile_base.full_body_controlled
        ):
            return []
        return [
            HoldPose(
                name="hold base",
                root_link=self.world.root,
                tip_link=robot.root,
            )
        ]

    def _only_allow_gripper_collision_rules(
        self, end_effector: EndEffector
    ) -> list[MotionStatechartNode]:
        """
        :param end_effector: The end effector that may collide with the environment.
        :return: Collision rules that only allow collisions between the end effector,
            together with whatever it holds, and the environment.
        """
        return [
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AllowCollisionForEndEffector(end_effector=end_effector)
                ]
            )
        ]


MotionType = TypeVar("MotionType", bound=BaseMotion)
