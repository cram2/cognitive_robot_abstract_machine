from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing_extensions import (
    Optional,
    Any, TYPE_CHECKING,
)

from krrood.entity_query_language.backends import QueryBackend, EntityQueryLanguageBackend
from pycram.plans.plan_entity import PlanEntity
from semantic_digital_twin.robots.abstract_robot import AbstractRobot

if TYPE_CHECKING:
    from pycram.plans.plan import Plan
    from semantic_digital_twin.world import World

try:
    import rclpy
except ImportError:
    rclpy = None

@dataclass
class Context(PlanEntity):
    """
    A dataclass for storing the context of a plan
    """

    world: World
    """
    The world in which the plan is executed
    """

    robot: AbstractRobot
    """
    The semantic robot annotation which should execute the plan
    """

    ros_node: Optional[rclpy.node.Node] = field(default=None)
    """
    A ROS node that should be used for communication in this plan
    """

    evaluate_conditions: bool = field(default=True)
    """
    Should pre -and postconditions of actions be evaluated in this plan
    """

    query_backend: QueryBackend = field(default_factory=EntityQueryLanguageBackend)
    """
    The backend used to answer queries about underspecified statements.
    """

    _debug: bool = field(default=False)
    """
    Should debug information be printed or visualized
    """

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value
        if self.debug and not self.ros_node:
            raise ValueError("Debug mode requires a ROS node")
        logging.getLogger("pycram").setLevel(logging.DEBUG if self.debug else logging.INFO)


    @classmethod
    def from_world(cls, world: World, plan: Plan = None, query_backend: Optional[QueryBackend] = None):
        """
        Create a context from a world by getting the first robot in the world. There is no super plan in this case.

        :param world: The world for which to create the context
        :param plan: The plan that manages this context
        :param query_backend: The query backend to use for answering queries
        :return: A context with the first robot in the world and no super plan
        """

        if query_backend is None:
            query_backend = EntityQueryLanguageBackend()

        result =  cls(
            world=world,
            robot=world.get_semantic_annotations_by_type(AbstractRobot)[0],
            query_backend=query_backend
        )
        if plan:
            plan.add_plan_entity(result)
        return result



