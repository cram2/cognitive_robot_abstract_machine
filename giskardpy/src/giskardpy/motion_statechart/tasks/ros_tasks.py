from dataclasses import dataclass, field

from rclpy.action import ActionClient
from typing_extensions import Any

from giskardpy.motion_statechart.context import ExecutionContext
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.graph_node import Task, MotionStatechartNode

import rclpy
import logging


logger = logging.getLogger(__name__)


@dataclass
class ActionServerTask(MotionStatechartNode):

    action_topic: str
    """
    Topic name for the action server.
    """

    goal_msg: Any
    """
    Fully specified goal message that can be send out. 
    """

    node_handle: rclpy.node.Node
    """
    A ROS node to create the action client.
    """

    _action_client: ActionClient = field(init=False)
    """
    ROS action client, is created in `on_start`.
    """

    def on_start(self, context: ExecutionContext):
        self._action_client = ActionClient(
            self.node_handle, self.goal_msg.__class__, self.action_topic
        )
        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()
        logger.debug("Sending goal to action server")
        result = self._action_client.send_goal(self.goal_msg)
        logger.info(f"Action server {self.action_topic} returned result: {result}")
        self._lifecycle_state = (
            LifeCycleValues.DONE if result.success else LifeCycleValues.FAILED
        )

    def on_end(self, context: ExecutionContext):
        pass


