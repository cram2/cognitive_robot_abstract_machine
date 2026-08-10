from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, Type

try:
    from rclpy.action import ActionClient
    from rclpy.node import Node
except ImportError:
    from semantic_digital_twin.utils import MockedNodeClass as Node

    ActionClient = None

from giskardpy.motion_statechart.context import ContextExtension
from giskardpy.motion_statechart.exceptions import ActionClientTypeMismatchError


@dataclass
class RosContextExtension(ContextExtension):
    ros_node: Node

    _action_clients: Dict[str, ActionClient] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Action clients keyed by action topic. Reused across every task that targets the
    same topic instead of constructing (and waiting on) a new client per task build.

    Safe to share between multiple tasks that are active at the same time: each
    ``send_goal_async``/``get_result_async`` call returns its own future tracking that
    specific goal, and callers (:class:`~giskardpy.motion_statechart.ros2_nodes.ros_tasks.ActionServerTask`)
    keep their own goal message and result on the task instance, not on the client. No
    state on the client itself is mutated per-goal.
    """

    def get_or_create_action_client(
        self, message_type: Type, action_topic: str
    ) -> ActionClient:
        """
        Returns the cached action client for ``action_topic``, creating and caching one
        the first time it is requested.

        :param message_type: the action message type to construct the client with.
        :param action_topic: the action server topic, used as the cache key.
        :return: the (possibly newly created) action client for this topic.
        """
        action_client = self._action_clients.get(action_topic)
        if action_client is not None:
            if action_client._action_type is not message_type:
                raise ActionClientTypeMismatchError(
                    action_topic=action_topic,
                    existing_message_type=action_client._action_type,
                    requested_message_type=message_type,
                )
            return action_client

        action_client = ActionClient(self.ros_node, message_type, action_topic)
        self._action_clients[action_topic] = action_client
        return action_client
