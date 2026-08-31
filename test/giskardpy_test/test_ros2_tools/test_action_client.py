from dataclasses import dataclass
from typing import Any

import pytest
from json_msgs.action import JsonAction
from rclpy.node import Node

from giskardpy.middleware.ros2 import ros2_interface, rospy
from giskardpy.middleware.ros2.ros2_interface import MyActionClient

# %% test doubles


@dataclass
class LateServerActionClient:
    """
    Stand-in for an action client whose server only shows up after a few waits, so that
    the retry loop of the client under test is actually entered.
    """

    node: Node
    """
    The node the client under test handed down.
    """

    action_type: Any
    """
    The action the client was built for.
    """

    action_name: str
    """
    Name of the action the client was built for.
    """

    waits_until_the_server_appears: int = 2
    """
    Number of waits that report no server before one becomes available.
    """

    performed_waits: int = 0
    """
    Number of waits that were asked for so far.
    """

    def wait_for_server(self, timeout_sec: float) -> bool:
        self.performed_waits += 1
        if self.performed_waits <= self.waits_until_the_server_appears:
            return False
        return True


# %% waiting for a late server


def test_waiting_for_a_late_server_uses_the_given_node(rclpy_node, monkeypatch):
    """
    A client built with an explicit node must report through that node while it waits,
    so that a server which starts late is waited out rather than turned into a crash.
    """
    monkeypatch.setattr(rospy, "node", None)
    monkeypatch.setattr(ros2_interface, "ActionClient", LateServerActionClient)

    client = MyActionClient(rclpy_node, JsonAction, "late_server_action")

    assert client.node_handle is rclpy_node
    assert (
        client._client.performed_waits
        == client._client.waits_until_the_server_appears + 1
    )
