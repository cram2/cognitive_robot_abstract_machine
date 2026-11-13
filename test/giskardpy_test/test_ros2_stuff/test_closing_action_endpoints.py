"""
What a closed action server, action client and Giskard leave in the ROS graph.

A server left in the graph keeps answering discovery for its action name, so a client
that starts later can send its goal to it instead of to the live server.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from time import monotonic, sleep

import pytest
from json_msgs.action import JsonAction
from rclpy.action import (
    get_action_client_names_and_types_by_node,
    get_action_server_names_and_types_by_node,
)
from rclpy.node import Node
from typing_extensions import Callable, Iterator

from giskardpy.middleware.ros2.action_server import ActionServerHandler
from giskardpy.middleware.ros2.exceptions import ActionClientClosedError
from giskardpy.middleware.ros2.action_client import GiskardActionClient
from semantic_digital_twin.adapters.ros import background_node

from .test_integration_stretch import StretchTester

# %% the ROS graph as a test sees it


@dataclass
class ObservedRosGraph:
    """
    The ROS graph as seen from one node, which takes in a change only after a moment.
    """

    observer: Node
    """
    The node whose view of the graph is read.
    """

    settling_time: timedelta = timedelta(seconds=5)
    """
    How long a change may take to show up in the view of :attr:`observer`.
    """

    def served_actions(self, node_name: str) -> list[str]:
        """
        :return: The actions the node called ``node_name`` serves.
        """
        return [
            action_name
            for action_name, _ in get_action_server_names_and_types_by_node(
                self.observer, node_name, self.observer.get_namespace()
            )
        ]

    def client_actions(self, node_name: str) -> list[str]:
        """
        :return: The actions the node called ``node_name`` has a client for.
        """
        return [
            action_name
            for action_name, _ in get_action_client_names_and_types_by_node(
                self.observer, node_name, self.observer.get_namespace()
            )
        ]

    def wait_until(self, condition: Callable[[], bool]) -> None:
        """
        Return once ``condition`` holds or :attr:`settling_time` has passed.
        """
        deadline = monotonic() + self.settling_time.total_seconds()
        while not condition() and monotonic() < deadline:
            sleep(0.01)


# %% closing an action server


def test_a_closed_action_server_no_longer_serves_its_action(
    init_background_node,
) -> None:
    graph = ObservedRosGraph(observer=background_node.get_node())
    server_node_name = background_node.get_node().get_name()
    action_server = ActionServerHandler(
        action_name="closed_server/command", action_type=JsonAction
    )
    graph.wait_until(
        lambda: "/closed_server/command" in graph.served_actions(server_node_name)
    )

    action_server.close()
    graph.wait_until(
        lambda: "/closed_server/command" not in graph.served_actions(server_node_name)
    )

    assert "/closed_server/command" not in graph.served_actions(server_node_name)


# %% closing an action client


@pytest.fixture()
def closed_client_action_server(init_background_node) -> Iterator[ActionServerHandler]:
    """
    A server for the action of the client under test, so that the client can be built.
    """
    action_server = ActionServerHandler(
        action_name="closed_client/command", action_type=JsonAction
    )
    yield action_server
    action_server.close()


def test_a_closed_action_client_no_longer_uses_its_action(
    closed_client_action_server: ActionServerHandler,
) -> None:
    graph = ObservedRosGraph(observer=background_node.get_node())
    client_node_name = background_node.get_node().get_name()
    action_client = GiskardActionClient(
        background_node.get_node(), JsonAction, "closed_client/command"
    )
    graph.wait_until(
        lambda: "/closed_client/command" in graph.client_actions(client_node_name)
    )

    action_client.close()
    graph.wait_until(
        lambda: "/closed_client/command" not in graph.client_actions(client_node_name)
    )

    assert "/closed_client/command" not in graph.client_actions(client_node_name)


def test_a_closed_action_client_refuses_to_send_a_goal(
    closed_client_action_server: ActionServerHandler,
) -> None:
    """
    Sending a goal through a closed client raises instead of waiting for an answer.
    """
    action_client = GiskardActionClient(
        background_node.get_node(), JsonAction, closed_client_action_server.action_name
    )
    action_client.close()

    with pytest.raises(ActionClientClosedError):
        action_client.send_goal(goal=JsonAction.Goal())


# %% closing a Giskard and its client


def test_a_closed_giskard_and_its_client_leave_the_graph(init_background_node) -> None:
    """
    The client's node is destroyed by closing it, so it has to disappear from the graph,
    and Giskard's node outlives Giskard, so it has to stop serving the command action.
    """
    graph = ObservedRosGraph(observer=background_node.get_node())
    tester = StretchTester()
    giskard_node_name = background_node.get_node().get_name()
    client_node_name = tester.api.node_handle.get_name()
    command_action_name = tester.giskard.motion_server.action_server.action_name

    tester.close()
    graph.wait_until(
        lambda: (
            client_node_name not in graph.observer.get_node_names()
            and not graph.served_actions(giskard_node_name)
        )
    )

    assert client_node_name not in graph.observer.get_node_names()
    assert f"/{command_action_name}" not in graph.served_actions(giskard_node_name)
