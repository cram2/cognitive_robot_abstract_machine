from dataclasses import dataclass, field, is_dataclass
from typing import Any

from pytest import MonkeyPatch
from rclpy.node import Node

from robokudo.scripts import query_test_client


@dataclass
class NodeForActionClient:
    """
    Record the node associated with an action client.
    """

    node: Node
    """Node supplied while constructing the action client."""

    action_type: Any
    """
    Type of action handled by the client.
    """

    action_name: str
    """Name through which the action is reached."""

    destroyed: bool = field(init=False, default=False)
    """
    Whether the action-client resource was destroyed.
    """

    def destroy(self) -> None:
        """
        Record destruction of the action-client resource.
        """
        self.destroyed = True


def test_action_client_is_associated_with_supplied_node(
    node: Node, monkeypatch: MonkeyPatch
) -> None:
    """
    The query client uses a supplied node without becoming that node.
    """
    monkeypatch.setattr(
        query_test_client,
        "ActionClient",
        NodeForActionClient,
    )

    action_client = query_test_client.RoboKudoActionClient(node=node)

    assert is_dataclass(action_client)
    assert not isinstance(action_client, Node)
    assert action_client.node is node
    assert action_client._action_client.node is node


def test_closing_action_client_does_not_destroy_supplied_node(
    node: Node, monkeypatch: MonkeyPatch
) -> None:
    """
    Closing the query client leaves its externally owned node usable.
    """
    monkeypatch.setattr(
        query_test_client,
        "ActionClient",
        NodeForActionClient,
    )
    action_client = query_test_client.RoboKudoActionClient(node=node)
    node_name = node.get_name()

    action_client.close()

    assert node.get_name() == node_name


def test_closing_action_client_destroys_client_resource(
    node: Node, monkeypatch: MonkeyPatch
) -> None:
    """
    Closing the query client releases its owned action-client resource.
    """
    monkeypatch.setattr(
        query_test_client,
        "ActionClient",
        NodeForActionClient,
    )
    action_client = query_test_client.RoboKudoActionClient(node=node)

    action_client.close()

    assert action_client._action_client.destroyed is True
