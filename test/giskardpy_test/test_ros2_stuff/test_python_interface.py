from __future__ import annotations

import pytest
from json_msgs.action import JsonAction

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.action_server import ActionServerHandler
from giskardpy.middleware.ros2.python_interface import (
    GiskardWrapper,
    GiskardWrapperNode,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.world import World

# %% fixtures


@pytest.fixture()
def giskard_command_server(init_rospy) -> ActionServerHandler:
    """
    A real "giskard/command" action server, just present enough for a client's own
    action client to find on construction.
    """
    return ActionServerHandler(action_name="giskard/command", action_type=JsonAction)


class TestGiskardWrapperClose:
    """
    Closing a client has to stop announcing its heartbeat, or the timer keeps ticking on
    a node nobody is using any more.
    """

    def test_close_stops_the_heartbeat(self, giskard_command_server):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)

        wrapper = GiskardWrapper(node_handle=rospy.get_node(), world=world)

        wrapper.close()

        assert wrapper.heartbeat_publisher.timer.is_canceled


class TestGiskardWrapperNodeClose:
    """
    A client that created its own node has to remove it from Giskard's shared executor
    and destroy it again, or every client leaves one more node in that executor's list
    forever.
    """

    def test_close_removes_the_node_from_the_shared_executor(
        self, giskard_command_server
    ):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)

        client = GiskardWrapperNode(node_name="test_client", world=world)
        assert client.node_handle in rospy.executor.get_nodes()

        client.close()

        assert client.node_handle not in rospy.executor.get_nodes()
