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

        wrapper = GiskardWrapper(
            node_handle=rospy.get_node(), world=world, announces_presence=True
        )

        wrapper.close()

        assert wrapper.heartbeat_publisher.timer.is_canceled


class TestAnnouncingPresence:
    """
    A client is only watched by Giskard once it announces itself, and it only does that
    when asked to: a goal that is stopped because heartbeats went missing is the wrong
    default for a client on a busy machine.
    """

    def test_a_client_does_not_announce_itself_unless_asked(
        self, giskard_command_server
    ):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)

        wrapper = GiskardWrapper(node_handle=rospy.get_node(), world=world)

        assert wrapper.heartbeat_publisher is None

    def test_a_client_that_never_announced_itself_can_be_closed(
        self, giskard_command_server
    ):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)
        wrapper = GiskardWrapper(node_handle=rospy.get_node(), world=world)

        wrapper.close()

        assert wrapper.heartbeat_publisher is None

    def test_a_client_asked_to_on_construction_announces_itself(
        self, giskard_command_server
    ):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)

        wrapper = GiskardWrapper(
            node_handle=rospy.get_node(), world=world, announces_presence=True
        )

        assert not wrapper.heartbeat_publisher.timer.is_canceled()
        wrapper.close()

    def test_a_client_can_start_announcing_itself_later(self, giskard_command_server):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)
        wrapper = GiskardWrapper(node_handle=rospy.get_node(), world=world)

        wrapper.announce_presence()

        assert not wrapper.heartbeat_publisher.timer.is_canceled()
        wrapper.close()

    def test_announcing_twice_keeps_the_first_heartbeat(self, giskard_command_server):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)
        wrapper = GiskardWrapper(
            node_handle=rospy.get_node(), world=world, announces_presence=True
        )
        first_publisher = wrapper.heartbeat_publisher

        wrapper.announce_presence()

        assert wrapper.heartbeat_publisher is first_publisher
        wrapper.close()


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
