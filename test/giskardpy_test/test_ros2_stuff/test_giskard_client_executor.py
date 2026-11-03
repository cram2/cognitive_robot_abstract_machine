from __future__ import annotations

from dataclasses import dataclass

import pytest

from cramph.executor import Executor, ExecutorExtension
from cramph.monitors import CountTicks
from cramph.statechart import Statechart
from semantic_digital_twin.adapters.ros import rospy
from giskardpy.middleware.ros2.action_server import ActionServerHandler, GoalOutcome
from giskardpy.middleware.ros2.exceptions import ExecutionAbortedException
from giskardpy.middleware.ros2.giskard_client_executor import (
    GiskardClientExecutor,
    GiskardClientExecutorNode,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.world import World

from .test_action_client import ServerAnsweringOneGoal, giskard_command_server

# %% mimics


@dataclass
class ExtensionCountingCompilesAndRuns(ExecutorExtension):
    """
    An executor extension that counts how often it was told that a statechart compiled
    and that a run stopped.
    """

    compile_count: int = 0
    """
    How often the extension was told that the statechart compiled.
    """

    run_count: int = 0
    """
    How often the extension was told that a run stopped.
    """

    def after_compile(self, executor: Executor) -> None:
        self.compile_count += 1

    def after_run(self, executor: Executor) -> None:
        self.run_count += 1


# %% fixtures


@pytest.fixture()
def extension() -> ExtensionCountingCompilesAndRuns:
    """
    :return: The extension of :func:`executor_with_a_statechart`.
    """
    return ExtensionCountingCompilesAndRuns()


@pytest.fixture()
def executor_with_a_statechart(
    giskard_command_server: ActionServerHandler,
    extension: ExtensionCountingCompilesAndRuns,
) -> GiskardClientExecutor:
    """
    :return: A client with `extension` that compiled a statechart of one node.
    """
    world = World()
    WorldSynchronizer(node=rospy.get_node(), _world=world)
    executor = GiskardClientExecutor(
        node_handle=rospy.get_node(), world=world, extensions=[extension]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(CountTicks(ticks=1))
    executor.compile(statechart)
    yield executor
    executor.close()


# %% extensions


def test_an_extension_is_told_that_the_statechart_compiled(
    executor_with_a_statechart: GiskardClientExecutor,
    extension: ExtensionCountingCompilesAndRuns,
) -> None:
    assert extension.compile_count == 1


def test_an_extension_is_told_that_an_aborted_goal_stopped(
    giskard_command_server: ActionServerHandler,
    executor_with_a_statechart: GiskardClientExecutor,
    extension: ExtensionCountingCompilesAndRuns,
) -> None:
    ServerAnsweringOneGoal(
        action_server=giskard_command_server,
        outcome_unless_canceled=GoalOutcome.ABORTED,
    )

    with pytest.raises(ExecutionAbortedException):
        executor_with_a_statechart.execute()

    assert extension.run_count == 1


# %% closing


class TestGiskardClientExecutorClose:
    """
    Closing a client has to stop announcing its heartbeat, or the timer keeps ticking on
    a node nobody is using any more.
    """

    def test_close_stops_the_heartbeat(self, giskard_command_server):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)

        executor = GiskardClientExecutor(
            node_handle=rospy.get_node(), world=world, announces_presence=True
        )

        executor.close()

        assert executor.heartbeat_publisher.timer.is_canceled


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

        executor = GiskardClientExecutor(node_handle=rospy.get_node(), world=world)

        assert executor.heartbeat_publisher is None

    def test_a_client_that_never_announced_itself_can_be_closed(
        self, giskard_command_server
    ):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)
        executor = GiskardClientExecutor(node_handle=rospy.get_node(), world=world)

        executor.close()

        assert executor.heartbeat_publisher is None

    def test_a_client_asked_to_on_construction_announces_itself(
        self, giskard_command_server
    ):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)

        executor = GiskardClientExecutor(
            node_handle=rospy.get_node(), world=world, announces_presence=True
        )

        assert not executor.heartbeat_publisher.timer.is_canceled()
        executor.close()

    def test_a_client_can_start_announcing_itself_later(self, giskard_command_server):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)
        executor = GiskardClientExecutor(node_handle=rospy.get_node(), world=world)

        executor.announce_presence()

        assert not executor.heartbeat_publisher.timer.is_canceled()
        executor.close()

    def test_announcing_twice_keeps_the_first_heartbeat(self, giskard_command_server):
        world = World()
        WorldSynchronizer(node=rospy.get_node(), _world=world)
        executor = GiskardClientExecutor(
            node_handle=rospy.get_node(), world=world, announces_presence=True
        )
        first_publisher = executor.heartbeat_publisher

        executor.announce_presence()

        assert executor.heartbeat_publisher is first_publisher
        executor.close()


class TestGiskardClientExecutorNodeClose:
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

        client = GiskardClientExecutorNode(node_name="test_client", world=world)
        assert client.node_handle in rospy.executor.get_nodes()

        client.close()

        assert client.node_handle not in rospy.executor.get_nodes()
