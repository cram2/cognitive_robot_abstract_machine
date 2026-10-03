from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Callable

import pytest
import rclpy
import std_msgs.msg
from rclpy.node import Node
from rclpy.timer import Timer

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.client_presence import (
    ClientHeartbeatPublisher,
    ClientWatchdog,
    HeartbeatPresence,
)
from giskardpy.middleware.ros2.exceptions import NoWatchedClientError
from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.adapters.ros.messages import MetaData

# %% helpers


@dataclass
class SteppingClock:
    """
    A clock that only moves when a test moves it.
    """

    now: float = 0.0
    """
    The time this clock currently reads.
    """

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """
        Let the given number of seconds pass.
        """
        self.now += seconds


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """
    Give the ros graph time to catch up with what a test just did.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def create_client(node_name: str = "some_client") -> MetaData:
    """
    The identity a client names itself with.
    """
    return MetaData(node_name=node_name, process_id=1234)


def heartbeat_of(client: MetaData) -> std_msgs.msg.String:
    """
    The message that client sends to announce itself.
    """
    return std_msgs.msg.String(data=json.dumps(to_json(client)))


def let_heartbeats_stop(presence: HeartbeatPresence) -> None:
    """
    Advance a presence check's clock past its timeout, as if its client's heartbeats had
    stopped arriving.
    """
    presence.clock.advance(presence.timeout.total_seconds() + 0.01)


@dataclass
class BusyDefaultCallbackGroup:
    """
    Keeps the default callback group of a node occupied until released, standing in for
    the stream of world updates a node applies while a motion runs.
    """

    node: Node
    """
    The node whose default callback group is kept busy.
    """

    started: Event = field(init=False, default_factory=Event)
    """
    Set once the blocking callback runs.
    """

    released: Event = field(init=False, default_factory=Event)
    """
    Set to let the blocking callback return.
    """

    timer: Timer = field(init=False)
    """
    The timer whose callback blocks.
    """

    def __post_init__(self):
        self.timer = self.node.create_timer(0.01, self.block)

    def block(self) -> None:
        """
        Occupy the default callback group until released.
        """
        self.started.set()
        self.released.wait()

    def release(self) -> None:
        """
        Let the blocking callback return and stop it from being called again.
        """
        self.released.set()
        self.node.destroy_timer(self.timer)


# %% the heartbeats a client sends


class TestClientHeartbeat:
    """
    The heartbeat has to reach Giskard, which means both sides have to agree on where it
    is sent and what it says.
    """

    def test_giskard_receives_the_heartbeat_of_a_client(self, rclpy_node: Node):
        presence = HeartbeatPresence(node=rclpy_node)
        client_node = rclpy.create_node("heartbeat_sender")
        client = MetaData(node_name=client_node.get_name(), process_id=7)
        publisher = ClientHeartbeatPublisher(
            node=client_node,
            client=client,
            giskard_node_name=rclpy_node.get_name(),
        )
        try:
            assert wait_until(
                lambda: publisher.publish() or client in presence.last_heartbeat
            )
        finally:
            publisher.stop()
            client_node.destroy_node()

        assert presence.start_watching(client)
        assert presence.is_client_present()

    def test_stop_does_nothing_the_second_time(self, rclpy_node: Node):
        """
        A client may stop announcing itself mid-test and its own teardown still calls
        stop() again, so a second call must not raise.
        """
        client_node = rclpy.create_node("heartbeat_sender")
        publisher = ClientHeartbeatPublisher(
            node=client_node,
            client=MetaData(node_name=client_node.get_name(), process_id=7),
            giskard_node_name=rclpy_node.get_name(),
        )

        publisher.stop()
        publisher.stop()

        client_node.destroy_node()

    def test_a_heartbeat_due_after_stop_is_dropped(self, rclpy_node: Node):
        """
        The timer may already have handed a heartbeat to an executor thread when the
        client stops, and a callback that raises there stops that executor for good.
        """
        client_node = rclpy.create_node("heartbeat_sender")
        publisher = ClientHeartbeatPublisher(
            node=client_node,
            client=MetaData(node_name=client_node.get_name(), process_id=7),
            giskard_node_name=rclpy_node.get_name(),
        )
        publisher.stop()

        publisher.publish()

        client_node.destroy_node()


# %% reading the heartbeats


class TestHeartbeatPresence:
    """
    A client counts as gone once its heartbeats stop arriving.
    """

    def test_a_client_that_just_announced_itself_is_present(self, rclpy_node: Node):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        client = create_client()
        presence.receive_heartbeat(heartbeat_of(client))

        assert presence.start_watching(client)
        assert presence.is_client_present()

    def test_a_client_that_stopped_announcing_itself_is_gone(self, rclpy_node: Node):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        client = create_client()
        presence.receive_heartbeat(heartbeat_of(client))
        presence.start_watching(client)

        let_heartbeats_stop(presence)

        assert not presence.is_client_present()

    def test_a_client_stays_present_while_it_keeps_announcing_itself(
        self, rclpy_node: Node
    ):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        client = create_client()
        presence.receive_heartbeat(heartbeat_of(client))
        presence.start_watching(client)

        for _ in range(5):
            presence.clock.advance(presence.timeout.total_seconds())
            presence.receive_heartbeat(heartbeat_of(client))

        assert presence.is_client_present()

    def test_a_client_that_never_announced_itself_is_not_watched(
        self, rclpy_node: Node
    ):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())

        assert not presence.start_watching(create_client())

    def test_the_heartbeat_of_one_client_says_nothing_about_another(
        self, rclpy_node: Node
    ):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        presence.receive_heartbeat(heartbeat_of(create_client("other_client")))

        assert not presence.start_watching(create_client("some_client"))

    def test_a_check_that_watches_nothing_cannot_be_asked(self, rclpy_node: Node):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())

        with pytest.raises(NoWatchedClientError):
            presence.is_client_present()


# %% watching the client of a goal


class TestClientWatchdog:
    """
    The watchdog reports on the client of the running goal, through its presence check.
    """

    def test_a_client_the_presence_check_does_not_recognize_is_never_reported_gone(
        self, rclpy_node: Node
    ):
        """
        Stopping a goal because the presence check does not recognize its client would
        break every client that Giskard simply cannot see.
        """
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        watchdog = ClientWatchdog(presence=presence)

        watchdog.watch(create_client())

        assert not watchdog.is_client_gone()

    def test_a_client_that_left_is_reported_gone(self, rclpy_node: Node):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        client = create_client()
        presence.receive_heartbeat(heartbeat_of(client))
        watchdog = ClientWatchdog(presence=presence)
        watchdog.watch(client)

        let_heartbeats_stop(presence)

        assert watchdog.is_client_gone()
        assert watchdog.client == client

    def test_a_client_recognized_after_its_goal_started_is_reported_gone_once_it_leaves(
        self, rclpy_node: Node
    ):
        """
        A client's first heartbeat can arrive after Giskard accepted its goal, and that
        client still has to be watched from then on.
        """
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        client = create_client()
        watchdog = ClientWatchdog(presence=presence)
        watchdog.watch(client)
        presence.receive_heartbeat(heartbeat_of(client))
        assert not watchdog.is_client_gone()

        let_heartbeats_stop(presence)

        assert watchdog.is_client_gone()
        assert watchdog.client == client

    def test_a_finished_goal_releases_its_check(self, rclpy_node: Node):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        client = create_client()
        presence.receive_heartbeat(heartbeat_of(client))
        watchdog = ClientWatchdog(presence=presence)
        watchdog.watch(client)

        watchdog.stop_watching()

        assert presence.watched_client is None
        assert not watchdog.is_client_gone()

    def test_a_finished_goal_is_not_watched_when_its_client_announces_itself_late(
        self, rclpy_node: Node
    ):
        presence = HeartbeatPresence(node=rclpy_node, clock=SteppingClock())
        client = create_client()
        watchdog = ClientWatchdog(presence=presence)
        watchdog.watch(client)
        watchdog.stop_watching()
        presence.receive_heartbeat(heartbeat_of(client))

        watchdog.is_client_gone()

        assert presence.watched_client is None

    def test_the_client_of_no_goal_cannot_be_asked_for(self, rclpy_node: Node):
        watchdog = ClientWatchdog(presence=HeartbeatPresence(node=rclpy_node))

        with pytest.raises(NoWatchedClientError):
            watchdog.client


# %% heartbeats while a node is busy


class TestHeartbeatWhileNodeIsBusy:
    """
    A node's other callbacks can keep it busy for seconds, for example while it applies
    a stream of world updates, which must not look like a client that left.
    """

    def test_giskard_receives_heartbeats_while_its_node_is_busy(self, init_rospy):
        giskard_node = rospy.get_node()
        presence = HeartbeatPresence(node=giskard_node)
        busy = BusyDefaultCallbackGroup(node=giskard_node)
        client_node = rclpy.create_node("heartbeat_sender")
        client = MetaData(node_name=client_node.get_name(), process_id=7)
        sender = client_node.create_publisher(
            std_msgs.msg.String,
            ClientHeartbeatPublisher.topic_name(giskard_node.get_name()),
            10,
        )
        try:
            assert wait_until(busy.started.is_set)

            assert wait_until(
                lambda: sender.publish(heartbeat_of(client))
                or client in presence.last_heartbeat
            )
        finally:
            busy.release()
            client_node.destroy_node()

    def test_a_client_keeps_announcing_itself_while_its_node_is_busy(self, init_rospy):
        giskard_node = rospy.get_node()
        presence = HeartbeatPresence(node=giskard_node)
        client_node = rclpy.create_node("heartbeat_sender")
        client = MetaData(node_name=client_node.get_name(), process_id=7)
        assert wait_until(lambda: rospy.executor is not None)
        rospy.executor.add_node(client_node)
        busy = BusyDefaultCallbackGroup(node=client_node)
        publisher = ClientHeartbeatPublisher(
            node=client_node,
            client=client,
            giskard_node_name=giskard_node.get_name(),
        )
        try:
            assert wait_until(busy.started.is_set)

            assert wait_until(lambda: client in presence.last_heartbeat)
        finally:
            busy.release()
            publisher.stop()
            rospy.executor.remove_node(client_node)
            client_node.destroy_node()
