from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Dict, Optional

import std_msgs.msg
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.timer import Timer

from giskardpy.middleware.ros2.exceptions import NoWatchedClientError
from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.ros.messages import MetaData

# %% the heartbeat a client sends


@dataclass
class ClientHeartbeatPublisher:
    """
    Announces that a client is still there, so that Giskard can stop a motion whose
    client died instead of running it to its end.
    """

    node: Node
    """
    Node of the client that announces itself.
    """

    client: MetaData
    """
    Identity of that client, the same one its goals name.
    """

    giskard_node_name: str
    """
    Node name of the Giskard the heartbeat is meant for.
    """

    period: timedelta = timedelta(seconds=1)
    """
    Time between two heartbeats.
    """

    publisher: Publisher = field(init=False)
    """
    Publisher the heartbeats go out on.
    """

    timer: Timer = field(init=False)
    """
    Timer that sends the heartbeats.
    """

    message: std_msgs.msg.String = field(init=False)
    """
    The heartbeat that is sent, built once because the identity never changes.
    """

    def __post_init__(self):
        self.message = std_msgs.msg.String(data=json.dumps(to_json(self.client)))
        self.publisher = self.node.create_publisher(
            std_msgs.msg.String,
            topic=self.topic_name(self.giskard_node_name),
            qos_profile=10,
        )
        self.timer = self.node.create_timer(self.period.total_seconds(), self.publish)

    @staticmethod
    def topic_name(giskard_node_name: str) -> str:
        """
        The topic the clients of the given Giskard announce themselves on.
        """
        return f"{giskard_node_name}/client_heartbeat"

    def publish(self) -> None:
        """
        Announce that this client is still there.
        """
        self.publisher.publish(self.message)

    def stop(self) -> None:
        """
        Stop announcing this client.
        """
        self.timer.cancel()
        self.node.destroy_timer(self.timer)
        self.node.destroy_publisher(self.publisher)


# %% whether a client is still there


@dataclass
class HeartbeatPresence:
    """
    Reads the heartbeats of the clients and considers one gone once its heartbeats stop
    arriving.

    This is the fast check: it notices a client that was killed within ``timeout``, and a
    client that is still running but no longer gets around to announcing itself.
    """

    node: Node
    """
    Node of Giskard, which the heartbeat topic is named after.
    """

    timeout: timedelta = timedelta(seconds=3)
    """
    Time without a heartbeat after which a client counts as gone.

    Heartbeats are received on a ros executor thread while the control loop runs on its
    own, so this has to stay well above the heartbeat period: a value close to it would
    turn a busy executor into a stopped robot.
    """

    clock: Callable[[], float] = field(default=time.monotonic, kw_only=True, repr=False)
    """
    Reads the time the heartbeats are dated with.

    Tests substitute a controllable clock here to advance simulated time
    deterministically instead of sleeping in real time.
    """

    watched_client: Optional[MetaData] = field(init=False, default=None)
    """
    The client this check reports on, or ``None`` while no goal is running.
    """

    last_heartbeat: Dict[MetaData, float] = field(init=False, default_factory=dict)
    """
    When each client announced itself last.
    """

    subscription: Subscription = field(init=False)
    """
    Subscription the heartbeats arrive on.
    """

    @property
    def client(self) -> MetaData:
        """
        The client that is being watched.

        :raises NoWatchedClientError: If nothing is being watched.
        """
        if self.watched_client is None:
            raise NoWatchedClientError(check_type=type(self))
        return self.watched_client

    def stop_watching(self) -> None:
        """
        Stop reporting on the client of the goal that just ended.
        """
        self.watched_client = None

    def __post_init__(self):
        self.subscription = self.node.create_subscription(
            std_msgs.msg.String,
            topic=ClientHeartbeatPublisher.topic_name(self.node.get_name()),
            callback=self.receive_heartbeat,
            qos_profile=10,
        )

    def receive_heartbeat(self, message: std_msgs.msg.String) -> None:
        """
        Note that the client that sent this heartbeat is still there.
        """
        client = from_json(json.loads(message.data))
        self.last_heartbeat[client] = self.clock()

    def has_recent_heartbeat(self, client: MetaData) -> bool:
        """
        Whether the given client announced itself within ``timeout``.
        """
        last_heartbeat = self.last_heartbeat.get(client)
        if last_heartbeat is None:
            return False
        return self.clock() - last_heartbeat <= self.timeout.total_seconds()

    def start_watching(self, client: MetaData) -> bool:
        if not self.has_recent_heartbeat(client):
            return False
        self.watched_client = client
        return True

    def is_client_present(self) -> bool:
        return self.has_recent_heartbeat(self.client)


# %% watching the client of a goal


@dataclass
class ClientWatchdog:
    """
    Watches the client of the running goal and reports when it is gone.

    A goal outlives its client for as long as nobody notices, which leaves the robot
    executing a plan that nobody is waiting for anymore.
    """

    presence: HeartbeatPresence
    """
    The check that reports on the client's continued presence.
    """

    @property
    def client(self) -> MetaData:
        """
        The client of the running goal.

        :raises NoWatchedClientError: If no goal is being watched.
        """
        if self.presence.watched_client is None:
            raise NoWatchedClientError(check_type=type(self))
        return self.presence.client

    def watch(self, client: MetaData) -> None:
        """
        Start watching the client of a goal, if the presence check recognizes it.

        A client the check does not recognize is not watched, so an unknown client can
        never make Giskard stop a goal it is still waiting for.
        """
        self.presence.start_watching(client)

    def stop_watching(self) -> None:
        """
        Stop watching the client of the goal that just ended.
        """
        if self.presence.watched_client is None:
            return
        self.presence.stop_watching()

    def is_client_gone(self) -> bool:
        """
        Whether the client of the running goal disconnected.
        """
        if self.presence.watched_client is None:
            return False
        return not self.presence.is_client_present()
