"""
Finding topics, waiting for messages and calling services and parameters on a ROS 2
graph, through a node someone else owns.

:class:`RosGraph` only uses a node: it never creates, spins or destroys one. Whoever runs
the process owns the node, for example :mod:`semantic_digital_twin.adapters.ros.background_node`
or an application that registers its own node in
:class:`~semantic_digital_twin.adapters.ros.node_registry.ROSNodeRegistry`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import timedelta
from time import sleep

import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy import Future
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String
from typing_extensions import Any, TypeAlias

from semantic_digital_twin.adapters.ros.exceptions import (
    AmbiguousTopicError,
    NoMessageReceivedError,
    ServiceUnavailableError,
    TopicNotFoundError,
)
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry

# %% types and futures

TopicNamesAndTypes: TypeAlias = list[tuple[str, list[str]]]
"""
Topics with the names of their types, as the graph queries of a node return them.
"""


def message_type_name(message_type: type) -> str:
    """
    :param message_type: A generated message, service or action class.
    :return: The name ROS 2 gives the type, e.g. ``std_msgs/msg/String``.
    """
    package, interface_kind = message_type.__module__.split(".")[:2]
    return f"{package}/{interface_kind}/{message_type.__name__}"


def wait_for_future_to_complete(future: Future) -> None:
    """
    Block until ``future`` is done or ROS shuts down.

    The future is completed by the executor that spins the node it belongs to, so this
    must not be called from one of that node's callbacks.

    :param future: The future to wait for.
    """
    while rclpy.ok() and not future.done():
        sleep(0.01)


# %% receiving one message


@dataclass
class _MessageInbox:
    """
    Keeps the first message a subscription receives.
    """

    received: threading.Event = field(default_factory=threading.Event)
    """
    Set once the first message arrived.
    """

    message: Any = None
    """
    The first message, ``None`` until it arrived.
    """

    def receive(self, message: Any) -> None:
        """
        Subscription callback that keeps ``message`` if it is the first one.

        :param message: A message of the subscription.
        """
        if self.received.is_set():
            return
        self.message = message
        self.received.set()


# %% the graph


@dataclass
class RosGraph:
    """
    The ROS 2 graph as seen from one node: its topics, messages, services and
    parameters.

    Messages and service responses arrive through the executor that spins the node, so
    the methods must not be called from one of the node's callbacks.
    """

    node: Node = field(default_factory=lambda: ROSNodeRegistry().get())
    """
    The node that looks at the graph, the node registered for the process by default.
    """

    # %% messages

    def wait_for_message(
        self,
        message_type: type,
        topic: str,
        *,
        qos_profile: QoSProfile | None = None,
        timeout: timedelta | None = None,
    ) -> Any:
        """
        Wait for the next message on a topic.

        :param message_type: The message type of the topic.
        :param topic: The topic to wait on.
        :param qos_profile: The quality of service of the subscription, a depth of 10 if
            ``None``.
        :param timeout: How long to wait, forever if ``None``.
        :return: The message.
        :raises NoMessageReceivedError: If no message arrives in time.
        """
        inbox = _MessageInbox()
        subscription = self.node.create_subscription(
            message_type, topic, inbox.receive, qos_profile or QoSProfile(depth=10)
        )
        has_received = inbox.received.wait(
            timeout=None if timeout is None else timeout.total_seconds()
        )
        self.node.destroy_subscription(subscription)
        if not has_received:
            raise NoMessageReceivedError(topic=topic)
        return inbox.message

    def robot_description(self, topic: str = "/robot_description") -> str:
        """
        Wait for the robot description a robot state publisher latches on ``topic``.

        :param topic: The topic the description is published on.
        :return: The robot description.
        """
        self.node.get_logger().info(f"Waiting for the robot description on {topic}.")
        qos_profile = QoSProfile(
            depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        return self.wait_for_message(String, topic, qos_profile=qos_profile).data

    # %% finding topics

    def publisher_of_node_with_type(
        self, node_name: str, topic_type: type, namespace: str = "/"
    ) -> str:
        """
        :param node_name: The name of the node that publishes the topic.
        :param topic_type: The message type of the topic.
        :param namespace: The namespace of the node.
        :return: The first topic of ``topic_type`` the node publishes.
        :raises TopicNotFoundError: If the node publishes no topic of that type.
        """
        return self._first_topic_of_node(
            self.node.get_publisher_names_and_types_by_node(node_name, namespace),
            topic_type,
            node_name,
        )

    def subscriber_of_node_with_type(
        self, node_name: str, topic_type: type, namespace: str = "/"
    ) -> str:
        """
        :param node_name: The name of the node that subscribes to the topic.
        :param topic_type: The message type of the topic.
        :param namespace: The namespace of the node.
        :return: The first topic of ``topic_type`` the node subscribes to.
        :raises TopicNotFoundError: If the node subscribes to no topic of that type.
        """
        return self._first_topic_of_node(
            self.node.get_subscriber_names_and_types_by_node(node_name, namespace),
            topic_type,
            node_name,
        )

    def publishers_of_type(self, topic_type: type) -> list[str]:
        """
        :param topic_type: The message type of the topics.
        :return: The topics of ``topic_type`` that some node publishes.
        """
        return [
            topic
            for topic in self._topics_of_type(
                self.node.get_topic_names_and_types(), topic_type
            )
            if self.node.get_publishers_info_by_topic(topic)
        ]

    def subscribers_of_type(self, topic_type: type) -> list[str]:
        """
        :param topic_type: The message type of the topics.
        :return: The topics of ``topic_type`` that some node subscribes to.
        """
        return [
            topic
            for topic in self._topics_of_type(
                self.node.get_topic_names_and_types(), topic_type
            )
            if self.node.get_subscriptions_info_by_topic(topic)
        ]

    def unique_publisher_of_type(self, topic_type: type) -> str:
        """
        :param topic_type: The message type of the topic.
        :return: The only topic of ``topic_type`` that some node publishes.
        :raises TopicNotFoundError: If no node publishes a topic of that type.
        :raises AmbiguousTopicError: If nodes publish several topics of that type.
        """
        return self._unique_topic(self.publishers_of_type(topic_type), topic_type)

    def unique_subscriber_of_type(self, topic_type: type) -> str:
        """
        :param topic_type: The message type of the topic.
        :return: The only topic of ``topic_type`` that some node subscribes to.
        :raises TopicNotFoundError: If no node subscribes to a topic of that type.
        :raises AmbiguousTopicError: If nodes subscribe to several topics of that type.
        """
        return self._unique_topic(self.subscribers_of_type(topic_type), topic_type)

    @staticmethod
    def _topics_of_type(
        topic_names_and_types: TopicNamesAndTypes, topic_type: type
    ) -> list[str]:
        """
        :param topic_names_and_types: The topics to look through.
        :param topic_type: The message type to look for.
        :return: The topics whose type is ``topic_type``.
        """
        type_name = message_type_name(topic_type)
        return [
            topic
            for topic, topic_types in topic_names_and_types
            if topic_types[0] == type_name
        ]

    @classmethod
    def _first_topic_of_node(
        cls,
        topic_names_and_types: TopicNamesAndTypes,
        topic_type: type,
        node_name: str,
    ) -> str:
        """
        :param topic_names_and_types: The topics of the node called ``node_name``.
        :param topic_type: The message type to look for.
        :param node_name: The name of the node the topics belong to.
        :return: The first topic whose type is ``topic_type``.
        :raises TopicNotFoundError: If no topic has that type.
        """
        topics = cls._topics_of_type(topic_names_and_types, topic_type)
        if not topics:
            raise TopicNotFoundError(
                message_type_name=message_type_name(topic_type), node_name=node_name
            )
        return topics[0]

    @staticmethod
    def _unique_topic(topics: list[str], topic_type: type) -> str:
        """
        :param topics: The topics of ``topic_type`` that were found.
        :param topic_type: The message type that was looked for.
        :return: The only topic in ``topics``.
        :raises TopicNotFoundError: If ``topics`` is empty.
        :raises AmbiguousTopicError: If ``topics`` holds several topics.
        """
        type_name = message_type_name(topic_type)
        if not topics:
            raise TopicNotFoundError(message_type_name=type_name)
        if len(topics) > 1:
            raise AmbiguousTopicError(message_type_name=type_name, topic_names=topics)
        return topics[0]

    # %% services and parameters

    def call_service(
        self,
        service_type: type,
        service_name: str,
        request: Any,
        wait_timeout: timedelta = timedelta(seconds=10),
    ) -> Any:
        """
        Call a service and wait for its response.

        :param service_type: The type of the service.
        :param service_name: The name of the service.
        :param request: The request to send.
        :param wait_timeout: How long to wait for the service to become available.
        :return: The response of the service.
        :raises ServiceUnavailableError: If the service does not become available in
            time.
        """
        client = self.node.create_client(service_type, service_name)
        self.node.get_logger().info(f"Waiting for the service {service_name}.")
        if not client.wait_for_service(timeout_sec=wait_timeout.total_seconds()):
            self.node.destroy_client(client)
            raise ServiceUnavailableError(service_name=service_name)
        response = client.call(request)
        self.node.destroy_client(client)
        return response

    def get_parameters(
        self, node_name: str, parameter_names: list[str]
    ) -> GetParameters.Response:
        """
        :param node_name: The node whose parameters are read, including its namespace.
        :param parameter_names: The parameters to read.
        :return: The values of the parameters, in the order of ``parameter_names``.
        :raises ServiceUnavailableError: If the node does not offer its parameters in
            time.
        """
        return self.call_service(
            GetParameters,
            f"{node_name}/get_parameters",
            GetParameters.Request(names=parameter_names),
        )
