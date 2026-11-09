"""
Finding topics, waiting for messages and calling services and parameters on a ROS 2
graph, through a node the caller passes.

Messages and service responses arrive through the executor that spins that node, so
these functions must not be called from one of its callbacks.
"""

from __future__ import annotations

import threading
from datetime import timedelta

from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String
from typing_extensions import Any

from semantic_digital_twin.adapters.ros.exceptions import (
    AmbiguousTopicError,
    NoMessageReceivedError,
    ServiceUnavailableError,
    TopicNotFoundError,
)

# %% messages


def message_type_name(message_type: type) -> str:
    """
    :param message_type: A generated message, service or action class.
    :return: The name ROS 2 gives the type, e.g. ``std_msgs/msg/String``.
    """
    package, interface_kind = message_type.__module__.split(".")[:2]
    return f"{package}/{interface_kind}/{message_type.__name__}"


def wait_for_message(
    node: Node,
    message_type: type,
    topic: str,
    *,
    qos_profile: QoSProfile | None = None,
    timeout: timedelta | None = None,
) -> Any:
    """
    Wait for the next message on a topic.

    :param node: The node that subscribes to the topic while waiting.
    :param message_type: The message type of the topic.
    :param topic: The topic to wait on.
    :param qos_profile: The quality of service of the subscription, a depth of 10 if
        ``None``.
    :param timeout: How long to wait, forever if ``None``.
    :return: The message.
    :raises NoMessageReceivedError: If no message arrives in time.
    """
    received = threading.Event()
    messages = []

    def receive(message: Any) -> None:
        messages.append(message)
        received.set()

    subscription = node.create_subscription(
        message_type, topic, receive, qos_profile or QoSProfile(depth=10)
    )
    has_received = received.wait(
        timeout=None if timeout is None else timeout.total_seconds()
    )
    node.destroy_subscription(subscription)
    if not has_received:
        raise NoMessageReceivedError(topic=topic)
    return messages[0]


def get_robot_description(node: Node, topic: str = "/robot_description") -> str:
    """
    Wait for the robot description a robot state publisher latches on ``topic``.

    :param node: The node that subscribes to the topic while waiting.
    :param topic: The topic the description is published on.
    :return: The robot description.
    """
    node.get_logger().info(f"Waiting for the robot description on {topic}.")
    qos_profile = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    return wait_for_message(node, String, topic, qos_profile=qos_profile).data


# %% finding topics


def search_for_publisher_of_node_with_type(
    node: Node, node_name: str, topic_type: type, namespace: str = "/"
) -> str:
    """
    :param node: The node that looks at the graph.
    :param node_name: The name of the node that publishes the topic.
    :param topic_type: The message type of the topic.
    :param namespace: The namespace of the node.
    :return: The first topic of ``topic_type`` the node publishes.
    :raises TopicNotFoundError: If the node publishes no topic of that type.
    """
    topics = _topics_of_type(
        node.get_publisher_names_and_types_by_node(node_name, namespace), topic_type
    )
    if not topics:
        raise TopicNotFoundError(
            message_type_name=message_type_name(topic_type), node_name=node_name
        )
    return topics[0]


def search_for_subscriber_of_node_with_type(
    node: Node, node_name: str, topic_type: type, namespace: str = "/"
) -> str:
    """
    :param node: The node that looks at the graph.
    :param node_name: The name of the node that subscribes to the topic.
    :param topic_type: The message type of the topic.
    :param namespace: The namespace of the node.
    :return: The first topic of ``topic_type`` the node subscribes to.
    :raises TopicNotFoundError: If the node subscribes to no topic of that type.
    """
    topics = _topics_of_type(
        node.get_subscriber_names_and_types_by_node(node_name, namespace), topic_type
    )
    if not topics:
        raise TopicNotFoundError(
            message_type_name=message_type_name(topic_type), node_name=node_name
        )
    return topics[0]


def search_for_publishers_of_type(node: Node, topic_type: type) -> list[str]:
    """
    :param node: The node that looks at the graph.
    :param topic_type: The message type of the topics.
    :return: The topics of ``topic_type`` that some node publishes.
    """
    return [
        topic
        for topic in _topics_of_type(node.get_topic_names_and_types(), topic_type)
        if node.get_publishers_info_by_topic(topic)
    ]


def search_for_subscribers_of_type(node: Node, topic_type: type) -> list[str]:
    """
    :param node: The node that looks at the graph.
    :param topic_type: The message type of the topics.
    :return: The topics of ``topic_type`` that some node subscribes to.
    """
    return [
        topic
        for topic in _topics_of_type(node.get_topic_names_and_types(), topic_type)
        if node.get_subscriptions_info_by_topic(topic)
    ]


def search_for_unique_publisher_of_type(node: Node, topic_type: type) -> str:
    """
    :param node: The node that looks at the graph.
    :param topic_type: The message type of the topic.
    :return: The only topic of ``topic_type`` that some node publishes.
    :raises TopicNotFoundError: If no node publishes a topic of that type.
    :raises AmbiguousTopicError: If nodes publish several topics of that type.
    """
    return _unique_topic(search_for_publishers_of_type(node, topic_type), topic_type)


def search_for_unique_subscriber_of_type(node: Node, topic_type: type) -> str:
    """
    :param node: The node that looks at the graph.
    :param topic_type: The message type of the topic.
    :return: The only topic of ``topic_type`` that some node subscribes to.
    :raises TopicNotFoundError: If no node subscribes to a topic of that type.
    :raises AmbiguousTopicError: If nodes subscribe to several topics of that type.
    """
    return _unique_topic(search_for_subscribers_of_type(node, topic_type), topic_type)


def _topics_of_type(
    topic_names_and_types: list[tuple[str, list[str]]], topic_type: type
) -> list[str]:
    """
    :param topic_names_and_types: Topics with their types, as the graph queries of a
        node return them.
    :param topic_type: The message type to look for.
    :return: The topics whose type is ``topic_type``.
    """
    type_name = message_type_name(topic_type)
    return [
        topic
        for topic, topic_types in topic_names_and_types
        if topic_types[0] == type_name
    ]


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
    node: Node,
    service_type: type,
    service_name: str,
    request: Any,
    wait_timeout: timedelta = timedelta(seconds=10),
) -> Any:
    """
    Call a service and wait for its response.

    :param node: The node that calls the service.
    :param service_type: The type of the service.
    :param service_name: The name of the service.
    :param request: The request to send.
    :param wait_timeout: How long to wait for the service to become available.
    :return: The response of the service.
    :raises ServiceUnavailableError: If the service does not become available in time.
    """
    client = node.create_client(service_type, service_name)
    node.get_logger().info(f"Waiting for the service {service_name}.")
    if not client.wait_for_service(timeout_sec=wait_timeout.total_seconds()):
        node.destroy_client(client)
        raise ServiceUnavailableError(service_name=service_name)
    response = client.call(request)
    node.destroy_client(client)
    return response


def get_parameters(
    node: Node, node_name: str, parameter_names: list[str]
) -> GetParameters.Response:
    """
    :param node: The node that asks for the parameters.
    :param node_name: The node whose parameters are read, including its namespace.
    :param parameter_names: The parameters to read.
    :return: The values of the parameters, in the order of ``parameter_names``.
    :raises ServiceUnavailableError: If the node does not offer its parameters in time.
    """
    return call_service(
        node,
        GetParameters,
        f"{node_name}/get_parameters",
        GetParameters.Request(names=parameter_names),
    )
