from datetime import timedelta

import pytest
from rcl_interfaces.msg import FloatingPointRange
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger
from typing_extensions import Iterator

from semantic_digital_twin.adapters.ros import rospy
from semantic_digital_twin.adapters.ros.exceptions import (
    AmbiguousTopicError,
    NoMessageReceivedError,
    ServiceUnavailableError,
    TopicNotFoundError,
)
from semantic_digital_twin.adapters.ros.ros_graph import RosGraph, message_type_name

# %% message types


def test_a_message_type_is_named_the_way_ros_names_it() -> None:
    assert message_type_name(String) == "std_msgs/msg/String"
    assert message_type_name(Trigger) == "std_srvs/srv/Trigger"


# %% the node


def test_a_graph_looks_through_the_registered_node_by_default(init_rospy) -> None:
    assert RosGraph().node is rospy.get_node()


# %% finding topics


@pytest.fixture()
def two_topics_of_one_type(init_rospy) -> Iterator[list[str]]:
    """
    Publishes two topics of a type nothing else in the graph publishes and yields their
    names.
    """
    node = rospy.get_node()
    topics = ["/first_range", "/second_range"]
    publishers = [
        node.create_publisher(FloatingPointRange, topic, 1) for topic in topics
    ]
    yield topics
    for publisher in publishers:
        node.destroy_publisher(publisher)


def test_looking_for_the_publisher_of_a_type_nobody_publishes_raises(
    init_rospy,
) -> None:
    with pytest.raises(TopicNotFoundError) as raised:
        RosGraph().unique_publisher_of_type(FloatingPointRange)

    assert raised.value.message_type_name == "rcl_interfaces/msg/FloatingPointRange"


def test_looking_for_the_publisher_of_a_type_two_topics_carry_raises(
    two_topics_of_one_type: list[str],
) -> None:
    with pytest.raises(AmbiguousTopicError) as raised:
        RosGraph().unique_publisher_of_type(FloatingPointRange)

    assert sorted(raised.value.topic_names) == two_topics_of_one_type


# %% waiting for messages


def test_a_latched_robot_description_is_received(init_rospy) -> None:
    node = rospy.get_node()
    topic = "/description_latched_by_the_test"
    publisher = node.create_publisher(
        String,
        topic,
        QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
    )
    publisher.publish(String(data="<robot/>"))

    description = RosGraph().robot_description(topic)
    node.destroy_publisher(publisher)

    assert description == "<robot/>"


def test_waiting_for_a_message_nobody_publishes_raises(init_rospy) -> None:
    with pytest.raises(NoMessageReceivedError) as raised:
        RosGraph().wait_for_message(
            String, "/topic_nobody_publishes_on", timeout=timedelta(seconds=0.1)
        )

    assert raised.value.topic == "/topic_nobody_publishes_on"


# %% calling services


def respond_called(
    request: Trigger.Request, response: Trigger.Response
) -> Trigger.Response:
    """
    Answers every trigger with the message ``called``.
    """
    response.message = "called"
    return response


def test_calling_a_service_returns_its_response(init_rospy) -> None:
    node = rospy.get_node()
    service = node.create_service(
        Trigger, "service_offered_by_the_test", respond_called
    )
    response = RosGraph().call_service(
        Trigger, "service_offered_by_the_test", Trigger.Request()
    )
    node.destroy_service(service)

    assert response.message == "called"


def test_calling_a_service_that_nobody_offers_raises(init_rospy) -> None:
    with pytest.raises(ServiceUnavailableError):
        RosGraph().call_service(
            Trigger,
            "service_that_nobody_offers",
            Trigger.Request(),
            wait_timeout=timedelta(seconds=0.1),
        )
