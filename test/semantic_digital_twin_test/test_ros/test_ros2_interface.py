from datetime import timedelta

import pytest
from rcl_interfaces.msg import FloatingPointRange
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
from semantic_digital_twin.adapters.ros.ros2_interface import (
    call_service,
    message_type_name,
    search_for_unique_publisher_of_type,
    wait_for_message,
)

# %% message types


def test_a_message_type_is_named_the_way_ros_names_it() -> None:
    assert message_type_name(String) == "std_msgs/msg/String"
    assert message_type_name(Trigger) == "std_srvs/srv/Trigger"


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
        search_for_unique_publisher_of_type(rospy.get_node(), FloatingPointRange)

    assert raised.value.message_type_name == "rcl_interfaces/msg/FloatingPointRange"


def test_looking_for_the_publisher_of_a_type_two_topics_carry_raises(
    two_topics_of_one_type: list[str],
) -> None:
    with pytest.raises(AmbiguousTopicError) as raised:
        search_for_unique_publisher_of_type(rospy.get_node(), FloatingPointRange)

    assert sorted(raised.value.topic_names) == two_topics_of_one_type


# %% waiting for messages


def test_waiting_for_a_message_nobody_publishes_raises(init_rospy) -> None:
    with pytest.raises(NoMessageReceivedError) as raised:
        wait_for_message(
            rospy.get_node(),
            String,
            "/topic_nobody_publishes_on",
            timeout=timedelta(seconds=0.1),
        )

    assert raised.value.topic == "/topic_nobody_publishes_on"


# %% calling services


def test_calling_a_service_returns_its_response(init_rospy) -> None:
    node = rospy.get_node()

    def respond(
        request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        response.message = "called"
        return response

    service = node.create_service(Trigger, "service_offered_by_the_test", respond)
    response = call_service(
        node, Trigger, "service_offered_by_the_test", Trigger.Request()
    )
    node.destroy_service(service)

    assert response.message == "called"


def test_calling_a_service_that_nobody_offers_raises(init_rospy) -> None:
    with pytest.raises(ServiceUnavailableError):
        call_service(
            rospy.get_node(),
            Trigger,
            "service_that_nobody_offers",
            Trigger.Request(),
            wait_timeout=timedelta(seconds=0.1),
        )
