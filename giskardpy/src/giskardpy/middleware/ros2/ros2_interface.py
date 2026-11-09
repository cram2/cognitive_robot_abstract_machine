from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing_extensions import Callable, Any

from action_msgs.msg import GoalStatus
from rcl_interfaces.srv._get_parameters import (
    GetParameters_Request,
    GetParameters_Response,
    GetParameters,
)
from rclpy import Future
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from std_msgs.msg import String

from semantic_digital_twin.adapters.ros import rospy
from giskardpy.middleware.ros2.event_loop_manager import get_event_loop
from giskardpy.middleware.ros2.exceptions import (
    ExecutionAbortedException,
    ExecutionCanceledException,
    GoalRequestNotAnsweredError,
    ServiceUnavailableError,
    UnexpectedGoalStatusError,
)
from giskardpy.middleware.ros2.utils.asynio_utils import wait_until_not_none
from krrood.adapters.exceptions import JSONSerializationError
from krrood.adapters.json_serializer import from_json


def msg_type_as_str(msg_type) -> str:
    type_str_parts = str(type(msg_type())).split(".")
    part1 = type_str_parts[0].split("'")[1]
    part2 = type_str_parts[1]
    part3 = type_str_parts[-1].split("'")[0]
    return f"{part1}/{part2}/{part3}"


def wait_for_message(
    msg_type,
    node: Node,
    topic: str,
    *,
    qos_profile: QoSProfile | int = QoSProfile(depth=10),
    time_to_wait=-1,
) -> tuple[bool, Any]:
    event = threading.Event()
    msg_holder = [None]

    def cb(msg):
        msg_holder[0] = msg
        event.set()

    sub = node.create_subscription(msg_type, topic, cb, qos_profile)
    try:
        timeout = None if time_to_wait < 0 else time_to_wait
        received = event.wait(timeout=timeout)
    finally:
        node.destroy_subscription(sub)

    if received:
        return True, msg_holder[0]
    return False, None


def get_robot_description(topic: str = "/robot_description") -> str:
    qos_profile = QoSProfile(depth=10)
    qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
    rospy.get_node().get_logger().info(f"Waiting for the robot description on {topic}.")
    return wait_for_message(String, rospy.get_node(), topic, qos_profile=qos_profile)[
        1
    ].data


def search_for_publisher_of_node_with_type(
    node_name: str, topic_type: type, namespace: str = "/"
) -> str:
    """
    :param node_name: The name of the node that publishes the topic.
    :param topic_type: The message type of the topic.
    :param namespace: The namespace of the node.
    :return: The first topic of `topic_type` the node publishes.
    """
    topics = rospy.get_node().get_publisher_names_and_types_by_node(
        node_name, namespace
    )
    return _search_in_topic_list(
        node_name=node_name, topic_list=topics, topic_type=topic_type
    )[0]


def search_for_subscriber_of_node_with_type(
    node_name: str, topic_type: type, namespace: str = "/"
) -> str:
    """
    :param node_name: The name of the node that subscribes to the topic.
    :param topic_type: The message type of the topic.
    :param namespace: The namespace of the node.
    :return: The first topic of `topic_type` the node subscribes to.
    """
    topics = rospy.get_node().get_subscriber_names_and_types_by_node(
        node_name, namespace
    )
    return _search_in_topic_list(
        node_name=node_name, topic_list=topics, topic_type=topic_type
    )[0]


def search_for_publishers_of_type(topic_type) -> list[str]:
    topics = _search_in_topic_list(
        topic_list=rospy.get_node().get_topic_names_and_types(), topic_type=topic_type
    )
    matches = []
    for topic_name in topics:
        if len(rospy.get_node().get_publishers_info_by_topic(topic_name)) > 0:
            matches.append(topic_name)
    return matches


def search_for_unique_publisher_of_type(topic_type) -> str:
    topic_names = search_for_publishers_of_type(topic_type)
    assert len(topic_names) == 1, (
        f"Found too many {msg_type_as_str(topic_type)} topics: {topic_names}."
    )
    return topic_names[0]


def search_for_unique_subscriber_of_type(topic_type) -> str:
    topic_names = search_for_subscribers_of_type(topic_type)
    assert len(topic_names) == 1, (
        f"Found too many {msg_type_as_str(topic_type)} topics: {topic_names}."
    )
    return topic_names[0]


def search_for_subscribers_of_type(topic_type) -> list[str]:
    topics = _search_in_topic_list(
        topic_list=rospy.get_node().get_topic_names_and_types(), topic_type=topic_type
    )
    matches = []
    for topic_name in topics:
        if len(rospy.get_node().get_subscriptions_info_by_topic(topic_name)) > 0:
            matches.append(topic_name)
    return matches


def call_service(
    service_type: type,
    service_name: str,
    request: Any,
    wait_timeout: timedelta = timedelta(seconds=10),
) -> Any:
    """
    Call a service through Giskard's node and wait for its response.

    .. warning:: The response is delivered by the executor that spins Giskard's node, so
        this must not be called from one of that node's callbacks.

    :param service_type: The type of the service.
    :param service_name: The name of the service.
    :param request: The request to send.
    :param wait_timeout: How long to wait for the service to become available.
    :return: The response of the service.
    :raises ServiceUnavailableError: If the service does not become available in time.
    """
    node = rospy.get_node()
    client = node.create_client(service_type, service_name)
    node.get_logger().info(f"Waiting for the service {service_name}.")
    if not client.wait_for_service(timeout_sec=wait_timeout.total_seconds()):
        node.destroy_client(client)
        raise ServiceUnavailableError(service_name=service_name)
    future = client.call_async(request)
    rospy.wait_for_future_to_complete(future)
    node.destroy_client(client)
    return future.result()


def get_parameters(
    parameters: list[str], node_name: str = "controller_manager"
) -> GetParameters_Response:
    request = GetParameters_Request()
    request.names = parameters
    return call_service(GetParameters, f"{node_name}/get_parameters", request)


def _search_in_topic_list(
    topic_list: list[tuple[str, list]], topic_type: str, node_name: str | None = None
) -> list[str]:
    matches = []
    for topic_name, topic_types in topic_list:
        if topic_types[0] == msg_type_as_str(topic_type):
            matches.append(topic_name)
    if matches:
        return matches
    if node_name is not None:
        raise AttributeError(f"Node {node_name} has no topic of type {topic_type}.")
    else:
        raise AttributeError(f"Didn't find topic of type {topic_type}.")


@dataclass
class MyActionClient:
    """
    Sends goals to one action and waits for their results, one goal at a time.

    Building it blocks until a server of the action is found.
    """

    node_handle: Node
    """
    The node the action client is created on.
    """

    action_type: Any
    """
    The ROS action type of the action.
    """

    action_name: str
    """
    The name of the action goals are sent to.
    """

    goal_response_timeout: timedelta = field(
        default=timedelta(seconds=30), kw_only=True
    )
    """
    How long :meth:`send_goal` waits for a server to accept or reject a goal.
    """

    result: Any | None = field(init=False, default=None)
    """
    The result of the goal sent last, until :meth:`get_result` takes it.
    """

    _goal_counter: int = field(init=False, default=-1)
    """
    Identifier of the goal sent last.
    """

    _current_goal_id: int | None = field(init=False, default=None)
    """
    Identifier of the goal that is still awaited, ``None`` once its result arrived.
    """

    _goal_handle: ClientGoalHandle | None = field(init=False, default=None)
    """
    Handle of the accepted goal that has not ended yet.
    """

    _result_future: Future | None = field(init=False, default=None)
    """
    Resolves with the result of the accepted goal.
    """

    _feedback_error: Exception | None = field(init=False, default=None)
    """
    The first error the feedback callback of the current goal raised.
    """

    _client: ActionClient | None = field(init=False, default=None)
    """
    The rclpy action client, ``None`` once :meth:`close` withdrew it.
    """

    def __post_init__(self) -> None:
        self._client = ActionClient(
            node=self.node_handle,
            action_type=self.action_type,
            action_name=self.action_name,
        )
        while not self._client.wait_for_server(timeout_sec=2):
            self.node_handle.get_logger().info(
                f"Waiting for {self.action_name} server..."
            )

    def close(self) -> None:
        """
        Withdraw the action client from the ROS graph, so that its node can be
        destroyed.

        Closing a client that is already closed does nothing.
        """
        if self._client is None:
            return
        self._client.destroy()
        self._client = None

    def send_goal_async(
        self, goal: Any, feedback_callback: Callable[[Any], None] | None = None
    ) -> Future:
        """
        Send `goal` without waiting for it to be accepted.

        :param goal: The goal to send.
        :param feedback_callback: Receives every feedback of the goal, see
            :meth:`_receive_feedback`.
        :return: The future that resolves once the server accepted or rejected the goal.
        """
        self._goal_counter += 1
        self._current_goal_id = self._goal_counter
        self._feedback_error = None
        if feedback_callback is not None:
            feedback_callback = partial(self._receive_feedback, feedback_callback)
        future = self._client.send_goal_async(goal, feedback_callback=feedback_callback)
        future.add_done_callback(self._receive_goal_response)
        return future

    def _receive_feedback(
        self, feedback_callback: Callable[[Any], None], feedback_message: Any
    ) -> None:
        """
        Hand `feedback_message` to `feedback_callback`.

        The first error the callback raises cancels the goal and is raised by
        :meth:`get_result`; the feedback that follows it is dropped.

        :param feedback_callback: The callback given with the goal.
        :param feedback_message: One feedback of the goal sent last.
        """
        if self._feedback_error is not None:
            return
        try:
            feedback_callback(feedback_message)
        except Exception as error:
            self._feedback_error = error
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()

    def send_goal(
        self, goal: Any, feedback_callback: Callable[[Any], None] | None = None
    ) -> Any:
        """
        Send `goal` and wait for its result.

        :param goal: The goal to send.
        :param feedback_callback: Receives every feedback of the goal, see
            :meth:`_receive_feedback`.
        :return: The result of the goal, see :meth:`get_result`.
        :raises GoalRequestNotAnsweredError: If no server accepts or rejects the goal
            within :attr:`goal_response_timeout`.
        """
        goal_response = self.send_goal_async(goal, feedback_callback=feedback_callback)
        answered = threading.Event()
        goal_response.add_done_callback(lambda _: answered.set())
        if not answered.wait(self.goal_response_timeout.total_seconds()):
            raise GoalRequestNotAnsweredError(
                action_name=self.action_name, timeout=self.goal_response_timeout
            )
        return get_event_loop().run_until_complete(self.get_result())

    async def get_result(self):
        """
        Wait for the result of the goal sent last.

        An aborted goal is returned like a succeeded one, so that the caller can take
        over its final state before raising the error it carries, see
        :meth:`create_abort_exception`.

        :raises ExecutionCanceledException: If the goal was canceled.
        :raises UnexpectedGoalStatusError: If the goal ended in a status a finished goal
            cannot have.
        :raises Exception: The error the feedback callback of the goal raised, see
            :meth:`_receive_feedback`.
        """
        goal_id = self._current_goal_id
        await wait_until_not_none(lambda: self.result)
        result = self.result
        self.result = None
        if self._feedback_error is not None:
            raise self._feedback_error
        match result.status:
            case GoalStatus.STATUS_ABORTED | GoalStatus.STATUS_SUCCEEDED:
                return result
            case GoalStatus.STATUS_CANCELED:
                raise ExecutionCanceledException(self.action_name, goal_id)
            case _:
                raise UnexpectedGoalStatusError(
                    action_name=self.action_name, status=result.status
                )

    @staticmethod
    def create_abort_exception(result: Any, **kwargs: Any) -> Exception:
        """
        Rebuild the exception that made the server abort the goal.

        The action status alone cannot tell a caller whether sending the goal again
        would help, so the error itself travels in the result payload.

        An error that cannot be rebuilt, because the client does not know its class or
        cannot construct it, is reported as a plain abort; the original failure is worth
        less than a caller that keeps working.

        :param result: The result of the aborted goal.
        :param kwargs: Handed to the deserialization of the error.
        :return: The rebuilt exception.
        """
        payload = json.loads(result.result.result)
        error = payload.get("error")
        if error is None:
            return ExecutionAbortedException()
        try:
            return from_json(error, **kwargs)
        except (JSONSerializationError, TypeError):
            return ExecutionAbortedException()

    def _receive_goal_response(self, future: Future) -> None:
        """
        Start waiting for the result of the goal sent last, if its server accepted it.

        :param future: Resolved with the handle of the goal the server answered.
        """
        goal_handle = future.result()
        goal_id = self._goal_counter  # Capture the current goal ID

        if not goal_handle.accepted:
            self.node_handle.get_logger().info(
                f"{self.action_name} Goal {goal_id} rejected"
            )
            return

        # Only process if this is still the current goal
        if goal_id != self._current_goal_id:
            self.node_handle.get_logger().debug(
                f"Ignoring accepted callback for old goal {goal_id}"
            )
            return

        self._goal_handle = goal_handle
        self.node_handle.get_logger().info(
            f"{self.action_name} Goal #{goal_id} accepted"
        )
        if self._feedback_error is not None:
            goal_handle.cancel_goal_async()

        self._result_future = self._goal_handle.get_result_async()
        self._result_future.add_done_callback(
            lambda result_future: self._receive_result(result_future, goal_id)
        )

    def _receive_result(self, future: Future, goal_id: int) -> None:
        """
        Keep the result of the goal ``goal_id`` for :meth:`get_result`, unless a newer
        goal was sent since.

        :param future: Resolved with the result of the goal.
        :param goal_id: Identifier of the goal the result belongs to.
        """
        # Only process if this is still the current goal
        if goal_id != self._current_goal_id:
            self.node_handle.get_logger().debug(
                f"Ignoring done callback for old goal {goal_id}"
            )
            self.result = None
            return

        self.node_handle.get_logger().info(
            f"{self.action_name} Goal #{goal_id} result received"
        )
        self.result = future.result()
        self._goal_handle = None
        self._current_goal_id = None
        self._result_future = None
