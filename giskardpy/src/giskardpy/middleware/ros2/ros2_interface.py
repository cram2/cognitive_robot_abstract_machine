import json
import threading
from datetime import timedelta
from typing import List, Tuple, Type, Union, Any

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

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.event_loop_manager import get_event_loop
from giskardpy.middleware.ros2.exceptions import (
    ExecutionAbortedException,
    ExecutionCanceledException,
    ServiceUnavailableError,
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
    node: "Node",
    topic: str,
    *,
    qos_profile: Union[QoSProfile, int] = QoSProfile(depth=10),
    time_to_wait=-1,
) -> Tuple[bool, Any]:
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
    node_name: str, topic_type, namespace: str = "/"
):
    topics = rospy.get_node().get_publisher_names_and_types_by_node(
        node_name, namespace
    )
    return _search_in_topic_list(
        node_name=node_name, topic_list=topics, topic_type=topic_type
    )[0]


def search_for_subscriber_of_node_with_type(
    node_name: str, topic_type, namespace: str = "/"
):
    topics = rospy.get_node().get_subscriber_names_and_types_by_node(
        node_name, namespace
    )
    return _search_in_topic_list(
        node_name=node_name, topic_list=topics, topic_type=topic_type
    )[0]


def search_for_publishers_of_type(topic_type) -> List[str]:
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
    assert (
        len(topic_names) == 1
    ), f"Found too many {msg_type_as_str(topic_type)} topics: {topic_names}."
    return topic_names[0]


def search_for_unique_subscriber_of_type(topic_type) -> str:
    topic_names = search_for_subscribers_of_type(topic_type)
    assert (
        len(topic_names) == 1
    ), f"Found too many {msg_type_as_str(topic_type)} topics: {topic_names}."
    return topic_names[0]


def search_for_subscribers_of_type(topic_type) -> List[str]:
    topics = _search_in_topic_list(
        topic_list=rospy.get_node().get_topic_names_and_types(), topic_type=topic_type
    )
    matches = []
    for topic_name in topics:
        if len(rospy.get_node().get_subscriptions_info_by_topic(topic_name)) > 0:
            matches.append(topic_name)
    return matches


def call_service(
    service_type: Type,
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
    parameters: List[str], node_name: str = "controller_manager"
) -> GetParameters_Response:
    request = GetParameters_Request()
    request.names = parameters
    return call_service(GetParameters, f"{node_name}/get_parameters", request)


def _search_in_topic_list(
    topic_list: List[Tuple[str, list]], topic_type: str, node_name: str | None = None
) -> List[str]:
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


def wait_for_publisher(publisher):
    return
    # while publisher.get_num_connections() == 0:
    #     rospy.sleep(0.1)


class MyActionClient:
    _goal_handle: ClientGoalHandle | None
    _result_future: Future | None
    _goal_counter: int

    def __init__(self, node_handle: Node, action_type, action_name: str):
        self._goal_counter = -1
        self._goal_handle = None
        self._goal_result = None
        self._result_future = None
        self._current_goal_id = None
        self.result = None
        self.node_handle = node_handle
        self.action_name = action_name
        self._client = ActionClient(
            node=node_handle, action_type=action_type, action_name=action_name
        )
        while not self._client.wait_for_server(timeout_sec=2):
            self.node_handle.get_logger().info(f"Waiting for {action_name} server...")

    def send_goal_async(self, goal) -> Future:
        self._goal_counter += 1
        self._current_goal_id = self._goal_counter
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self.__goal_accepted_cb)
        return future

    def send_goal(self, goal):
        async def muh():
            rospy.wait_for_future_to_complete(self.send_goal_async(goal))
            result = await self.get_result()
            return result

        return get_event_loop().run_until_complete(muh())

    async def get_result(self):
        goal_id = self._current_goal_id
        await wait_until_not_none(lambda: self.result)
        result = self.result
        self.result = None
        match result.status:
            case GoalStatus.STATUS_ABORTED:
                raise self.create_abort_exception(result)
            case GoalStatus.STATUS_SUCCEEDED:
                return result
            case GoalStatus.STATUS_CANCELED:
                raise ExecutionCanceledException(self._client._action_name, goal_id)
            case _:
                raise Exception(f"Unexpected status {result.status}")

    @staticmethod
    def create_abort_exception(result: Any) -> Exception:
        """
        Rebuild the exception that made the server abort the goal.

        The action status alone cannot tell a caller whether sending the goal again
        would help, so the error itself travels in the result payload.

        An error that cannot be rebuilt, because the client does not know its class or
        cannot construct it, is reported as a plain abort; the original failure is worth
        less than a caller that keeps working.
        """
        payload = json.loads(result.result.result)
        error = payload.get("error")
        if error is None:
            return ExecutionAbortedException()
        try:
            return from_json(error)
        except (JSONSerializationError, TypeError):
            return ExecutionAbortedException()

    def __goal_accepted_cb(self, future: Future):
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

        self._result_future = self._goal_handle.get_result_async()
        self._result_future.add_done_callback(lambda f: self.__goal_done_cb(f, goal_id))

    def __goal_done_cb(self, future: Future, goal_id: int):
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
