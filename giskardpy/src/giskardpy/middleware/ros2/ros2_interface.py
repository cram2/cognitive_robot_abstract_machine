import threading
from typing import List, Optional, Tuple, Union, Any

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
from giskardpy.middleware.ros2.utils.asynio_utils import wait_until_not_none

from giskardpy.middleware.ros2.exceptions import (
    ExecutionAbortedException,
    ExecutionCanceledException,
)
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.event_loop_manager import get_event_loop


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
    return wait_for_message(String, rospy.node, topic, qos_profile=qos_profile)[1].data


def search_for_publisher_of_node_with_type(node_name: str, topic_type):
    topics = rospy.node.get_publisher_names_and_types_by_node(node_name, "/")
    return _search_in_topic_list(
        node_name=node_name, topic_list=topics, topic_type=topic_type
    )[0]


def search_for_subscriber_of_node_with_type(node_name: str, topic_type):
    topics = rospy.node.get_subscriber_names_and_types_by_node(node_name, "/")
    return _search_in_topic_list(
        node_name=node_name, topic_list=topics, topic_type=topic_type
    )[0]


def search_for_publishers_of_type(topic_type) -> List[str]:
    topics = _search_in_topic_list(
        topic_list=rospy.node.get_topic_names_and_types(), topic_type=topic_type
    )
    matches = []
    for topic_name in topics:
        if len(rospy.node.get_publishers_info_by_topic(topic_name)) > 0:
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
        topic_list=rospy.node.get_topic_names_and_types(), topic_type=topic_type
    )
    matches = []
    for topic_name in topics:
        if len(rospy.node.get_subscriptions_info_by_topic(topic_name)) > 0:
            matches.append(topic_name)
    return matches


def get_parameters(
    parameters: List[str], node_name: str = "controller_manager"
) -> GetParameters_Response:
    from controller_manager import controller_manager_services

    req = GetParameters_Request()
    req.names = parameters
    return controller_manager_services.service_caller(
        node=rospy.node,
        service_name=f"{node_name}/get_parameters",
        service_type=GetParameters,
        request=req,
        service_timeout=10,
    )


def _search_in_topic_list(
    topic_list: List[Tuple[str, list]], topic_type: str, node_name: Optional[str] = None
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
    _goal_handle: Optional[ClientGoalHandle]
    _result_future: Optional[Future]
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
            rospy.node.get_logger().info(f"Waiting for {action_name} server...")

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
                raise ExecutionAbortedException()
            case GoalStatus.STATUS_SUCCEEDED:
                return result
            case GoalStatus.STATUS_CANCELED:
                raise ExecutionCanceledException(self._client._action_name, goal_id)
            case _:
                raise Exception(f"Unexpected status {result.status}")

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
