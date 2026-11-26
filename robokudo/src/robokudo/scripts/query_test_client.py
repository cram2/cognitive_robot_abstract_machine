"""
Action client that sends test queries to the RoboKudo query action server.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from queue import Queue
from typing_extensions import TYPE_CHECKING, Any, Protocol

import rclpy
from action_msgs.msg import GoalStatus
from rclpy import Context
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry

from robokudo_msgs.action import Query
from krrood.patterns.called_positionally import called_positionally


if TYPE_CHECKING:
    from rclpy.action.client import ClientGoalHandle
    from rclpy.task import Future
    from rclpy.timer import Timer
    from robokudo_msgs.action._query import Query_FeedbackMessage, Query_Result

logger = logging.getLogger(__name__)


class ResultQueue(Protocol):
    """
    Queue-like object receiving the action client's final result dictionary.
    """

    @called_positionally
    def put(self, item: dict[str, Any]) -> None:
        """
        Store one result dictionary.
        """
        ...


class ReadinessSignal(Protocol):
    """
    Signal that the action client is ready to discover its server.
    """

    def set(self) -> None:
        """
        Report that client construction has completed.
        """
        ...


class PrettyResultPrinter:
    """
    Formats a query result for the console.
    """

    def pretty_print_result(self, *, result: Query_Result) -> str:
        """
        :return: *result* as indented JSON.
        """
        result_dict = message_to_ordereddict(result)
        return json.dumps(result_dict, indent=2)


@dataclass(eq=False, kw_only=True)
class RoboKudoActionClient:
    """
    Send dynamic goals to the RoboKudo query action server.
    """

    node: Node
    """
    ROS node providing communication and timer resources.

    Make sure to spin it externally.
    """

    preempt_timer: float | None = None
    """
    Delay before automatically cancelling an active goal.
    """

    _action_client: ActionClient = field(init=False)
    """
    ROS client communicating with the query action server.
    """

    _cancel_timer: Timer | None = field(init=False, default=None)
    """
    Timer for cancellation of the active goal.
    """

    _goal_handle: ClientGoalHandle | None = field(init=False, default=None)
    """
    Handle of the active goal.
    """

    done: bool = field(init=False, default=False)
    """
    Whether processing of the current goal has finished.
    """

    last_feedback: Query_FeedbackMessage | None = field(init=False, default=None)
    """
    Most recently received feedback.
    """

    goal_status: int | None = field(init=False, default=None)
    """
    Final status of the current goal.
    """

    goal_result: Query_Result | None = field(init=False, default=None)
    """
    Final result of the current goal.
    """

    cancel_response: Future | None = field(init=False, default=None)
    """
    Response to the latest cancellation request.
    """

    def __post_init__(self) -> None:
        """
        Create the ROS action client associated with the supplied node.
        """
        self._action_client = ActionClient(
            node=self.node, action_type=Query, action_name="/robokudo/query"
        )

    def send_goal(self, *, goal_type: str) -> None:
        """
        Waits for the action server and sends a dynamic goal.

        :param goal_type: Content of the type field in the goal sent to the action
            server.
        """
        logger.info("Waiting for action server...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            logger.error("Action server not available!")
            self.done = True
            return

        # Create and send a goal
        goal_msg = Query.Goal()
        goal_msg.obj.type = goal_type

        logger.info(f"Sending goal request with type: '{goal_type}'")
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    @called_positionally
    def goal_response_callback(self, future: Future) -> None:
        """
        Handles the response from the action server regarding goal acceptance.

        :param future: The future of the async goal task returned by the action client
            upon sending a goal.
        """
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            logger.error("Goal rejected by the action server.")
            self.done = True
            return

        logger.info("Goal accepted by the action server.")

        # Scheduling of cancellation after X seconds
        if self.preempt_timer is not None:
            self._cancel_timer = self.node.create_timer(
                self.preempt_timer, self.cancel_goal
            )

        # Wait for the result
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    @called_positionally
    def feedback_callback(self, feedback_msg: Query_FeedbackMessage) -> None:
        """
        Processes feedback messages from the action server.

        :param feedback_msg: The feedback message returned by the action client.
        """
        feedback = feedback_msg.feedback
        self.last_feedback = feedback
        logger.info(f"Received feedback: {feedback.feedback}")

    def cancel_goal(self) -> None:
        """
        Sends a cancel request for the active goal.
        """
        if not self._goal_handle:
            logger.error("No active goal to cancel.")
            return

        logger.info("Sending cancel request...")
        cancel_future = self._goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

        # Stop the cancel timer
        if self._cancel_timer:
            self._cancel_timer.cancel()

    @called_positionally
    def cancel_done_callback(self, future: Future) -> None:
        """
        Handles the response from the action server regarding goal cancellation.

        :param future: The future of the async goal task returned by the action client
            upon cancelling a goal.
        """
        cancel_response = future.result()
        self.cancel_response = cancel_response
        if len(cancel_response.goals_canceling) > 0:
            logger.info("Goal cancellation accepted by the server.")
        else:
            logger.warning("Goal cancellation was not successful.")

    @called_positionally
    def result_callback(self, future: Future) -> None:
        """
        Processes the result from the action server.

        :param future: The future of the async goal task returned by the action client
            upon getting the goal result.
        """
        try:
            result_future = future.result()
            status = result_future.status
            result = result_future.result

            prp = PrettyResultPrinter()
            pretty_result_str = prp.pretty_print_result(result=result)

            self.goal_status = status
            self.goal_result = result

            if status == GoalStatus.STATUS_CANCELED:
                logger.info("Goal was successfully canceled by the server.")
            elif status == GoalStatus.STATUS_SUCCEEDED:
                logger.info("Goal succeeded.")
            else:
                logger.info(f"Goal finished with status code: {status}")

            logger.info(f"Formatted Result: \n{pretty_result_str}")
        except Exception as e:
            logger.error(f"Error receiving result: {e}")
        finally:
            logger.info("Shutting down after receiving the final result.")
            self.done = True

    def close(self) -> None:
        """
        Release owned ROS resources without destroying the associated node.
        """
        if self._goal_handle is not None and not self.done:
            self.cancel_goal()
        self._action_client.destroy()


def main_cli(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RoboKudo Action Client")
    parser.add_argument(
        "--preempt_timer",
        type=float,
        default=None,
        help="Time in seconds before preempting the goal",
    )
    cli_args = parser.parse_args()
    rclpy.init(args=args)
    node_registry = ROSNodeRegistry()
    node = Node("robokudo_query_test_client")
    node_registry.register(node)
    action_client = RoboKudoActionClient(
        node=node, preempt_timer=cli_args.preempt_timer
    )

    try:
        # Accept user input for the goal dynamically
        action_client.send_goal(goal_type="test")

        # Keep the node alive until the action is done
        while rclpy.ok() and not action_client.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        action_client.close()
        node_registry.clear(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(
    timeout_seconds: float = 20.0,
    result: ResultQueue | None = None,
    readiness: ReadinessSignal | None = None,
) -> None:
    """
    Send one query and report its result.

    :param timeout_seconds: Maximum query duration.
    :param result: Destination for the query result.
    :param readiness: Notification that the client can begin server discovery.
    """
    timeout_deadline = time.monotonic() + timeout_seconds
    result_queue = result if result is not None else Queue()
    ros_context = Context()
    rclpy.init(context=ros_context)
    node_registry = ROSNodeRegistry()
    node = Node("robokudo_query_test_client", context=ros_context)
    node_registry.register(node)
    action_client = RoboKudoActionClient(node=node, preempt_timer=None)
    executor = SingleThreadedExecutor(context=ros_context)
    executor.add_node(node)
    if readiness is not None:
        readiness.set()
    result_dict: dict[str, Any] = dict()
    result_dict["timed_out"] = False

    try:
        # Accept user input for the goal dynamically
        action_client.send_goal(goal_type="test")

        # Keep the node alive until the action is done
        while rclpy.ok(context=ros_context) and not action_client.done:
            if not time.monotonic() < timeout_deadline:
                result_dict["timed_out"] = True
                break

            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        result_dict["last_feedback"] = action_client.last_feedback
        result_dict["goal_status"] = action_client.goal_status
        result_dict["goal_result"] = action_client.goal_result
        result_dict["cancel_response"] = action_client.cancel_response
        result_queue.put(result_dict)
        action_client.close()
        executor.remove_node(node)
        executor.shutdown()
        node_registry.clear(node)
        node.destroy_node()
        if rclpy.ok(context=ros_context):
            rclpy.shutdown(context=ros_context)


if __name__ == "__main__":
    main_cli()
