from __future__ import annotations

import argparse
import json
import logging
import time
from queue import Queue

import rclpy
from action_msgs.msg import GoalStatus
from rclpy import Context
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from typing_extensions import TYPE_CHECKING, Optional, List, Dict, Any

from robokudo_msgs.action import Query

from robokudo.defs import PACKAGE_NAME

if TYPE_CHECKING:
    from rclpy.action.client import ClientGoalHandle
    from rclpy.task import Future
    from rclpy.timer import Timer
    from robokudo_msgs.action._query import Query_FeedbackMessage, Query_Result


class PrettyResultPrinter:
    def pretty_print_result(self, result: Query_Result) -> str:
        result_dict = message_to_ordereddict(result)
        return json.dumps(result_dict, indent=2)


class RoboKudoActionClient(Node):
    """
    Debug Action Client for the Query Action Server.
    Allows sending dynamic goals and handles feedback, result, and cancellation.
    """

    def __init__(self, preempt_timer: Optional[float] = None) -> None:
        """Create a new RoboKudo action client.

        :param preempt_timer: Optional preempt timer to cancel the goal automatically after x seconds.
        """
        super().__init__("robokudo_query_test_client")
        self.rk_logger = logging.getLogger(PACKAGE_NAME)
        self._action_client: ActionClient = ActionClient(
            self, Query, "/robokudo/query"
        )  # Connect to the action server

        self._preempt_timer: float = preempt_timer
        self._cancel_timer: Optional[Timer] = None

        self._goal_handle: Optional[ClientGoalHandle] = None
        self.done: bool = False  # Flag to indicate completion

        # Results
        self.last_feedback: Optional[Query_FeedbackMessage] = None
        self.goal_status: Optional[GoalStatus] = None
        self.goal_result: Optional[Query_Result] = None
        self.cancel_response: Optional[Future] = None

    def send_goal(self, goal_type: str) -> None:
        """Waits for the action server and sends a dynamic goal.

        :param goal_type: Content of the type field in the goal sent to the action server.
        """
        self.rk_logger.info("Waiting for action server...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.rk_logger.error("Action server not available!")
            self.done = True
            return

        # Create and send a goal
        goal_msg = Query.Goal()
        goal_msg.obj.type = goal_type

        self.rk_logger.info(f"Sending goal request with type: '{goal_type}'")
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future) -> None:
        """Handles the response from the action server regarding goal acceptance.

        :param future: The future of the async goal task returned by the action client upon sending a goal.
        """
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.rk_logger.error("Goal rejected by the action server.")
            self.done = True
            return

        self.rk_logger.info("Goal accepted by the action server.")

        # Scheduling of cancellation after X seconds
        if self._preempt_timer is not None:
            self._cancel_timer = self.create_timer(
                self._preempt_timer, self.cancel_goal
            )

        # Wait for the result
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg: Query_FeedbackMessage) -> None:
        """Processes feedback messages from the action server.

        :param feedback_msg: The feedback message returned by the action client.
        """
        feedback = feedback_msg.feedback
        self.last_feedback = feedback
        self.rk_logger.info(f"Received feedback: {feedback.feedback}")

    def cancel_goal(self) -> None:
        """Sends a cancel request for the active goal."""
        if not self._goal_handle:
            self.rk_logger.error("No active goal to cancel.")
            return

        self.rk_logger.info("Sending cancel request...")
        cancel_future = self._goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

        # Stop the cancel timer
        if self._cancel_timer:
            self._cancel_timer.cancel()

    def cancel_done_callback(self, future: Future) -> None:
        """Handles the response from the action server regarding goal cancellation.

        :param future: The future of the async goal task returned by the action client upon cancelling a goal.
        """
        cancel_response = future.result()
        self.cancel_response = cancel_response
        if len(cancel_response.goals_canceling) > 0:
            self.rk_logger.info("Goal cancellation accepted by the server.")
        else:
            self.rk_logger.warning("Goal cancellation was not successful.")
        # self.done = True
        # self.rk_logger.info("Shutting down after cancellation is accepted.")
        # rclpy.shutdown()

    def result_callback(self, future: Future) -> None:
        """Processes the result from the action server.

        :param future: The future of the async goal task returned by the action client upon getting the goal result.
        """
        try:
            result_future = future.result()
            status = result_future.status
            result = result_future.result

            prp = PrettyResultPrinter()
            pretty_result_str = prp.pretty_print_result(result)

            self.goal_status = status
            self.goal_result = result

            if status == GoalStatus.STATUS_CANCELED:
                self.rk_logger.info("Goal was successfully canceled by the server.")
            elif status == GoalStatus.STATUS_SUCCEEDED:
                self.rk_logger.info("Goal succeeded.")
            else:
                self.rk_logger.info(f"Goal finished with status code: {status}")

            self.rk_logger.info(f"Formatted Result: \n{pretty_result_str}")
        except Exception as e:
            self.rk_logger.error(f"Error receiving result: {e}")
        finally:
            self.rk_logger.info("Shutting down after receiving the final result.")
            self.done = True

    # def result_callback(self, future):
    #     """Processes the result from the action server."""
    #     try:
    #         result = future.result().result
    #         # self.rk_logger.info(f"Result received: {result}")
    #         prp = PrettyResultPrinter()
    #         pretty_result_str = prp.pretty_print_result(result)
    #         self.rk_logger.info(f'Formatted Result: \n{pretty_result_str}')
    #     except Exception as e:
    #         self.rk_logger.error(f"Error receiving result: {e}")
    #     finally:
    #         self.rk_logger.info("Shutting down after receiving the result.")
    #         self.done = True

    def destroy_node(self) -> None:
        """Cancel the goal and destroys the ROS node."""
        self.cancel_goal()
        super().destroy_node()


def main_cli(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="RoboKudo Action Client")
    parser.add_argument(
        "--preempt_timer",
        type=float,
        default=None,
        help="Time in seconds before preempting the goal",
    )
    cli_args = parser.parse_args()
    rclpy.init(args=args)
    action_client = RoboKudoActionClient(preempt_timer=cli_args.preempt_timer)

    try:
        # Accept user input for the goal dynamically
        action_client.send_goal(goal_type="test")

        # Keep the node alive until the action is done
        while rclpy.ok() and not action_client.done:
            rclpy.spin_once(action_client, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        action_client.destroy_node()


def main(timeout_seconds: float = 20.0, result: Queue = Queue()) -> None:
    timeout_deadline = time.monotonic() + timeout_seconds
    ctx = Context()
    rclpy.init(context=ctx)
    action_client = RoboKudoActionClient(preempt_timer=None)
    result_dict: Dict[str, Any] = dict()
    result_dict["timed_out"] = False

    try:
        # Accept user input for the goal dynamically
        action_client.send_goal(goal_type="test")

        # Keep the node alive until the action is done
        while rclpy.ok() and not action_client.done:
            if not time.monotonic() < timeout_deadline:
                result_dict["timed_out"] = True
                break

            rclpy.spin_once(action_client, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # if rclpy.ok():
        #     rclpy.shutdown()

        result_dict["last_feedback"] = action_client.last_feedback
        result_dict["goal_status"] = action_client.goal_status
        result_dict["goal_result"] = action_client.goal_result
        result_dict["cancel_response"] = action_client.cancel_response
        result.put(result_dict)
        if rclpy.ok(context=ctx):
            rclpy.shutdown(context=ctx)
        action_client.destroy_node()


if __name__ == "__main__":
    main_cli()
