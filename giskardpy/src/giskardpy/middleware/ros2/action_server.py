from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue, Empty
from time import sleep
from typing import Any

from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle

from giskardpy.data_types.exceptions import (
    MissingActionResultError,
    MissingGoalOutcomeError,
)
from giskardpy.middleware.ros2 import rospy


class GoalOutcome(Enum):
    """
    How a goal ended, in the terms rclpy expects.
    """

    SUCCEEDED = auto()
    """
    The motion reached its end.
    """

    ABORTED = auto()
    """
    The motion failed.
    """

    CANCELED = auto()
    """
    The motion was stopped by the client or superseded by a new goal.
    """

    def report_to(self, goal_handle: ServerGoalHandle) -> None:
        """
        Transition the goal handle into the state matching this outcome.
        """
        match self:
            case GoalOutcome.SUCCEEDED:
                goal_handle.succeed()
            case GoalOutcome.ABORTED:
                goal_handle.abort()
            case GoalOutcome.CANCELED:
                goal_handle.canceled()


@dataclass
class ActionServerHandler:
    """
    Hands goals from rclpy's executor threads over to the thread that runs the motion
    server.

    ``execute_cb`` runs on an rclpy executor thread and blocks on a queue, while the
    motion server polls :meth:`has_goal` and answers with :meth:`send_result`. Goal
    execution therefore stays on the motion server's own thread.
    """

    action_name: str
    """
    Name under which the action is advertised.
    """

    action_type: Any
    """
    The ROS action type this server offers.
    """

    goal_id: int = field(init=False, default=-1)
    """
    Number of goals accepted so far, used to identify goals in logs and feedback.
    """

    goal_msg: Any | None = field(init=False, default=None)
    """
    Request of the currently accepted goal.
    """

    goal_handle: ServerGoalHandle | None = field(init=False, default=None)
    """
    Handle of the currently accepted goal.
    """

    cancel_requested: bool = field(init=False, default=False)
    """
    Set when a new goal arrives while another one is still running.
    """

    outcome: GoalOutcome | None = field(init=False, default=None)
    """
    How the current goal ended, reported to rclpy once the result was handed back.
    """

    goal_queue: Queue = field(init=False, default_factory=lambda: Queue(1))
    """
    Handover of incoming goals to the motion server thread.
    """

    result_queue: Queue = field(init=False, default_factory=lambda: Queue(1))
    """
    Handover of results back to the rclpy executor thread.
    """

    _result_msg: Any | None = field(init=False, default=None)
    """
    Result of the currently accepted goal.
    """

    _action_server: ActionServer = field(init=False)
    """
    The rclpy action server this handler wraps.
    """

    def __post_init__(self):
        self._action_server = ActionServer(
            node=rospy.node,
            action_type=self.action_type,
            action_name=self.action_name,
            execute_callback=self.execute_cb,
            goal_callback=self.default_goal_callback,
            cancel_callback=self.cancel_callback,
        )

    def loginfo(self, message: str) -> None:
        """
        Log a message tagged with the action name and the current goal id.
        """
        rospy.node.get_logger().info(
            f"{self.action_name}(Goal #{self.goal_id}): {message}"
        )

    def default_goal_callback(self, goal_request: Any) -> GoalResponse:
        """
        Accept every goal and cancel a running one to make room for the new goal.
        """
        if self.goal_handle is not None:
            self.loginfo(
                f"New Goal requested while Goal #{self.goal_id} is being processed. "
                f"Cancelling old Goal."
            )
            self.cancel_requested = True
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        """
        Accept every cancel request.
        """
        self.loginfo("Cancel request received.")
        return CancelResponse.ACCEPT

    async def execute_cb(self, goal_handle: ServerGoalHandle) -> Any:
        """
        Queue the goal for the motion server thread and wait for its result.
        """
        while self.goal_handle is not None:
            sleep(0.1)
        self.goal_queue.put(goal_handle)
        result_msg = self.result_queue.get()
        self.loginfo("Sending response.")
        outcome = self.outcome
        self.goal_msg = None
        self.goal_handle = None
        self.result_msg = None
        self.cancel_requested = False
        self.outcome = None
        self.report_outcome(goal_handle, outcome)
        return result_msg

    def report_outcome(
        self, goal_handle: ServerGoalHandle, outcome: GoalOutcome | None
    ) -> None:
        """
        Tell rclpy how the goal ended.

        The handler releases the goal before this runs, so that the next goal waiting in
        :meth:`execute_cb` is not blocked while rclpy transitions this one.

        :raises MissingGoalOutcomeError: If the goal is answered without an outcome.
        """
        if outcome is None:
            raise MissingGoalOutcomeError(
                action_server_name=self.action_name, goal_id=self.goal_id
            )
        outcome.report_to(goal_handle)

    def accept_goal(self) -> None:
        """
        Take the next queued goal and make it the current one.
        """
        try:
            self.goal_handle = self.goal_queue.get_nowait()
        except Empty:
            return
        self.goal_msg = self.goal_handle.request
        self.goal_id += 1
        self.loginfo("Accepted")

    @property
    def result_msg(self) -> Any:
        """
        The result built for the current goal.

        :raises MissingActionResultError: If no result was set for the current goal.
        """
        if self._result_msg is None:
            raise MissingActionResultError(
                action_server_name=self.action_name, goal_id=self.goal_id
            )
        return self._result_msg

    @result_msg.setter
    def result_msg(self, value: Any | None) -> None:
        self._result_msg = value

    def has_goal(self) -> bool:
        """
        Whether a goal is waiting to be accepted.
        """
        return not self.goal_queue.empty()

    def send_feedback(self, message: Any) -> None:
        """
        Publish feedback for the current goal.
        """
        self.goal_handle.publish_feedback(message)

    def set_canceled(self) -> None:
        self.outcome = GoalOutcome.CANCELED

    def set_aborted(self) -> None:
        self.outcome = GoalOutcome.ABORTED

    def set_succeeded(self) -> None:
        self.outcome = GoalOutcome.SUCCEEDED

    def send_result(self) -> None:
        """
        Hand the result back to the waiting rclpy executor thread.
        """
        self.result_queue.put(self.result_msg)

    def is_cancel_requested(self) -> bool:
        """
        Whether the current goal was canceled by the client or superseded by a new goal.
        """
        return self.cancel_requested or self.goal_handle.is_cancel_requested
