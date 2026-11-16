"""
An action client that waits for the result of a Giskard goal and rebuilds the error an
aborted goal carries.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial

from action_msgs.msg import GoalStatus
from rclpy import Future
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from typing_extensions import Callable, Any

from giskardpy.middleware.ros2.event_loop_manager import get_event_loop
from giskardpy.middleware.ros2.exceptions import (
    ActionClientClosedError,
    ExecutionAbortedException,
    ExecutionCanceledException,
    GoalRequestNotAnsweredError,
    UnexpectedGoalStatusError,
)
from giskardpy.middleware.ros2.utils.asynio_utils import wait_until_not_none
from krrood.adapters.exceptions import JSONSerializationError
from krrood.adapters.json_serializer import from_json


# %% action client


@dataclass
class GiskardActionClient:
    """
    Sends goals to a Giskard action server one at a time and waits for their results.

    Only the goal sent last is followed; results and acceptances of earlier goals are
    ignored. An aborted goal carries its error as JSON in the result, which
    :meth:`create_abort_exception` turns back into the exception.

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

    _current_goal_lock: threading.Lock = field(
        init=False, default_factory=threading.Lock
    )
    """
    Keeps a goal from being given up while its acceptance makes it the current one.
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
        :raises ActionClientClosedError: If the client was closed.
        """
        if self._client is None:
            raise ActionClientClosedError(action_name=self.action_name)
        self._goal_counter += 1
        self._current_goal_id = self._goal_counter
        self._feedback_error = None
        if feedback_callback is not None:
            feedback_callback = partial(self._receive_feedback, feedback_callback)
        future = self._client.send_goal_async(goal, feedback_callback=feedback_callback)
        future.add_done_callback(
            partial(self._receive_goal_response, goal_id=self._goal_counter)
        )
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
        :raises ActionClientClosedError: If the client was closed.
        :raises GoalRequestNotAnsweredError: If no server accepts or rejects the goal
            within :attr:`goal_response_timeout`. The goal is given up, so a server that
            accepts it later has it canceled.
        """
        goal_response = self.send_goal_async(goal, feedback_callback=feedback_callback)
        answered = threading.Event()
        goal_response.add_done_callback(lambda _: answered.set())
        answered.wait(self.goal_response_timeout.total_seconds())
        with self._current_goal_lock:
            is_given_up = not goal_response.done()
            if is_given_up:
                self._current_goal_id = None
        if is_given_up:
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

    def _receive_goal_response(self, future: Future, goal_id: int) -> None:
        """
        Start waiting for the result of the goal ``goal_id`` if its server accepted it.

        A goal that is accepted after it was given up or superseded by a newer goal is
        canceled, so that it does not run although nobody waits for it.

        :param future: Resolved with the handle of the goal the server answered.
        :param goal_id: Identifier of the goal the answer belongs to.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node_handle.get_logger().info(
                f"{self.action_name} Goal {goal_id} rejected"
            )
            return

        with self._current_goal_lock:
            is_current = goal_id == self._current_goal_id
            if is_current:
                self._goal_handle = goal_handle
        if not is_current:
            self.node_handle.get_logger().info(
                f"{self.action_name} Goal #{goal_id} accepted after it was given up, "
                f"canceling it"
            )
            goal_handle.cancel_goal_async()
            return

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
