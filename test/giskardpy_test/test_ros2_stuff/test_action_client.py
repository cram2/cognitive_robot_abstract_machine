from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Thread
from time import sleep

import pytest
from json_msgs.action import JsonAction
from rclpy.action import ActionServer
from rclpy.node import Node
from typing_extensions import Any, Iterator

from semantic_digital_twin.adapters.ros import rospy
from giskardpy.middleware.ros2.action_server import ActionServerHandler, GoalOutcome
from giskardpy.middleware.ros2.exceptions import GoalRequestNotAnsweredError
from giskardpy.middleware.ros2.ros2_interface import MyActionClient

# %% mimics


class FeedbackDeliberatelyRejected(Exception):
    """
    Raised by a feedback callback that fails on purpose.
    """


@dataclass
class ServerAnsweringOneGoal:
    """
    Stands in for the motion server: accepts one goal, optionally sends one feedback,
    and answers the goal once its client canceled it or the server ran out of patience.
    """

    action_server: ActionServerHandler
    """
    The action server the goal arrives at.
    """

    outcome_unless_canceled: GoalOutcome
    """
    How the goal ends if its client does not cancel it.
    """

    feedback_json: str | None = None
    """
    The feedback sent once the goal is accepted, None to send none.
    """

    patience: timedelta = timedelta(seconds=5)
    """
    How long the server waits for the goal to arrive, and then for its cancellation.
    """

    cancel_was_requested: bool = field(init=False, default=False)
    """
    Whether the client asked to cancel the goal before it was answered.
    """

    thread: Thread = field(init=False)
    """
    The thread serving the goal.
    """

    def __post_init__(self) -> None:
        self.thread = Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        """
        Accept the goal, send the feedback and answer.
        """
        self._wait_until(self.action_server.has_goal)
        self.action_server.accept_goal()
        if self.feedback_json is not None:
            self.action_server.send_feedback(
                JsonAction.Feedback(feedback=self.feedback_json)
            )
            self._wait_until(self.action_server.is_cancel_requested)
        self.cancel_was_requested = self.action_server.is_cancel_requested()
        self.action_server.outcome = (
            GoalOutcome.CANCELED
            if self.cancel_was_requested
            else self.outcome_unless_canceled
        )
        self.action_server.result_message = JsonAction.Result(result="{}")
        self.action_server.send_result()

    def _wait_until(self, condition: Any) -> None:
        """
        :param condition: Called until it returns True or :attr:`patience` ran out.
        """
        deadline = datetime.now() + self.patience
        while not condition() and datetime.now() < deadline:
            sleep(0.01)


def reject_feedback(feedback_message: Any) -> None:
    """
    :param feedback_message: The feedback that is rejected.
    :raises FeedbackDeliberatelyRejected: Always.
    """
    raise FeedbackDeliberatelyRejected()


# %% fixtures


@pytest.fixture()
def giskard_command_server(init_rospy) -> Iterator[ActionServerHandler]:
    """
    A real "giskard/command" action server that a client can send goals to.
    """
    action_server = ActionServerHandler(
        action_name="giskard/command", action_type=JsonAction
    )
    yield action_server
    action_server.close()


@pytest.fixture()
def server_never_answering(init_rospy) -> Iterator[ActionServer]:
    """
    An action server on a node that nothing spins, so it is found by a client but never
    answers a goal request, like the server of a node whose executor stopped.
    """
    node = Node("server_never_answering")
    action_server = ActionServer(
        node=node,
        action_type=JsonAction,
        action_name="never_answered/command",
        execute_callback=lambda goal_handle: JsonAction.Result(),
    )
    yield action_server
    action_server.destroy()
    node.destroy_node()


@pytest.fixture()
def server_sending_feedback(giskard_command_server) -> ServerAnsweringOneGoal:
    """
    :return: A server that sends one feedback for the goal it receives.
    """
    return ServerAnsweringOneGoal(
        action_server=giskard_command_server,
        outcome_unless_canceled=GoalOutcome.SUCCEEDED,
        feedback_json="{}",
    )


# %% a feedback callback that fails


def test_an_error_of_the_feedback_callback_is_raised_to_the_sender_of_the_goal(
    server_sending_feedback: ServerAnsweringOneGoal,
) -> None:
    client = MyActionClient(rospy.get_node(), JsonAction, "giskard/command")

    with pytest.raises(FeedbackDeliberatelyRejected):
        client.send_goal(JsonAction.Goal(), feedback_callback=reject_feedback)


def test_a_goal_whose_feedback_callback_failed_is_canceled(
    server_sending_feedback: ServerAnsweringOneGoal,
) -> None:
    client = MyActionClient(rospy.get_node(), JsonAction, "giskard/command")

    with pytest.raises(FeedbackDeliberatelyRejected):
        client.send_goal(JsonAction.Goal(), feedback_callback=reject_feedback)

    server_sending_feedback.thread.join()
    assert server_sending_feedback.cancel_was_requested


# %% a goal request nobody answers


def test_a_goal_request_nobody_answers_raises_once_the_response_timeout_passed(
    server_never_answering: ActionServer,
) -> None:
    client = MyActionClient(
        rospy.get_node(),
        JsonAction,
        "never_answered/command",
        goal_response_timeout=timedelta(seconds=0.5),
    )

    with pytest.raises(GoalRequestNotAnsweredError):
        client.send_goal(JsonAction.Goal())
