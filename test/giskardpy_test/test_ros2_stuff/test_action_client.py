from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Thread
from time import sleep

import pytest
from json_msgs.action import JsonAction
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from typing_extensions import Any, Iterator

from semantic_digital_twin.adapters.ros import rospy
from giskardpy.middleware.ros2.action_server import ActionServerHandler, GoalOutcome
from giskardpy.middleware.ros2.action_client import GiskardActionClient
from giskardpy.middleware.ros2.exceptions import GoalRequestNotAnsweredError

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


@dataclass
class ServerNeverAnswering:
    """
    An action server on a node that nothing spins, so a client finds it but never gets
    an answer to a goal request, like the server of a node whose executor stopped.
    """

    action_name: str
    """
    The action the server offers.
    """

    node: Node = field(init=False)
    """
    The node the server lives on.
    """

    action_server: ActionServer = field(init=False)
    """
    The server that never answers.
    """

    def __post_init__(self) -> None:
        self.node = Node("server_never_answering")
        self.action_server = ActionServer(
            node=self.node,
            action_type=JsonAction,
            action_name=self.action_name,
            execute_callback=lambda goal_handle: JsonAction.Result(),
        )

    def close(self) -> None:
        """
        Withdraw the server and its node from the ROS graph.
        """
        self.action_server.destroy()
        self.node.destroy_node()


@dataclass
class ServerAcceptingLate:
    """
    An action server that accepts every goal only after :attr:`acceptance_delay`, and
    then runs it until its client cancels it or :attr:`patience` ran out.
    """

    action_name: str
    """
    The action the server offers.
    """

    acceptance_delay: timedelta
    """
    How long the server takes to accept a goal.
    """

    patience: timedelta = timedelta(seconds=5)
    """
    How long an accepted goal waits to be canceled before it succeeds.
    """

    canceled_goals: list[str] = field(init=False, default_factory=list)
    """
    The goals whose client canceled them, in the order they ended.
    """

    is_closing: bool = field(init=False, default=False)
    """
    Set by :meth:`close` so that every running goal ends at once.
    """

    node: Node = field(init=False)
    """
    The node the server lives on.
    """

    action_server: ActionServer = field(init=False)
    """
    The server that accepts late.
    """

    executor: MultiThreadedExecutor = field(init=False)
    """
    Spins :attr:`node`, so that goal requests, cancel requests and running goals are
    served at the same time.
    """

    spin_thread: Thread = field(init=False)
    """
    The thread that runs :attr:`executor`.
    """

    def __post_init__(self) -> None:
        self.node = Node("server_accepting_late")
        self.action_server = ActionServer(
            node=self.node,
            action_type=JsonAction,
            action_name=self.action_name,
            execute_callback=self._run_goal,
            goal_callback=self._accept_late,
            cancel_callback=lambda goal_handle: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def _accept_late(self, goal_request: JsonAction.Goal) -> GoalResponse:
        """
        Accept ``goal_request`` once :attr:`acceptance_delay` has passed.
        """
        sleep(self.acceptance_delay.total_seconds())
        return GoalResponse.ACCEPT

    def _run_goal(self, goal_handle: ServerGoalHandle) -> JsonAction.Result:
        """
        Wait for the goal to be canceled, and record it in :attr:`canceled_goals` if it
        is.
        """
        deadline = datetime.now() + self.patience
        while (
            not goal_handle.is_cancel_requested
            and not self.is_closing
            and datetime.now() < deadline
        ):
            sleep(0.01)
        if goal_handle.is_cancel_requested:
            self.canceled_goals.append(goal_handle.request.goal)
            goal_handle.canceled()
        else:
            goal_handle.succeed()
        return JsonAction.Result(result="{}")

    def wait_until_a_goal_was_canceled(self) -> None:
        """
        Return once a goal was canceled or :attr:`patience` has passed.
        """
        deadline = datetime.now() + self.patience
        while not self.canceled_goals and datetime.now() < deadline:
            sleep(0.01)

    def close(self) -> None:
        """
        End every running goal and withdraw the server and its node from the ROS graph.
        """
        self.is_closing = True
        self.executor.shutdown()
        self.spin_thread.join()
        self.action_server.destroy()
        self.node.destroy_node()


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
def server_never_answering(init_rospy) -> Iterator[ServerNeverAnswering]:
    """
    A server that is found but never answers, withdrawn again after the test.
    """
    server = ServerNeverAnswering(action_name="never_answered/command")
    yield server
    server.close()


@pytest.fixture()
def client_of_server_never_answering(
    server_never_answering: ServerNeverAnswering,
) -> Iterator[GiskardActionClient]:
    """
    A client that gives up on a goal request after half a second.
    """
    client = GiskardActionClient(
        rospy.get_node(),
        JsonAction,
        server_never_answering.action_name,
        goal_response_timeout=timedelta(seconds=0.5),
    )
    yield client
    client.close()


@pytest.fixture()
def server_accepting_late(init_rospy) -> Iterator[ServerAcceptingLate]:
    """
    A server that takes a second to accept a goal, withdrawn again after the test.
    """
    server = ServerAcceptingLate(
        action_name="accepted_late/command", acceptance_delay=timedelta(seconds=1)
    )
    yield server
    server.close()


@pytest.fixture()
def client_of_server_accepting_late(
    server_accepting_late: ServerAcceptingLate,
) -> Iterator[GiskardActionClient]:
    """
    A client that gives up on a goal request well before the server accepts it.
    """
    client = GiskardActionClient(
        rospy.get_node(),
        JsonAction,
        server_accepting_late.action_name,
        goal_response_timeout=server_accepting_late.acceptance_delay / 4,
    )
    yield client
    client.close()


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
    client = GiskardActionClient(rospy.get_node(), JsonAction, "giskard/command")

    with pytest.raises(FeedbackDeliberatelyRejected):
        client.send_goal(JsonAction.Goal(), feedback_callback=reject_feedback)


def test_a_goal_whose_feedback_callback_failed_is_canceled(
    server_sending_feedback: ServerAnsweringOneGoal,
) -> None:
    client = GiskardActionClient(rospy.get_node(), JsonAction, "giskard/command")

    with pytest.raises(FeedbackDeliberatelyRejected):
        client.send_goal(JsonAction.Goal(), feedback_callback=reject_feedback)

    server_sending_feedback.thread.join()
    assert server_sending_feedback.cancel_was_requested


# %% a goal request nobody answers


def test_a_goal_request_nobody_answers_raises_once_the_response_timeout_passed(
    client_of_server_never_answering: GiskardActionClient,
) -> None:
    with pytest.raises(GoalRequestNotAnsweredError):
        client_of_server_never_answering.send_goal(JsonAction.Goal())


# %% a goal accepted after its client gave up on it


def test_a_goal_accepted_after_its_response_timeout_is_canceled(
    server_accepting_late: ServerAcceptingLate,
    client_of_server_accepting_late: GiskardActionClient,
) -> None:
    with pytest.raises(GoalRequestNotAnsweredError):
        client_of_server_accepting_late.send_goal(JsonAction.Goal(goal="abandoned"))

    server_accepting_late.wait_until_a_goal_was_canceled()

    assert server_accepting_late.canceled_goals == ["abandoned"]


def test_a_late_acceptance_does_not_take_the_place_of_the_goal_sent_after_it(
    server_accepting_late: ServerAcceptingLate,
    client_of_server_accepting_late: GiskardActionClient,
) -> None:
    """
    The goal sent next is still awaiting its own answer when the abandoned goal is
    accepted, so only the abandoned goal may be canceled.
    """
    with pytest.raises(GoalRequestNotAnsweredError):
        client_of_server_accepting_late.send_goal(JsonAction.Goal(goal="abandoned"))
    client_of_server_accepting_late.send_goal_async(JsonAction.Goal(goal="next"))

    server_accepting_late.wait_until_a_goal_was_canceled()

    assert server_accepting_late.canceled_goals == ["abandoned"]
