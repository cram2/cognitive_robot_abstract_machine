import asyncio
from dataclasses import dataclass, field
from threading import Thread
from time import sleep
from typing import Any, List

import pytest

from giskardpy.data_types.exceptions import (
    MissingActionResultError,
    MissingGoalOutcomeError,
)
from giskardpy.middleware.ros2.action_server import ActionServerHandler, GoalOutcome

# %% mimics


@dataclass
class HandlerWithoutRosAdvertisement(ActionServerHandler):
    """
    Exercises the handler's goal bookkeeping without advertising a ROS action.
    """

    messages: List[str] = field(default_factory=list)
    """
    Everything the handler logged, instead of writing it to a ROS logger.
    """

    def __post_init__(self):
        pass

    def loginfo(self, message: str, goal_id: int | None = None) -> None:
        self.messages.append(message)


@dataclass
class GoalStateRecorder:
    """
    Stands in for a goal handle and records which state transition was requested.
    """

    transitions: List[str] = field(default_factory=list)
    """
    The name of every transition that was requested, in order.
    """

    request: Any = None
    """
    The goal message this handle carries.
    """

    is_cancel_requested: bool = False
    """
    Whether the client asked rclpy to cancel this goal.
    """

    def succeed(self) -> None:
        self.transitions.append("succeed")

    def abort(self) -> None:
        self.transitions.append("abort")

    def canceled(self) -> None:
        self.transitions.append("canceled")


# %% reporting an outcome


def test_succeeded_is_reported_as_success():
    goal_handle = GoalStateRecorder()

    GoalOutcome.SUCCEEDED.report_to(goal_handle)

    assert goal_handle.transitions == ["succeed"]


def test_aborted_is_reported_as_abort():
    goal_handle = GoalStateRecorder()

    GoalOutcome.ABORTED.report_to(goal_handle)

    assert goal_handle.transitions == ["abort"]


def test_a_goal_its_client_asked_to_cancel_is_reported_as_canceled():
    goal_handle = GoalStateRecorder(is_cancel_requested=True)

    GoalOutcome.CANCELED.report_to(goal_handle)

    assert goal_handle.transitions == ["canceled"]


def test_a_goal_superseded_by_a_newer_one_is_reported_as_aborted():
    goal_handle = GoalStateRecorder(is_cancel_requested=False)

    GoalOutcome.CANCELED.report_to(goal_handle)

    assert goal_handle.transitions == ["abort"]


def test_every_outcome_reports_exactly_one_transition():
    for outcome in GoalOutcome:
        goal_handle = GoalStateRecorder()

        outcome.report_to(goal_handle)

        assert len(goal_handle.transitions) == 1


# %% errors identify the goal they are about


def test_answering_a_goal_without_an_outcome_reports_which_goal_it_was():
    handler = HandlerWithoutRosAdvertisement(
        action_name="giskard/command", action_type=None
    )
    handler.goal_id = 3
    handler.result_message = "result"

    with pytest.raises(MissingGoalOutcomeError) as error:
        handler.send_result()

    assert error.value.action_server_name == handler.action_name
    assert error.value.goal_id == handler.goal_id


def test_reading_an_unset_result_reports_which_goal_it_was():
    handler = HandlerWithoutRosAdvertisement(
        action_name="giskard/command", action_type=None
    )
    handler.goal_id = 7

    with pytest.raises(MissingActionResultError) as error:
        handler.result_message

    assert error.value.action_server_name == handler.action_name
    assert error.value.goal_id == handler.goal_id


def test_a_set_result_is_returned_unchanged():
    handler = HandlerWithoutRosAdvertisement(
        action_name="giskard/command", action_type=None
    )

    handler.result_message = "result"

    assert handler.result_message == "result"


def test_missing_outcome_message_names_the_action_server_and_the_goal():
    error = MissingGoalOutcomeError(action_server_name="giskard/command", goal_id=3)

    assert "'giskard/command'" in str(error)
    assert "#3" in str(error)


def test_missing_result_message_names_the_action_server_and_the_goal():
    error = MissingActionResultError(action_server_name="giskard/command", goal_id=3)

    assert "'giskard/command'" in str(error)
    assert "#3" in str(error)


# %% handing a goal over between the rclpy thread and the motion server thread


def answer_the_next_goal(
    handler: ActionServerHandler, outcome: GoalOutcome, result: Any
) -> None:
    """
    Do what the motion server does for one goal: accept it and answer it.
    """
    while not handler.has_goal():
        sleep(0.001)
    handler.accept_goal()
    handler.outcome = outcome
    handler.result_message = result
    handler.send_result()


def test_answering_a_goal_resets_the_handler_before_the_answer_is_released():
    handler = HandlerWithoutRosAdvertisement(
        action_name="giskard/command", action_type=None
    )
    goal_handle = GoalStateRecorder(request="goal")
    handler.goal_queue.put(goal_handle)
    handler.accept_goal()
    handler.cancel_requested = True
    handler.set_canceled()
    handler.result_message = "result"

    handler.send_result()

    assert handler.goal_handle is None
    assert handler.goal_msg is None
    assert handler.cancel_requested is False
    assert handler.outcome is None
    with pytest.raises(MissingActionResultError):
        handler.result_message


def test_the_callback_returns_the_result_and_reports_the_outcome_of_its_goal():
    handler = HandlerWithoutRosAdvertisement(
        action_name="giskard/command", action_type=None
    )
    goal_handle = GoalStateRecorder(request="goal")
    returned: List[Any] = []
    callback = Thread(
        target=lambda: returned.append(
            asyncio.run(handler.execute_callback(goal_handle))
        ),
        daemon=True,
    )
    callback.start()

    answer_the_next_goal(handler, GoalOutcome.SUCCEEDED, "result")
    callback.join(timeout=5)

    assert returned == ["result"]
    assert goal_handle.transitions == ["succeed"]


def test_a_goal_accepted_while_the_previous_one_is_being_answered_is_kept():
    """
    The rclpy thread answering the previous goal must not touch the goal the motion
    server has moved on to, however late it runs.
    """
    handler = HandlerWithoutRosAdvertisement(
        action_name="giskard/command", action_type=None
    )
    previous = GoalStateRecorder(request="previous")
    callback = Thread(
        target=lambda: asyncio.run(handler.execute_callback(previous)), daemon=True
    )
    callback.start()
    answer_the_next_goal(handler, GoalOutcome.CANCELED, "result")
    following = GoalStateRecorder(request="following")
    handler.goal_queue.put(following)
    handler.accept_goal()

    callback.join(timeout=5)

    assert handler.goal_handle is following
    assert handler.goal_msg == following.request
