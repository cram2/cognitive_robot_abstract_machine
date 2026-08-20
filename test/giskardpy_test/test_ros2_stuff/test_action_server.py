from dataclasses import dataclass, field
from typing import List

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

    def __post_init__(self):
        pass


@dataclass
class GoalStateRecorder:
    """
    Stands in for a goal handle and records which state transition was requested.
    """

    transitions: List[str] = field(default_factory=list)
    """
    The name of every transition that was requested, in order.
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


def test_canceled_is_reported_as_cancellation():
    goal_handle = GoalStateRecorder()

    GoalOutcome.CANCELED.report_to(goal_handle)

    assert goal_handle.transitions == ["canceled"]


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

    with pytest.raises(MissingGoalOutcomeError) as error:
        handler.report_outcome(GoalStateRecorder(), None)

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
