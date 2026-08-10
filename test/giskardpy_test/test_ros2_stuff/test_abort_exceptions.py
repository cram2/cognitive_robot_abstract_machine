import json
from dataclasses import dataclass
from typing import Any

from giskardpy.middleware.ros2.exceptions import (
    ExecutionAbortedException,
    ExecutionCanceledException,
    WorldModelModifiedDuringMotionError,
)
from giskardpy.middleware.ros2.ros2_interface import MyActionClient
from krrood.adapters.json_serializer import to_json

# %% mimics


@dataclass
class ResultMessageMimic:
    """
    Stands in for the result message of the action, whose payload is nested twice.
    """

    result: Any
    """
    The wrapped result carrying the json payload.
    """

    @classmethod
    def with_payload(cls, payload: dict) -> "ResultMessageMimic":
        """
        Build a result message around the given payload.
        """
        return cls(result=PayloadMimic(result=json.dumps(payload)))


@dataclass
class PayloadMimic:
    """
    Stands in for the result of the action, which carries the states as json.
    """

    result: str
    """
    The states of the finished goal as json.
    """


# %% rebuilding the reported error


def test_a_reported_error_is_rebuilt():
    result = ResultMessageMimic.with_payload(
        {"error": to_json(WorldModelModifiedDuringMotionError())}
    )

    error = MyActionClient.create_abort_exception(result)

    assert isinstance(error, WorldModelModifiedDuringMotionError)


def test_a_rebuilt_error_keeps_its_fields():
    result = ResultMessageMimic.with_payload(
        {
            "error": to_json(
                ExecutionCanceledException(action_server_name="a", goal_id=2)
            )
        }
    )

    error = MyActionClient.create_abort_exception(result)

    assert error.action_server_name == "a"
    assert error.goal_id == 2


def test_a_failure_without_a_reported_error_is_a_plain_abort():
    result = ResultMessageMimic.with_payload({})

    error = MyActionClient.create_abort_exception(result)

    assert isinstance(error, ExecutionAbortedException)


def test_an_error_of_an_unknown_class_is_a_plain_abort():
    result = ResultMessageMimic.with_payload(
        {"error": {"__json_type__": "a.module.the.client.does.not.have.Boom"}}
    )

    error = MyActionClient.create_abort_exception(result)

    assert isinstance(error, ExecutionAbortedException)


def test_an_error_that_cannot_be_constructed_is_a_plain_abort():
    result = ResultMessageMimic.with_payload(
        {"error": to_json(ErrorNeedingTwoArguments("first", "second"))}
    )

    error = MyActionClient.create_abort_exception(result)

    assert isinstance(error, ExecutionAbortedException)


class ErrorNeedingTwoArguments(Exception):
    """
    An exception that the generic exception deserializer cannot rebuild, because it does
    not take a single message.
    """

    def __init__(self, first: str, second: str):
        super().__init__(first, second)
