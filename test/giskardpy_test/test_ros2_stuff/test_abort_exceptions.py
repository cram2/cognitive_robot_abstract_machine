from __future__ import annotations

import json
from dataclasses import dataclass
from typing_extensions import Any

from giskardpy.middleware.ros2.exceptions import (
    ExecutionAbortedException,
    ExecutionCanceledException,
    UnserializableGoalError,
    WorldModelModifiedDuringMotionError,
)
from giskardpy.middleware.ros2.ros2_interface import MyActionClient
from krrood.adapters.json_serializer import to_json
from krrood.exceptions import DataclassException
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

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
    def with_payload(cls, payload: dict) -> ResultMessageMimic:
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


@dataclass
class ErrorAboutABody(DataclassException):
    """
    An error that refers to a body of the world, as one about a part of the robot does.
    """

    body: Body
    """
    The body the error is about.
    """

    def error_message(self) -> str:
        return f"Something went wrong with {self.body.name}."

    def suggest_correction(self) -> str:
        return "Nothing to correct, this error only exists for a test."


# %% rebuilding the reported error


def test_a_reported_error_is_rebuilt():
    result = ResultMessageMimic.with_payload(
        {"error": to_json(WorldModelModifiedDuringMotionError())}
    )

    error = MyActionClient.create_abort_exception(result)

    assert isinstance(error, WorldModelModifiedDuringMotionError)


def test_an_error_about_a_body_is_rebuilt_with_the_body_of_the_client_world() -> None:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body := Body(name=PrefixedName("arm")))
    result = ResultMessageMimic.with_payload(
        {"error": to_json(ErrorAboutABody(body=body))}
    )

    error = MyActionClient.create_abort_exception(
        result, **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs()
    )

    assert error.body is body


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


def test_an_error_that_could_not_be_serialized_arrives_with_its_message():
    reported = UnserializableGoalError(error_class_name="a.module.Boom", message="boom")
    result = ResultMessageMimic.with_payload({"error": to_json(reported)})

    error = MyActionClient.create_abort_exception(result)

    assert isinstance(error, UnserializableGoalError)
    assert error.message == reported.message


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
