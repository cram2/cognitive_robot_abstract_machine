"""
Sending statecharts through JSON in tests, and the cases that put every node class of a
package through it.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from abc import ABC
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType

from typing_extensions import Any, Callable, Self

import pytest

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from krrood.exceptions import DataclassException
from krrood.utils import recursive_subclasses
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.world import World

# %% a statechart and what a receiver reads from its JSON


@dataclass
class StatechartSentThroughJSON:
    """
    A statechart together with the one a receiver in the same world reads from its JSON.
    """

    sent: Statechart
    """
    The statechart the sender wrote.
    """

    sent_document: dict[str, Any]
    """
    The JSON document the receiver got, with its numbers rounded.
    """

    received: Statechart
    """
    The statechart the receiver read.
    """

    @classmethod
    def received_in(cls, sent: Statechart, context: StatechartContext) -> Self:
        """
        :param sent: The statechart to send.
        :param context: The context the receiver builds its statechart in, on the world
            `sent` refers to.
        :return: `sent` with what a receiver reads from the JSON text of it.
        """
        sent_text = json.dumps(sent.to_json())
        tracker = WorldEntityWithIDKwargsTracker.from_world(context.world)
        received = Statechart.from_json(
            json.loads(sent_text),
            context=context,
            world=context.world,
            **tracker.create_kwargs(),
        )
        return cls(
            sent=sent,
            sent_document=cls._read_with_rounded_numbers(sent_text),
            received=received,
        )

    def write_received_document(self) -> dict[str, Any]:
        """
        :return: The JSON document a receiver of :attr:`received` would get, with its
            numbers rounded like those of :attr:`sent_document`.
        """
        return self._read_with_rounded_numbers(json.dumps(self.received.to_json()))

    @staticmethod
    def _read_with_rounded_numbers(
        text: str, decimal_places: int = 12
    ) -> dict[str, Any]:
        """
        .. note:: A rotation is written as the quaternion of its matrix, which differs in
            the last digit every time it is written again.

        :param text: A JSON document.
        :param decimal_places: The decimal places kept of every number.
        :return: The document with its numbers rounded.
        """
        return json.loads(
            text, parse_float=lambda number: round(float(number), decimal_places)
        )


# %% one case per node class


class WorldFixture(StrEnum):
    """
    The fixtures giving the worlds the nodes of a case refer to.
    """

    MINI_WORLD = "mini_world"
    CYLINDER_BOT_WORLD = "cylinder_bot_world"
    CYLINDER_BOT_DIFF_WORLD = "cylinder_bot_diff_world"
    SELF_COLLISION_BOT_WORLD = "self_collision_bot_world"
    PRISMATIC_BOT2 = "prismatic_bot2"
    PR2_WORLD = "pr2_world_state_reset"
    PR2_APARTMENT_CONTEXT = "pr2_apartment_context"


@dataclass
class NodeTypeMissingFromCaseError(DataclassException):
    """
    Raised when the nodes of a case do not hold a node of the class the case is for.
    """

    case: NodeRoundTripCase
    """
    The case whose nodes lack its class.
    """

    def error_message(self) -> str:
        return (
            f"The nodes built for {self.case.node_type.__name__} hold no node of that "
            f"class."
        )

    def suggest_correction(self) -> str:
        return "Build a node of that class in the case, or one that expands to it."


@dataclass
class NodeRoundTripCase:
    """
    The nodes that send one node class through JSON.
    """

    node_type: type[StatechartNode]
    """
    The node class the case sends.
    """

    build_nodes: Callable[[World], list[StatechartNode]]
    """
    Builds the top level nodes of the sent statechart, among which, or among whose
    descendants, is a node of :attr:`node_type`.
    """

    world_fixture: WorldFixture = WorldFixture.MINI_WORLD
    """
    The fixture giving the world the nodes refer to.
    """

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        :param context: The context of the sender.
        :return: The statechart to send, holding a node of :attr:`node_type`.
        :raises NodeTypeMissingFromCaseError: If no node of the statechart is of
            :attr:`node_type`.
        """
        statechart = Statechart(context=context)
        statechart.add_nodes(self.build_nodes(context.world))
        if self.node_type not in {type(node) for node in statechart.nodes}:
            raise NodeTypeMissingFromCaseError(case=self)
        return statechart


def concrete_node_types(package: ModuleType) -> set[type[StatechartNode]]:
    """
    :param package: A package or module defining statechart nodes.
    :return: The node classes defined in `package` that can be instantiated.
    """
    if package.__spec__.submodule_search_locations is not None:
        for module in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
            importlib.import_module(module.name)
    return {
        node_type
        for node_type in recursive_subclasses(StatechartNode)
        if node_type.__module__.startswith(package.__name__)
        and not inspect.isabstract(node_type)
        and ABC not in node_type.__bases__
    }


# %% the tests every package runs over its own cases


class NodeRoundTripTests:
    """
    Sends every node class of a package through JSON.

    The module of a subclass gives the fixtures the tests read: ``case``, parametrized
    over its cases, ``cases``, ``node_types_needing_a_case``, ``sending_context`` and
    ``receiving_executor``, the last two on the world of ``case``.
    """

    @pytest.fixture()
    def round_trip(
        self,
        case: NodeRoundTripCase,
        sending_context: StatechartContext,
        receiving_executor: StatechartExecutor,
    ) -> StatechartSentThroughJSON:
        """
        :return: The statechart of `case`, sent to `receiving_executor`.

        .. note:: The sender releases what it registered in the world it shares with the
            receiver once its statechart is sent.
        """
        sent = case.build_statechart(sending_context)
        round_trip = StatechartSentThroughJSON.received_in(
            sent, receiving_executor.context
        )
        sending_context.cleanup()
        return round_trip

    def test_node_survives_json_round_trip(
        self, round_trip: StatechartSentThroughJSON
    ) -> None:
        """
        What a receiver read writes the same JSON as what was sent, so nothing a node
        serializes is lost or changed on the way.
        """
        assert round_trip.write_received_document() == round_trip.sent_document

    def test_node_received_from_json_compiles(
        self,
        round_trip: StatechartSentThroughJSON,
        receiving_executor: StatechartExecutor,
    ) -> None:
        receiving_executor.compile(round_trip.received)

        assert receiving_executor.statechart is round_trip.received

    def test_every_node_class_has_a_round_trip_case(
        self,
        cases: list[NodeRoundTripCase],
        node_types_needing_a_case: set[type[StatechartNode]],
    ) -> None:
        assert {case.node_type for case in cases} == node_types_needing_a_case
