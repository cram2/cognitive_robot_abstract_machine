"""
Every node class of cramph sent through JSON inside a statechart.
"""

from __future__ import annotations

import pytest

import cramph
from cramph import nodes_for_testing
from cramph.composites import (
    CancelledWhenTrue,
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
    Parallel,
    PausedUntilTrue,
    PausedWhileTrue,
    RepeatUntil,
    Sequence,
    StatementNode,
    StoppedWhenTrue,
    TryAll,
    TryInOrder,
)
from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.exceptions import NodeNotFoundError
from cramph.executor import StatechartExecutor
from cramph.monitors import (
    CheckTickCount,
    CountNodeResets,
    CountSeconds,
    CountSimulationTimeSeconds,
    CountTicks,
    Print,
    Pulse,
)
from cramph.node import (
    CancelStatechart,
    EndedByOwner,
    EndStatechart,
    StatechartNode,
    StructureCopyCompositeNode,
    StructureCopyNode,
    SucceedsOnObservingTrue,
)
from cramph.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
    NodeFailingOnObservingFalse,
)
from cramph.threaded_nodes import FunctionCall
from cramph.world_modification_nodes import MoveBranch
from krrood.entity_query_language.factories import a
from semantic_digital_twin.world import World

from .test_composite_node_choosing_its_child import ChooserAnsweringInTurn
from ...statechart_json_round_trip import (
    NodeRoundTripCase,
    NodeRoundTripTests,
    concrete_node_types,
)

# %% nodes of the cases


def _node_that_can_fail() -> NodeFailingOnObservingFalse:
    """
    :return: A node that can fail, as the nodes moving on after a failure need it.
    """
    return NodeFailingOnObservingFalse(observation=ObservationStateValues.TRUE)


def _context(world: World) -> StatechartContext:
    """
    :return: A context on `world` holding what the nodes of the cases read.
    """
    context = StatechartContext(world=world)
    context.add_extension(
        ChildChooserAccess(choosers=[ChooserAnsweringInTurn(children=[])])
    )
    return context


def _watched_node_and_its_reset_counter(world: World) -> list[StatechartNode]:
    watched = ConstTrueNode()
    return [watched, CountNodeResets(node=watched, target=2)]


def _cases() -> list[NodeRoundTripCase]:
    """
    :return: One case per node class of cramph.
    """
    return [
        NodeRoundTripCase(EndedByOwner, lambda world: [EndedByOwner()]),
        NodeRoundTripCase(
            SucceedsOnObservingTrue, lambda world: [SucceedsOnObservingTrue()]
        ),
        NodeRoundTripCase(StructureCopyNode, lambda world: [StructureCopyNode()]),
        NodeRoundTripCase(
            StructureCopyCompositeNode, lambda world: [StructureCopyCompositeNode()]
        ),
        NodeRoundTripCase(EndStatechart, lambda world: [EndStatechart()]),
        NodeRoundTripCase(
            CancelStatechart,
            lambda world: [
                never_true := ConstFalseNode(),
                CancelStatechart.when_true(
                    never_true, exception=NodeNotFoundError(name="node")
                ),
            ],
        ),
        NodeRoundTripCase(CheckTickCount, lambda world: [CheckTickCount(threshold=3)]),
        NodeRoundTripCase(Print, lambda world: [Print(message="message")]),
        NodeRoundTripCase(CountSeconds, lambda world: [CountSeconds(seconds=1.5)]),
        NodeRoundTripCase(
            CountSimulationTimeSeconds,
            lambda world: [CountSimulationTimeSeconds(seconds=1.5)],
        ),
        NodeRoundTripCase(CountTicks, lambda world: [CountTicks(ticks=3)]),
        NodeRoundTripCase(Pulse, lambda world: [Pulse(length=2)]),
        NodeRoundTripCase(CountNodeResets, _watched_node_and_its_reset_counter),
        NodeRoundTripCase(
            MoveBranch,
            lambda world: [
                MoveBranch(
                    branch_root=world.get_kinematic_structure_entity_by_name("tip"),
                    new_parent=world.get_kinematic_structure_entity_by_name("root"),
                )
            ],
        ),
        NodeRoundTripCase(
            Sequence, lambda world: [Sequence(nodes=[ConstTrueNode(), ConstTrueNode()])]
        ),
        NodeRoundTripCase(
            Parallel,
            lambda world: [
                Parallel(nodes=[ConstTrueNode(), ConstFalseNode()], minimum_success=1)
            ],
        ),
        NodeRoundTripCase(
            TryInOrder,
            lambda world: [
                TryInOrder(nodes=[_node_that_can_fail(), _node_that_can_fail()])
            ],
        ),
        NodeRoundTripCase(
            TryAll, lambda world: [TryAll(nodes=[ConstTrueNode(), ConstTrueNode()])]
        ),
        NodeRoundTripCase(
            CompositeNodeChoosingItsChild,
            lambda world: [CompositeNodeChoosingItsChild()],
        ),
        NodeRoundTripCase(
            StatementNode,
            lambda world: [StatementNode(statement=a(ConstTrueNode)())],
        ),
        NodeRoundTripCase(
            RepeatUntil,
            lambda world: [
                RepeatUntil(
                    task=_node_that_can_fail(), stop_retry_monitor=ConstFalseNode()
                )
            ],
        ),
        NodeRoundTripCase(
            PausedWhileTrue,
            lambda world: [
                PausedWhileTrue(
                    monitor=ConstFalseNode(), monitored_node=ConstTrueNode()
                )
            ],
        ),
        NodeRoundTripCase(
            PausedUntilTrue,
            lambda world: [
                PausedUntilTrue(monitor=ConstTrueNode(), monitored_node=ConstTrueNode())
            ],
        ),
        NodeRoundTripCase(
            StoppedWhenTrue,
            lambda world: [
                StoppedWhenTrue(
                    monitor=ConstFalseNode(), monitored_node=ConstTrueNode()
                )
            ],
        ),
        NodeRoundTripCase(
            CancelledWhenTrue,
            lambda world: [
                CancelledWhenTrue(
                    monitor=ConstFalseNode(),
                    monitored_node=ConstTrueNode(),
                    exception=NodeNotFoundError(name="node"),
                )
            ],
        ),
    ]


# %% sender and receiver


@pytest.fixture(params=_cases(), ids=lambda case: case.node_type.__name__)
def case(request: pytest.FixtureRequest) -> NodeRoundTripCase:
    return request.param


@pytest.fixture()
def cases() -> list[NodeRoundTripCase]:
    return _cases()


@pytest.fixture()
def node_types_needing_a_case() -> set[type[StatechartNode]]:
    """
    :return: The node classes of cramph, apart from those it gives tests to build on
        and those holding a Python callable, which cannot be sent.
    """
    return (
        concrete_node_types(cramph)
        - concrete_node_types(nodes_for_testing)
        - {FunctionCall}
    )


@pytest.fixture()
def sending_context(
    case: NodeRoundTripCase, request: pytest.FixtureRequest
) -> StatechartContext:
    return _context(request.getfixturevalue(case.world_fixture))


@pytest.fixture()
def receiving_executor(
    case: NodeRoundTripCase, request: pytest.FixtureRequest
) -> StatechartExecutor:
    return StatechartExecutor(_context(request.getfixturevalue(case.world_fixture)))


# %% round trips


class TestCramphNodeRoundTrips(NodeRoundTripTests):
    pass
