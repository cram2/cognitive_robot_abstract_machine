"""
Tests that every node has a framework that ticks it and every composite node a framework
that may change its children, and that a statechart lets only that framework change
them.
"""

from __future__ import annotations

import json

import pytest
from typing_extensions import Any

from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValue
from cramph.exceptions import ChildrenOwnedByAnotherFrameworkError
from cramph.node import ReadingFramework, StructureCopyCompositeNode
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from cramph.statechart import Statechart
from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.world import World

from ..dataset.frameworks import (
    Framework,
    SequenceThePerceptionTicks,
    StepTheMotionTicks,
)

# %% helpers


def through_json_string(*, document: dict[str, Any]) -> dict[str, Any]:
    """
    :return: `document` after a trip through a JSON string, as another process gets it.
    """
    return json.loads(json.dumps(document))


def statechart_of(*, framework: Framework | None) -> Statechart:
    """
    :return: An empty statechart that `framework` runs.
    """
    return Statechart(
        context=StatechartContext(world=World(), framework_name=framework)
    )


def step(*, name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A structure node, which succeeds once it observes True.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValue.TRUE
    )


def sequence_perception_ticks(
    *, structuring_framework: Framework | None = None
) -> SequenceThePerceptionTicks:
    """
    :return: A sequence of two structure nodes that the perception framework ticks.
    """
    return SequenceThePerceptionTicks(
        name="look around",
        nodes=[step(name="look left"), step(name="look right")],
        structuring_framework=structuring_framework,
    )


# %% who ticks a node


def test_a_structure_node_below_a_node_a_framework_ticks_has_no_tick_owner() -> None:
    """
    Every framework builds and ticks the structure nodes of cramph, wherever they sit.
    """
    sequence = sequence_perception_ticks()
    statechart_of(framework=Framework.PLANNING).add_node(node=sequence)

    assert [child.tick_owner for child in sequence.children] == [None, None]


def test_a_node_below_a_node_another_framework_ticks_keeps_the_framework_of_its_class() -> (
    None
):
    """
    The framework a node class names outranks the one of the parent.
    """
    sequence = SequenceThePerceptionTicks(
        name="look and reach",
        nodes=[
            step(name="look"),
            StepTheMotionTicks(name="reach", observation=ObservationStateValue.TRUE),
        ],
    )
    statechart_of(framework=Framework.PLANNING).add_node(node=sequence)

    assert [child.tick_owner for child in sequence.children] == [
        None,
        StepTheMotionTicks.ticking_framework,
    ]


def test_a_structure_node_below_a_node_another_framework_ticks_is_read_as_itself() -> (
    None
):
    """
    The reader holds a stand-in for the node it does not tick, and the structure nodes
    below it.
    """
    statechart = statechart_of(framework=Framework.PLANNING)
    statechart.add_node(node=sequence_perception_ticks())

    read = Statechart.from_json(
        data=through_json_string(document=statechart.to_json()),
        context=StatechartContext(world=World(), framework_name=Framework.MOTION),
        **ReadingFramework(framework_name=Framework.MOTION).create_kwargs(),
    )

    assert [type(node) for node in read.nodes] == [
        StructureCopyCompositeNode,
        NodeSucceedingOnObservingTrue,
        NodeSucceedingOnObservingTrue,
    ]


# %% who may change the children of a composite node


def test_a_composite_node_naming_no_structuring_framework_is_owned_by_its_tick_owner() -> (
    None
):
    """
    The framework that ticks a composite node may change its children.
    """
    sequence = sequence_perception_ticks()

    assert sequence.structure_owner == sequence.ticking_framework


def test_a_composite_node_naming_a_structuring_framework_is_owned_by_it() -> None:
    """
    The framework holding what a choice is made from may change the children of a node
    another framework ticks.
    """
    sequence = sequence_perception_ticks(structuring_framework=Framework.PLANNING)

    assert sequence.structure_owner == Framework.PLANNING


def test_the_structure_owner_survives_a_trip_through_json() -> None:
    """
    A framework reading the node knows who may change its children.
    """
    sequence = sequence_perception_ticks(structuring_framework=Framework.PLANNING)
    statechart_of(framework=Framework.PLANNING).add_node(node=sequence)

    read = from_json(
        through_json_string(document=to_json(sequence)),
        **ReadingFramework(framework_name=Framework.PERCEPTION).create_kwargs(),
    )

    assert read.structure_owner == sequence.structure_owner


def test_a_framework_may_not_change_the_children_of_a_node_another_framework_owns() -> (
    None
):
    """
    The motion framework cannot add a step to a sequence the perception framework ticks.
    """
    sequence = sequence_perception_ticks()
    statechart = statechart_of(framework=Framework.MOTION)
    statechart.add_node(node=sequence)

    with pytest.raises(ChildrenOwnedByAnotherFrameworkError):
        sequence.add_node(node=step(name="look up"))


def test_a_refused_change_of_children_leaves_the_statechart_usable() -> None:
    """
    The change is refused before anything changed, so the statechart is not broken.
    """
    sequence = sequence_perception_ticks()
    statechart = statechart_of(framework=Framework.MOTION)
    statechart.add_node(node=sequence)

    with pytest.raises(ChildrenOwnedByAnotherFrameworkError):
        sequence.add_node(node=step(name="look up"))

    statechart.check_not_broken()
    assert [child.name for child in sequence.children] == ["look left", "look right"]


def test_the_owning_framework_may_change_the_children() -> None:
    """
    The perception framework adds a step to a sequence it ticks.
    """
    sequence = sequence_perception_ticks()
    statechart = statechart_of(framework=Framework.PERCEPTION)
    statechart.add_node(node=sequence)

    sequence.add_node(node=(looking_up := step(name="look up")))

    assert sequence.children[-1] is looking_up


def test_any_framework_may_change_the_children_of_a_node_no_framework_owns() -> None:
    """
    A node naming no framework is changed as before.
    """
    sequence = Sequence(name="steps", nodes=[step(name="first")])
    statechart = statechart_of(framework=Framework.MOTION)
    statechart.add_node(node=sequence)

    sequence.add_node(node=(second := step(name="second")))

    assert sequence.children[-1] is second


def test_a_statechart_naming_no_framework_may_change_the_children_of_every_node() -> (
    None
):
    """
    A statechart that runs every node in its own process changes them as before.
    """
    sequence = sequence_perception_ticks()
    statechart = statechart_of(framework=None)
    statechart.add_node(node=sequence)

    sequence.add_node(node=(looking_up := step(name="look up")))

    assert sequence.children[-1] is looking_up


def test_a_framework_adds_a_node_another_framework_owns_with_its_children() -> None:
    """
    The framework sending a statechart builds every node in it, including the children a
    node takes while it joins.
    """
    sequence = sequence_perception_ticks()

    statechart_of(framework=Framework.PLANNING).add_node(node=sequence)

    assert [child.name for child in sequence.children] == ["look left", "look right"]


def test_a_framework_follows_the_change_of_children_the_owner_made() -> None:
    """
    A framework that does not own a node takes over the changes its owner made to it.
    """
    owner = statechart_of(framework=Framework.PERCEPTION)
    owner.add_node(node=(sequence := sequence_perception_ticks()))
    sequence.add_node(node=step(name="look up"))
    follower = statechart_of(framework=Framework.MOTION)

    for block in owner.structure_change_recorder.change_blocks:
        follower.apply_change_block_from_json(
            data=through_json_string(document=block.to_json()),
            **ReadingFramework(framework_name=Framework.MOTION).create_kwargs(),
        )

    assert [child.name for child in follower.nodes[0].children] == [
        "look left",
        "look right",
        "look up",
    ]
