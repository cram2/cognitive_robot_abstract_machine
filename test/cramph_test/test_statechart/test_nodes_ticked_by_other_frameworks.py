"""
Tests that a framework reading a statechart builds only the nodes it ticks itself, and
a stand-in for every node another framework ticks, without importing that node's class.
"""

from __future__ import annotations

import json
from enum import StrEnum

from typing_extensions import Any

from cramph.composites import CompositeNodeChoosingItsChild, StatementNode
from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValue, TransitionKind
from cramph.node import (
    NodeJSONKey,
    ReadingFramework,
    StatechartNode,
    StructureCopyCompositeNode,
    StructureCopyNode,
)
from cramph.nodes_for_testing import (
    CompositeNodeWithChainedChildren,
    ConstTrueNode,
    NodeObservingTheOppositeOfAnObservationPredicate,
    NodeSucceedingOnObservingTrue,
)
from cramph.statechart import Statechart
from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import from_json, to_json
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.world import World

# %% mimics


class Framework(StrEnum):
    """
    The frameworks sharing the statecharts of these tests.
    """

    PERCEPTION = "perception"
    """
    Ticks the nodes that look at the scene.
    """

    MOTION = "motion"
    """
    Ticks the nodes that move the robot.
    """


# %% helpers


def through_json_string(*, document: dict[str, Any]) -> dict[str, Any]:
    """
    :return: `document` after a trip through a JSON string, as another process gets it.
    """
    return json.loads(json.dumps(document))


def reading_kwargs(*, framework: Framework) -> dict[str, Any]:
    """
    :return: The keyword arguments with which `framework` reads a document.
    """
    return ReadingFramework(framework_name=framework).create_kwargs()


def read_node(*, node: StatechartNode, framework: Framework) -> StatechartNode:
    """
    :return: `node` as `framework` reads it from another process.
    """
    return from_json(
        through_json_string(document=to_json(node)),
        **reading_kwargs(framework=framework),
    )


def node_looking_at_the_scene() -> NodeSucceedingOnObservingTrue:
    """
    :return: A node the perception framework ticks.
    """
    return NodeSucceedingOnObservingTrue(
        name="look at the scene",
        observation=ObservationStateValue.TRUE,
        ticking_framework=Framework.PERCEPTION,
    )


def node_reaching() -> NodeSucceedingOnObservingTrue:
    """
    :return: A node the motion framework ticks.
    """
    return NodeSucceedingOnObservingTrue(
        name="reach",
        observation=ObservationStateValue.TRUE,
        ticking_framework=Framework.MOTION,
    )


def empty_statechart() -> Statechart:
    """
    :return: A statechart without nodes, in a context of its own.
    """
    return Statechart(context=StatechartContext(world=World()))


def read_statechart(*, statechart: Statechart, framework: Framework) -> Statechart:
    """
    :return: `statechart` as `framework` reads it from another process.
    """
    return Statechart.from_json(
        data=through_json_string(document=statechart.to_json()),
        context=StatechartContext(world=World()),
        **reading_kwargs(framework=framework),
    )


def statechart_with_a_perception_node_that_chose_its_child() -> Statechart:
    """
    :return: A statechart holding a node the perception framework ticks, which chose
        one child.
    """
    statechart = empty_statechart()
    choosing = CompositeNodeChoosingItsChild(
        name="choose a view", ticking_framework=Framework.PERCEPTION
    )
    statechart.add_node(node=choosing)
    with statechart.modify():
        choosing.adopt_chosen_child(child=ConstTrueNode(name="look left"))
    return statechart


def statement_of_one_node() -> Match[NodeSucceedingOnObservingTrue]:
    """
    :return: A statement with a single candidate.
    """
    return a(NodeSucceedingOnObservingTrue)(
        name=variable(str, domain=["look"]), observation=ObservationStateValue.TRUE
    )


# %% a single node


def test_a_node_another_framework_ticks_is_read_as_a_stand_in() -> None:
    """
    The motion framework does not build a node the perception framework ticks.
    """
    node = node_looking_at_the_scene()
    read = read_node(node=node, framework=Framework.MOTION)
    assert type(read) is StructureCopyNode


def test_a_stand_in_keeps_the_id_name_and_framework_of_its_node() -> None:
    """
    Documents naming the node by its id find the stand-in, and the stand-in still says
    which framework ticks the node.
    """
    node = node_looking_at_the_scene()
    read = read_node(node=node, framework=Framework.MOTION)
    assert (read._node_id, read.name, read.ticking_framework) == (
        node._node_id,
        node.name,
        node.ticking_framework,
    )


def test_a_stand_in_keeps_the_plot_specifications_of_its_node() -> None:
    """
    A plot of the reading framework's statechart draws the stand-in like its node.
    """
    node = node_looking_at_the_scene()
    read = read_node(node=node, framework=Framework.MOTION)
    assert read.plot_specifications == node.plot_specifications


def test_a_composite_node_another_framework_ticks_is_read_as_a_composite_stand_in() -> (
    None
):
    """
    A stand-in of a composite node can hold the children the document gives it.
    """
    node = CompositeNodeWithChainedChildren(
        name="search the shelf", ticking_framework=Framework.PERCEPTION
    )
    empty_statechart().add_node(node=node)
    read = read_node(node=node, framework=Framework.MOTION)
    assert type(read) is StructureCopyCompositeNode


def test_the_class_of_a_node_another_framework_ticks_is_never_imported() -> None:
    """
    Reading succeeds although the class of the node lives in a module that does not
    exist in the reading process.
    """
    document = through_json_string(document=to_json(node_looking_at_the_scene()))
    document[NodeJSONKey.TICKED_NODE][JSONField.TYPE] = (
        "a_framework_this_process_never_imports.nodes.LookAtTheScene"
    )
    read = from_json(document, **reading_kwargs(framework=Framework.MOTION))
    assert type(read) is StructureCopyNode


def test_a_node_the_reading_framework_ticks_is_read_as_itself() -> None:
    """
    The framework that ticks a node builds it with its own class and fields.
    """
    node = node_reaching()
    read = read_node(node=node, framework=Framework.MOTION)
    assert (type(read), read.observation, read._node_id, read.ticking_framework) == (
        type(node),
        node.observation,
        node._node_id,
        node.ticking_framework,
    )


def test_a_node_no_framework_ticks_is_read_as_itself_by_every_framework() -> None:
    """
    A node without a ticking framework is read as before.
    """
    node = NodeSucceedingOnObservingTrue(
        name="shared", observation=ObservationStateValue.TRUE
    )
    read = read_node(node=node, framework=Framework.PERCEPTION)
    assert type(read) is type(node)


def test_a_reader_naming_no_framework_reads_a_stand_in() -> None:
    """
    Without a :class:`~cramph.node.ReadingFramework`, no framework-ticked node is built.
    """
    read = from_json(
        through_json_string(document=to_json(node_reaching())),
    )
    assert type(read) is StructureCopyNode


def test_a_stand_in_child_keeps_its_parent() -> None:
    """
    A stand-in is a child of the node its node is a child of, not a top level node.
    """
    statechart = empty_statechart()
    parent = CompositeNodeWithChainedChildren(name="search the shelf")
    statechart.add_node(node=parent)
    parent.children[0].ticking_framework = Framework.PERCEPTION
    read_parent, read_stand_in, _ = read_statechart(
        statechart=statechart, framework=Framework.MOTION
    ).nodes
    assert read_stand_in.parent_node is read_parent


# %% a statechart


def test_a_condition_reading_a_node_another_framework_ticks_reads_its_stand_in() -> (
    None
):
    """
    A node the reader ticks can start once a node of another framework succeeded.
    """
    statechart = empty_statechart()
    looking = node_looking_at_the_scene()
    reaching = node_reaching()
    statechart.add_nodes(nodes=[looking, reaching])
    reaching.start_condition = looking.is_succeeded
    read = Statechart.from_json(
        data=through_json_string(document=statechart.to_json()),
        context=StatechartContext(world=World()),
        **reading_kwargs(framework=Framework.MOTION),
    )
    read_looking, read_reaching = read.nodes
    start_condition = next(
        condition
        for condition in read_reaching.conditions
        if condition.kind == TransitionKind.START
    )
    assert start_condition.node_dependencies == [read_looking]


def test_a_node_another_framework_ticks_joins_through_a_change_block_as_a_stand_in() -> (
    None
):
    """
    Change blocks deliver nodes of other frameworks as stand-ins too.
    """
    sender = empty_statechart()
    sender.add_node(node=node_reaching())
    sender.add_node(node=node_looking_at_the_scene())
    receiver = empty_statechart()
    for block in sender.structure_change_recorder.change_blocks:
        receiver.apply_change_block_from_json(
            data=through_json_string(document=block.to_json()),
            **reading_kwargs(framework=Framework.MOTION),
        )
    assert [type(node) for node in receiver.nodes] == [
        NodeSucceedingOnObservingTrue,
        StructureCopyNode,
    ]


def test_a_node_referring_to_a_node_another_framework_ticks_refers_to_its_stand_in() -> (
    None
):
    """
    A node read before the node it refers to shares the stand-in later read for it.
    """
    statechart = empty_statechart()
    looking = node_looking_at_the_scene()
    watching = NodeObservingTheOppositeOfAnObservationPredicate(
        name="watch the scene",
        watched_node=looking,
        ticking_framework=Framework.MOTION,
    )
    statechart.add_nodes(nodes=[watching, looking])
    read = Statechart.from_json(
        data=through_json_string(document=statechart.to_json()),
        context=StatechartContext(world=World()),
        **reading_kwargs(framework=Framework.MOTION),
    )
    read_watching, read_looking = read.nodes
    assert read_watching.watched_node is read_looking


# %% a node choosing its child


def test_a_node_choosing_its_child_another_framework_ticks_is_read_as_a_stand_in() -> (
    None
):
    """
    The chosen children of a node another framework ticks are not restored on its
    stand-in, which chooses nothing.
    """
    read = read_statechart(
        statechart=statechart_with_a_perception_node_that_chose_its_child(),
        framework=Framework.MOTION,
    )
    assert [type(node) for node in read.nodes] == [
        StructureCopyCompositeNode,
        ConstTrueNode,
    ]


def test_a_node_choosing_its_child_joins_through_a_change_block_as_a_stand_in() -> None:
    """
    Change blocks naming the chosen children of a stand-in apply too.
    """
    sender = statechart_with_a_perception_node_that_chose_its_child()
    receiver = empty_statechart()
    for block in sender.structure_change_recorder.change_blocks:
        receiver.apply_change_block_from_json(
            data=through_json_string(document=block.to_json()),
            **reading_kwargs(framework=Framework.MOTION),
        )
    assert [type(node) for node in receiver.nodes] == [
        StructureCopyCompositeNode,
        ConstTrueNode,
    ]


def test_a_node_choosing_its_child_the_reading_framework_ticks_keeps_its_choice() -> (
    None
):
    """
    The framework ticking a node choosing its child gets the children it chose.
    """
    read_choosing, read_child = read_statechart(
        statechart=statechart_with_a_perception_node_that_chose_its_child(),
        framework=Framework.PERCEPTION,
    ).nodes
    assert read_choosing.chosen_children == [read_child]


def test_a_statement_another_framework_ticks_is_read_as_a_stand_in() -> None:
    """
    A framework not ticking a statement neither grounds it nor chooses its children.
    """
    node = StatementNode(
        statement=statement_of_one_node(), ticking_framework=Framework.PERCEPTION
    )
    read = read_node(node=node, framework=Framework.MOTION)
    assert type(read) is StructureCopyCompositeNode


def test_a_statement_the_reading_framework_ticks_is_read_as_a_node_choosing_its_child() -> (
    None
):
    """
    The framework ticking a statement receives the children it chose rather than the
    statement, as it does for a statement no framework ticks.
    """
    node = StatementNode(
        statement=statement_of_one_node(), ticking_framework=Framework.PERCEPTION
    )
    read = read_node(node=node, framework=Framework.PERCEPTION)
    assert type(read) is CompositeNodeChoosingItsChild
